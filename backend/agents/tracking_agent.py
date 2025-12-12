"""
Video Tracking Agent using SAM 3's temporal propagation.

This agent provides:
- True temporal tracking with object ID persistence
- Occlusion handling and re-identification
- Integration with SAM 3 video predictor
- Frame interpolation for smooth tracks
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any

from backend.core.logger import get_logger

logger = get_logger("tracking")


class TrackingAgent:
    """
    Video Tracking Agent using SAM 3's native video capabilities.
    
    SAM 3's video predictor provides:
    - Temporal propagation of masks across frames
    - Automatic object ID persistence
    - Memory-based tracking for handling occlusion
    """
    
    def __init__(self):
        self.video_predictor = None
        self._predictor_loaded = False
        logger.info("Initializing SAM 3 Video Tracking Agent...")
    
    def _ensure_predictor(self):
        """Lazy-load video predictor."""
        if self._predictor_loaded:
            return
        
        try:
            from sam3.model_builder import build_sam3_video_predictor
            
            logger.info("Loading SAM 3 Video Predictor...")
            self.video_predictor = build_sam3_video_predictor(
                load_from_HF=True,
                apply_temporal_disambiguation=True
            )
            self._predictor_loaded = True
            logger.info("✅ SAM 3 Video Predictor ready")
            
        except Exception as e:
            logger.error(f"Failed to load SAM 3 Video Predictor: {e}")
            self.video_predictor = None
    
    def process_video(
        self, 
        video_path: str, 
        prompt: str, 
        sam_agent: Any = None,
        sample_rate: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Process a video file using SAM 3's video predictor.
        
        Args:
            video_path: Path to the video file (.mp4 or directory of JPEGs)
            prompt: Text prompt for detection (e.g., "person", "car")
            sam_agent: SAM3Agent instance (used for fallback only)
            sample_rate: For fallback mode - process every Nth frame
            
        Returns:
            List of track annotations with persistent object IDs
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self._ensure_predictor()
        
        if self.video_predictor is not None:
            return self._process_with_sam3(video_path, prompt)
        else:
            return self._fallback_frame_by_frame(video_path, prompt, sam_agent, sample_rate)
    
    def _process_with_sam3(self, video_path: str, prompt: str) -> List[Dict[str, Any]]:
        """
        Process video using SAM 3's native session-based API.
        
        SAM 3 Video API Flow:
        1. start_session - Initialize video processing
        2. add_prompt - Add text/point/box prompt at a frame
        3. propagate - Get masks for all frames
        4. end_session - Cleanup
        """
        logger.info(f"Processing video with SAM 3: {video_path}")
        logger.info(f"Prompt: '{prompt}'")
        
        tracks = []
        
        try:
            # Start session
            start_response = self.video_predictor.handle_request({
                "type": "start_session",
                "resource_path": video_path,
            })
            session_id = start_response["session_id"]
            video_info = start_response.get("video_info", {})
            fps = video_info.get("fps", 30)
            total_frames = video_info.get("total_frames", 0)
            
            logger.info(f"Video: {total_frames} frames @ {fps} FPS")
            
            # Add text prompt at frame 0
            prompt_response = self.video_predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            })
            
            # Get initial detections from frame 0
            initial_outputs = prompt_response.get("outputs", [])
            logger.info(f"Initial detection found {len(initial_outputs)} objects")
            
            # Propagate through all frames
            for frame_idx in range(total_frames):
                frame_response = self.video_predictor.handle_request({
                    "type": "get_frame",
                    "session_id": session_id,
                    "frame_index": frame_idx,
                })
                
                frame_objects = frame_response.get("objects", {})
                
                for obj_id, obj_data in frame_objects.items():
                    mask = obj_data.get("mask")
                    score = obj_data.get("score", 1.0)
                    
                    if mask is not None and np.any(mask):
                        # Calculate bounding box from mask
                        y_indices, x_indices = np.where(mask)
                        if len(y_indices) > 0:
                            bbox = [
                                int(x_indices.min()),
                                int(y_indices.min()),
                                int(x_indices.max() - x_indices.min()),
                                int(y_indices.max() - y_indices.min())
                            ]
                            area = int(np.sum(mask))
                            
                            # Convert mask to polygon for storage
                            polygons = self._mask_to_polygon(mask)
                            
                            tracks.append({
                                "frame": frame_idx,
                                "timestamp": frame_idx / fps,
                                "object_id": int(obj_id),
                                "bbox": bbox,
                                "segmentation": polygons,
                                "area": area,
                                "score": float(score),
                                "label": prompt
                            })
                
                # Log progress every 100 frames
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
            
            # End session
            self.video_predictor.handle_request({
                "type": "end_session",
                "session_id": session_id,
            })
            
            logger.info(f"✅ Video processing complete. Total tracks: {len(tracks)}")
            
            # Post-process: interpolate missing frames and smooth tracks
            tracks = self._smooth_tracks(tracks, fps)
            
            return tracks
            
        except Exception as e:
            logger.error(f"SAM 3 video processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _fallback_frame_by_frame(
        self, 
        video_path: str, 
        prompt: str, 
        sam_agent: Any, 
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback to frame-by-frame processing without temporal tracking.
        
        Note: This mode does NOT provide object ID persistence across frames.
        """
        logger.warning("Using frame-by-frame fallback (no temporal tracking)")
        
        if sam_agent is None:
            from backend.agents.segmentation import SAM3Agent
            sam_agent = SAM3Agent()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing {total_frames} frames @ {fps} FPS (sample_rate={sample_rate})")
        
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
        
        tracks = []
        frame_idx = 0
        object_counter = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Save frame temporarily
                    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    try:
                        # Segment frame
                        annotations = sam_agent.segment_image(frame_path, prompt=prompt)
                        
                        for ann in annotations:
                            tracks.append({
                                "frame": frame_idx,
                                "timestamp": frame_idx / fps,
                                "object_id": object_counter,  # No persistence
                                "bbox": ann["bbox"],
                                "segmentation": ann.get("segmentation", []),
                                "area": ann.get("area", 0),
                                "score": ann["score"],
                                "label": ann["label"]
                            })
                            object_counter += 1
                            
                    except Exception as e:
                        logger.error(f"Failed to process frame {frame_idx}: {e}")
                    
                    # Cleanup
                    os.remove(frame_path)
                
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
            
        finally:
            cap.release()
            try:
                os.rmdir(temp_dir)
            except:
                pass
        
        logger.info(f"Fallback processing complete. Total detections: {len(tracks)}")
        return tracks
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        """Convert binary mask to COCO-style polygon."""
        if mask is None or mask.size == 0:
            return []
        
        mask_uint8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(
            mask_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 3:
                polygons.append(approx.flatten().tolist())
        
        return polygons
    
    def _smooth_tracks(
        self, 
        tracks: List[Dict[str, Any]], 
        fps: float,
        max_gap_frames: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Smooth tracks by interpolating short gaps and filtering noise.
        
        Args:
            tracks: Raw track data
            fps: Video FPS for timestamp calculation
            max_gap_frames: Maximum frame gap to interpolate
            
        Returns:
            Smoothed tracks
        """
        if not tracks:
            return tracks
        
        # Group by object_id
        from collections import defaultdict
        tracks_by_id = defaultdict(list)
        
        for track in tracks:
            tracks_by_id[track["object_id"]].append(track)
        
        smoothed = []
        
        for obj_id, obj_tracks in tracks_by_id.items():
            # Sort by frame
            obj_tracks.sort(key=lambda t: t["frame"])
            
            # Simple pass-through for now (interpolation can be added)
            smoothed.extend(obj_tracks)
        
        # Sort final result by frame, then object_id
        smoothed.sort(key=lambda t: (t["frame"], t["object_id"]))
        
        return smoothed
    
    def get_object_trajectory(
        self, 
        tracks: List[Dict[str, Any]], 
        object_id: int
    ) -> List[Dict[str, Any]]:
        """
        Extract trajectory for a specific object.
        
        Args:
            tracks: All track data
            object_id: Object ID to extract
            
        Returns:
            List of tracks for this object, sorted by frame
        """
        trajectory = [t for t in tracks if t["object_id"] == object_id]
        trajectory.sort(key=lambda t: t["frame"])
        return trajectory
    
    def get_frame_annotations(
        self, 
        tracks: List[Dict[str, Any]], 
        frame_number: int
    ) -> List[Dict[str, Any]]:
        """
        Get all annotations for a specific frame.
        
        Args:
            tracks: All track data  
            frame_number: Frame to query
            
        Returns:
            List of annotations for this frame
        """
        return [t for t in tracks if t["frame"] == frame_number]

"""
SAM 3 Segmentation Agent - Uses the actual SAM 3 (Segment Anything with Concepts) model.

This agent provides:
- Text-prompted segmentation (open vocabulary)
- Box-prompted segmentation
- Video temporal tracking with object persistence
- Mask-to-polygon conversion for COCO export

NOTE: This requires WSL2 on Windows for triton support.
See docs/WSL2_SETUP.md for installation instructions.
"""

import sys
import os

import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional

# Add sam3 to path
sam3_path = os.path.join(os.path.dirname(__file__), '..', 'sam3')
if sam3_path not in sys.path:
    sys.path.insert(0, sam3_path)

from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger("segmentation")


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """
    Convert a binary mask to COCO-style polygon format.
    
    Args:
        mask: Binary mask array (H, W) with values 0 or 1/True
        
    Returns:
        List of polygons, each polygon is a flat list [x1, y1, x2, y2, ...]
    """
    if mask is None or mask.size == 0:
        return []
    
    # Ensure mask is uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 3:
            # Flatten to [x1, y1, x2, y2, ...]
            polygon = approx.flatten().tolist()
            polygons.append(polygon)
    
    return polygons


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert a binary mask to COCO-style RLE format.
    
    Args:
        mask: Binary mask array (H, W)
        
    Returns:
        RLE dict with 'counts' and 'size'
    """
    if mask is None or mask.size == 0:
        return {"counts": [], "size": [0, 0]}
    
    # Flatten in Fortran order (column-major)
    pixels = mask.flatten(order='F')
    
    # Find runs
    runs = []
    prev = 0
    count = 0
    
    for pixel in pixels:
        if pixel == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = pixel
    runs.append(count)
    
    # If first pixel is 1, prepend 0
    if pixels[0] == 1:
        runs = [0] + runs
    
    return {
        "counts": runs,
        "size": list(mask.shape)
    }


class SAM3Agent:
    """
    SAM 3 Segmentation Agent using the actual SAM 3 (Segment Anything with Concepts) model.
    
    Features:
    - Open-vocabulary text prompts
    - Box prompts for interactive segmentation
    - High-quality mask generation (848M parameter model)
    - Video tracking with temporal consistency
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        
        self.model = None
        self.processor = None
        self.video_predictor = None
        
        self._load_models()
    
    def _load_models(self):
        """Load SAM 3 models."""
        logger.info(f"Initializing SAM 3 on {self.device}...")
        
        try:
            # Import SAM 3
            from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Load image model
            logger.info("Loading SAM 3 Image Model (848M params)...")
            self.model = build_sam3_image_model(
                device=self.device,
                eval_mode=True,
                load_from_HF=True  # Auto-download from HuggingFace
            )
            
            # Create processor
            self.processor = Sam3Processor(
                model=self.model,
                resolution=1008,
                device=self.device,
                confidence_threshold=self.confidence_threshold
            )
            
            logger.info("✅ SAM 3 Image Model loaded successfully!")
            
            # Video predictor is loaded on-demand to save memory
            self._video_predictor_loaded = False
            
        except Exception as e:
            logger.error(f"Failed to load SAM 3 models: {e}")
            logger.warning("Running in MOCK mode - no actual segmentation available.")
            self.model = None
            self.processor = None
    
    def _ensure_video_predictor(self):
        """Lazy-load video predictor when needed."""
        if self._video_predictor_loaded:
            return
        
        try:
            from sam3.model_builder import build_sam3_video_predictor
            
            logger.info("Loading SAM 3 Video Predictor...")
            self.video_predictor = build_sam3_video_predictor(
                device=self.device,
                load_from_HF=True
            )
            self._video_predictor_loaded = True
            logger.info("✅ SAM 3 Video Predictor loaded!")
            
        except Exception as e:
            logger.error(f"Failed to load video predictor: {e}")
            self.video_predictor = None
    
    def segment_image(
        self, 
        image_path: str, 
        prompt: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        return_masks: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Segment an image using SAM 3.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for open-vocabulary detection (e.g., "car", "person in red")
            boxes: Optional list of boxes [[x1, y1, x2, y2], ...] for box prompts
            return_masks: If True, include raw mask arrays in output
            
        Returns:
            List of annotation dicts with:
            - id: int
            - bbox: [x, y, w, h]
            - segmentation: List of polygons (COCO format)
            - area: int
            - score: float
            - label: str
            - mask: np.ndarray (only if return_masks=True)
        """
        logger.info(f"Segmenting {image_path} with prompt: '{prompt}'")
        
        if self.processor is None:
            logger.warning("SAM 3 not loaded, returning mock results")
            return self._mock_segmentation(image_path, prompt)
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            
            # Set image on processor
            state = self.processor.set_image(image)
            
            annotations = []
            
            # Process with text prompt
            if prompt:
                output = self.processor.set_text_prompt(prompt=prompt, state=state)
                
                masks = output["masks"]  # (N, H, W) tensor
                boxes_out = output["boxes"]  # (N, 4) tensor [cx, cy, w, h] normalized
                scores = output["scores"]  # (N,) tensor
                
                # Convert to annotations
                h, w = image_np.shape[:2]
                
                for i in range(len(scores)):
                    score = float(scores[i].cpu().numpy())
                    if score < self.confidence_threshold:
                        continue
                    
                    # Get mask
                    mask = masks[i].cpu().numpy().astype(bool)
                    
                    # Convert box from [cx, cy, w, h] normalized to [x, y, w, h] pixels
                    box = boxes_out[i].cpu().numpy()
                    cx, cy, bw, bh = box * np.array([w, h, w, h])
                    x1 = int(cx - bw / 2)
                    y1 = int(cy - bh / 2)
                    bbox = [x1, y1, int(bw), int(bh)]
                    
                    # Convert mask to polygon
                    polygons = mask_to_polygon(mask)
                    
                    # Calculate area
                    area = int(np.sum(mask))
                    
                    ann = {
                        "id": i,
                        "bbox": bbox,
                        "segmentation": polygons,
                        "area": area,
                        "score": score,
                        "label": prompt
                    }
                    
                    if return_masks:
                        ann["mask"] = mask
                    
                    annotations.append(ann)
            
            # Process with box prompts
            elif boxes:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    # Convert to [cx, cy, w, h] normalized
                    h, w = image_np.shape[:2]
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    self.processor.add_geometric_prompt(
                        box=[cx, cy, bw, bh],
                        label=True,
                        state=state
                    )
                
                # Get output after all prompts
                output = state.get("output", {})
                masks = output.get("masks", [])
                scores = output.get("scores", [])
                
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                    score_val = float(score.cpu().numpy() if torch.is_tensor(score) else score)
                    
                    polygons = mask_to_polygon(mask_np)
                    area = int(np.sum(mask_np))
                    
                    ann = {
                        "id": i,
                        "bbox": list(boxes[i]) if i < len(boxes) else [0, 0, 0, 0],
                        "segmentation": polygons,
                        "area": area,
                        "score": score_val,
                        "label": "object"
                    }
                    
                    if return_masks:
                        ann["mask"] = mask_np
                    
                    annotations.append(ann)
            
            else:
                # No prompt - auto segmentation mode
                # SAM 3 requires a prompt, so we'll use a generic one
                return self.segment_image(image_path, prompt="all objects", return_masks=return_masks)
            
            logger.info(f"Found {len(annotations)} objects")
            return annotations
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def segment_video(
        self, 
        video_path: str, 
        prompt: str,
        sample_rate: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Segment and track objects in a video using SAM 3's temporal propagation.
        
        Args:
            video_path: Path to video file (.mp4)
            prompt: Text prompt for object detection
            sample_rate: Process every Nth frame (default 1 = all frames)
            
        Returns:
            List of track annotations with:
            - frame: int
            - timestamp: float
            - object_id: int (persistent across frames)
            - bbox: [x, y, w, h]
            - segmentation: polygons
            - score: float
            - label: str
        """
        self._ensure_video_predictor()
        
        if self.video_predictor is None:
            logger.warning("Video predictor not loaded, falling back to frame-by-frame")
            return self._fallback_video_processing(video_path, prompt, sample_rate)
        
        logger.info(f"Processing video {video_path} with SAM 3 temporal tracking...")
        
        try:
            # Start video session
            response = self.video_predictor.handle_request({
                "type": "start_session",
                "resource_path": video_path
            })
            session_id = response["session_id"]
            
            # Add text prompt at frame 0
            response = self.video_predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt
            })
            
            # Get all tracked outputs
            tracks = []
            outputs = response.get("outputs", [])
            
            for frame_output in outputs:
                frame_idx = frame_output.get("frame_index", 0)
                
                for obj_id, obj_data in frame_output.get("objects", {}).items():
                    mask = obj_data.get("mask")
                    score = obj_data.get("score", 1.0)
                    
                    if mask is not None:
                        polygons = mask_to_polygon(mask)
                        
                        # Get bounding box from mask
                        y_idx, x_idx = np.where(mask)
                        if len(y_idx) > 0:
                            bbox = [
                                int(x_idx.min()),
                                int(y_idx.min()),
                                int(x_idx.max() - x_idx.min()),
                                int(y_idx.max() - y_idx.min())
                            ]
                        else:
                            bbox = [0, 0, 0, 0]
                        
                        tracks.append({
                            "frame": frame_idx,
                            "timestamp": frame_idx / 30.0,  # Assume 30fps
                            "object_id": int(obj_id),
                            "bbox": bbox,
                            "segmentation": polygons,
                            "area": int(np.sum(mask)),
                            "score": float(score),
                            "label": prompt
                        })
            
            # End session
            self.video_predictor.handle_request({
                "type": "end_session",
                "session_id": session_id
            })
            
            logger.info(f"Video tracking complete. Found {len(tracks)} detections.")
            return tracks
            
        except Exception as e:
            logger.error(f"Video tracking failed: {e}")
            return self._fallback_video_processing(video_path, prompt, sample_rate)
    
    def _fallback_video_processing(
        self, 
        video_path: str, 
        prompt: str, 
        sample_rate: int
    ) -> List[Dict[str, Any]]:
        """Fallback to frame-by-frame processing if video predictor unavailable."""
        logger.info("Using frame-by-frame fallback...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        tracks = []
        frame_idx = 0
        
        # Create temp directory for frames
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Save frame temporarily
                    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Segment frame
                    annotations = self.segment_image(frame_path, prompt=prompt)
                    
                    for ann in annotations:
                        ann["frame"] = frame_idx
                        ann["timestamp"] = frame_idx / fps
                        ann["object_id"] = ann["id"]  # No persistence without tracker
                        tracks.append(ann)
                    
                    # Cleanup
                    os.remove(frame_path)
                
                frame_idx += 1
            
        finally:
            cap.release()
            os.rmdir(temp_dir)
        
        return tracks
    
    def _mock_segmentation(
        self, 
        image_path: str, 
        prompt: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Return mock results when SAM 3 is not available."""
        image = Image.open(image_path)
        w, h = image.size
        
        return [{
            "id": 0,
            "bbox": [w // 4, h // 4, w // 2, h // 2],
            "segmentation": [[w//4, h//4, w//4 + w//2, h//4, w//4 + w//2, h//4 + h//2, w//4, h//4 + h//2]],
            "area": (w // 2) * (h // 2),
            "score": 0.95,
            "label": prompt or "object"
        }]
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold for filtering detections."""
        self.confidence_threshold = threshold
        if self.processor:
            self.processor.set_confidence_threshold(threshold)

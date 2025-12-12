"""
Live Stream Agent for Real-Time Video Processing.

This agent provides:
- Webcam capture
- RTSP stream support
- Real-time frame processing
- WebSocket streaming to frontend
- Integration with SAM3 and counting

Use Cases:
- Live traffic monitoring
- Real-time people counting
- Security camera annotation
- Live retail analytics
"""

import cv2
import base64
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue

import numpy as np

from backend.core.logger import get_logger

logger = get_logger("live_stream")


class StreamSource(str, Enum):
    """Video stream sources."""
    WEBCAM = "webcam"
    RTSP = "rtsp"
    FILE = "file"
    HTTP = "http"


@dataclass 
class StreamConfig:
    """Configuration for a live stream."""
    source_type: StreamSource
    source_url: str
    name: str = "Live Stream"
    fps_limit: int = 15
    width: int = 1280
    height: int = 720
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0


@dataclass
class FrameResult:
    """Result of processing a frame."""
    frame_number: int
    timestamp: float
    detections: List[Dict[str, Any]] = field(default_factory=list)
    counting_results: Dict[str, Any] = field(default_factory=dict)
    annotated_frame: Optional[bytes] = None  # JPEG encoded


class LiveStreamAgent:
    """
    Live Stream Agent for real-time video processing.
    
    Features:
    - Multi-source support (webcam, RTSP, file, HTTP)
    - Frame-by-frame processing with callbacks
    - Real-time annotation overlay
    - Integration with counting agent
    - WebSocket-ready frame output
    """
    
    def __init__(self):
        """Initialize Live Stream Agent."""
        self.streams: Dict[str, 'StreamSession'] = {}
        self.frame_callbacks: List[Callable] = []
        
        logger.info("LiveStreamAgent initialized")
    
    def add_stream(
        self,
        stream_id: str,
        source_type: str,
        source_url: str,
        name: str = "Live Stream",
        fps_limit: int = 15
    ) -> Dict[str, Any]:
        """
        Add a new stream source.
        
        Args:
            stream_id: Unique identifier for the stream
            source_type: One of 'webcam', 'rtsp', 'file', 'http'
            source_url: URL or device index (0 for default webcam)
            name: Human-readable name
            fps_limit: Max frames per second to process
            
        Returns:
            Stream configuration
        """
        config = StreamConfig(
            source_type=StreamSource(source_type),
            source_url=source_url,
            name=name,
            fps_limit=fps_limit
        )
        
        session = StreamSession(stream_id, config)
        self.streams[stream_id] = session
        
        logger.info(f"Added stream: {name} ({stream_id}) - {source_type}://{source_url}")
        
        return {
            "stream_id": stream_id,
            "name": name,
            "source_type": source_type,
            "source_url": source_url,
            "status": "added"
        }
    
    def start_stream(self, stream_id: str) -> Dict[str, Any]:
        """Start capturing from a stream."""
        if stream_id not in self.streams:
            return {"error": "Stream not found"}
        
        session = self.streams[stream_id]
        session.start()
        
        return {"stream_id": stream_id, "status": "started"}
    
    def stop_stream(self, stream_id: str) -> Dict[str, Any]:
        """Stop a stream."""
        if stream_id not in self.streams:
            return {"error": "Stream not found"}
        
        session = self.streams[stream_id]
        session.stop()
        
        return {"stream_id": stream_id, "status": "stopped"}
    
    def get_frame(self, stream_id: str, annotate: bool = True) -> Optional[bytes]:
        """
        Get the latest frame from a stream.
        
        Args:
            stream_id: Stream to get frame from
            annotate: Whether to include annotations
            
        Returns:
            JPEG encoded frame bytes or None
        """
        if stream_id not in self.streams:
            return None
        
        session = self.streams[stream_id]
        frame = session.get_latest_frame()
        
        if frame is None:
            return None
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def get_frame_base64(self, stream_id: str) -> Optional[str]:
        """Get latest frame as base64 string for WebSocket."""
        frame_bytes = self.get_frame(stream_id)
        if frame_bytes:
            return base64.b64encode(frame_bytes).decode('utf-8')
        return None
    
    def get_stream_info(self, stream_id: str) -> Dict[str, Any]:
        """Get information about a stream."""
        if stream_id not in self.streams:
            return {"error": "Stream not found"}
        
        session = self.streams[stream_id]
        return session.get_info()
    
    def list_streams(self) -> List[Dict[str, Any]]:
        """List all streams."""
        return [session.get_info() for session in self.streams.values()]
    
    def remove_stream(self, stream_id: str):
        """Remove a stream."""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            del self.streams[stream_id]
            logger.info(f"Removed stream: {stream_id}")


class StreamSession:
    """
    Individual stream session handling capture and processing.
    """
    
    def __init__(self, stream_id: str, config: StreamConfig):
        """Initialize stream session."""
        self.stream_id = stream_id
        self.config = config
        
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        
        self.frame_queue: Queue = Queue(maxsize=10)
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        self.frame_count = 0
        self.fps_actual = 0.0
        self.last_frame_time = 0.0
        self.start_time = 0.0
        
        # Detection results
        self.latest_detections: List[Dict] = []
        self.counting_results: Dict = {}
    
    def start(self):
        """Start the capture thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info(f"Stream {self.stream_id} started")
    
    def stop(self):
        """Stop the capture thread."""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        logger.info(f"Stream {self.stream_id} stopped")
    
    def _capture_loop(self):
        """Main capture loop running in thread."""
        # Determine source
        source = self._get_source()
        
        # Open capture
        self.capture = cv2.VideoCapture(source)
        
        if not self.capture.isOpened():
            logger.error(f"Failed to open stream: {source}")
            self.is_running = False
            return
        
        # Set resolution if supported
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        
        frame_interval = 1.0 / self.config.fps_limit
        
        while self.is_running:
            loop_start = time.time()
            
            ret, frame = self.capture.read()
            
            if not ret:
                if self.config.auto_reconnect:
                    logger.warning(f"Stream {self.stream_id} disconnected, reconnecting...")
                    time.sleep(self.config.reconnect_delay)
                    self.capture.release()
                    self.capture = cv2.VideoCapture(source)
                    continue
                else:
                    break
            
            # Update frame
            with self.frame_lock:
                self.latest_frame = frame.copy()
                self.frame_count += 1
                self.last_frame_time = time.time()
            
            # Calculate actual FPS
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps_actual = self.frame_count / elapsed
            
            # Rate limit
            process_time = time.time() - loop_start
            sleep_time = max(0, frame_interval - process_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if self.capture:
            self.capture.release()
    
    def _get_source(self):
        """Get the capture source."""
        if self.config.source_type == StreamSource.WEBCAM:
            # Webcam index
            try:
                return int(self.config.source_url)
            except ValueError:
                return 0
        elif self.config.source_type == StreamSource.RTSP:
            return self.config.source_url
        elif self.config.source_type == StreamSource.HTTP:
            return self.config.source_url
        else:  # FILE
            return self.config.source_url
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None
    
    def get_annotated_frame(
        self,
        detections: List[Dict],
        counting_lines: List[Dict] = None,
        counting_zones: List[Dict] = None
    ) -> Optional[np.ndarray]:
        """
        Get frame with annotations drawn.
        
        Args:
            detections: List of detection dicts with bbox, label, score
            counting_lines: Optional counting lines to draw
            counting_zones: Optional counting zones to draw
        """
        frame = self.get_latest_frame()
        if frame is None:
            return None
        
        # Color palette
        colors = [
            (129, 140, 248),  # indigo
            (244, 114, 182),  # pink
            (52, 211, 153),   # emerald
            (251, 191, 36),   # amber
            (96, 165, 250),   # blue
            (167, 139, 250),  # violet
        ]
        
        # Draw detections
        for i, det in enumerate(detections):
            color = colors[i % len(colors)]
            bbox = det.get("bbox", [])
            
            if len(bbox) >= 4:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Label
                label = det.get("label", "object")
                score = det.get("score", 0)
                text = f"{label} {score*100:.0f}%"
                
                cv2.rectangle(frame, (x, y - 25), (x + len(text) * 10, y), color, -1)
                cv2.putText(frame, text, (x + 5, y - 7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw counting lines
        if counting_lines:
            for line in counting_lines:
                start = tuple(line.get("start", (0, 0)))
                end = tuple(line.get("end", (0, 0)))
                cv2.line(frame, start, end, (0, 255, 0), 2)
                
                # Line name
                mid_x = (start[0] + end[0]) // 2
                mid_y = (start[1] + end[1]) // 2
                cv2.putText(frame, line.get("name", ""), (mid_x, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw counting zones
        if counting_zones:
            for zone in counting_zones:
                vertices = zone.get("vertices", [])
                if len(vertices) >= 3:
                    pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (255, 165, 0), 2)
                    
                    # Zone name
                    if vertices:
                        cv2.putText(frame, zone.get("name", ""), 
                                   (vertices[0][0], vertices[0][1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps_actual:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def get_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            "stream_id": self.stream_id,
            "name": self.config.name,
            "source_type": self.config.source_type.value,
            "source_url": self.config.source_url,
            "is_running": self.is_running,
            "frame_count": self.frame_count,
            "fps_actual": round(self.fps_actual, 1),
            "fps_limit": self.config.fps_limit,
            "resolution": f"{self.config.width}x{self.config.height}",
            "uptime": round(time.time() - self.start_time, 1) if self.start_time else 0
        }


# Global instance
live_stream_agent = LiveStreamAgent()

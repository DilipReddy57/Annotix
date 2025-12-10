from typing import List, Dict, Any
from backend.core.logger import get_logger

logger = get_logger("tracking")

class TrackingAgent:
    def __init__(self):
        self.tracks = {} # id -> history
        self.next_id = 0

    def update_tracks(self, frame_id: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates object tracks across frames using simple IoU matching.
        """
        # Mock tracking logic: just assign IDs sequentially for now
        # In production: SORT or DeepSORT implementation
        for det in detections:
            det["track_id"] = self.next_id
            self.next_id += 1
        return detections

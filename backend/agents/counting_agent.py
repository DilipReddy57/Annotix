"""
Counting Agent for Multi-Directional Object Counting.

This agent provides:
- Line crossing detection with direction
- Zone-based counting (polygon regions)
- Object trajectory tracking
- Multi-directional counting (up/down, left/right, in/out)

Use Cases:
- Traffic counting (vehicles in each lane, direction)
- People counting (entrance/exit, footfall)
- Retail analytics (aisle traffic, checkout lines)
- Security (perimeter crossing detection)
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum

import numpy as np

from backend.core.logger import get_logger
from backend.core.config import settings

logger = get_logger("counting_agent")


class Direction(str, Enum):
    """Crossing directions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    IN = "in"
    OUT = "out"
    UNKNOWN = "unknown"


@dataclass
class CountingLine:
    """A line for counting crossings."""
    id: str
    name: str
    start_point: Tuple[int, int]  # (x, y)
    end_point: Tuple[int, int]    # (x, y)
    positive_direction: Direction  # Direction that counts as positive
    negative_direction: Direction  # Direction that counts as negative
    
    def __post_init__(self):
        # Calculate line equation: ax + by + c = 0
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        self.a = y2 - y1
        self.b = x1 - x2
        self.c = (x2 * y1) - (x1 * y2)


@dataclass
class CountingZone:
    """A polygon zone for counting objects inside."""
    id: str
    name: str
    vertices: List[Tuple[int, int]]  # List of (x, y) vertices
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(self.vertices)
        inside = False
        
        p1x, p1y = self.vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


@dataclass
class ObjectTrack:
    """Track an object's movement across frames."""
    object_id: int
    label: str
    positions: List[Tuple[int, int, int]] = field(default_factory=list)  # (frame, x, y)
    lines_crossed: Dict[str, str] = field(default_factory=dict)  # line_id -> direction
    first_seen: int = 0
    last_seen: int = 0


class CountingAgent:
    """
    Multi-Directional Object Counting Agent.
    
    Features:
    - Define counting lines (cross to count)
    - Define counting zones (inside to count)
    - Track object trajectories
    - Count in multiple directions
    - Export counting statistics
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize Counting Agent.
        
        Args:
            persist_path: Path to persist counting data
        """
        self.persist_path = persist_path or os.path.join(
            settings.DATA_DIR, "counting_data.json"
        )
        
        # Counting elements
        self.lines: Dict[str, CountingLine] = {}
        self.zones: Dict[str, CountingZone] = {}
        
        # Object tracking
        self.tracks: Dict[int, ObjectTrack] = {}
        self.next_object_id = 1
        
        # Counting results
        self.line_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.zone_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Frame-by-frame counts for time series
        self.frame_counts: List[Dict[str, Any]] = []
        
        # Load existing data
        self._load_data()
        
        logger.info(f"CountingAgent initialized with {len(self.lines)} lines, {len(self.zones)} zones")
    
    def _load_data(self):
        """Load saved counting configuration."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                
                for line_data in data.get("lines", []):
                    line = CountingLine(**line_data)
                    self.lines[line.id] = line
                
                for zone_data in data.get("zones", []):
                    zone = CountingZone(**zone_data)
                    self.zones[zone.id] = zone
                
                self.line_counts = defaultdict(lambda: defaultdict(int), data.get("line_counts", {}))
                self.zone_counts = defaultdict(lambda: defaultdict(int), data.get("zone_counts", {}))
                
                logger.info(f"Loaded counting config from {self.persist_path}")
            except Exception as e:
                logger.error(f"Failed to load counting data: {e}")
    
    def _save_data(self):
        """Save counting configuration."""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            
            data = {
                "lines": [asdict(l) for l in self.lines.values()],
                "zones": [asdict(z) for z in self.zones.values()],
                "line_counts": dict(self.line_counts),
                "zone_counts": dict(self.zone_counts),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Remove calculated attributes from lines
            for line_data in data["lines"]:
                line_data.pop("a", None)
                line_data.pop("b", None)
                line_data.pop("c", None)
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save counting data: {e}")
    
    # === Line Management ===
    
    def add_line(
        self,
        name: str,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        positive_direction: str = "down",
        negative_direction: str = "up"
    ) -> str:
        """
        Add a counting line.
        
        Args:
            name: Human-readable name
            start_point: (x, y) start coordinates
            end_point: (x, y) end coordinates
            positive_direction: Direction for +1 count
            negative_direction: Direction for -1 count
            
        Returns:
            Line ID
        """
        line_id = str(uuid.uuid4())[:8]
        
        line = CountingLine(
            id=line_id,
            name=name,
            start_point=start_point,
            end_point=end_point,
            positive_direction=Direction(positive_direction),
            negative_direction=Direction(negative_direction)
        )
        
        self.lines[line_id] = line
        self.line_counts[line_id] = defaultdict(int)
        
        self._save_data()
        logger.info(f"Added counting line: {name} ({line_id})")
        
        return line_id
    
    def add_horizontal_line(
        self,
        name: str,
        y: int,
        x_start: int = 0,
        x_end: int = 1920
    ) -> str:
        """Add a horizontal counting line (up/down detection)."""
        return self.add_line(
            name=name,
            start_point=(x_start, y),
            end_point=(x_end, y),
            positive_direction="down",
            negative_direction="up"
        )
    
    def add_vertical_line(
        self,
        name: str,
        x: int,
        y_start: int = 0,
        y_end: int = 1080
    ) -> str:
        """Add a vertical counting line (left/right detection)."""
        return self.add_line(
            name=name,
            start_point=(x, y_start),
            end_point=(x, y_end),
            positive_direction="right",
            negative_direction="left"
        )
    
    # === Zone Management ===
    
    def add_zone(
        self,
        name: str,
        vertices: List[Tuple[int, int]]
    ) -> str:
        """
        Add a counting zone (polygon).
        
        Args:
            name: Zone name
            vertices: List of (x, y) vertices defining the polygon
            
        Returns:
            Zone ID
        """
        zone_id = str(uuid.uuid4())[:8]
        
        zone = CountingZone(
            id=zone_id,
            name=name,
            vertices=vertices
        )
        
        self.zones[zone_id] = zone
        self.zone_counts[zone_id] = defaultdict(int)
        
        self._save_data()
        logger.info(f"Added counting zone: {name} ({zone_id})")
        
        return zone_id
    
    def add_rectangular_zone(
        self,
        name: str,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> str:
        """Add a rectangular counting zone."""
        vertices = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height)
        ]
        return self.add_zone(name, vertices)
    
    # === Counting Logic ===
    
    def _get_signed_distance(self, line: CountingLine, x: int, y: int) -> float:
        """Get signed distance from point to line."""
        return (line.a * x + line.b * y + line.c) / np.sqrt(line.a**2 + line.b**2 + 1e-8)
    
    def _check_line_crossing(
        self,
        line: CountingLine,
        prev_pos: Tuple[int, int],
        curr_pos: Tuple[int, int]
    ) -> Optional[Direction]:
        """
        Check if object crossed a line between two positions.
        
        Returns:
            Direction of crossing, or None if no crossing
        """
        prev_dist = self._get_signed_distance(line, prev_pos[0], prev_pos[1])
        curr_dist = self._get_signed_distance(line, curr_pos[0], curr_pos[1])
        
        # Check for sign change (crossing)
        if prev_dist * curr_dist < 0:
            # Determine direction based on sign change
            if prev_dist < 0 and curr_dist > 0:
                return line.positive_direction
            else:
                return line.negative_direction
        
        return None
    
    def process_frame(
        self,
        frame_number: int,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a frame of detections for counting.
        
        Args:
            frame_number: Current frame number
            detections: List of detections with:
                - object_id: Unique tracking ID
                - label: Object class
                - bbox: [x, y, w, h]
                - center: Optional (x, y) center point
                
        Returns:
            Dict with frame counting results
        """
        frame_results = {
            "frame": frame_number,
            "crossings": [],
            "zone_occupancy": {},
            "total_objects": len(detections)
        }
        
        for det in detections:
            object_id = det.get("object_id", self.next_object_id)
            label = det.get("label", "object")
            
            # Get center point
            if "center" in det:
                cx, cy = det["center"]
            elif "bbox" in det:
                x, y, w, h = det["bbox"]
                cx, cy = int(x + w/2), int(y + h/2)
            else:
                continue
            
            # Update or create track
            if object_id not in self.tracks:
                self.tracks[object_id] = ObjectTrack(
                    object_id=object_id,
                    label=label,
                    first_seen=frame_number
                )
                self.next_object_id = max(self.next_object_id, object_id + 1)
            
            track = self.tracks[object_id]
            track.last_seen = frame_number
            
            # Check line crossings if we have previous position
            if track.positions:
                prev_frame, prev_x, prev_y = track.positions[-1]
                
                for line_id, line in self.lines.items():
                    # Skip if already counted for this line
                    if line_id in track.lines_crossed:
                        continue
                    
                    direction = self._check_line_crossing(
                        line,
                        (prev_x, prev_y),
                        (cx, cy)
                    )
                    
                    if direction:
                        track.lines_crossed[line_id] = direction.value
                        self.line_counts[line_id][direction.value] += 1
                        self.line_counts[line_id][f"{label}_{direction.value}"] += 1
                        
                        frame_results["crossings"].append({
                            "line_id": line_id,
                            "line_name": line.name,
                            "object_id": object_id,
                            "label": label,
                            "direction": direction.value
                        })
                        
                        logger.debug(f"Object {object_id} ({label}) crossed {line.name} going {direction.value}")
            
            # Add current position to track
            track.positions.append((frame_number, cx, cy))
            
            # Keep only last 30 frames of positions
            if len(track.positions) > 30:
                track.positions = track.positions[-30:]
            
            # Check zone occupancy
            for zone_id, zone in self.zones.items():
                if zone.contains_point(cx, cy):
                    if zone_id not in frame_results["zone_occupancy"]:
                        frame_results["zone_occupancy"][zone_id] = {
                            "name": zone.name,
                            "count": 0,
                            "labels": defaultdict(int)
                        }
                    frame_results["zone_occupancy"][zone_id]["count"] += 1
                    frame_results["zone_occupancy"][zone_id]["labels"][label] += 1
        
        # Update zone counts
        for zone_id, occupancy in frame_results["zone_occupancy"].items():
            self.zone_counts[zone_id]["current"] = occupancy["count"]
            self.zone_counts[zone_id]["max"] = max(
                self.zone_counts[zone_id].get("max", 0),
                occupancy["count"]
            )
        
        self.frame_counts.append(frame_results)
        
        return frame_results
    
    def get_line_counts(self, line_id: Optional[str] = None) -> Dict[str, Any]:
        """Get counting results for lines."""
        if line_id:
            line = self.lines.get(line_id)
            if not line:
                return {"error": "Line not found"}
            
            counts = dict(self.line_counts.get(line_id, {}))
            return {
                "line_id": line_id,
                "name": line.name,
                "counts": counts,
                "total": sum(v for k, v in counts.items() if k in ["up", "down", "left", "right", "in", "out"])
            }
        
        # All lines
        results = {}
        for lid, line in self.lines.items():
            counts = dict(self.line_counts.get(lid, {}))
            results[lid] = {
                "name": line.name,
                "counts": counts,
                "total": sum(v for k, v in counts.items() if k in ["up", "down", "left", "right", "in", "out"])
            }
        
        return results
    
    def get_zone_counts(self, zone_id: Optional[str] = None) -> Dict[str, Any]:
        """Get counting results for zones."""
        if zone_id:
            zone = self.zones.get(zone_id)
            if not zone:
                return {"error": "Zone not found"}
            
            return {
                "zone_id": zone_id,
                "name": zone.name,
                "counts": dict(self.zone_counts.get(zone_id, {}))
            }
        
        return {zid: {"name": z.name, "counts": dict(self.zone_counts.get(zid, {}))} 
                for zid, z in self.zones.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get overall counting summary."""
        total_crossings = sum(
            sum(v for k, v in counts.items() if k in ["up", "down", "left", "right", "in", "out"])
            for counts in self.line_counts.values()
        )
        
        return {
            "lines": len(self.lines),
            "zones": len(self.zones),
            "tracked_objects": len(self.tracks),
            "total_crossings": total_crossings,
            "frames_processed": len(self.frame_counts),
            "line_counts": self.get_line_counts(),
            "zone_counts": self.get_zone_counts()
        }
    
    def reset_counts(self):
        """Reset all counting data (keep lines and zones)."""
        self.tracks.clear()
        self.line_counts = defaultdict(lambda: defaultdict(int))
        self.zone_counts = defaultdict(lambda: defaultdict(int))
        self.frame_counts.clear()
        logger.info("Counting data reset")
    
    def export_counts(self, output_path: str):
        """Export counting results to JSON."""
        data = {
            "summary": self.get_summary(),
            "lines": {lid: asdict(l) for lid, l in self.lines.items()},
            "zones": {zid: asdict(z) for zid, z in self.zones.items()},
            "frame_data": self.frame_counts[-100:],  # Last 100 frames
            "exported_at": datetime.utcnow().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported counts to {output_path}")

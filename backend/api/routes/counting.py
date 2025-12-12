"""
Counting API Routes for Multi-Directional Object Counting.

Provides endpoints for:
- Managing counting lines and zones
- Getting counting results
- Processing video frames
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple

from backend.core.logger import get_logger
from backend.agents.counting_agent import CountingAgent

router = APIRouter()
logger = get_logger("counting")

# Initialize counting agent
counting_agent = CountingAgent()


# Request Models
class LineRequest(BaseModel):
    name: str
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    positive_direction: str = "down"
    negative_direction: str = "up"


class HorizontalLineRequest(BaseModel):
    name: str
    y: int
    x_start: int = 0
    x_end: int = 1920


class VerticalLineRequest(BaseModel):
    name: str
    x: int
    y_start: int = 0
    y_end: int = 1080


class ZoneRequest(BaseModel):
    name: str
    vertices: List[Tuple[int, int]]


class RectZoneRequest(BaseModel):
    name: str
    x: int
    y: int
    width: int
    height: int


class FrameDetection(BaseModel):
    object_id: int
    label: str
    bbox: List[float]  # [x, y, w, h]
    center: Optional[Tuple[int, int]] = None


class FrameRequest(BaseModel):
    frame_number: int
    detections: List[FrameDetection]


# === Line Endpoints ===

@router.post("/lines")
async def add_counting_line(request: LineRequest):
    """Add a counting line with custom direction."""
    line_id = counting_agent.add_line(
        name=request.name,
        start_point=request.start_point,
        end_point=request.end_point,
        positive_direction=request.positive_direction,
        negative_direction=request.negative_direction
    )
    return {"line_id": line_id, "name": request.name}


@router.post("/lines/horizontal")
async def add_horizontal_line(request: HorizontalLineRequest):
    """Add a horizontal line (counts up/down crossings)."""
    line_id = counting_agent.add_horizontal_line(
        name=request.name,
        y=request.y,
        x_start=request.x_start,
        x_end=request.x_end
    )
    return {"line_id": line_id, "name": request.name, "directions": ["up", "down"]}


@router.post("/lines/vertical")
async def add_vertical_line(request: VerticalLineRequest):
    """Add a vertical line (counts left/right crossings)."""
    line_id = counting_agent.add_vertical_line(
        name=request.name,
        x=request.x,
        y_start=request.y_start,
        y_end=request.y_end
    )
    return {"line_id": line_id, "name": request.name, "directions": ["left", "right"]}


@router.get("/lines")
async def list_lines():
    """List all counting lines."""
    lines = []
    for lid, line in counting_agent.lines.items():
        lines.append({
            "id": lid,
            "name": line.name,
            "start": line.start_point,
            "end": line.end_point,
            "positive_direction": line.positive_direction.value,
            "negative_direction": line.negative_direction.value
        })
    return {"lines": lines, "count": len(lines)}


@router.delete("/lines/{line_id}")
async def delete_line(line_id: str):
    """Delete a counting line."""
    if line_id in counting_agent.lines:
        del counting_agent.lines[line_id]
        counting_agent._save_data()
        return {"deleted": line_id}
    raise HTTPException(status_code=404, detail="Line not found")


# === Zone Endpoints ===

@router.post("/zones")
async def add_counting_zone(request: ZoneRequest):
    """Add a polygon counting zone."""
    zone_id = counting_agent.add_zone(
        name=request.name,
        vertices=request.vertices
    )
    return {"zone_id": zone_id, "name": request.name}


@router.post("/zones/rect")
async def add_rectangular_zone(request: RectZoneRequest):
    """Add a rectangular counting zone."""
    zone_id = counting_agent.add_rectangular_zone(
        name=request.name,
        x=request.x,
        y=request.y,
        width=request.width,
        height=request.height
    )
    return {"zone_id": zone_id, "name": request.name}


@router.get("/zones")
async def list_zones():
    """List all counting zones."""
    zones = []
    for zid, zone in counting_agent.zones.items():
        zones.append({
            "id": zid,
            "name": zone.name,
            "vertices": zone.vertices
        })
    return {"zones": zones, "count": len(zones)}


@router.delete("/zones/{zone_id}")
async def delete_zone(zone_id: str):
    """Delete a counting zone."""
    if zone_id in counting_agent.zones:
        del counting_agent.zones[zone_id]
        counting_agent._save_data()
        return {"deleted": zone_id}
    raise HTTPException(status_code=404, detail="Zone not found")


# === Counting Endpoints ===

@router.post("/process")
async def process_frame(request: FrameRequest):
    """
    Process a frame of detections for counting.
    
    Detects line crossings and zone occupancy.
    """
    detections = [d.dict() for d in request.detections]
    
    result = counting_agent.process_frame(
        frame_number=request.frame_number,
        detections=detections
    )
    
    return result


@router.get("/counts/lines")
async def get_line_counts(line_id: Optional[str] = None):
    """Get counting results for lines."""
    return counting_agent.get_line_counts(line_id)


@router.get("/counts/zones")
async def get_zone_counts(zone_id: Optional[str] = None):
    """Get counting results for zones."""
    return counting_agent.get_zone_counts(zone_id)


@router.get("/summary")
async def get_counting_summary():
    """Get overall counting summary."""
    return counting_agent.get_summary()


@router.post("/reset")
async def reset_counts():
    """Reset all counting data (keeps lines and zones)."""
    counting_agent.reset_counts()
    return {"status": "reset", "message": "Counting data cleared"}


@router.get("/export")
async def export_counts():
    """Export counting data as JSON."""
    return {
        "summary": counting_agent.get_summary(),
        "lines": counting_agent.get_line_counts(),
        "zones": counting_agent.get_zone_counts()
    }

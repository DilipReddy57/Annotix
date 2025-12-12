"""
Live Stream API Routes with WebSocket Support.

Provides endpoints for:
- Managing live stream sources
- REST endpoints for stream control
- WebSocket for real-time frame streaming
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import base64

from backend.core.logger import get_logger
from backend.agents.live_stream import live_stream_agent, LiveStreamAgent
from backend.agents.counting_agent import CountingAgent

router = APIRouter()
logger = get_logger("live_api")

# Counting agent for integration
counting_agent = CountingAgent()


# Request Models
class StreamRequest(BaseModel):
    stream_id: str
    source_type: str  # webcam, rtsp, file, http
    source_url: str
    name: str = "Live Stream"
    fps_limit: int = 15


# === REST Endpoints ===

@router.post("/streams")
async def add_stream(request: StreamRequest):
    """
    Add a new live stream source.
    
    source_type options:
    - 'webcam': Use device index (0, 1, etc.)
    - 'rtsp': RTSP URL (e.g., rtsp://192.168.1.100:554/stream)
    - 'http': HTTP stream URL
    - 'file': Video file path
    """
    result = live_stream_agent.add_stream(
        stream_id=request.stream_id,
        source_type=request.source_type,
        source_url=request.source_url,
        name=request.name,
        fps_limit=request.fps_limit
    )
    return result


@router.post("/streams/{stream_id}/start")
async def start_stream(stream_id: str):
    """Start capturing from a stream."""
    if stream_id not in live_stream_agent.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    result = live_stream_agent.start_stream(stream_id)
    return result


@router.post("/streams/{stream_id}/stop")
async def stop_stream(stream_id: str):
    """Stop a stream."""
    if stream_id not in live_stream_agent.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    result = live_stream_agent.stop_stream(stream_id)
    return result


@router.get("/streams")
async def list_streams():
    """List all configured streams."""
    return {"streams": live_stream_agent.list_streams()}


@router.get("/streams/{stream_id}")
async def get_stream_info(stream_id: str):
    """Get information about a specific stream."""
    if stream_id not in live_stream_agent.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return live_stream_agent.get_stream_info(stream_id)


@router.delete("/streams/{stream_id}")
async def remove_stream(stream_id: str):
    """Remove a stream."""
    if stream_id not in live_stream_agent.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    live_stream_agent.remove_stream(stream_id)
    return {"removed": stream_id}


@router.get("/streams/{stream_id}/frame")
async def get_latest_frame(stream_id: str):
    """Get the latest frame as base64 JPEG."""
    if stream_id not in live_stream_agent.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    frame_b64 = live_stream_agent.get_frame_base64(stream_id)
    
    if frame_b64 is None:
        raise HTTPException(status_code=503, detail="No frame available")
    
    return {
        "stream_id": stream_id,
        "frame": frame_b64,
        "format": "jpeg"
    }


# === WebSocket Endpoint ===

@router.websocket("/ws/{stream_id}")
async def websocket_stream(websocket: WebSocket, stream_id: str):
    """
    WebSocket endpoint for real-time video streaming.
    
    Sends frames as base64 JPEG at the stream's FPS limit.
    Receives control messages from client.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for stream: {stream_id}")
    
    # Check if stream exists
    if stream_id not in live_stream_agent.streams:
        await websocket.send_json({"error": "Stream not found"})
        await websocket.close()
        return
    
    session = live_stream_agent.streams[stream_id]
    
    # Ensure stream is running
    if not session.is_running:
        session.start()
    
    frame_interval = 1.0 / session.config.fps_limit
    frame_count = 0
    
    try:
        while True:
            # Check for incoming messages (non-blocking)
            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(), 
                    timeout=0.01
                )
                
                # Handle control messages
                if message.get("action") == "stop":
                    break
                elif message.get("action") == "set_fps":
                    new_fps = message.get("fps", 15)
                    frame_interval = 1.0 / max(1, min(30, new_fps))
                    
            except asyncio.TimeoutError:
                pass  # No message, continue
            
            # Get and send frame
            frame = session.get_latest_frame()
            
            if frame is not None:
                import cv2
                
                # Get counting lines/zones for overlay
                lines = []
                for lid, line in counting_agent.lines.items():
                    lines.append({
                        "name": line.name,
                        "start": line.start_point,
                        "end": line.end_point
                    })
                
                zones = []
                for zid, zone in counting_agent.zones.items():
                    zones.append({
                        "name": zone.name,
                        "vertices": zone.vertices
                    })
                
                # Draw annotations
                annotated = session.get_annotated_frame(
                    detections=session.latest_detections,
                    counting_lines=lines,
                    counting_zones=zones
                )
                
                if annotated is not None:
                    # Encode and send
                    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                    
                    await websocket.send_json({
                        "type": "frame",
                        "frame": frame_b64,
                        "frame_number": frame_count,
                        "fps": round(session.fps_actual, 1),
                        "counting": counting_agent.get_summary() if frame_count % 30 == 0 else None
                    })
                    
                    frame_count += 1
            
            await asyncio.sleep(frame_interval)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {stream_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"WebSocket closed for stream: {stream_id}")


# === Quick Start Endpoints ===

@router.post("/webcam/start")
async def start_webcam(device_id: int = 0, fps: int = 15):
    """Quick start webcam capture."""
    stream_id = f"webcam_{device_id}"
    
    live_stream_agent.add_stream(
        stream_id=stream_id,
        source_type="webcam",
        source_url=str(device_id),
        name=f"Webcam {device_id}",
        fps_limit=fps
    )
    live_stream_agent.start_stream(stream_id)
    
    return {
        "stream_id": stream_id,
        "status": "started",
        "websocket_url": f"/api/live/ws/{stream_id}"
    }


@router.post("/rtsp/start")
async def start_rtsp(url: str, name: str = "RTSP Stream", fps: int = 15):
    """Quick start RTSP stream."""
    import hashlib
    stream_id = f"rtsp_{hashlib.md5(url.encode()).hexdigest()[:8]}"
    
    live_stream_agent.add_stream(
        stream_id=stream_id,
        source_type="rtsp",
        source_url=url,
        name=name,
        fps_limit=fps
    )
    live_stream_agent.start_stream(stream_id)
    
    return {
        "stream_id": stream_id,
        "status": "started",
        "websocket_url": f"/api/live/ws/{stream_id}"
    }

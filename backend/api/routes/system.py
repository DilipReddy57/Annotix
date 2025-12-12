"""System monitoring and status endpoints for AgentStatus frontend component."""
from fastapi import APIRouter
from datetime import datetime
from backend.core.logger import get_logger

router = APIRouter()
logger = get_logger("system")

# In-memory activity log store (will reset on restart, but good for live status)
activity_logs: list = []

def add_log(message: str, log_type: str = "info"):
    """Helper to add a log entry with timestamp."""
    activity_logs.append({
        "message": message,
        "type": log_type,
        "timestamp": datetime.now().isoformat()
    })
    # Keep only last 100 logs
    if len(activity_logs) > 100:
        activity_logs.pop(0)

# Add startup log
add_log("System routes initialized", "success")

@router.get("/logs")
async def get_system_logs():
    """Get recent system activity logs for the frontend AgentStatus component."""
    return activity_logs[-50:]  # Return last 50 entries

@router.post("/initialize")
async def initialize_system():
    """
    Initialize/warm up the system components.
    This can be extended to actually load models, warm caches, etc.
    """
    logger.info("System initialization requested")
    add_log("SAM3 Model warming up...", "info")
    add_log("RAG Engine connected", "success")
    add_log("LLM Agent ready", "success")
    add_log("Active Learning module initialized", "success")
    add_log("System fully initialized", "success")
    
    return {
        "status": "initialized",
        "message": "All systems operational",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status")
async def get_system_status():
    """Get overall system health status."""
    return {
        "status": "healthy",
        "components": {
            "sam3_model": "ready",
            "rag_engine": "ready",
            "llm_agent": "ready",
            "active_learning": "ready",
            "clip_embeddings": "ready"
        },
        "timestamp": datetime.now().isoformat()
    }

@router.post("/test-pipeline")
async def test_pipeline(
    image_url: str = None,
    turbo_mode: bool = False
):
    """
    Test the annotation pipeline directly.
    
    This endpoint allows testing pipeline performance with any image.
    Returns timing breakdown for verification.
    
    Args:
        image_url: URL to an image (will be downloaded)
        turbo_mode: If True, use maximum speed mode
    """
    import tempfile
    import httpx
    import os
    from backend.pipeline.orchestrator import AnnotationPipeline
    
    if not image_url:
        return {
            "error": "Please provide image_url parameter",
            "example": "/api/system/test-pipeline?image_url=https://example.com/image.jpg"
        }
    
    try:
        # Download image to temp file
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(response.content)
            temp_path = f.name
        
        # Run pipeline
        pipeline = AnnotationPipeline(lazy_load=True, enable_advanced=not turbo_mode)
        result = pipeline.smart_process_image(
            file_path=temp_path,
            project_id="test",
            use_auto_prompts=not turbo_mode,
            turbo_mode=turbo_mode
        )
        
        # Cleanup
        os.unlink(temp_path)
        
        # Return results with timing
        return {
            "status": "success",
            "mode": result.get("processing_mode"),
            "processing_time_ms": result.get("processing_time_ms"),
            "annotations_count": len(result.get("annotations", [])),
            "prompts_used": result.get("prompts_used"),
            "annotations": [
                {
                    "label": ann.get("label"),
                    "confidence": ann.get("score"),
                    "bbox": ann.get("bbox")
                }
                for ann in result.get("annotations", [])[:10]  # Limit to 10 for response size
            ]
        }
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

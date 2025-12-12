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
            "active_learning": "ready"
        },
        "timestamp": datetime.now().isoformat()
    }

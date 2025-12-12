from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.concurrency import run_in_threadpool
import shutil
import os
import uuid
import aiofiles
from sqlmodel import Session, select, func, text
from backend.core.config import settings
from backend.core.logger import get_logger
from backend.pipeline.orchestrator import AnnotationPipeline
from backend.core.database import get_session
from backend.core.models import Image, Annotation

router = APIRouter()
logger = get_logger("api")

# Initialize Pipeline (Singleton)
pipeline = AnnotationPipeline()
results_store = {}

@router.post("/upload")
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    task_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{task_id}_{file.filename}")
    
    try:
        # Use aiofiles for non-blocking file I/O
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                await out_file.write(content)
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")
    
    results_store[task_id] = {"status": "pending", "image_path": file_path}
    
    # Run pipeline in background
    background_tasks.add_task(run_pipeline_task, task_id, file_path)
    
    return {"task_id": task_id, "status": "queued"}

@router.post("/segment/text")
async def segment_text(task_id: str, prompt: str):
    if task_id not in results_store:
        raise HTTPException(status_code=404, detail="Task not found")
        
    result = results_store[task_id]
    image_path = result.get("image_path")
    
    if not image_path:
        raise HTTPException(status_code=400, detail="Image path missing")
        
    try:
        # Run pipeline in threadpool to avoid blocking event loop
        output = await run_in_threadpool(pipeline.process_image, image_path, prompt=prompt)
        result["annotations"] = output["annotations"]
        result["status"] = "completed"
        results_store[task_id] = result
        return output
    except Exception as e:
        logger.error(f"Text segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{task_id}")
async def get_result(task_id: str):
    return results_store.get(task_id, {"status": "not_found"})

@router.get("/results")
async def list_results():
    return results_store

@router.get("/analytics")
async def get_analytics(session: Session = Depends(get_session)):
    # Calculate analytics from DB
    total_images = session.exec(select(func.count(Image.id))).one()
    total_annotations = session.exec(select(func.count(Annotation.id))).one()
    
    # Class distribution using SQL Group By for efficiency
    # Note: If database is SQLite and label is a column, this works.
    # If label is inside JSON, we need native SQL. Assuming Annotation has 'label' column as per model.
    try:
        statement = select(Annotation.label, func.count(Annotation.id)).group_by(Annotation.label).order_by(func.count(Annotation.id).desc()).limit(10)
        results = session.exec(statement).all()
        sorted_classes = {label: count for label, count in results}
        categories_count = session.exec(select(func.count(func.distinct(Annotation.label)))).one()
    except Exception:
        # Fallback for complex schemas or if label is missing (safer than crashing)
        sorted_classes = {}
        categories_count = 0
        
    # ROI Metrics (Mock calculation based on counts)
    human_cost_per_ann = 0.50 # $0.50 per annotation
    agent_cost_per_ann = 0.01 # $0.01 per annotation
    
    human_cost_est = total_annotations * human_cost_per_ann
    agent_cost_est = total_annotations * agent_cost_per_ann
    savings = human_cost_est - agent_cost_est
    
    time_per_ann_human = 30 # seconds
    time_per_ann_agent = 2 # seconds
    time_saved_seconds = total_annotations * (time_per_ann_human - time_per_ann_agent)
    time_saved_hours = round(time_saved_seconds / 3600, 2)
    
    return {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "class_distribution": sorted_classes,
        "categories": categories_count,
        "roi_metrics": {
            "human_cost_est": round(human_cost_est, 2),
            "agent_cost_est": round(agent_cost_est, 2),
            "savings": round(savings, 2),
            "time_saved_hours": time_saved_hours
        }
    }

def run_pipeline_task(task_id: str, file_path: str):
    try:
        output = pipeline.process_image(file_path)
        results_store[task_id].update({
            "status": "completed",
            "annotations": output["annotations"]
        })
    except Exception as e:
        logger.error(f"Pipeline failed for {task_id}: {e}")
        results_store[task_id]["status"] = "failed"
        results_store[task_id]["error"] = str(e)

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from typing import List, Dict
import shutil
import os
import json
from sqlmodel import Session, select
from backend.core.config import settings
from backend.core.models import Project, Image, Annotation, Video
from backend.core.database import get_session, engine
from backend.core.logger import get_logger
from backend.pipeline.orchestrator import AnnotationPipeline

router = APIRouter()
logger = get_logger("projects")
pipeline = AnnotationPipeline()

@router.post("/", response_model=Project)
async def create_project(name: str, description: str = None, session: Session = Depends(get_session)):
    project = Project(name=name, description=description)
    session.add(project)
    session.commit()
    session.refresh(project)
    
    # Create project directory
    project_dir = os.path.join(settings.UPLOAD_DIR, project.id)
    os.makedirs(project_dir, exist_ok=True)
    
    logger.info(f"Created project: {project.name} ({project.id})")
    return project

@router.get("/", response_model=List[Project])
async def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project)).all()
    return projects

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.post("/{project_id}/upload")
async def upload_images(
    project_id: str, 
    files: List[UploadFile] = File(...), 
    auto_annotate: bool = False,
    background_tasks: BackgroundTasks = None,
    session: Session = Depends(get_session)
):
    """
    Upload images to a project.
    
    Args:
        project_id: The project ID
        files: List of image files to upload
        auto_annotate: If True, automatically queue annotation for each image
        background_tasks: FastAPI background tasks for async annotation
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_dir = os.path.join(settings.UPLOAD_DIR, project_id)
    os.makedirs(project_dir, exist_ok=True)
    uploaded_images = []
    
    for file in files:
        file_path = os.path.join(project_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        image = Image(
            project_id=project_id,
            filename=file.filename,
            status="processing" if auto_annotate else "pending",
            width=0,
            height=0
        )
        session.add(image)
        session.commit()
        session.refresh(image)
        uploaded_images.append(image)
        
        # Auto-annotate if requested
        if auto_annotate and background_tasks:
            background_tasks.add_task(run_annotation_task, project_id, image.id)
            logger.info(f"Queued auto-annotation for {image.filename}")
        
    return {
        "status": "success", 
        "uploaded": len(uploaded_images),
        "auto_annotate": auto_annotate,
        "images": [{"id": img.id, "filename": img.filename, "status": img.status} for img in uploaded_images]
    }

@router.post("/{project_id}/images/{image_id}/annotate")
async def annotate_image(project_id: str, image_id: str, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    image = session.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    image.status = "processing"
    session.add(image)
    session.commit()
    
    # Run pipeline
    background_tasks.add_task(run_annotation_task, project_id, image_id)
    
    return {"status": "queued", "image_id": image_id}

def run_annotation_task(project_id: str, image_id: str):
    # Create a new session for the background task
    with Session(engine) as session:
        image = session.get(Image, image_id)
        if not image:
            logger.error(f"Image {image_id} not found in background task")
            return

        try:
            file_path = os.path.join(settings.UPLOAD_DIR, project_id, image.filename)
            
            # Call the orchestrator
            result = pipeline.process_image(file_path)
            
            # Convert result annotations to our model
            for ann_data in result["annotations"]:
                ann = Annotation(
                    image_id=image.id,
                    label=ann_data["label"],
                    confidence=ann_data["score"],
                    bbox_json=json.dumps(ann_data["bbox"]),
                    segmentation_json=json.dumps(ann_data.get("segmentation", {})) 
                )
                session.add(ann)
                
            image.status = "completed"
            session.add(image)
            session.commit()
            logger.info(f"Annotated image {image.id} with {len(result['annotations'])} objects")
            
        except Exception as e:
            logger.error(f"Annotation failed for {image.id}: {e}")
            image.status = "error"
            session.add(image)
            session.commit()

@router.post("/{project_id}/videos/upload")
async def upload_video(project_id: str, file: UploadFile = File(...), session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_dir = os.path.join(settings.UPLOAD_DIR, project_id)
    file_path = os.path.join(project_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    video = Video(
        project_id=project_id,
        filename=file.filename,
        status="pending"
    )
    session.add(video)
    session.commit()
    session.refresh(video)
    
    return video

@router.post("/{project_id}/images/{image_id}/segment")
async def segment_image_interactive(
    project_id: str, 
    image_id: str, 
    prompt: str,
    session: Session = Depends(get_session)
):
    """
    Interactive segmentation for a specific image in a project.
    Returns annotations for preview (does not save to DB immediately).
    """
    image = session.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    project = session.get(Project, project_id)
    if not project:
         raise HTTPException(status_code=404, detail="Project not found")

    file_path = os.path.join(settings.UPLOAD_DIR, project_id, image.filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file missing on disk")

    try:
        # Run specific SAM 3 agent directly for low latency
        # We access the agent from the pipeline singleton
        annotations = pipeline.sam_agent.segment_image(file_path, prompt=prompt)
        return annotations
    except Exception as e:
        logger.error(f"Interactive segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{project_id}/videos", response_model=List[Video])
async def list_videos(project_id: str, session: Session = Depends(get_session)):
    videos = session.exec(select(Video).where(Video.project_id == project_id)).all()
    return videos

@router.post("/{project_id}/videos/{video_id}/annotate")
async def annotate_video(
    project_id: str, 
    video_id: str, 
    prompt: str,
    session: Session = Depends(get_session)
):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
        
    project = session.get(Project, project_id)
    if not project:
         raise HTTPException(status_code=404, detail="Project not found")

    file_path = os.path.join(settings.UPLOAD_DIR, project_id, video.filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file missing on disk")

    try:
        # Run video pipeline
        output = pipeline.process_video(file_path, prompt)
        
        # In a real app, we would save these temporal annotations to the DB.
        # For this MVP, we return them directly for frontend visualization.
        return output
    except Exception as e:
        logger.error(f"Video annotation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

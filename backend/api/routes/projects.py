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

@router.post("/import-dataset")
async def import_dataset(
    request: Dict,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """
    Import a dataset from an external URL (Kaggle, HuggingFace, GitHub, etc.)
    
    Args:
        request: Dict containing 'url' and 'source' (kaggle, huggingface, github, url)
    """
    url = request.get("url", "")
    source = request.get("source", "url")
    
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    logger.info(f"Importing dataset from {source}: {url}")
    
    # Create a new project for this import
    project_name = f"Import from {source.title()} - {url.split('/')[-1][:30]}"
    project = Project(name=project_name, description=f"Imported from {url}")
    session.add(project)
    session.commit()
    session.refresh(project)
    
    project_dir = os.path.join(settings.UPLOAD_DIR, project.id)
    os.makedirs(project_dir, exist_ok=True)
    
    # Queue background task for actual download
    background_tasks.add_task(
        download_dataset_task, 
        project.id, 
        url, 
        source, 
        project_dir
    )
    
    return {
        "status": "importing",
        "project_id": project.id,
        "project_name": project_name,
        "source": source,
        "message": f"Dataset import started. Check project '{project_name}' for progress."
    }

def download_dataset_task(project_id: str, url: str, source: str, project_dir: str):
    """Background task to download dataset from URL."""
    import requests
    import zipfile
    import io
    import glob
    from sqlmodel import Session as SqlSession
    
    logger.info(f"Starting dataset download for project {project_id} from {url}")
    
    downloaded_files = []
    
    try:
        if source == "kaggle":
            # Try to use Kaggle API if available
            try:
                import kaggle
                # Extract dataset identifier from URL: kaggle.com/datasets/{user}/{dataset}
                parts = url.replace("https://", "").replace("http://", "").split("/")
                if "datasets" in parts:
                    idx = parts.index("datasets")
                    if len(parts) > idx + 2:
                        dataset_id = f"{parts[idx+1]}/{parts[idx+2]}"
                        kaggle.api.dataset_download_files(dataset_id, path=project_dir, unzip=True)
                        logger.info(f"Downloaded Kaggle dataset: {dataset_id}")
                        downloaded_files = glob.glob(os.path.join(project_dir, "**/*.*"), recursive=True)
            except ImportError:
                logger.warning("Kaggle package not installed. Run: pip install kaggle")
            except Exception as ke:
                logger.warning(f"Kaggle download failed: {ke}. Make sure ~/.kaggle/kaggle.json exists")
            
        elif source == "huggingface":
            # Try HuggingFace datasets
            try:
                from datasets import load_dataset
                # Extract dataset name from URL
                parts = url.replace("https://", "").replace("http://", "").split("/")
                if "datasets" in parts:
                    idx = parts.index("datasets")
                    if len(parts) > idx + 2:
                        dataset_id = f"{parts[idx+1]}/{parts[idx+2]}"
                        ds = load_dataset(dataset_id, split="train[:100]")  # Limit to 100 samples
                        # Save images if it's an image dataset
                        if "image" in ds.features:
                            for i, item in enumerate(ds):
                                img = item["image"]
                                img_path = os.path.join(project_dir, f"image_{i:04d}.jpg")
                                img.save(img_path)
                                downloaded_files.append(img_path)
                        logger.info(f"Downloaded {len(downloaded_files)} images from HuggingFace")
            except ImportError:
                logger.warning("datasets package not installed. Run: pip install datasets")
            except Exception as he:
                logger.warning(f"HuggingFace download failed: {he}")
            
        elif source == "github":
            # GitHub raw files or releases
            if "raw.githubusercontent.com" in url or url.endswith(('.zip', '.tar.gz')):
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                if url.endswith('.zip'):
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        zf.extractall(project_dir)
                        downloaded_files = [os.path.join(project_dir, f) for f in zf.namelist()]
                        logger.info(f"Extracted {len(zf.namelist())} files from GitHub archive")
                else:
                    filename = url.split('/')[-1]
                    filepath = os.path.join(project_dir, filename)
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded_files.append(filepath)
                    logger.info(f"Downloaded {filename} from GitHub")
        else:
            # Direct URL download
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            filename = url.split('/')[-1] or "downloaded_file"
            filepath = os.path.join(project_dir, filename)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # If it's a zip, extract it
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filepath) as zf:
                    zf.extractall(project_dir)
                    downloaded_files = [os.path.join(project_dir, f) for f in zf.namelist()]
                os.remove(filepath)
                logger.info(f"Extracted archive to {project_dir}")
            else:
                downloaded_files.append(filepath)
                logger.info(f"Downloaded {filename}")
        
        # Register downloaded images in the database
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        with SqlSession(engine) as session:
            for filepath in downloaded_files:
                ext = os.path.splitext(filepath)[1].lower()
                if ext in image_extensions:
                    filename = os.path.basename(filepath)
                    image = Image(
                        project_id=project_id,
                        filename=filename,
                        status="pending",
                        width=0,
                        height=0
                    )
                    session.add(image)
            session.commit()
            logger.info(f"Registered {len(downloaded_files)} images in database")
                
        logger.info(f"Dataset import completed for project {project_id}")
        
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")

@router.get("/stats")
async def get_dashboard_stats(session: Session = Depends(get_session)):
    """Get dashboard statistics for home page."""
    projects = session.exec(select(Project)).all()
    images = session.exec(select(Image)).all()
    annotations = session.exec(select(Annotation)).all()
    
    # Get unique labels
    labels = set()
    for ann in annotations:
        if ann.label:
            labels.add(ann.label)
    
    # Recent activity (last 5 images by id)
    recent = sorted(images, key=lambda x: x.id, reverse=True)[:5]
    recent_activity = []
    for img in recent:
        project = session.get(Project, img.project_id)
        recent_activity.append({
            "id": img.id,
            "type": "image",
            "name": img.filename,
            "project": project.name if project else "Unknown",
            "time": "Recently"
        })
    
    return {
        "totalProjects": len(projects),
        "totalAssets": len(images),
        "totalAnnotations": len(annotations),
        "totalClasses": len(labels),
        "todayAnnotations": min(len(annotations), 24),
        "recentActivity": recent_activity
    }

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.get("/{project_id}/images")
async def get_project_images(project_id: str, session: Session = Depends(get_session)):
    """Get all images for a project with their annotations."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    
    result = []
    for img in images:
        annotations = session.exec(select(Annotation).where(Annotation.image_id == img.id)).all()
        result.append({
            "id": img.id,
            "filename": img.filename,
            "image_path": img.image_path,
            "status": img.status,
            "width": img.width,
            "height": img.height,
            "annotations": [
                {"id": a.id, "label": a.label, "bbox": a.bbox, "confidence": a.confidence}
                for a in annotations
            ]
        })
    
    return {"project_id": project_id, "images": result, "count": len(result)}

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

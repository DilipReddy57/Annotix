from fastapi import APIRouter, HTTPException, Body, Depends
from sqlmodel import Session, select
from backend.core.models import Project, Image
from backend.core.database import get_session
from backend.core.logger import get_logger

router = APIRouter()
logger = get_logger("qa")

@router.get("/{project_id}/queue")
async def get_review_queue(project_id: str, session: Session = Depends(get_session)):
    """
    Returns a list of images that need review, sorted by lowest confidence.
    Criteria: Status is 'processing' (manual flag) or 'completed' with low score.
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Fetch all images for the project
    # In a real app, we would write a more complex SQL query to filter this efficiently
    # For now, we'll fetch and filter in Python to keep logic consistent with MVP
    images = session.exec(select(Image).where(Image.project_id == project_id)).all()
    
    queue = []
    
    for img in images:
        # Logic: If any annotation has low score (< 0.8) or image is flagged
        needs_review = False
        min_score = 1.0
        
        if img.status == "flagged":
            needs_review = True
            
        # Accessing img.annotations triggers lazy loading if not eager loaded
        # This is fine for moderate dataset sizes
        for ann in img.annotations:
            if ann.confidence < 0.8:
                needs_review = True
            if ann.confidence < min_score:
                min_score = ann.confidence
                
        if needs_review and img.status != "reviewed" and img.status != "rejected":
            queue.append({
                "image": img,
                "priority_score": min_score
            })
            
    # Sort by lowest confidence first
    queue.sort(key=lambda x: x["priority_score"])
    
    return [item["image"] for item in queue]

@router.post("/images/{image_id}/review")
async def submit_review(image_id: str, decision: str = Body(..., embed=True), notes: str = Body(None, embed=True), session: Session = Depends(get_session)):
    """
    Submits a review decision for an image.
    Decision: 'approve', 'reject', 'flag'
    """
    image = session.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    if decision == "approve":
        image.status = "reviewed"
    elif decision == "reject":
        image.status = "rejected"
    elif decision == "flag":
        image.status = "flagged"
        
    session.add(image)
    session.commit()
    session.refresh(image)
        
    logger.info(f"Image {image_id} reviewed: {decision}")
    return {"status": "success", "new_status": image.status}

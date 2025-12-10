from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any
from sqlmodel import Session
from backend.core.models import Project, Image, Annotation
from backend.core.database import get_session
from backend.core.logger import get_logger
import datetime
import json

router = APIRouter()
logger = get_logger("export")

@router.get("/{project_id}/export/coco")
async def export_coco(project_id: str, session: Session = Depends(get_session)):
    """
    Generates a COCO format JSON for the project.
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    coco_output = {
        "info": {
            "description": project.name,
            "date_created": datetime.datetime.now().isoformat()
        },
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Build Categories
    category_map = {} # name -> id
    next_cat_id = 1
    
    # Build Images & Annotations
    ann_id_counter = 1
    
    for img in project.images:
        # Only export valid images
        if img.status == "rejected":
            continue
            
        coco_img = {
            "id": img.id, # Use UUID string or hash to int if strict COCO needed
            "file_name": img.filename,
            "width": img.width,
            "height": img.height
        }
        coco_output["images"].append(coco_img)
        
        for ann in img.annotations:
            if ann.label not in category_map:
                category_map[ann.label] = next_cat_id
                coco_output["categories"].append({
                    "id": next_cat_id,
                    "name": ann.label,
                    "supercategory": "object"
                })
                next_cat_id += 1
            
            # Parse JSON strings back to objects
            bbox = json.loads(ann.bbox_json) if ann.bbox_json else []
            segmentation = json.loads(ann.segmentation_json) if ann.segmentation_json else []
            
            coco_ann = {
                "id": ann_id_counter,
                "image_id": img.id,
                "category_id": category_map[ann.label],
                "bbox": bbox,
                "area": 0, # Calculate if needed
                "segmentation": segmentation,
                "iscrowd": 0,
                "score": ann.confidence
            }
            coco_output["annotations"].append(coco_ann)
            ann_id_counter += 1
            
    return JSONResponse(content=coco_output)

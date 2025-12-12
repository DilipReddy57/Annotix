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


@router.get("/{project_id}/export/yolo")
async def export_yolo(project_id: str, session: Session = Depends(get_session)):
    """
    Generates YOLO format annotations for the project.
    Returns a JSON with:
    - data_yaml: Content for data.yaml file
    - labels: Dict mapping image filename to YOLO format labels
    - classes: List of class names
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Build class mapping
    class_names = []
    class_map = {}  # name -> index
    
    # First pass: collect all unique labels
    for img in project.images:
        if img.status == "rejected":
            continue
        for ann in img.annotations:
            if ann.label not in class_map:
                class_map[ann.label] = len(class_names)
                class_names.append(ann.label)
    
    # Second pass: generate YOLO format labels
    labels = {}
    
    for img in project.images:
        if img.status == "rejected":
            continue
        
        img_labels = []
        img_width = img.width if img.width > 0 else 1
        img_height = img.height if img.height > 0 else 1
        
        for ann in img.annotations:
            bbox = json.loads(ann.bbox_json) if ann.bbox_json else []
            
            if len(bbox) >= 4:
                # COCO bbox is [x, y, width, height]
                # YOLO format is [class_id, x_center, y_center, width, height] (normalized)
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Normalize to 0-1
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                # Clamp values to 0-1
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))
                
                class_id = class_map[ann.label]
                img_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Use filename without extension + .txt
        label_filename = img.filename.rsplit('.', 1)[0] + '.txt'
        labels[label_filename] = "\n".join(img_labels)
    
    # Generate data.yaml content
    data_yaml = f"""# ANNOTIX YOLO Dataset
# Generated: {datetime.datetime.now().isoformat()}

path: ./dataset
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""

    logger.info(f"Exported YOLO format for project {project_id}: {len(labels)} images, {len(class_names)} classes")
    
    return JSONResponse(content={
        "format": "yolo",
        "project_name": project.name,
        "num_images": len(labels),
        "num_classes": len(class_names),
        "classes": class_names,
        "data_yaml": data_yaml,
        "labels": labels
    })


@router.get("/{project_id}/analytics")
async def get_project_analytics(project_id: str, session: Session = Depends(get_session)):
    """
    Get analytics data for a project including:
    - Annotation counts per class
    - Confidence distribution
    - Status breakdown
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Initialize analytics
    class_counts = {}
    confidence_buckets = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0
    }
    status_counts = {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "error": 0,
        "rejected": 0
    }
    total_annotations = 0
    
    for img in project.images:
        # Count status
        status = img.status or "pending"
        if status in status_counts:
            status_counts[status] += 1
        
        for ann in img.annotations:
            total_annotations += 1
            
            # Count by class
            label = ann.label or "unknown"
            class_counts[label] = class_counts.get(label, 0) + 1
            
            # Bucket by confidence
            conf = ann.confidence or 0
            if conf < 0.2:
                confidence_buckets["0.0-0.2"] += 1
            elif conf < 0.4:
                confidence_buckets["0.2-0.4"] += 1
            elif conf < 0.6:
                confidence_buckets["0.4-0.6"] += 1
            elif conf < 0.8:
                confidence_buckets["0.6-0.8"] += 1
            else:
                confidence_buckets["0.8-1.0"] += 1
    
    # Sort class counts by value
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "project_id": project_id,
        "project_name": project.name,
        "total_images": len(project.images),
        "total_annotations": total_annotations,
        "class_distribution": [{"name": k, "count": v} for k, v in sorted_classes],
        "confidence_distribution": [{"range": k, "count": v} for k, v in confidence_buckets.items()],
        "status_breakdown": [{"status": k, "count": v} for k, v in status_counts.items() if v > 0],
        "avg_annotations_per_image": round(total_annotations / max(len(project.images), 1), 2)
    }


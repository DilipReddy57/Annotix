import json
import os
from typing import Dict, Any, List
from backend.core.config import settings
from backend.core.logger import get_logger

logger = get_logger("aggregator")

class AggregatorAgent:
    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR
        self.annotations_file = os.path.join(self.output_dir, "annotations.json")
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, "r") as f:
                return json.load(f)
        return {"images": [], "annotations": [], "categories": []}

    def add_image(self, file_path: str, width: int, height: int) -> int:
        image_id = len(self.data["images"]) + 1
        self.data["images"].append({
            "id": image_id,
            "file_name": os.path.basename(file_path),
            "width": width,
            "height": height
        })
        return image_id

    def add_annotation(self, image_id: int, label: str, bbox: List[float], segmentation: Any, score: float):
        ann_id = len(self.data["annotations"]) + 1
        
        # Ensure category exists
        cat_id = self._get_category_id(label)
        
        self.data["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": bbox,
            "segmentation": segmentation,
            "score": score,
            "iscrowd": 0
        })

    def _get_category_id(self, label: str) -> int:
        for cat in self.data["categories"]:
            if cat["name"] == label:
                return cat["id"]
        
        new_id = len(self.data["categories"]) + 1
        self.data["categories"].append({"id": new_id, "name": label, "supercategory": "object"})
        return new_id

    def save_json(self):
        with open(self.annotations_file, "w") as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"Saved annotations to {self.annotations_file}")

    def get_analytics(self) -> Dict[str, Any]:
        """
        Computes dataset statistics for the dashboard.
        """
        total_images = len(self.data["images"])
        total_annotations = len(self.data["annotations"])
        
        # Class distribution
        class_counts = {}
        for ann in self.data["annotations"]:
            cat_id = ann["category_id"]
            cat_name = next((c["name"] for c in self.data["categories"] if c["id"] == cat_id), "unknown")
            class_counts[cat_name] = class_counts.get(cat_name, 0) + 1
            
        # Sort by frequency
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "class_distribution": dict(sorted_classes[:10]), # Top 10
            "categories": len(self.data["categories"]),
            "roi_metrics": self.calculate_roi(total_annotations)
        }

    def calculate_roi(self, total_annotations: int) -> Dict[str, float]:
        """
        Calculates cost savings compared to human annotation.
        Assumptions:
        - Human speed: 20 seconds per annotation
        - Human cost: $0.05 per annotation (approx $10/hr)
        - Agent cost: $0.001 per annotation (compute)
        """
        human_time_hours = (total_annotations * 20) / 3600
        human_cost = total_annotations * 0.05
        
        agent_time_hours = (total_annotations * 0.5) / 3600 # 0.5s per ann
        agent_cost = total_annotations * 0.001
        
        savings = human_cost - agent_cost
        
        return {
            "human_cost_est": round(human_cost, 2),
            "agent_cost_est": round(agent_cost, 2),
            "savings": round(savings, 2),
            "time_saved_hours": round(human_time_hours - agent_time_hours, 2)
        }

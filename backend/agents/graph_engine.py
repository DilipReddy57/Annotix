from typing import List, Dict, Any
from backend.core.logger import get_logger

logger = get_logger("graph_engine")

class SpatialGraphEngine:
    def build_graph(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Constructs a scene graph based on spatial relationships between objects.
        Returns a list of relationships: {"subject": id, "predicate": "left_of", "object": id}
        """
        relationships = []
        logger.info(f"Building spatial graph for {len(annotations)} objects...")
        
        for i, subj in enumerate(annotations):
            for j, obj in enumerate(annotations):
                if i == j:
                    continue
                
                rel = self._compute_relationship(subj["bbox"], obj["bbox"])
                if rel:
                    relationships.append({
                        "subject_id": subj.get("id", i),
                        "subject_label": subj.get("label", "object"),
                        "predicate": rel,
                        "object_id": obj.get("id", j),
                        "object_label": obj.get("label", "object")
                    })
        
        return relationships

    def _compute_relationship(self, bbox1: List[float], bbox2: List[float]) -> str:
        """
        Determines the spatial relationship between two bounding boxes [x, y, w, h].
        """
        # Convert to [x1, y1, x2, y2]
        x1_min, y1_min, w1, h1 = bbox1
        x1_max, y1_max = x1_min + w1, y1_min + h1
        
        x2_min, y2_min, w2, h2 = bbox2
        x2_max, y2_max = x2_min + w2, y2_min + h2
        
        # Center points
        c1_x, c1_y = x1_min + w1/2, y1_min + h1/2
        c2_x, c2_y = x2_min + w2/2, y2_min + h2/2
        
        # Overlap check (IoU-like)
        intersect_x_min = max(x1_min, x2_min)
        intersect_y_min = max(y1_min, y2_min)
        intersect_x_max = min(x1_max, x2_max)
        intersect_y_max = min(y1_max, y2_max)
        
        if intersect_x_max > intersect_x_min and intersect_y_max > intersect_y_min:
            return "overlapping"
            
        # Directional checks
        if c1_x < x2_min: return "left_of"
        if c1_x > x2_max: return "right_of"
        if c1_y < y2_min: return "above"
        if c1_y > y2_max: return "below"
        
        return None

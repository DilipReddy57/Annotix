from typing import Dict, Any
from backend.core.logger import get_logger

logger = get_logger("qa")

class QAAgent:
    def validate_annotation(self, mask: Any, bbox: Any, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the quality of an annotation and computes a fused confidence score.
        """
        # Extract individual scores
        sam_score = classification.get("sam_score", 1.0) # IoU from SAM
        class_score = classification.get("confidence", 1.0) # CLIP confidence
        
        # 1. Heuristic Checks
        valid = True
        reason = None
        
        # Check 1: Area (Mock)
        # if area < 10: valid = False
        
        # 2. Confidence Fusion
        final_score = self.calculate_confidence(sam_score, class_score)
        
        # Threshold check
        if final_score < 0.6:
            valid = False
            reason = "low_fused_confidence"
            logger.warning(f"Annotation rejected. Score: {final_score:.2f}")

        return {
            "valid": valid, 
            "reason": reason, 
            "score": final_score
        }

    def calculate_confidence(self, sam_score: float, class_score: float) -> float:
        """
        Fuses multiple confidence metrics into a single quality score.
        Weights:
        - SAM 3 IoU: 40%
        - Classifier Confidence: 60%
        """
        # Weighted fusion
        fused_score = (sam_score * 0.4) + (class_score * 0.6)
        return round(fused_score, 4)

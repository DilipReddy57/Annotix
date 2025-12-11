"""
Instance Learning Agent for Few-Shot Learning from User Corrections.

This agent provides:
- Learning from user label corrections
- Few-shot adaptation to new labels
- Confidence boosting for validated annotations
- Correction history tracking

Features:
- Real-time learning from corrections
- Similarity-based propagation
- Automatic relabeling suggestions
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np

from backend.core.logger import get_logger
from backend.core.config import settings

logger = get_logger("instance_learner")


@dataclass
class CorrectionEntry:
    """A single correction entry."""
    id: str
    original_label: str
    corrected_label: str
    embedding: Optional[List[float]]
    image_id: str
    project_id: str
    timestamp: str
    bbox: Optional[List[float]] = None
    confidence_boost: float = 0.1
    propagated: bool = False


class InstanceLearner:
    """
    Instance Learning Agent for few-shot learning from user corrections.
    
    Learns from:
    - User renames (label corrections)
    - User approvals (confidence boost)
    - User rejections (negative examples)
    
    Provides:
    - Automatic relabeling suggestions
    - Confidence adjustments
    - Similar instance identification
    """
    
    def __init__(
        self, 
        persist_path: Optional[str] = None,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize Instance Learner.
        
        Args:
            persist_path: Path to persist corrections (JSON file)
            similarity_threshold: Threshold for similarity-based propagation
        """
        self.persist_path = persist_path or os.path.join(
            settings.DATA_DIR, "instance_corrections.json"
        )
        self.similarity_threshold = similarity_threshold
        
        # Correction storage
        self.corrections: Dict[str, CorrectionEntry] = {}
        self.label_mapping: Dict[str, str] = {}  # original -> corrected
        self.label_counts: Dict[str, int] = defaultdict(int)
        self.project_corrections: Dict[str, List[str]] = defaultdict(list)
        
        # Approval/rejection tracking
        self.approved_labels: Dict[str, int] = defaultdict(int)
        self.rejected_labels: Dict[str, int] = defaultdict(int)
        
        # Load existing corrections
        self._load_corrections()
        
        logger.info(f"InstanceLearner initialized with {len(self.corrections)} corrections")
    
    def _load_corrections(self):
        """Load corrections from persistent storage."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                
                for entry_data in data.get("corrections", []):
                    entry = CorrectionEntry(**entry_data)
                    self.corrections[entry.id] = entry
                    self.label_mapping[entry.original_label.lower()] = entry.corrected_label
                    self.label_counts[entry.corrected_label] += 1
                    self.project_corrections[entry.project_id].append(entry.id)
                
                self.label_mapping.update(data.get("label_mapping", {}))
                self.approved_labels.update(data.get("approved_labels", {}))
                self.rejected_labels.update(data.get("rejected_labels", {}))
                
                logger.info(f"Loaded {len(self.corrections)} corrections from {self.persist_path}")
            except Exception as e:
                logger.error(f"Failed to load corrections: {e}")
    
    def _save_corrections(self):
        """Save corrections to persistent storage."""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            
            data = {
                "corrections": [asdict(c) for c in self.corrections.values()],
                "label_mapping": dict(self.label_mapping),
                "approved_labels": dict(self.approved_labels),
                "rejected_labels": dict(self.rejected_labels),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.corrections)} corrections")
        except Exception as e:
            logger.error(f"Failed to save corrections: {e}")
    
    def record_correction(
        self,
        original_label: str,
        corrected_label: str,
        image_id: str,
        project_id: str,
        embedding: Optional[np.ndarray] = None,
        bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Record a user correction (label rename).
        
        Args:
            original_label: Original label from detector
            corrected_label: User's corrected label
            image_id: Image where correction was made
            project_id: Project ID
            embedding: Visual embedding for similarity search
            bbox: Bounding box coordinates
            
        Returns:
            Dict with correction info and propagation suggestions
        """
        entry_id = str(uuid.uuid4())[:8]
        
        entry = CorrectionEntry(
            id=entry_id,
            original_label=original_label,
            corrected_label=corrected_label,
            embedding=embedding.tolist() if embedding is not None else None,
            image_id=image_id,
            project_id=project_id,
            timestamp=datetime.utcnow().isoformat(),
            bbox=bbox
        )
        
        self.corrections[entry_id] = entry
        self.label_mapping[original_label.lower()] = corrected_label
        self.label_counts[corrected_label] += 1
        self.project_corrections[project_id].append(entry_id)
        
        self._save_corrections()
        
        logger.info(f"Recorded correction: '{original_label}' -> '{corrected_label}'")
        
        # Find similar entries that might need same correction
        suggestions = self._find_similar_corrections(embedding, original_label, project_id)
        
        return {
            "correction_id": entry_id,
            "from": original_label,
            "to": corrected_label,
            "learned": True,
            "propagation_suggestions": suggestions
        }
    
    def _find_similar_corrections(
        self,
        embedding: Optional[np.ndarray],
        original_label: str,
        project_id: str
    ) -> List[Dict[str, Any]]:
        """Find other annotations that might need the same correction."""
        if embedding is None:
            return []
        
        suggestions = []
        query_vec = np.array(embedding)
        
        for cid, entry in self.corrections.items():
            if entry.embedding is None:
                continue
            if entry.project_id != project_id:
                continue
            if entry.original_label.lower() == original_label.lower():
                continue
            
            entry_vec = np.array(entry.embedding)
            similarity = np.dot(query_vec, entry_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(entry_vec) + 1e-8
            )
            
            if similarity > self.similarity_threshold:
                suggestions.append({
                    "correction_id": cid,
                    "original_label": entry.original_label,
                    "similarity": float(similarity)
                })
        
        return sorted(suggestions, key=lambda x: x["similarity"], reverse=True)[:5]
    
    def apply_learned_correction(self, label: str) -> Tuple[str, float]:
        """
        Apply learned correction to a label.
        
        Args:
            label: Input label
            
        Returns:
            Tuple of (corrected_label, confidence_boost)
        """
        label_lower = label.lower()
        
        if label_lower in self.label_mapping:
            corrected = self.label_mapping[label_lower]
            count = self.label_counts.get(corrected, 1)
            # Confidence boost increases with more examples
            boost = min(0.1 + (count * 0.02), 0.3)
            return corrected, boost
        
        return label, 0.0
    
    def record_approval(self, label: str, project_id: str):
        """Record that a label was approved by user."""
        self.approved_labels[label.lower()] += 1
        self._save_corrections()
        logger.debug(f"Recorded approval for '{label}'")
    
    def record_rejection(self, label: str, project_id: str):
        """Record that a label was rejected by user."""
        self.rejected_labels[label.lower()] += 1
        self._save_corrections()
        logger.debug(f"Recorded rejection for '{label}'")
    
    def get_confidence_adjustment(self, label: str) -> float:
        """
        Get confidence adjustment based on approval/rejection history.
        
        Args:
            label: Label to check
            
        Returns:
            Confidence adjustment (-0.2 to +0.2)
        """
        label_lower = label.lower()
        approvals = self.approved_labels.get(label_lower, 0)
        rejections = self.rejected_labels.get(label_lower, 0)
        
        if approvals + rejections == 0:
            return 0.0
        
        approval_ratio = approvals / (approvals + rejections)
        # Scale to -0.2 to +0.2
        return (approval_ratio - 0.5) * 0.4
    
    def get_learned_mappings(self, project_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get all learned label mappings.
        
        Args:
            project_id: Optional filter by project
            
        Returns:
            Dict mapping original labels to corrected labels
        """
        if project_id is None:
            return dict(self.label_mapping)
        
        # Filter by project
        project_mappings = {}
        for cid in self.project_corrections.get(project_id, []):
            if cid in self.corrections:
                entry = self.corrections[cid]
                project_mappings[entry.original_label.lower()] = entry.corrected_label
        
        return project_mappings
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_corrections": len(self.corrections),
            "unique_mappings": len(self.label_mapping),
            "label_counts": dict(self.label_counts),
            "approved_labels": dict(self.approved_labels),
            "rejected_labels": dict(self.rejected_labels),
            "projects": list(self.project_corrections.keys())
        }
    
    def export_mappings(self, output_path: str):
        """Export label mappings to JSON."""
        data = {
            "mappings": self.label_mapping,
            "statistics": {
                "corrections": len(self.corrections),
                "approvals": dict(self.approved_labels),
                "rejections": dict(self.rejected_labels)
            },
            "exported_at": datetime.utcnow().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.label_mapping)} mappings to {output_path}")

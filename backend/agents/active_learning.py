"""
Active Learning Agent for intelligent sample selection.

Features:
- Identifies most valuable images to annotate next
- Uncertainty sampling based on model confidence
- Diversity sampling for balanced dataset coverage
- Query strategies for efficient annotation workflows
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import json

from backend.core.logger import get_logger
from backend.core.config import settings

logger = get_logger("active_learning")


class ActiveLearningAgent:
    """
    Active Learning Agent for intelligent sample prioritization.
    
    Strategies:
    1. Uncertainty Sampling - Prioritize images where model is least confident
    2. Diversity Sampling - Select diverse samples for better coverage
    3. Query-by-Committee - Use disagreement between multiple models
    4. Expected Model Change - Select samples that would most change the model
    """
    
    def __init__(self):
        self.annotation_history: List[Dict[str, Any]] = []
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.confidence_cache: Dict[str, float] = {}
        
    def add_to_history(
        self,
        image_path: str,
        annotations: List[Dict[str, Any]],
        embedding: Optional[np.ndarray] = None
    ):
        """
        Add an annotated image to the history.
        
        Args:
            image_path: Path to the annotated image
            annotations: List of annotations
            embedding: Optional image embedding for diversity calculation
        """
        avg_confidence = np.mean([
            ann.get("score", 0.5) for ann in annotations
        ]) if annotations else 0.5
        
        labels = [ann.get("label", "unknown") for ann in annotations]
        
        self.annotation_history.append({
            "image_path": image_path,
            "num_annotations": len(annotations),
            "avg_confidence": avg_confidence,
            "labels": labels,
            "timestamp": len(self.annotation_history)
        })
        
        if embedding is not None:
            self.embedding_cache[image_path] = embedding
        
        self.confidence_cache[image_path] = avg_confidence
    
    def calculate_uncertainty_score(
        self,
        predictions: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate uncertainty score for predictions.
        
        Uses entropy-based uncertainty:
        - High entropy = High uncertainty = More valuable to annotate
        
        Args:
            predictions: List of predictions with scores
            
        Returns:
            Uncertainty score (higher = more uncertain)
        """
        if not predictions:
            return 1.0  # No predictions = high uncertainty
        
        scores = [p.get("score", 0.5) for p in predictions]
        
        # Entropy-based uncertainty
        scores = np.array(scores)
        scores = np.clip(scores, 1e-10, 1 - 1e-10)  # Avoid log(0)
        
        entropy = -np.mean(scores * np.log(scores) + (1 - scores) * np.log(1 - scores))
        
        # Also consider disagreement (variance in scores)
        variance = np.var(scores)
        
        # Combined uncertainty
        uncertainty = 0.7 * entropy + 0.3 * variance
        
        return float(uncertainty)
    
    def calculate_diversity_score(
        self,
        image_embedding: np.ndarray,
        existing_embeddings: List[np.ndarray]
    ) -> float:
        """
        Calculate how diverse/different an image is from existing annotations.
        
        Args:
            image_embedding: Embedding of the candidate image
            existing_embeddings: Embeddings of already annotated images
            
        Returns:
            Diversity score (higher = more different from existing)
        """
        if not existing_embeddings:
            return 1.0  # First image is maximally diverse
        
        # Calculate cosine similarities to all existing
        similarities = []
        for existing in existing_embeddings:
            sim = np.dot(image_embedding, existing) / (
                np.linalg.norm(image_embedding) * np.linalg.norm(existing) + 1e-10
            )
            similarities.append(sim)
        
        # Diversity = 1 - max_similarity (most different from nearest neighbor)
        max_similarity = max(similarities)
        diversity = 1 - max_similarity
        
        return float(diversity)
    
    def calculate_label_balance_score(
        self,
        predicted_labels: List[str]
    ) -> float:
        """
        Score based on whether predicted labels are underrepresented.
        
        Args:
            predicted_labels: Labels predicted for this image
            
        Returns:
            Balance score (higher = contains underrepresented classes)
        """
        # Count existing labels
        label_counts = defaultdict(int)
        for entry in self.annotation_history:
            for label in entry.get("labels", []):
                label_counts[label] += 1
        
        if not label_counts:
            return 1.0  # No history, any labels are valuable
        
        # Calculate rarity score for predicted labels
        total_annotations = sum(label_counts.values())
        rarity_scores = []
        
        for label in predicted_labels:
            count = label_counts.get(label, 0)
            # Rarer labels get higher scores
            rarity = 1 - (count / (total_annotations + 1))
            rarity_scores.append(rarity)
        
        return float(np.mean(rarity_scores)) if rarity_scores else 0.5
    
    def rank_images_for_annotation(
        self,
        candidate_images: List[str],
        predictions_per_image: Dict[str, List[Dict[str, Any]]],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        strategy: str = "combined",
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate images by annotation priority.
        
        Args:
            candidate_images: List of image paths to rank
            predictions_per_image: Predictions for each image
            embeddings: Optional embeddings per image
            strategy: "uncertainty", "diversity", "balance", or "combined"
            weights: Optional custom weights for combined strategy
            
        Returns:
            Ranked list of images with scores
        """
        if weights is None:
            weights = {
                "uncertainty": 0.4,
                "diversity": 0.3,
                "balance": 0.3
            }
        
        # Get existing embeddings
        existing_embeddings = list(self.embedding_cache.values())
        
        ranked = []
        
        for image_path in candidate_images:
            predictions = predictions_per_image.get(image_path, [])
            
            # Calculate component scores
            uncertainty = self.calculate_uncertainty_score(predictions)
            
            diversity = 0.5
            if embeddings and image_path in embeddings:
                diversity = self.calculate_diversity_score(
                    embeddings[image_path],
                    existing_embeddings
                )
            
            predicted_labels = [p.get("label", "unknown") for p in predictions]
            balance = self.calculate_label_balance_score(predicted_labels)
            
            # Combined score based on strategy
            if strategy == "uncertainty":
                final_score = uncertainty
            elif strategy == "diversity":
                final_score = diversity
            elif strategy == "balance":
                final_score = balance
            else:  # combined
                final_score = (
                    weights["uncertainty"] * uncertainty +
                    weights["diversity"] * diversity +
                    weights["balance"] * balance
                )
            
            ranked.append({
                "image_path": image_path,
                "priority_score": final_score,
                "uncertainty_score": uncertainty,
                "diversity_score": diversity,
                "balance_score": balance,
                "predicted_labels": predicted_labels,
                "num_predictions": len(predictions)
            })
        
        # Sort by priority (descending)
        ranked.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return ranked
    
    def get_next_batch(
        self,
        candidate_images: List[str],
        predictions_per_image: Dict[str, List[Dict[str, Any]]],
        batch_size: int = 10,
        embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> List[str]:
        """
        Get the next batch of images to annotate.
        
        Args:
            candidate_images: Pool of unannotated images
            predictions_per_image: Initial predictions for each
            batch_size: Number of images to select
            embeddings: Optional embeddings
            
        Returns:
            List of image paths to annotate next
        """
        # Filter out already annotated
        annotated = set(e["image_path"] for e in self.annotation_history)
        candidates = [img for img in candidate_images if img not in annotated]
        
        if not candidates:
            logger.info("All candidate images have been annotated!")
            return []
        
        # Rank and select top batch_size
        ranked = self.rank_images_for_annotation(
            candidates,
            predictions_per_image,
            embeddings,
            strategy="combined"
        )
        
        selected = [r["image_path"] for r in ranked[:batch_size]]
        
        logger.info(f"Selected {len(selected)} images for next annotation batch")
        logger.info(f"Top priority image: {selected[0] if selected else 'None'}")
        
        return selected
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        if not self.annotation_history:
            return {
                "total_annotated": 0,
                "labels": {},
                "avg_confidence": 0.0
            }
        
        # Label distribution
        label_counts = defaultdict(int)
        for entry in self.annotation_history:
            for label in entry.get("labels", []):
                label_counts[label] += 1
        
        # Confidence over time
        confidences = [e["avg_confidence"] for e in self.annotation_history]
        
        return {
            "total_annotated": len(self.annotation_history),
            "total_annotations": sum(e["num_annotations"] for e in self.annotation_history),
            "labels": dict(label_counts),
            "num_classes": len(label_counts),
            "avg_confidence": float(np.mean(confidences)),
            "confidence_trend": confidences[-10:] if len(confidences) > 10 else confidences,
            "underrepresented_classes": [
                label for label, count in label_counts.items()
                if count < np.mean(list(label_counts.values())) * 0.5
            ]
        }
    
    def suggest_prompts_for_balance(self, top_k: int = 5) -> List[str]:
        """
        Suggest prompts to improve class balance.
        
        Returns:
            List of suggested prompts for underrepresented classes
        """
        stats = self.get_statistics()
        underrepresented = stats.get("underrepresented_classes", [])
        
        if not underrepresented:
            return ["Focus on any visible objects"]
        
        return underrepresented[:top_k]

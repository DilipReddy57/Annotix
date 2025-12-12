"""
Real Embedding Extractor using CLIP.

This provides ACTUAL visual embeddings for RAG - not random noise.
CLIP is fast (~50ms per crop) and provides semantically meaningful embeddings.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple
from functools import lru_cache

from backend.core.logger import get_logger

logger = get_logger("embedding_extractor")


class EmbeddingExtractor:
    """
    Fast visual embedding extraction using CLIP.
    
    Why CLIP:
    - ~50ms per image crop (fast)
    - Semantically meaningful (a "cat" embedding is similar to other cats)
    - Works with any image, no training needed
    - Model is ~400MB, reasonable for most systems
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - only load CLIP once."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        self._load_failed = False
        
        # Lazy load - don't block startup
        logger.info("EmbeddingExtractor initialized (CLIP will load on first use)")
    
    def _ensure_loaded(self) -> bool:
        """Lazy load CLIP model."""
        if self.model is not None:
            return True
        if self._load_failed:
            return False
            
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info("Loading CLIP model for real embeddings...")
            
            # Use smaller CLIP variant for speed
            model_name = "openai/clip-vit-base-patch32"
            
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            logger.info(f"✅ CLIP loaded on {self.device}")
            return True
            
        except Exception as e:
            logger.warning(f"CLIP not available: {e}")
            logger.warning("RAG will use label-based text embeddings as fallback")
            self._load_failed = True
            return False
    
    def extract_embedding(
        self, 
        image: Image.Image,
        bbox: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """
        Extract real visual embedding from image or crop.
        
        Args:
            image: PIL Image
            bbox: Optional [x, y, w, h] to crop before embedding
            
        Returns:
            512-dim numpy array, or None if extraction failed
        """
        if not self._ensure_loaded():
            return None
            
        try:
            # Crop if bbox provided
            if bbox and len(bbox) == 4:
                x, y, w, h = bbox
                # Ensure valid crop
                img_w, img_h = image.size
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = max(1, min(w, img_w - x))
                h = max(1, min(h, img_h - y))
                image = image.crop((x, y, x + w, y + h))
            
            # Process with CLIP
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                features = self.model.get_image_features(**inputs)
                
                # Normalize embedding
                embedding = features.cpu().numpy()[0]
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return None
    
    def extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Extract text embedding for label-based similarity.
        
        Args:
            text: Label text (e.g., "cat", "person")
            
        Returns:
            512-dim numpy array
        """
        if not self._ensure_loaded():
            return None
            
        try:
            with torch.no_grad():
                inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
                features = self.model.get_text_features(**inputs)
                
                embedding = features.cpu().numpy()[0]
                embedding = embedding / np.linalg.norm(embedding)
                
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Text embedding failed: {e}")
            return None
    
    def batch_extract(
        self, 
        image: Image.Image, 
        bboxes: List[List[int]]
    ) -> List[Optional[np.ndarray]]:
        """
        Batch extract embeddings for multiple crops - more efficient.
        
        Args:
            image: Full PIL Image
            bboxes: List of [x, y, w, h] bounding boxes
            
        Returns:
            List of embeddings (None for failed extractions)
        """
        if not self._ensure_loaded() or not bboxes:
            return [None] * len(bboxes)
            
        try:
            # Crop all regions
            crops = []
            for bbox in bboxes:
                x, y, w, h = bbox
                img_w, img_h = image.size
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                w = max(1, min(w, img_w - x))
                h = max(1, min(h, img_h - y))
                crop = image.crop((x, y, x + w, y + h))
                crops.append(crop)
            
            # Batch process
            with torch.no_grad():
                inputs = self.processor(images=crops, return_tensors="pt", padding=True).to(self.device)
                features = self.model.get_image_features(**inputs)
                
                embeddings = features.cpu().numpy()
                # Normalize each
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
                
            return [emb.astype(np.float32) for emb in embeddings]
            
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            return [None] * len(bboxes)


# Singleton accessor
def get_embedding_extractor() -> EmbeddingExtractor:
    return EmbeddingExtractor()

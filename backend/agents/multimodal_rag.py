"""
Enhanced Multi-Modal RAG Agent for rich annotation intelligence.

Features:
- Dual embedding storage (visual + text)
- Semantic label understanding via LLM
- Cross-modal retrieval
- Intelligent label hierarchy and ontology
"""

import os
import json
import uuid
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict

from backend.core.logger import get_logger
from backend.core.config import settings

logger = get_logger("multimodal_rag")


class MultiModalRAGAgent:
    """
    Enhanced Multi-Modal RAG Agent combining visual and text understanding.
    
    Features:
    - Dual collection: visual_embeddings + text_embeddings
    - Semantic label understanding
    - Cross-modal search
    - Label ontology management
    - LLM-enhanced label correction
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        use_llm: bool = True
    ):
        """
        Initialize Multi-Modal RAG Agent.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            use_llm: Whether to use LLM for semantic understanding
        """
        self.persist_directory = persist_directory or os.path.join(
            settings.DATA_DIR, "chromadb_multimodal"
        )
        
        self.client = None
        self.visual_collection = None
        self.text_collection = None
        self.llm_agent = None
        
        # Label ontology (hierarchical labels)
        self.label_ontology: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_chromadb()
        self._load_ontology()
        
        if use_llm:
            self._initialize_llm()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with dual collections."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Visual embeddings collection (from SAM3 encoder)
            self.visual_collection = self.client.get_or_create_collection(
                name="visual_embeddings",
                metadata={
                    "description": "Visual feature embeddings from SAM3",
                    "hnsw:space": "cosine"
                }
            )
            
            # Text embeddings collection (from text encoder)
            self.text_collection = self.client.get_or_create_collection(
                name="text_embeddings",
                metadata={
                    "description": "Text description embeddings",
                    "hnsw:space": "cosine"
                }
            )
            
            logger.info(f"✅ Multi-Modal ChromaDB initialized")
            logger.info(f"   Visual entries: {self.visual_collection.count()}")
            logger.info(f"   Text entries: {self.text_collection.count()}")
            
        except ImportError:
            logger.warning("ChromaDB not installed, using memory fallback")
            self._use_fallback()
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            self._use_fallback()
    
    def _use_fallback(self):
        """Fallback to in-memory storage."""
        self.client = None
        self.visual_collection = None
        self.text_collection = None
        self._memory_store = {
            "visual": [],
            "text": []
        }
    
    def _initialize_llm(self):
        """Initialize LLM for semantic understanding."""
        try:
            from backend.agents.llm_agent import LLMAgent
            self.llm_agent = LLMAgent(provider="gemini")
            logger.info("✅ LLM Agent connected for semantic understanding")
        except Exception as e:
            logger.warning(f"LLM Agent not available: {e}")
            self.llm_agent = None
    
    def _load_ontology(self):
        """Load label ontology for hierarchical understanding."""
        # Default ontology with parent-child relationships
        self.label_ontology = {
            # Vehicles
            "vehicle": {
                "children": ["car", "truck", "motorcycle", "bicycle", "bus"],
                "synonyms": ["transport", "automobile"]
            },
            "car": {
                "parent": "vehicle",
                "synonyms": ["auto", "automobile", "sedan", "hatchback", "suv"]
            },
            "truck": {
                "parent": "vehicle",
                "synonyms": ["lorry", "pickup", "van"]
            },
            
            # People
            "person": {
                "children": ["man", "woman", "child"],
                "synonyms": ["human", "people", "pedestrian", "individual"]
            },
            
            # Animals
            "animal": {
                "children": ["dog", "cat", "bird", "horse"],
                "synonyms": ["creature", "pet"]
            },
            "dog": {
                "parent": "animal",
                "synonyms": ["puppy", "canine", "pup"]
            },
            
            # Objects
            "furniture": {
                "children": ["chair", "table", "sofa", "bed"],
                "synonyms": ["furnishing"]
            }
        }
        
        # Load custom ontology if exists
        ontology_path = os.path.join(settings.DATA_DIR, "label_ontology.json")
        if os.path.exists(ontology_path):
            try:
                with open(ontology_path, "r") as f:
                    custom = json.load(f)
                    self.label_ontology.update(custom)
                    logger.info(f"Loaded custom ontology with {len(custom)} entries")
            except Exception as e:
                logger.warning(f"Failed to load custom ontology: {e}")
    
    def get_canonical_label(self, label: str) -> str:
        """
        Get canonical label using ontology and synonyms.
        
        Args:
            label: Input label
            
        Returns:
            Canonical normalized label
        """
        label_lower = label.lower().strip()
        
        # Direct match
        if label_lower in self.label_ontology:
            return label_lower
        
        # Check synonyms
        for canonical, info in self.label_ontology.items():
            synonyms = info.get("synonyms", [])
            if label_lower in [s.lower() for s in synonyms]:
                return canonical
        
        return label_lower
    
    def get_parent_label(self, label: str) -> Optional[str]:
        """Get parent category from ontology."""
        canonical = self.get_canonical_label(label)
        info = self.label_ontology.get(canonical, {})
        return info.get("parent")
    
    def add_multimodal_entry(
        self,
        entry_id: Optional[str] = None,
        visual_embedding: Optional[np.ndarray] = None,
        text_embedding: Optional[np.ndarray] = None,
        label: str = "object",
        description: Optional[str] = None,
        image_id: str = "",
        project_id: Optional[str] = None,
        bbox: Optional[List[int]] = None,
        confidence: float = 1.0,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a multi-modal entry with both visual and text embeddings.
        
        Args:
            entry_id: Optional custom ID
            visual_embedding: Visual feature embedding
            text_embedding: Text description embedding
            label: Object label
            description: Text description of the object
            image_id: Source image ID
            project_id: Project ID
            bbox: Bounding box
            confidence: Detection confidence
            additional_metadata: Extra metadata
            
        Returns:
            Entry ID
        """
        entry_id = entry_id or str(uuid.uuid4())
        canonical_label = self.get_canonical_label(label)
        parent_label = self.get_parent_label(canonical_label)
        
        base_metadata = {
            "label": canonical_label,
            "original_label": label,
            "parent_label": parent_label or "",
            "description": description or "",
            "image_id": image_id,
            "project_id": project_id or "default",
            "confidence": confidence,
            "bbox": json.dumps(bbox) if bbox else "null"
        }
        
        if additional_metadata:
            base_metadata.update({
                k: str(v) for k, v in additional_metadata.items()
            })
        
        # Add visual embedding
        if visual_embedding is not None and self.visual_collection is not None:
            try:
                self.visual_collection.add(
                    embeddings=[visual_embedding.tolist()],
                    metadatas=[base_metadata],
                    ids=[f"vis_{entry_id}"]
                )
            except Exception as e:
                logger.error(f"Failed to add visual embedding: {e}")
        
        # Add text embedding (or use document for auto-embedding)
        if self.text_collection is not None:
            try:
                doc_text = description or f"{canonical_label} with {confidence:.2f} confidence"
                
                if text_embedding is not None:
                    self.text_collection.add(
                        embeddings=[text_embedding.tolist()],
                        metadatas=[base_metadata],
                        ids=[f"txt_{entry_id}"]
                    )
                else:
                    # Let ChromaDB generate embedding from document
                    self.text_collection.add(
                        documents=[doc_text],
                        metadatas=[base_metadata],
                        ids=[f"txt_{entry_id}"]
                    )
            except Exception as e:
                logger.error(f"Failed to add text embedding: {e}")
        
        logger.debug(f"Added multi-modal entry: {canonical_label} ({entry_id})")
        return entry_id
    
    def retrieve_by_visual_similarity(
        self,
        visual_embedding: np.ndarray,
        n_results: int = 10,
        filter_project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar entries by visual similarity.
        
        Args:
            visual_embedding: Query visual embedding
            n_results: Number of results
            filter_project: Optional project filter
            
        Returns:
            List of similar entries with metadata
        """
        if self.visual_collection is None or self.visual_collection.count() == 0:
            return []
        
        try:
            where = {"project_id": filter_project} if filter_project else None
            
            results = self.visual_collection.query(
                query_embeddings=[visual_embedding.tolist()],
                n_results=n_results,
                where=where
            )
            
            return self._format_results(results)
            
        except Exception as e:
            logger.error(f"Visual retrieval failed: {e}")
            return []
    
    def retrieve_by_text_query(
        self,
        query_text: str,
        n_results: int = 10,
        filter_project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve entries by text semantic search.
        
        Args:
            query_text: Text query (e.g., "red car", "person walking")
            n_results: Number of results
            filter_project: Optional project filter
            
        Returns:
            List of matching entries
        """
        if self.text_collection is None or self.text_collection.count() == 0:
            return []
        
        try:
            where = {"project_id": filter_project} if filter_project else None
            
            results = self.text_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            
            return self._format_results(results)
            
        except Exception as e:
            logger.error(f"Text retrieval failed: {e}")
            return []
    
    def retrieve_cross_modal(
        self,
        visual_embedding: Optional[np.ndarray] = None,
        text_query: Optional[str] = None,
        n_results: int = 10,
        visual_weight: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Cross-modal retrieval combining visual and text search.
        
        Args:
            visual_embedding: Optional visual embedding
            text_query: Optional text query
            n_results: Number of results
            visual_weight: Weight for visual vs text (0-1)
            
        Returns:
            Fused results from both modalities
        """
        visual_results = []
        text_results = []
        
        if visual_embedding is not None:
            visual_results = self.retrieve_by_visual_similarity(
                visual_embedding, n_results=n_results * 2
            )
        
        if text_query:
            text_results = self.retrieve_by_text_query(
                text_query, n_results=n_results * 2
            )
        
        # Fuse results
        all_results = {}
        
        for i, r in enumerate(visual_results):
            entry_id = r.get("id", "").replace("vis_", "")
            score = (1 - r.get("distance", 0)) * visual_weight
            
            if entry_id not in all_results:
                all_results[entry_id] = r.copy()
                all_results[entry_id]["fused_score"] = score
            else:
                all_results[entry_id]["fused_score"] += score
        
        text_weight = 1 - visual_weight
        for i, r in enumerate(text_results):
            entry_id = r.get("id", "").replace("txt_", "")
            score = (1 - r.get("distance", 0)) * text_weight
            
            if entry_id not in all_results:
                all_results[entry_id] = r.copy()
                all_results[entry_id]["fused_score"] = score
            else:
                all_results[entry_id]["fused_score"] += score
        
        # Sort by fused score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.get("fused_score", 0),
            reverse=True
        )
        
        return sorted_results[:n_results]
    
    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results into clean list."""
        formatted = []
        
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for i in range(len(ids)):
            entry = {
                "id": ids[i],
                "distance": distances[i] if i < len(distances) else 0,
                "similarity": 1 - distances[i] if i < len(distances) else 1,
                **metadatas[i]
            }
            formatted.append(entry)
        
        return formatted
    
    def suggest_label_with_llm(
        self,
        visual_embedding: np.ndarray,
        proposed_label: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to suggest or refine labels based on context.
        
        Args:
            visual_embedding: Visual embedding
            proposed_label: Initially proposed label
            image_path: Optional path to image for LLM analysis
            
        Returns:
            Dict with suggested label and reasoning
        """
        # Get similar annotations from knowledge base
        similar = self.retrieve_by_visual_similarity(visual_embedding, n_results=5)
        
        # Analyze context
        label_votes = Counter()
        for s in similar:
            if s.get("similarity", 0) > 0.8:
                label_votes[s.get("label", "unknown")] += s.get("similarity", 0)
        
        most_common = label_votes.most_common(1)
        kb_suggestion = most_common[0][0] if most_common else proposed_label
        
        result = {
            "proposed": proposed_label,
            "kb_suggestion": kb_suggestion,
            "final": kb_suggestion,
            "similar_entries": len(similar),
            "confidence": "high" if label_votes.get(kb_suggestion, 0) > 2 else "medium",
            "reasoning": f"Based on {len(similar)} similar annotations"
        }
        
        # Optionally use LLM for refinement
        if self.llm_agent and image_path:
            try:
                llm_result = self.llm_agent.refine_prompt(proposed_label, image_path)
                if llm_result.get("refined") and llm_result["refined"] != proposed_label:
                    result["llm_suggestion"] = llm_result["refined"]
                    result["reasoning"] += f". LLM suggests: {llm_result.get('reasoning', '')}"
            except Exception as e:
                logger.debug(f"LLM refinement skipped: {e}")
        
        return result
    
    def get_ontology_statistics(self) -> Dict[str, Any]:
        """Get statistics about label usage by ontology."""
        stats = {
            "by_parent": defaultdict(int),
            "by_label": defaultdict(int),
            "total_visual": 0,
            "total_text": 0
        }
        
        if self.visual_collection:
            stats["total_visual"] = self.visual_collection.count()
            
            try:
                all_entries = self.visual_collection.get(include=["metadatas"])
                for meta in all_entries.get("metadatas", []):
                    label = meta.get("label", "unknown")
                    parent = meta.get("parent_label", "")
                    
                    stats["by_label"][label] += 1
                    if parent:
                        stats["by_parent"][parent] += 1
            except:
                pass
        
        if self.text_collection:
            stats["total_text"] = self.text_collection.count()
        
        stats["by_parent"] = dict(stats["by_parent"])
        stats["by_label"] = dict(stats["by_label"])
        
        return stats

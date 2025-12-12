"""
RAG (Retrieval-Augmented Generation) Agent for Annotation Intelligence.

This agent provides:
- Visual embedding storage using ChromaDB
- Label consistency across annotations
- Similar annotation retrieval for quality assurance
- Knowledge base building from annotation history
"""

import os
import json
import uuid
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from backend.core.logger import get_logger
from backend.core.config import settings

logger = get_logger("rag")


class RAGAgent:
    """
    RAG Agent for intelligent annotation management.
    
    Uses ChromaDB for vector storage and retrieval, enabling:
    - Label normalization (e.g., "auto" -> "car")
    - Visual similarity search
    - Cross-project knowledge transfer
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize RAG Agent with ChromaDB backend.
        
        Args:
            persist_directory: Directory to persist ChromaDB data.
                             Defaults to DATA_DIR/chromadb
        """
        self.persist_directory = persist_directory or os.path.join(
            settings.DATA_DIR, "chromadb"
        )
        
        self.client = None
        self.collection = None
        self.label_synonyms: Dict[str, str] = {}  # Maps synonyms to canonical labels
        
        self._initialize_chromadb()
        self._load_label_synonyms()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection for annotations
            self.collection = self.client.get_or_create_collection(
                name="annotation_embeddings",
                metadata={
                    "description": "Visual embeddings for annotation consistency",
                    "hnsw:space": "cosine"  # Use cosine similarity
                }
            )
            
            logger.info(f"✅ ChromaDB initialized at {self.persist_directory}")
            logger.info(f"   Collection has {self.collection.count()} entries")
            
        except ImportError:
            logger.warning("ChromaDB not installed. RAG will run in memory-only mode.")
            logger.warning("Install with: pip install chromadb")
            self._use_fallback_storage()
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._use_fallback_storage()
    
    def _use_fallback_storage(self):
        """Use simple in-memory storage as fallback."""
        self.client = None
        self.collection = None
        self._memory_store: List[Dict[str, Any]] = []
        logger.info("Using in-memory fallback storage")
    
    def _load_label_synonyms(self):
        """Load label synonym mapping from config or defaults."""
        # Common label synonyms for annotation consistency
        self.label_synonyms = {
            # Vehicles
            "auto": "car",
            "automobile": "car",
            "vehicle": "car",
            "truck": "truck",
            "lorry": "truck",
            "bike": "bicycle",
            "motorbike": "motorcycle",
            
            # People
            "human": "person",
            "man": "person",
            "woman": "person",
            "child": "person",
            "pedestrian": "person",
            
            # Animals
            "dog": "dog",
            "puppy": "dog",
            "cat": "cat",
            "kitten": "cat",
            
            # Common objects
            "cellphone": "phone",
            "mobile": "phone",
            "laptop": "computer",
            "notebook": "computer",
        }
        
        # Try to load custom synonyms from config
        synonyms_path = os.path.join(settings.DATA_DIR, "label_synonyms.json")
        if os.path.exists(synonyms_path):
            try:
                with open(synonyms_path, 'r') as f:
                    custom_synonyms = json.load(f)
                    self.label_synonyms.update(custom_synonyms)
                    logger.info(f"Loaded {len(custom_synonyms)} custom synonyms")
            except Exception as e:
                logger.warning(f"Failed to load custom synonyms: {e}")
    
    def normalize_label(self, label: str) -> str:
        """
        Normalize a label using synonym mapping.
        
        Args:
            label: Raw label from detection
            
        Returns:
            Canonical label
        """
        label_lower = label.lower().strip()
        return self.label_synonyms.get(label_lower, label_lower)
    
    def add_entry(
        self,
        embedding: Optional[np.ndarray],
        label: str,
        image_id: str,
        project_id: Optional[str] = None,
        bbox: Optional[List[int]] = None,
        confidence: float = 1.0,
        annotation_id: Optional[str] = None
    ) -> str:
        """
        Add an annotation entry to the knowledge base.
        
        Args:
            embedding: Visual embedding vector (e.g., from SAM 3 encoder)
            label: Label for this annotation
            image_id: ID of the source image
            project_id: Optional project ID
            bbox: Optional bounding box [x, y, w, h]
            confidence: Confidence score
            annotation_id: Optional custom annotation ID
            
        Returns:
            ID of the added entry
        """
        entry_id = annotation_id or str(uuid.uuid4())
        canonical_label = self.normalize_label(label)
        
        metadata = {
            "label": canonical_label,
            "original_label": label,
            "image_id": image_id,
            "project_id": project_id or "unknown",
            "confidence": confidence,
            "bbox": json.dumps(bbox) if bbox else "null"
        }
        
        if self.collection is not None:
            try:
                # Generate embedding if not provided
                if embedding is None:
                    # Use label as text embedding via ChromaDB's default embedding
                    self.collection.add(
                        documents=[f"{canonical_label} detected with {confidence:.2f} confidence"],
                        metadatas=[metadata],
                        ids=[entry_id]
                    )
                else:
                    self.collection.add(
                        embeddings=[embedding.tolist()],
                        metadatas=[metadata],
                        ids=[entry_id]
                    )
                
                logger.debug(f"Added entry: {canonical_label} from image {image_id}")
                
            except Exception as e:
                logger.error(f"Failed to add entry to ChromaDB: {e}")
        else:
            # Fallback to memory storage
            self._memory_store.append({
                "id": entry_id,
                "embedding": embedding.tolist() if embedding is not None else None,
                **metadata
            })
        
        return entry_id
    
    def retrieve_consistent_label(
        self,
        embedding: Optional[np.ndarray],
        proposed_label: str,
        n_results: int = 5,
        similarity_threshold: float = 0.85,
        llm_agent: Any = None
    ) -> str:
        """
        Retrieve the most consistent label based on visual similarity.
        
        Novel Approach:
        If 'llm_agent' is provided, we use "LLM Arbitration":
        1. Retrieve similar past annotations.
        2. Ask LLM: "Given these past examples, what is the consistent label for a new object called '{proposed_label}'?"
        This handles complex taxonomy better than simple voting.
        """
        # First, normalize the proposed label
        canonical_proposed = self.normalize_label(proposed_label)
        
        if embedding is None or self.collection is None:
            return canonical_proposed
        
        if self.collection.count() == 0:
            return canonical_proposed
        
        try:
            # Query similar annotations
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=min(n_results, self.collection.count())
            )
            
            if not results["metadatas"] or not results["metadatas"][0]:
                return canonical_proposed
            
            metadatas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]
            
            # --- Novel Approach: LLM Arbitration ---
            if llm_agent:
                try:
                    # Filter relevant examples
                    context_examples = []
                    for i, meta in enumerate(metadatas):
                        dist = distances[i] if i < len(distances) else 1.0
                        sim = 1 - dist
                        if sim >= similarity_threshold:
                            context_examples.append(f"- Label: '{meta.get('label')}' (Confidence: {meta.get('confidence', 0):.2f})")
                    
                    if context_examples:
                        # Ask LLM to arbitrate
                        prompt = f"""We are enforcing label consistency for an annotation project.
Current Detection: "{proposed_label}"
Similar Past Objects have been labeled as:
{chr(10).join(context_examples)}

Task: strict_consistency
Rule: If the specific term appears in history (e.g., 'cracked_screen' vs 'screen'), prefer the specific one.
Return ONLY the best label string. Nothing else."""
                        
                        # Use a lightweight generation (we assume llm_agent has a 'generate_text' or similar, 
                        # but we'll use the 'refine_prompt' structure or direct call if available.
                        # Since LLMAgent usage varies, we'll try to map it or skip if method missing.
                        if hasattr(llm_agent, "model") and llm_agent.model:
                            # Implement Gemini call for arbitration
                            response = llm_agent.model.generate_content(prompt)
                            refined_label = response.text.strip()
                            
                            # Sanity check: ensure it's not a sentence
                            if len(refined_label.split()) < 5:
                                logger.info(f"LLM Arbitration: '{proposed_label}' -> '{refined_label}'")
                                return refined_label
                            else:
                                logger.warning(f"LLM returned verbose label: {refined_label}. Falling back to voting.") 
                except Exception as e:
                    logger.warning(f"LLM Arbitration failed: {e}")
            
            # --- Standard Approach: Weighted Voting ---
            
            # Count labels from similar annotations
            label_votes: Counter = Counter()
            
            for i, meta in enumerate(metadatas):
                distance = distances[i] if i < len(distances) else 1.0
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= similarity_threshold:
                    label = meta.get("label", "unknown")
                    # Weight by similarity and confidence
                    weight = similarity * meta.get("confidence", 1.0)
                    label_votes[label] += weight
            
            if not label_votes:
                return canonical_proposed
            
            # Get most voted label
            most_common = label_votes.most_common(1)[0]
            best_label, best_score = most_common
            
            # Only override if significantly better
            proposed_score = label_votes.get(canonical_proposed, 0)
            if best_score > proposed_score * 1.5:
                logger.info(f"RAG correction: '{proposed_label}' -> '{best_label}' (score: {best_score:.2f})")
                return best_label
            
            return canonical_proposed
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return canonical_proposed
    
    def find_similar_annotations(
        self,
        embedding: np.ndarray,
        n_results: int = 10,
        filter_project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find annotations with similar visual features.
        
        Args:
            embedding: Query embedding
            n_results: Number of results to return
            filter_project_id: Optional project ID to filter results
            
        Returns:
            List of similar annotation metadata
        """
        if self.collection is None or self.collection.count() == 0:
            return []
        
        try:
            where_filter = None
            if filter_project_id:
                where_filter = {"project_id": filter_project_id}
            
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=n_results,
                where=where_filter
            )
            
            similar = []
            for i, meta in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i] if "distances" in results else 0
                similar.append({
                    **meta,
                    "similarity": 1 - distance,
                    "id": results["ids"][0][i]
                })
            
            return similar
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_label_statistics(self, project_id: Optional[str] = None) -> Dict[str, int]:
        """
        Get label frequency statistics.
        
        Args:
            project_id: Optional project to filter by
            
        Returns:
            Dict mapping labels to counts
        """
        if self.collection is None:
            return {}
        
        try:
            where_filter = {"project_id": project_id} if project_id else None
            
            results = self.collection.get(
                where=where_filter,
                include=["metadatas"]
            )
            
            label_counts: Counter = Counter()
            for meta in results["metadatas"]:
                label = meta.get("label", "unknown")
                label_counts[label] += 1
            
            return dict(label_counts)
            
        except Exception as e:
            logger.error(f"Failed to get label statistics: {e}")
            return {}
    
    def clear_project_data(self, project_id: str):
        """
        Remove all entries for a specific project.
        
        Args:
            project_id: Project ID to clear
        """
        if self.collection is None:
            return
        
        try:
            # Get IDs to delete
            results = self.collection.get(
                where={"project_id": project_id}
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Cleared {len(results['ids'])} entries for project {project_id}")
                
        except Exception as e:
            logger.error(f"Failed to clear project data: {e}")
    
    def export_knowledge_base(self, output_path: str):
        """
        Export the entire knowledge base to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if self.collection is None:
            logger.warning("No collection to export")
            return
        
        try:
            results = self.collection.get(include=["metadatas", "embeddings"])
            
            export_data = {
                "count": len(results["ids"]),
                "entries": []
            }
            
            for i, entry_id in enumerate(results["ids"]):
                entry = {
                    "id": entry_id,
                    **results["metadatas"][i]
                }
                if results.get("embeddings"):
                    entry["embedding_size"] = len(results["embeddings"][i])
                export_data["entries"].append(entry)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {export_data['count']} entries to {output_path}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")

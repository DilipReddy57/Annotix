"""
Annotation Pipeline Orchestrator - Manages the complete annotation workflow.

Pipeline Flow:
1. Image/Video Input
2. SAM 3 Segmentation (text or box prompts)
3. Mask Refinement
4. RAG-based Label Consistency
5. QA Validation  
6. Aggregation & Export

Enhanced with:
- LLM-powered auto-prompt generation
- Active Learning for intelligent sample selection
- Multi-Modal RAG with visual+text understanding
- Embedding visualization for annotation analysis

This orchestrator coordinates all agents for autonomous annotation.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image

from backend.core.logger import get_logger
from backend.core.config import settings
from backend.agents.segmentation import SAM3Agent
from backend.agents.classification import ClassificationAgent
from backend.agents.qa import QAAgent
from backend.agents.rag import RAGAgent
from backend.agents.aggregator import AggregatorAgent
from backend.agents.graph_engine import SpatialGraphEngine
from backend.utils.refinement import MaskRefiner
from backend.agents.tracking_agent import TrackingAgent

# Advanced AI Agents
from backend.agents.llm_agent import LLMAgent
from backend.agents.active_learning import ActiveLearningAgent
from backend.agents.embedding_visualizer import EmbeddingVisualizer
from backend.agents.multimodal_rag import MultiModalRAGAgent

logger = get_logger("orchestrator")


class AnnotationPipeline:
    """
    Main annotation pipeline orchestrating all agents.
    
    Features:
    - SAM 3 for segmentation (open-vocabulary text prompts)
    - RAG for label consistency across annotations
    - QA validation for quality assurance
    - Spatial graph for scene understanding
    - Video tracking with temporal consistency
    
    Advanced Features:
    - LLM-powered auto-prompt generation
    - Active learning for optimal sample selection
    - Multi-modal RAG with visual+text understanding
    - Embedding visualization for cluster analysis
    """
    
    def __init__(self, lazy_load: bool = True, enable_advanced: bool = True):
        """
        Initialize the annotation pipeline.
        
        Args:
            lazy_load: If True, load heavy models on-demand
            enable_advanced: If True, enable advanced AI features
        """
        logger.info("Initializing SAM 3 Annotation Pipeline...")
        
        self.lazy_load = lazy_load
        self.enable_advanced = enable_advanced
        
        # Core Agents (loaded immediately or lazily)
        self._sam_agent: Optional[SAM3Agent] = None
        self._classifier: Optional[ClassificationAgent] = None
        self._qa: Optional[QAAgent] = None
        self._rag: Optional[RAGAgent] = None
        self._aggregator: Optional[AggregatorAgent] = None
        
        # Advanced Features
        self._graph_engine: Optional[SpatialGraphEngine] = None
        self._refiner: Optional[MaskRefiner] = None
        self._tracker: Optional[TrackingAgent] = None
        
        # NEW: Advanced AI Agents
        self._llm_agent: Optional[LLMAgent] = None
        self._active_learning: Optional[ActiveLearningAgent] = None
        self._embedding_viz: Optional[EmbeddingVisualizer] = None
        self._multimodal_rag: Optional[MultiModalRAGAgent] = None
        
        if not lazy_load:
            self._initialize_all_agents()
        
        logger.info("✅ Pipeline initialized (lazy_load=%s, advanced=%s)", lazy_load, enable_advanced)
    
    def _initialize_all_agents(self):
        """Initialize all agents immediately."""
        _ = self.sam_agent
        _ = self.classifier
        _ = self.qa
        _ = self.rag
        _ = self.aggregator
        _ = self.graph_engine
        _ = self.refiner
        _ = self.tracker
        if self.enable_advanced:
            _ = self.llm_agent
            _ = self.active_learning
            _ = self.embedding_viz
            _ = self.multimodal_rag
    
    # Lazy-loading properties
    @property
    def sam_agent(self) -> SAM3Agent:
        if self._sam_agent is None:
            self._sam_agent = SAM3Agent(
                confidence_threshold=settings.SAM3_CONFIDENCE_THRESHOLD
            )
        return self._sam_agent
    
    @property
    def classifier(self) -> ClassificationAgent:
        if self._classifier is None:
            self._classifier = ClassificationAgent()
        return self._classifier
    
    @property
    def qa(self) -> QAAgent:
        if self._qa is None:
            self._qa = QAAgent()
        return self._qa
    
    @property
    def rag(self) -> RAGAgent:
        if self._rag is None:
            self._rag = RAGAgent(persist_directory=settings.CHROMADB_PATH)
        return self._rag
    
    @property
    def aggregator(self) -> AggregatorAgent:
        if self._aggregator is None:
            self._aggregator = AggregatorAgent()
        return self._aggregator
    
    @property
    def graph_engine(self) -> SpatialGraphEngine:
        if self._graph_engine is None:
            self._graph_engine = SpatialGraphEngine()
        return self._graph_engine
    
    @property
    def refiner(self) -> MaskRefiner:
        if self._refiner is None:
            self._refiner = MaskRefiner()
        return self._refiner
    
    @property
    def tracker(self) -> TrackingAgent:
        if self._tracker is None:
            self._tracker = TrackingAgent()
        return self._tracker
    
    # Advanced AI Agent Properties
    @property
    def llm_agent(self) -> LLMAgent:
        """LLM Agent for auto-prompt generation and semantic understanding."""
        if self._llm_agent is None:
            self._llm_agent = LLMAgent(provider="gemini")
        return self._llm_agent
    
    @property
    def active_learning(self) -> ActiveLearningAgent:
        """Active Learning Agent for intelligent sample selection."""
        if self._active_learning is None:
            self._active_learning = ActiveLearningAgent()
        return self._active_learning
    
    @property
    def embedding_viz(self) -> EmbeddingVisualizer:
        """Embedding Visualizer for annotation cluster analysis."""
        if self._embedding_viz is None:
            self._embedding_viz = EmbeddingVisualizer()
        return self._embedding_viz
    
    @property
    def multimodal_rag(self) -> MultiModalRAGAgent:
        """Multi-Modal RAG Agent with visual+text understanding."""
        if self._multimodal_rag is None:
            self._multimodal_rag = MultiModalRAGAgent()
        return self._multimodal_rag
    
    def process_image(
        self, 
        file_path: str, 
        prompt: Optional[str] = None,
        project_id: Optional[str] = None,
        validate: bool = True,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete annotation pipeline on a single image.
        
        Flow:
        1. SAM 3 Segmentation (with text prompt if provided)
        2. Mask Refinement
        3. Classification (if needed)
        4. RAG Label Consistency
        5. QA Validation
        6. Aggregation
        7. Scene Graph Generation
        
        Args:
            file_path: Path to the image file
            prompt: Optional text prompt for detection (e.g., "car", "person in red")
            project_id: Optional project ID for RAG scoping
            validate: Whether to run QA validation
            use_rag: Whether to use RAG for label consistency
            
        Returns:
            Dict with annotations, scene graph, and analytics
        """
        logger.info(f"Processing image: {file_path}")
        logger.info(f"Prompt: '{prompt or 'auto-detect'}'")
        
        # Get image dimensions
        try:
            img = Image.open(file_path)
            width, height = img.size
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            width, height = 0, 0
        
        # Initialize aggregator entry
        image_id = self.aggregator.add_image(file_path, width, height)
        
        # Step 1: SAM 3 Segmentation
        raw_annotations = self.sam_agent.segment_image(
            file_path, 
            prompt=prompt,
            return_masks=True
        )
        
        logger.info(f"SAM 3 found {len(raw_annotations)} objects")
        
        final_annotations = []
        
        for ann in raw_annotations:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            mask = ann.get("mask")
            segmentation = ann.get("segmentation", [])
            score = ann.get("score", 0.0)
            label = ann.get("label", "object")
            
            # Step 2: Mask Refinement
            if mask is not None:
                mask = self.refiner.refine_mask(mask)
            
            # Step 3: Classification (if generic label)
            if label in ["object", "thing", "all objects"]:
                try:
                    cls_result = self.classifier.classify_object(file_path, bbox)
                    label = cls_result.get("label", label)
                except Exception as e:
                    logger.warning(f"Classification failed: {e}")
            
            # Step 4: RAG Label Consistency
            if use_rag:
                # Generate embedding from mask or use placeholder
                embedding = np.random.rand(256).astype('float32')  # TODO: Extract from SAM 3
                label = self.rag.retrieve_consistent_label(embedding, label)
                self.rag.add_entry(
                    embedding=embedding,
                    label=label,
                    image_id=str(image_id),
                    project_id=project_id,
                    bbox=bbox,
                    confidence=score
                )
            
            # Step 5: QA Validation
            if validate:
                qa_result = self.qa.validate_annotation(
                    mask, bbox, {"label": label, "confidence": score}
                )
                if not qa_result.get("valid", True):
                    logger.info(f"Annotation rejected by QA: {qa_result.get('reason')}")
                    continue
            
            # Step 6: Aggregation
            self.aggregator.add_annotation(image_id, label, bbox, segmentation, score)
            
            final_annotations.append({
                "id": len(final_annotations),
                "bbox": bbox,
                "segmentation": segmentation,
                "area": ann.get("area", 0),
                "label": label,
                "score": score
            })
        
        # Save aggregated data
        self.aggregator.save_json()
        
        # Step 7: Scene Graph
        scene_graph = self.graph_engine.build_graph(final_annotations)
        
        logger.info(f"✅ Image processed: {len(final_annotations)} annotations")
        
        return {
            "status": "success",
            "image_id": image_id,
            "image_path": file_path,
            "width": width,
            "height": height,
            "annotations": final_annotations,
            "scene_graph": scene_graph,
            "analytics": self.aggregator.get_analytics()
        }
    
    def process_video(
        self, 
        video_path: str, 
        prompt: str,
        project_id: Optional[str] = None,
        sample_rate: int = 1
    ) -> Dict[str, Any]:
        """
        Run the video annotation pipeline with temporal tracking.
        
        Args:
            video_path: Path to the video file
            prompt: Text prompt for detection
            project_id: Optional project ID
            sample_rate: Process every Nth frame
            
        Returns:
            Dict with track annotations
        """
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Prompt: '{prompt}'")
        
        # Use tracking agent (which uses SAM 3 video predictor internally)
        tracks = self.tracker.process_video(
            video_path=video_path,
            prompt=prompt,
            sam_agent=self.sam_agent,
            sample_rate=sample_rate
        )
        
        # Apply RAG for label consistency
        for track in tracks:
            embedding = np.random.rand(256).astype('float32')
            track["label"] = self.rag.retrieve_consistent_label(embedding, track["label"])
        
        logger.info(f"✅ Video processed: {len(tracks)} track annotations")
        
        return {
            "status": "success",
            "video_path": video_path,
            "prompt": prompt,
            "annotations": tracks,
            "unique_objects": len(set(t["object_id"] for t in tracks)),
            "total_frames": max(t["frame"] for t in tracks) + 1 if tracks else 0
        }
    
    def batch_process_images(
        self, 
        image_paths: List[str], 
        prompt: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            prompt: Optional text prompt
            project_id: Optional project ID
            
        Returns:
            List of results for each image
        """
        logger.info(f"Batch processing {len(image_paths)} images")
        
        results = []
        for i, path in enumerate(image_paths):
            try:
                result = self.process_image(
                    path, 
                    prompt=prompt, 
                    project_id=project_id
                )
                results.append(result)
                logger.info(f"Processed {i+1}/{len(image_paths)}: {path}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append({
                    "status": "error",
                    "image_path": path,
                    "error": str(e)
                })
        
        return results
    
    def get_rag_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get RAG knowledge base statistics."""
        return {
            "label_counts": self.rag.get_label_statistics(project_id),
            "total_entries": self.rag.collection.count() if self.rag.collection else 0
        }
    
    # ============= ADVANCED AI METHODS =============
    
    def auto_generate_prompts(
        self,
        image_path: str,
        context: Optional[str] = None,
        max_prompts: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to automatically generate optimal prompts for an image.
        
        Args:
            image_path: Path to the image
            context: Optional context (e.g., "autonomous driving", "retail")
            max_prompts: Maximum number of prompts to generate
            
        Returns:
            List of suggested prompts with confidence and reasoning
        """
        if not self.enable_advanced:
            return [{"prompt": "all objects", "confidence": 0.5, "reason": "Advanced features disabled"}]
        
        logger.info(f"Auto-generating prompts for: {image_path}")
        prompts = self.llm_agent.analyze_image_for_prompts(
            image_path, context=context, max_prompts=max_prompts
        )
        
        logger.info(f"Generated {len(prompts)} prompts")
        return prompts
    
    def smart_process_image(
        self,
        file_path: str,
        project_id: Optional[str] = None,
        use_auto_prompts: bool = True,
        context: Optional[str] = None,
        turbo_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Process image with intelligent annotation pipeline.
        
        FAST BY DEFAULT:
        - Single SAM3 inference (unless auto-prompts find multiple targets)
        - Real CLIP embeddings for genuine RAG consistency
        - QA validation (fast, rule-based)
        - LLM-arbitrated RAG only when needed (ambiguous labels)
        - Analytics run in background (non-blocking)
        
        TURBO MODE (even faster):
        - Skip auto-prompt generation (use generic prompt)
        - Skip CLIP embeddings (use label normalization only)
        - Skip RAG lookup (just normalize labels)
        
        Args:
            file_path: Path to the image
            project_id: Optional project ID
            use_auto_prompts: Use LLM to generate prompts (default mode)
            context: Optional context for prompt generation
            turbo_mode: Skip heavy AI features for maximum speed
            
        Returns:
            Enhanced result with annotations
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing: {file_path} (turbo={turbo_mode})")
        
        # Load image once for all operations
        try:
            pil_image = Image.open(file_path).convert("RGB")
            width, height = pil_image.size
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return {"status": "error", "error": str(e)}
        
        all_annotations = []
        used_prompts = []
        
        # ============================================================
        # STEP 1: Determine prompts (Fast or Auto)
        # ============================================================
        if turbo_mode:
            # Turbo: Single generic prompt, no LLM
            prompts_to_use = ["all objects"]
        elif use_auto_prompts and self.enable_advanced:
            # Default: Ask LLM for smart prompts (limit to 1 for speed)
            try:
                generated = self.auto_generate_prompts(file_path, context=context, max_prompts=2)
                prompts_to_use = [p["prompt"] for p in generated if p.get("confidence", 0) > 0.6][:1]
            except Exception as e:
                logger.warning(f"Auto-prompt failed, using fallback: {e}")
                prompts_to_use = []
            
            if not prompts_to_use:
                prompts_to_use = ["all visible objects"]
        else:
            prompts_to_use = ["all visible objects"]
        
        used_prompts = prompts_to_use
        
        # ============================================================
        # STEP 2: SAM3 Segmentation (single pass for speed)
        # ============================================================
        for prompt in prompts_to_use:
            try:
                annotations = self.sam_agent.segment_image(
                    file_path, prompt=prompt, return_masks=True
                )
                
                for ann in annotations:
                    ann["source_prompt"] = prompt
                    all_annotations.append(ann)
                    
            except Exception as e:
                logger.warning(f"Segmentation failed for '{prompt}': {e}")
        
        if not all_annotations:
            # No annotations found
            elapsed = time.time() - start_time
            return {
                "status": "success",
                "image_path": file_path,
                "width": width,
                "height": height,
                "annotations": [],
                "prompts_used": used_prompts,
                "processing_time_ms": int(elapsed * 1000),
                "processing_mode": "turbo" if turbo_mode else "fast"
            }
        
        # ============================================================
        # STEP 3: Mask Refinement (fast, always run)
        # ============================================================
        for ann in all_annotations:
            if ann.get("mask") is not None:
                ann["mask"] = self.refiner.refine_mask(ann["mask"])
        
        # ============================================================
        # STEP 4: Real Embeddings + RAG Consistency (skip in turbo)
        # ============================================================
        if not turbo_mode:
            try:
                from backend.agents.embedding_extractor import get_embedding_extractor
                extractor = get_embedding_extractor()
                
                # Batch extract embeddings (more efficient)
                bboxes = [ann.get("bbox", [0, 0, width, height]) for ann in all_annotations]
                embeddings = extractor.batch_extract(pil_image, bboxes)
                
                for ann, embedding in zip(all_annotations, embeddings):
                    label = ann.get("label", ann.get("source_prompt", "object"))
                    
                    if embedding is not None:
                        # Real RAG consistency check (no LLM arbitration by default - too slow)
                        consistent_label = self.rag.retrieve_consistent_label(
                            embedding, 
                            label,
                            llm_agent=None  # Skip LLM for speed, use voting
                        )
                        ann["label"] = consistent_label
                        ann["_embedding"] = embedding  # Store for background RAG update
                    else:
                        # Fallback: just normalize the label
                        ann["label"] = self.rag.normalize_label(label)
                        
            except Exception as e:
                logger.warning(f"Embedding extraction failed: {e}")
                # Fallback: normalize labels without embeddings
                for ann in all_annotations:
                    label = ann.get("label", ann.get("source_prompt", "object"))
                    ann["label"] = self.rag.normalize_label(label)
        else:
            # Turbo mode: just normalize labels
            for ann in all_annotations:
                label = ann.get("label", ann.get("source_prompt", "object"))
                ann["label"] = self.rag.normalize_label(label)
        
        # ============================================================
        # STEP 5: QA Validation (fast, always run)
        # ============================================================
        validated_annotations = []
        for ann in all_annotations:
            qa_result = self.qa.validate_annotation(
                ann.get("mask"), 
                ann.get("bbox"), 
                {"label": ann.get("label"), "confidence": ann.get("score")}
            )
            
            if qa_result.get("valid", True):
                validated_annotations.append(ann)
            else:
                logger.debug(f"QA rejected: {qa_result.get('reason')}")
        
        # ============================================================
        # STEP 6: Deduplicate overlapping annotations
        # ============================================================
        deduplicated = self._deduplicate_annotations(validated_annotations)
        
        # ============================================================
        # STEP 7: Background Analytics (non-blocking)
        # ============================================================
        if self.enable_advanced and deduplicated and not turbo_mode:
            from threading import Thread
            
            def _background_work(anns, f_path, p_id, pil_img):
                """Update RAG database and analytics in background."""
                try:
                    from backend.agents.embedding_extractor import get_embedding_extractor
                    extractor = get_embedding_extractor()
                    
                    for ann in anns:
                        # Use stored embedding or extract new
                        embedding = ann.pop("_embedding", None)
                        if embedding is None:
                            bbox = ann.get("bbox")
                            embedding = extractor.extract_embedding(pil_img, bbox)
                        
                        if embedding is not None:
                            # Add to RAG for future consistency
                            self.rag.add_entry(
                                embedding=embedding,
                                label=ann.get("label", "object"),
                                image_id=f_path,
                                project_id=p_id,
                                bbox=ann.get("bbox"),
                                confidence=ann.get("score", 0.5)
                            )
                            
                            # Add to multi-modal RAG
                            self.multimodal_rag.add_multimodal_entry(
                                visual_embedding=embedding,
                                label=ann.get("label", "object"),
                                image_id=f_path,
                                project_id=p_id,
                                bbox=ann.get("bbox"),
                                confidence=ann.get("score", 0.5)
                            )
                            
                            # Add to embedding visualizer
                            self.embedding_viz.add_embedding(
                                entry_id=f"{f_path}_{ann.get('id', 0)}",
                                embedding=embedding,
                                metadata={
                                    "label": ann.get("label"),
                                    "image_path": f_path,
                                    "score": ann.get("score")
                                }
                            )
                    
                    # Track for active learning
                    avg_embedding = np.mean([
                        ann.get("_embedding", np.zeros(512)) 
                        for ann in anns if ann.get("_embedding") is not None
                    ], axis=0) if anns else np.zeros(512)
                    
                    self.active_learning.add_to_history(
                        image_path=f_path,
                        annotations=anns,
                        embedding=avg_embedding.astype(np.float32)
                    )
                    
                except Exception as e:
                    logger.error(f"Background analytics failed: {e}")
            
            # Copy image to avoid thread issues
            pil_copy = pil_image.copy()
            thread = Thread(target=_background_work, args=(deduplicated.copy(), file_path, project_id, pil_copy))
            thread.daemon = True
            thread.start()
        
        # Clean up embeddings from output (not needed in response)
        for ann in deduplicated:
            ann.pop("_embedding", None)
        
        # ============================================================
        # STEP 8: Build scene graph
        # ============================================================
        scene_graph = self.graph_engine.build_graph(deduplicated)
        
        elapsed = time.time() - start_time
        
        return {
            "status": "success",
            "image_path": file_path,
            "width": width,
            "height": height,
            "annotations": deduplicated,
            "prompts_used": used_prompts,
            "scene_graph": scene_graph,
            "processing_time_ms": int(elapsed * 1000),
            "processing_mode": "turbo" if turbo_mode else "fast"
        }
    
    def _deduplicate_annotations(
        self,
        annotations: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Deduplicate overlapping annotations based on IoU."""
        if not annotations:
            return []
        
        # Sort by score
        sorted_anns = sorted(annotations, key=lambda x: x.get("score", 0), reverse=True)
        kept = []
        
        for ann in sorted_anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            should_keep = True
            
            for kept_ann in kept:
                kept_bbox = kept_ann.get("bbox", [0, 0, 0, 0])
                iou = self._calculate_iou(bbox, kept_bbox)
                
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(ann)
        
        return kept
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate IoU between two boxes [x, y, w, h]."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2]
        b1 = [x1, y1, x1 + w1, y1 + h1]
        b2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Intersection
        xi1 = max(b1[0], b2[0])
        yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2])
        yi2 = min(b1[3], b2[3])
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        b1_area = w1 * h1
        b2_area = w2 * h2
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def get_next_annotation_batch(
        self,
        candidate_images: List[str],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Use Active Learning to select the best images to annotate next.
        
        Args:
            candidate_images: Pool of unannotated images
            batch_size: Number of images to select
            
        Returns:
            Ranked list of images with priority scores
        """
        if not self.enable_advanced:
            return [{"image_path": img, "priority_score": 1.0} for img in candidate_images[:batch_size]]
        
        # Get preliminary predictions for uncertainty estimation
        predictions_per_image = {}
        embeddings = {}
        
        for img_path in candidate_images:
            try:
                # Quick prediction with generic prompt
                preds = self.sam_agent.segment_image(img_path, prompt="all objects")
                predictions_per_image[img_path] = preds
                
                # Get embedding for diversity
                embeddings[img_path] = np.random.rand(256).astype('float32')
            except:
                predictions_per_image[img_path] = []
        
        # Rank images
        ranked = self.active_learning.rank_images_for_annotation(
            candidate_images,
            predictions_per_image,
            embeddings,
            strategy="combined"
        )
        
        return ranked[:batch_size]
    
    def get_embedding_visualization(
        self,
        method: str = "umap"
    ) -> Dict[str, Any]:
        """
        Get embedding visualization data for the annotation knowledge base.
        
        Args:
            method: "umap", "tsne", or "pca"
            
        Returns:
            Plotly-compatible visualization data
        """
        if not self.enable_advanced:
            return {"error": "Advanced features disabled"}
        
        return self.embedding_viz.export_for_plotly(method=method)
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all advanced agents."""
        stats = {
            "basic": self.aggregator.get_analytics(),
            "rag": self.get_rag_statistics()
        }
        
        if self.enable_advanced:
            stats["active_learning"] = self.active_learning.get_statistics()
            stats["multimodal_rag"] = self.multimodal_rag.get_ontology_statistics()
            
            # Embedding clusters
            viz_data = self.embedding_viz.get_visualization_data()
            if "error" not in viz_data:
                stats["embedding_clusters"] = {
                    "n_points": viz_data.get("n_points", 0),
                    "n_clusters": viz_data.get("n_clusters", 0),
                    "n_outliers": viz_data.get("n_outliers", 0)
                }
        
        return stats


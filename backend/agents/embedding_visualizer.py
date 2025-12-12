"""
Embedding Visualizer for annotation cluster analysis.

Features:
- UMAP/t-SNE dimensionality reduction
- Cluster visualization and analysis
- Label distribution mapping
- Interactive embedding exploration
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json

from backend.core.logger import get_logger
from backend.core.config import settings

logger = get_logger("embedding_viz")


class EmbeddingVisualizer:
    """
    Embedding Visualizer for understanding annotation distributions.
    
    Features:
    - Dimensionality reduction (UMAP, t-SNE, PCA)
    - Cluster detection and labeling
    - Outlier identification
    - Similarity heatmaps
    """
    
    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._reducer = None
        self._reduced_embeddings: Optional[np.ndarray] = None
    
    def add_embedding(
        self,
        entry_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add an embedding with optional metadata.
        
        Args:
            entry_id: Unique identifier
            embedding: Embedding vector
            metadata: Optional metadata (label, image_path, bbox, etc.)
        """
        self.embeddings[entry_id] = embedding
        self.metadata[entry_id] = metadata or {}
        
        # Invalidate cached reduction
        self._reduced_embeddings = None
    
    def reduce_dimensions(
        self,
        method: str = "umap",
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            method: "umap", "tsne", or "pca"
            n_components: Number of output dimensions (2 or 3)
            **kwargs: Additional parameters for the reducer
            
        Returns:
            Reduced embeddings array (n_samples, n_components)
        """
        if len(self.embeddings) < 2:
            logger.warning("Need at least 2 embeddings for reduction")
            return np.array([])
        
        # Stack embeddings
        ids = list(self.embeddings.keys())
        X = np.array([self.embeddings[i] for i in ids])
        
        if method == "umap":
            reduced = self._reduce_umap(X, n_components, **kwargs)
        elif method == "tsne":
            reduced = self._reduce_tsne(X, n_components, **kwargs)
        elif method == "pca":
            reduced = self._reduce_pca(X, n_components)
        else:
            logger.warning(f"Unknown method: {method}, using PCA")
            reduced = self._reduce_pca(X, n_components)
        
        self._reduced_embeddings = reduced
        self._reduced_ids = ids
        
        return reduced
    
    def _reduce_umap(
        self,
        X: np.ndarray,
        n_components: int,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        **kwargs
    ) -> np.ndarray:
        """Reduce using UMAP."""
        try:
            import umap
            
            reducer = umap.UMAP(
                n_neighbors=min(n_neighbors, len(X) - 1),
                min_dist=min_dist,
                n_components=n_components,
                random_state=42,
                **kwargs
            )
            return reducer.fit_transform(X)
            
        except ImportError:
            logger.warning("UMAP not installed. Run: pip install umap-learn")
            logger.info("Falling back to PCA")
            return self._reduce_pca(X, n_components)
    
    def _reduce_tsne(
        self,
        X: np.ndarray,
        n_components: int,
        perplexity: float = 30.0,
        **kwargs
    ) -> np.ndarray:
        """Reduce using t-SNE."""
        try:
            from sklearn.manifold import TSNE
            
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(X) - 1),
                random_state=42,
                **kwargs
            )
            return reducer.fit_transform(X)
            
        except ImportError:
            logger.warning("scikit-learn not installed for t-SNE")
            return self._reduce_pca(X, n_components)
    
    def _reduce_pca(self, X: np.ndarray, n_components: int) -> np.ndarray:
        """Reduce using PCA (always available via numpy)."""
        # Center the data
        X_centered = X - X.mean(axis=0)
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Project to n_components
        return U[:, :n_components] * S[:n_components]
    
    def detect_clusters(
        self,
        n_clusters: Optional[int] = None,
        method: str = "kmeans"
    ) -> Dict[str, int]:
        """
        Detect clusters in the embedding space.
        
        Args:
            n_clusters: Number of clusters (auto-detected if None)
            method: "kmeans", "dbscan", or "hdbscan"
            
        Returns:
            Dict mapping entry_id to cluster label
        """
        if len(self.embeddings) < 2:
            return {}
        
        ids = list(self.embeddings.keys())
        X = np.array([self.embeddings[i] for i in ids])
        
        if method == "kmeans":
            labels = self._cluster_kmeans(X, n_clusters)
        elif method == "dbscan":
            labels = self._cluster_dbscan(X)
        else:
            labels = self._cluster_kmeans(X, n_clusters)
        
        return {ids[i]: int(labels[i]) for i in range(len(ids))}
    
    def _cluster_kmeans(
        self,
        X: np.ndarray,
        n_clusters: Optional[int]
    ) -> np.ndarray:
        """Cluster using K-means."""
        try:
            from sklearn.cluster import KMeans
            
            if n_clusters is None:
                # Estimate optimal clusters using elbow method (simplified)
                n_clusters = min(10, len(X) // 5)
                n_clusters = max(2, n_clusters)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(X)
            
        except ImportError:
            logger.warning("scikit-learn not installed for clustering")
            return np.zeros(len(X))
    
    def _cluster_dbscan(self, X: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN."""
        try:
            from sklearn.cluster import DBSCAN
            
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            return dbscan.fit_predict(X)
            
        except ImportError:
            return np.zeros(len(X))
    
    def find_outliers(self, threshold: float = 2.0) -> List[str]:
        """
        Find outlier embeddings (far from cluster centers).
        
        Args:
            threshold: Z-score threshold for outliers
            
        Returns:
            List of outlier entry IDs
        """
        if len(self.embeddings) < 3:
            return []
        
        ids = list(self.embeddings.keys())
        X = np.array([self.embeddings[i] for i in ids])
        
        # Calculate centroid
        centroid = X.mean(axis=0)
        
        # Calculate distances
        distances = np.linalg.norm(X - centroid, axis=1)
        
        # Z-score
        z_scores = (distances - distances.mean()) / (distances.std() + 1e-10)
        
        outliers = [ids[i] for i in range(len(ids)) if z_scores[i] > threshold]
        
        return outliers
    
    def get_similarity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate pairwise cosine similarity matrix.
        
        Returns:
            Tuple of (similarity_matrix, entry_ids)
        """
        if len(self.embeddings) < 2:
            return np.array([]), []
        
        ids = list(self.embeddings.keys())
        X = np.array([self.embeddings[i] for i in ids])
        
        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = X / (norms + 1e-10)
        
        # Cosine similarity
        similarity = X_normalized @ X_normalized.T
        
        return similarity, ids
    
    def get_visualization_data(
        self,
        method: str = "umap",
        color_by: str = "label"
    ) -> Dict[str, Any]:
        """
        Get data ready for visualization.
        
        Args:
            method: Reduction method
            color_by: Metadata field to use for coloring
            
        Returns:
            Dict with coordinates, labels, and metadata for plotting
        """
        if len(self.embeddings) < 2:
            return {"error": "Not enough embeddings"}
        
        # Reduce dimensions
        coords = self.reduce_dimensions(method=method, n_components=2)
        ids = self._reduced_ids
        
        # Prepare visualization data
        points = []
        for i, entry_id in enumerate(ids):
            meta = self.metadata.get(entry_id, {})
            points.append({
                "id": entry_id,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "label": meta.get("label", "unknown"),
                "image_path": meta.get("image_path", ""),
                "color_value": meta.get(color_by, "default")
            })
        
        # Cluster info
        clusters = self.detect_clusters()
        for point in points:
            point["cluster"] = clusters.get(point["id"], -1)
        
        # Outliers
        outliers = set(self.find_outliers())
        for point in points:
            point["is_outlier"] = point["id"] in outliers
        
        return {
            "points": points,
            "method": method,
            "color_by": color_by,
            "n_points": len(points),
            "n_clusters": len(set(clusters.values())),
            "n_outliers": len(outliers)
        }
    
    def export_for_plotly(self, method: str = "umap") -> Dict[str, Any]:
        """
        Export data in Plotly-compatible format.
        
        Returns:
            Dict with trace data for Plotly
        """
        viz_data = self.get_visualization_data(method=method)
        
        if "error" in viz_data:
            return viz_data
        
        points = viz_data["points"]
        
        # Group by label
        label_groups = {}
        for p in points:
            label = p["label"]
            if label not in label_groups:
                label_groups[label] = {"x": [], "y": [], "text": [], "ids": []}
            label_groups[label]["x"].append(p["x"])
            label_groups[label]["y"].append(p["y"])
            label_groups[label]["text"].append(f"ID: {p['id']}<br>Cluster: {p['cluster']}")
            label_groups[label]["ids"].append(p["id"])
        
        traces = []
        for label, data in label_groups.items():
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": label,
                "x": data["x"],
                "y": data["y"],
                "text": data["text"],
                "hoverinfo": "text+name",
                "marker": {"size": 8}
            })
        
        return {
            "traces": traces,
            "layout": {
                "title": f"Embedding Visualization ({method.upper()})",
                "xaxis": {"title": "Dimension 1"},
                "yaxis": {"title": "Dimension 2"},
                "hovermode": "closest"
            }
        }

import numpy as np
from typing import Tuple


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100):
        """
        Initialize K-means clustering algorithm.
        
        Args:
            n_clusters (int): Number of clusters
            max_iter (int): Maximum number of iterations
            
        Raises:
            ValueError: If n_clusters <= 0
        """
        if n_clusters <= 0:
            raise ValueError("Number of clusters must be positive")
            
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        
    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """Initialize cluster centroids randomly."""
        n_samples = data.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return data[random_indices].copy()
    
    def _assign_clusters(self, data: np.ndarray) -> np.ndarray:
        """Assign each data point to the nearest centroid."""
        distances = np.sqrt(((data[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _update_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids based on mean of assigned points."""
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                new_centroids[k] = np.mean(data[labels == k], axis=0)
            else:
                new_centroids[k] = self.centroids[k]
        return new_centroids

    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit k-means clustering to the data.
        
        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple[np.ndarray, float]: Cluster labels and final SSE
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(data)
        
        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # Assign clusters
            labels = self._assign_clusters(data)
            
            # Update centroids
            self.centroids = self._update_centroids(data, labels)
            
            # Check convergence
            if np.all(old_centroids == self.centroids):
                break
        
        # Calculate final SSE
        sse = self._calculate_sse(data, labels)
        
        return labels, sse
    
    def _calculate_sse(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Sum of Squared Errors."""
        sse = 0
        for k in range(self.n_clusters):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - self.centroids[k]) ** 2)
        return sse

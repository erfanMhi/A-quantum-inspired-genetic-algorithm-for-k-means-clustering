import numpy as np
import pytest
from qiga_kmeans.algorithms.kmeans import KMeans


@pytest.fixture
def sample_data():
    """Create a simple dataset with clear clusters."""
    np.random.seed(42)
    cluster1 = np.random.normal(0, 0.5, (100, 2))
    cluster2 = np.random.normal(4, 0.5, (100, 2))
    cluster3 = np.random.normal([0, 4], 0.5, (100, 2))
    return np.vstack([cluster1, cluster2, cluster3])


def test_kmeans_initialization():
    """Test KMeans initialization with valid parameters."""
    kmeans = KMeans(n_clusters=3, max_iter=100)
    assert kmeans.n_clusters == 3
    assert kmeans.max_iter == 100
    assert kmeans.centroids is None


def test_kmeans_invalid_clusters():
    """Test KMeans initialization with invalid parameters."""
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
    with pytest.raises(ValueError):
        KMeans(n_clusters=-1)


def test_kmeans_fit(sample_data):
    """Test KMeans fitting process."""
    kmeans = KMeans(n_clusters=3, max_iter=100)
    labels, sse = kmeans.fit(sample_data)
    
    # Check output shapes and types
    assert isinstance(labels, np.ndarray)
    assert isinstance(sse, float)
    assert labels.shape == (300,)
    assert kmeans.centroids.shape == (3, 2)
    
    # Check labels are valid
    assert np.all(labels >= 0)
    assert np.all(labels < 3)
    assert len(np.unique(labels)) == 3
    
    # Check SSE is positive
    assert sse > 0


def test_kmeans_convergence(sample_data):
    """Test KMeans convergence by comparing consecutive iterations."""
    kmeans = KMeans(n_clusters=3, max_iter=100)
    labels1, sse1 = kmeans.fit(sample_data)
    centroids1 = kmeans.centroids.copy()
    
    # Run again with same initialization
    kmeans.centroids = centroids1.copy()
    labels2, sse2 = kmeans.fit(sample_data)
    
    # Check if results are consistent
    np.testing.assert_array_equal(labels1, labels2)
    assert abs(sse1 - sse2) < 1e-10


def test_kmeans_different_initializations(sample_data):
    """Test that different initializations converge to similar SSE values."""
    np.random.seed(42)
    n_runs = 5
    sse_values = []
    
    for _ in range(n_runs):
        kmeans = KMeans(n_clusters=3)
        _, sse = kmeans.fit(sample_data)
        sse_values.append(sse)
    
    # Check that SSE values are similar (within reasonable range)
    sse_mean = np.mean(sse_values)
    sse_std = np.std(sse_values)
    assert sse_std / sse_mean < 0.1  # Relative standard deviation < 10% 
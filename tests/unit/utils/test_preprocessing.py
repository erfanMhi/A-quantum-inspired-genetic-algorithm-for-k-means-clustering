import numpy as np
import pytest
from qiga_kmeans.utils.preprocessing import minmax_normalize


def test_minmax_normalize_basic():
    """Test basic minmax normalization."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    normalized = minmax_normalize(data)
    
    # Check shape preservation
    assert normalized.shape == data.shape
    
    # Check range [0, 1]
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)
    
    # Check min and max values
    assert np.allclose(np.min(normalized, axis=0), 0)
    assert np.allclose(np.max(normalized, axis=0), 1)


def test_minmax_normalize_constant():
    """Test normalization with constant features."""
    data = np.array([[1, 2], [1, 2], [1, 2]])
    normalized = minmax_normalize(data)
    
    # Check shape preservation
    assert normalized.shape == data.shape
    
    # Check constant features are handled properly
    assert np.all(normalized[:, 0] == 0)  # First column is constant
    assert np.all(normalized[:, 1] == 0)  # Second column is constant


def test_minmax_normalize_single_value():
    """Test normalization with single value."""
    data = np.array([[1]])
    normalized = minmax_normalize(data)
    
    assert normalized.shape == data.shape
    assert normalized[0, 0] == 0


def test_minmax_normalize_negative_values():
    """Test normalization with negative values."""
    data = np.array([[-1, -2], [0, 0], [1, 2]])
    normalized = minmax_normalize(data)
    
    assert normalized.shape == data.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)
    
    # Check specific values
    assert np.allclose(normalized[0], [0, 0])    # Minimum values
    assert np.allclose(normalized[1], [0.5, 0.5])  # Middle values
    assert np.allclose(normalized[2], [1, 1])    # Maximum values


def test_minmax_normalize_different_ranges():
    """Test normalization with features having different ranges."""
    data = np.array([[1, 100], [2, 200], [3, 300]])
    normalized = minmax_normalize(data)
    
    assert normalized.shape == data.shape
    assert np.all(normalized >= 0)
    assert np.all(normalized <= 1)
    
    # Check that both features are properly normalized despite different scales
    assert np.allclose(np.min(normalized, axis=0), [0, 0])
    assert np.allclose(np.max(normalized, axis=0), [1, 1])


def test_minmax_normalize_type_preservation():
    """Test that normalization preserves float type."""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    normalized = minmax_normalize(data)
    
    assert normalized.dtype == np.float32


def test_minmax_normalize_invalid_input():
    """Test normalization with invalid inputs."""
    # Test with empty array
    with pytest.raises(ValueError):
        minmax_normalize(np.array([]))
    
    # Test with 1D array
    with pytest.raises(ValueError):
        minmax_normalize(np.array([1, 2, 3]))
    
    # Test with 3D array
    with pytest.raises(ValueError):
        minmax_normalize(np.ones((2, 2, 2))) 
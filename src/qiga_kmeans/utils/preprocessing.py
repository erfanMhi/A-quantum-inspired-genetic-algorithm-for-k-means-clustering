import numpy as np


def minmax_normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize data using min-max scaling.
    
    Args:
        data (np.ndarray): Input data to normalize
        
    Returns:
        np.ndarray: Normalized data
        
    Raises:
        ValueError: If data is empty or not 2D
    """
    if data.size == 0:
        raise ValueError("Empty array")
    if data.ndim != 2:
        raise ValueError("Input must be a 2D array")
    
    # Preserve input dtype
    dtype = data.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32
    
    data = data.astype(dtype)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)
    
    normalized_data = (data - min_vals) / range_vals
    return normalized_data

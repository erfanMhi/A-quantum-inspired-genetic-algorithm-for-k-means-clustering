import numpy as np


def generate_sda_dataset(num: int, cluster_members_num: int = 100) -> np.ndarray:
    """
    Generate synthetic datasets SDA1, SDA2, or SDA3 for clustering experiments.
    
    Args:
        num (int): Dataset number (1, 2, or 3) to generate
        cluster_members_num (int): Number of members per cluster (default: 100)
    
    Returns:
        np.ndarray: Generated dataset with shape (n_samples, 2)
    """
    np.random.seed(0)
    
    if num == 1:
        # SDA1: 3 clusters
        return np.concatenate([
            np.random.uniform(0, 20, (cluster_members_num, 2)),
            np.random.uniform(40, 60, (cluster_members_num, 2)),
            np.random.uniform(80, 100, (cluster_members_num, 2))
        ])
    
    elif num == 2:
        # SDA2: 4 clusters
        return np.concatenate([
            np.random.uniform(0, 20, (cluster_members_num, 2)),
            np.random.uniform(40, 60, (cluster_members_num, 2)),
            np.random.uniform(80, 100, (cluster_members_num, 2)),
            np.array([[np.random.uniform(0, 20), np.random.uniform(80, 100)] 
                     for _ in range(cluster_members_num)])
        ])
    
    elif num == 3:
        # SDA3: 8 clusters
        return np.concatenate([
            np.random.uniform(0, 20, (cluster_members_num, 2)),
            np.random.uniform(40, 60, (cluster_members_num, 2)),
            np.random.uniform(80, 100, (cluster_members_num, 2)),
            np.array([[np.random.uniform(80, 100), np.random.uniform(0, 20)] 
                     for _ in range(cluster_members_num)]),
            np.array([[np.random.uniform(0, 20), np.random.uniform(180, 200)] 
                     for _ in range(cluster_members_num)]),
            np.array([[np.random.uniform(180, 200), np.random.uniform(0, 20)] 
                     for _ in range(cluster_members_num)]),
            np.array([[np.random.uniform(180, 200), np.random.uniform(80, 100)] 
                     for _ in range(cluster_members_num)]),
            np.array([[np.random.uniform(180, 200), np.random.uniform(180, 200)] 
                     for _ in range(cluster_members_num)])
        ])
    
    else:
        raise ValueError("Dataset number must be 1, 2, or 3")

import numpy as np
import pytest
from qiga_kmeans.utils.data_generator import generate_sda_dataset


def test_generate_sda1():
    """Test generation of SDA1 dataset."""
    data = generate_sda_dataset(num=1, cluster_members_num=100)
    
    # Check shape
    assert isinstance(data, np.ndarray)
    assert data.shape == (300, 2)  # 3 clusters * 100 members
    
    # Check data ranges
    cluster1 = data[:100]
    cluster2 = data[100:200]
    cluster3 = data[200:]
    
    assert np.all(cluster1 >= 0) and np.all(cluster1 <= 20)
    assert np.all(cluster2 >= 40) and np.all(cluster2 <= 60)
    assert np.all(cluster3 >= 80) and np.all(cluster3 <= 100)


def test_generate_sda2():
    """Test generation of SDA2 dataset."""
    data = generate_sda_dataset(num=2, cluster_members_num=100)
    
    # Check shape
    assert isinstance(data, np.ndarray)
    assert data.shape == (400, 2)  # 4 clusters * 100 members
    
    # Check data ranges for first three clusters (same as SDA1)
    cluster1 = data[:100]
    cluster2 = data[100:200]
    cluster3 = data[200:300]
    cluster4 = data[300:]
    
    assert np.all(cluster1 >= 0) and np.all(cluster1 <= 20)
    assert np.all(cluster2 >= 40) and np.all(cluster2 <= 60)
    assert np.all(cluster3 >= 80) and np.all(cluster3 <= 100)
    
    # Check fourth cluster
    assert np.all(cluster4[:, 0] >= 0) and np.all(cluster4[:, 0] <= 20)
    assert np.all(cluster4[:, 1] >= 80) and np.all(cluster4[:, 1] <= 100)


def test_generate_sda3():
    """Test generation of SDA3 dataset."""
    data = generate_sda_dataset(num=3, cluster_members_num=100)
    
    # Check shape
    assert isinstance(data, np.ndarray)
    assert data.shape == (800, 2)  # 8 clusters * 100 members
    
    # Check data ranges for first three clusters (same as SDA1)
    cluster1 = data[:100]
    cluster2 = data[100:200]
    cluster3 = data[200:300]
    
    assert np.all(cluster1 >= 0) and np.all(cluster1 <= 20)
    assert np.all(cluster2 >= 40) and np.all(cluster2 <= 60)
    assert np.all(cluster3 >= 80) and np.all(cluster3 <= 100)


def test_invalid_dataset_number():
    """Test invalid dataset number."""
    with pytest.raises(ValueError):
        generate_sda_dataset(num=0)
    
    with pytest.raises(ValueError):
        generate_sda_dataset(num=4)


def test_custom_cluster_size():
    """Test custom cluster size."""
    members = 50
    data = generate_sda_dataset(num=1, cluster_members_num=members)
    
    assert data.shape == (members * 3, 2)  # 3 clusters * 50 members


def test_reproducibility():
    """Test reproducibility with same seed."""
    np.random.seed(42)
    data1 = generate_sda_dataset(num=1)
    
    np.random.seed(42)
    data2 = generate_sda_dataset(num=1)
    
    assert np.array_equal(data1, data2) 
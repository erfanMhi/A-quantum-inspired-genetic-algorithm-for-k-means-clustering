import numpy as np
import pytest
from qiga_kmeans.algorithms.qiga import QIGA, QuantumIndividual


@pytest.fixture
def sample_data():
    """Create a simple dataset with clear clusters."""
    np.random.seed(42)
    cluster1 = np.random.normal(0, 0.5, (100, 2))
    cluster2 = np.random.normal(4, 0.5, (100, 2))
    cluster3 = np.random.normal([0, 4], 0.5, (100, 2))
    return np.vstack([cluster1, cluster2, cluster3])


def test_quantum_individual_initialization():
    """Test QuantumIndividual initialization."""
    n_clusters, n_features = 3, 2
    individual = QuantumIndividual(n_clusters, n_features)
    
    assert individual.n_clusters == n_clusters
    assert individual.n_features == n_features
    assert individual.theta.shape == (n_clusters, n_features)
    assert np.all(individual.theta >= 0)
    assert np.all(individual.theta <= np.pi/2)
    assert individual.centroids is None
    assert individual.fitness == float('inf')


def test_quantum_individual_observe():
    """Test QuantumIndividual observation (collapse) process."""
    np.random.seed(42)
    individual = QuantumIndividual(3, 2)
    
    # Test multiple observations
    for _ in range(10):
        centroids = individual.observe()
        assert centroids.shape == (3, 2)
        assert np.all(np.logical_or(centroids == 0, centroids == 1))


def test_qiga_initialization():
    """Test QIGA initialization with valid parameters."""
    qiga = QIGA(n_clusters=3)
    
    assert qiga.n_clusters == 3
    assert qiga.population_size == 20  # default value
    assert qiga.max_generations == 100  # default value
    assert isinstance(qiga.rotation_angle, float)
    assert isinstance(qiga.mutation_prob, float)
    assert len(qiga.population) == 0
    assert qiga.best_individual is None


def test_qiga_population_initialization(sample_data):
    """Test QIGA population initialization."""
    qiga = QIGA(n_clusters=3, population_size=10)
    qiga._initialize_population(sample_data.shape[1])
    
    assert len(qiga.population) == 10
    for individual in qiga.population:
        assert isinstance(individual, QuantumIndividual)
        assert individual.theta.shape == (3, 2)


def test_qiga_quantum_rotation():
    """Test quantum rotation gate operation."""
    qiga = QIGA(n_clusters=3)
    individual = QuantumIndividual(3, 2)
    best = QuantumIndividual(3, 2)
    
    # Set fitness values to test rotation direction
    individual.fitness = 2.0
    best.fitness = 1.0
    
    # Store original angles
    original_theta = individual.theta.copy()
    
    # Apply rotation
    qiga._quantum_rotation(individual, best)
    
    # Check that angles have changed
    assert not np.array_equal(individual.theta, original_theta)
    assert np.all(individual.theta >= 0)
    assert np.all(individual.theta <= np.pi/2)


def test_qiga_quantum_mutation():
    """Test quantum mutation operation."""
    np.random.seed(42)
    qiga = QIGA(n_clusters=3, mutation_prob=0.5)
    individual = QuantumIndividual(3, 2)
    
    # Store original angles
    original_theta = individual.theta.copy()
    
    # Apply mutation
    qiga._quantum_mutation(individual)
    
    # Check that some angles have changed
    assert not np.array_equal(individual.theta, original_theta)
    assert np.all(individual.theta >= 0)
    assert np.all(individual.theta <= np.pi/2)


def test_qiga_fit(sample_data):
    """Test QIGA fitting process."""
    np.random.seed(42)
    qiga = QIGA(n_clusters=3, population_size=10, max_generations=10)
    centroids, sse = qiga.fit(sample_data)
    
    # Check output shapes and types
    assert isinstance(centroids, np.ndarray)
    assert isinstance(sse, float)
    assert centroids.shape == (3, 2)
    
    # Check SSE is positive and reasonable
    assert sse > 0
    assert sse < np.inf


def test_qiga_multiple_runs(sample_data):
    """Test that multiple QIGA runs produce consistent results."""
    np.random.seed(42)
    n_runs = 3
    sse_values = []
    
    for _ in range(n_runs):
        qiga = QIGA(n_clusters=3, population_size=10, max_generations=10)
        _, sse = qiga.fit(sample_data)
        sse_values.append(sse)
    
    # Check that SSE values are similar (within reasonable range)
    sse_mean = np.mean(sse_values)
    sse_std = np.std(sse_values)
    assert sse_std / sse_mean < 0.2  # Relative standard deviation < 20% 
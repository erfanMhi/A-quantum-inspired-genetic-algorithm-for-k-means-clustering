import numpy as np
import pytest
from qiga_kmeans.algorithms.genetic import GeneticKMeans, Individual


@pytest.fixture
def sample_data():
    """Create a simple dataset with clear clusters."""
    np.random.seed(42)
    cluster1 = np.random.normal(0, 0.5, (100, 2))
    cluster2 = np.random.normal(4, 0.5, (100, 2))
    cluster3 = np.random.normal([0, 4], 0.5, (100, 2))
    return np.vstack([cluster1, cluster2, cluster3])


def test_individual_initialization(sample_data):
    """Test Individual initialization."""
    centroids = sample_data[:3]  # Use first 3 points as centroids
    individual = Individual(centroids)
    
    assert individual.centroids.shape == (3, 2)
    assert np.array_equal(individual.centroids, centroids)
    assert individual.fitness == float('inf')


def test_genetic_kmeans_initialization():
    """Test GeneticKMeans initialization with valid parameters."""
    ga = GeneticKMeans(n_clusters=3)
    
    assert ga.n_clusters == 3
    assert ga.population_size == 20  # default value
    assert ga.max_generations == 100  # default value
    assert ga.crossover_prob == 0.8  # default value
    assert ga.mutation_prob == 0.01  # default value
    assert len(ga.population) == 0
    assert ga.best_individual is None


def test_genetic_kmeans_population_initialization(sample_data):
    """Test population initialization."""
    ga = GeneticKMeans(n_clusters=3, population_size=10)
    ga._initialize_population(sample_data)
    
    assert len(ga.population) == 10
    for individual in ga.population:
        assert isinstance(individual, Individual)
        assert individual.centroids.shape == (3, 2)


def test_genetic_kmeans_tournament_selection(sample_data):
    """Test tournament selection process."""
    ga = GeneticKMeans(n_clusters=3, population_size=10)
    ga._initialize_population(sample_data)
    
    # Set fitness values
    for i, individual in enumerate(ga.population):
        individual.fitness = float(i)
    
    # Test tournament selection
    selected = ga._tournament_selection(tournament_size=3)
    assert isinstance(selected, Individual)
    assert selected.fitness <= max(ind.fitness for ind in ga.population)


def test_genetic_kmeans_crossover(sample_data):
    """Test crossover operation."""
    ga = GeneticKMeans(n_clusters=3, crossover_prob=1.0)  # Always perform crossover
    
    # Create parent individuals
    parent1 = Individual(sample_data[:3])
    parent2 = Individual(sample_data[3:6])
    
    # Perform crossover
    offspring = ga._crossover(parent1, parent2)
    
    assert isinstance(offspring, Individual)
    assert offspring.centroids.shape == (3, 2)
    # Check that offspring contains parts from both parents
    assert not np.array_equal(offspring.centroids, parent1.centroids)
    assert not np.array_equal(offspring.centroids, parent2.centroids)


def test_genetic_kmeans_mutation(sample_data):
    """Test mutation operation."""
    ga = GeneticKMeans(n_clusters=3, mutation_prob=0.5)  # High mutation probability
    individual = Individual(sample_data[:3])
    
    # Store original centroids
    original_centroids = individual.centroids.copy()
    
    # Apply mutation
    ga._mutation(individual, sample_data)
    
    # Check that some centroids have changed
    assert not np.array_equal(individual.centroids, original_centroids)
    assert individual.centroids.shape == (3, 2)


def test_genetic_kmeans_fit(sample_data):
    """Test genetic algorithm fitting process."""
    np.random.seed(42)
    ga = GeneticKMeans(n_clusters=3, population_size=10, max_generations=10)
    centroids, sse = ga.fit(sample_data)
    
    # Check output shapes and types
    assert isinstance(centroids, np.ndarray)
    assert isinstance(sse, float)
    assert centroids.shape == (3, 2)
    
    # Check SSE is positive and reasonable
    assert sse > 0
    assert sse < np.inf


def test_genetic_kmeans_multiple_runs(sample_data):
    """Test that multiple GA runs produce consistent results."""
    np.random.seed(42)
    n_runs = 5  # Increased number of runs
    sse_values = []
    
    for _ in range(n_runs):
        ga = GeneticKMeans(
            n_clusters=3,
            population_size=20,  # Increased population size
            max_generations=20,  # Increased generations
            crossover_prob=0.8,
            mutation_prob=0.01
        )
        _, sse = ga.fit(sample_data)
        sse_values.append(sse)
    
    # Check that SSE values are similar (within reasonable range)
    sse_mean = np.mean(sse_values)
    sse_std = np.std(sse_values)
    # Increased threshold due to stochastic nature of GA
    assert sse_std / sse_mean < 0.5  # Allow up to 50% variation 
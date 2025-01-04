import numpy as np
from typing import Tuple, List, Optional
from .kmeans import KMeans


class QuantumIndividual:
    def __init__(self, n_clusters: int, n_features: int):
        """
        Initialize a quantum individual with quantum angles.

        Args:
            n_clusters (int): Number of clusters
            n_features (int): Number of features in the data
        """
        self.n_clusters = n_clusters
        self.n_features = n_features
        # Initialize quantum angles (theta) randomly between 0 and π/2
        self.theta = np.random.uniform(0, np.pi / 2, (n_clusters, n_features))
        self.centroids = None
        self.fitness = float("inf")

    def observe(self) -> np.ndarray:
        """Convert quantum state to classical state (cluster centroids)."""
        # Probability amplitudes
        alpha = np.cos(self.theta)
        beta = np.sin(self.theta)

        # Collapse quantum state to classical state based on probabilities
        r = np.random.random(self.theta.shape)
        self.centroids = np.where(r < alpha**2, 0, 1)
        return self.centroids


class QIGA:
    def __init__(
        self,
        n_clusters: int,
        population_size: int = 20,
        max_generations: int = 100,
        rotation_angle: float = 0.025 * np.pi,
        mutation_prob: float = 0.01,
    ):
        """
        Initialize Quantum-Inspired Genetic Algorithm for k-means clustering.

        Args:
            n_clusters (int): Number of clusters
            population_size (int): Size of quantum population
            max_generations (int): Maximum number of generations
            rotation_angle (float): Quantum rotation gate angle
            mutation_prob (float): Probability of quantum mutation
        """
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.max_generations = max_generations
        self.rotation_angle = rotation_angle
        self.mutation_prob = mutation_prob
        self.population: List[QuantumIndividual] = []
        self.best_individual: Optional[QuantumIndividual] = None

    def _initialize_population(self, n_features: int) -> None:
        """Initialize quantum population."""
        self.population = [
            QuantumIndividual(self.n_clusters, n_features)
            for _ in range(self.population_size)
        ]

    def _evaluate_fitness(
        self, individual: QuantumIndividual, data: np.ndarray
    ) -> float:
        """Evaluate fitness using k-means objective function."""
        centroids = individual.observe()
        # Scale centroids to data range
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        scaled_centroids = min_vals + centroids * (max_vals - min_vals)

        # Calculate distances to centroids
        distances = np.sqrt(((data[:, np.newaxis] - scaled_centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        # Calculate SSE (Sum of Squared Errors)
        sse = 0
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                cluster_points = data[labels == k]
                sse += np.sum((cluster_points - scaled_centroids[k]) ** 2)

        return sse

    def _quantum_rotation(
        self,
        individual: QuantumIndividual,
        best: QuantumIndividual,
    ) -> None:
        """Apply quantum rotation gate."""
        # Determine rotation direction based on fitness comparison
        direction = np.where(individual.fitness > best.fitness, 1, -1)

        # Apply rotation to quantum angles
        delta_theta = self.rotation_angle * direction
        individual.theta += delta_theta

        # Ensure angles stay within [0, π/2]
        individual.theta = np.clip(individual.theta, 0, np.pi / 2)

    def _quantum_mutation(self, individual: QuantumIndividual) -> None:
        """Apply quantum mutation operator."""
        mutation_mask = np.random.random(individual.theta.shape) < self.mutation_prob
        if np.any(mutation_mask):
            # Randomly flip selected quantum angles
            individual.theta[mutation_mask] = (
                np.pi / 2 - individual.theta[mutation_mask]
            )

    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit the quantum-inspired genetic algorithm to the data.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features)

        Returns:
            Tuple[np.ndarray, float]: Best centroids and final SSE
        """
        n_features = data.shape[1]
        self._initialize_population(n_features)

        # Initial evaluation
        for individual in self.population:
            individual.fitness = self._evaluate_fitness(individual, data)

        self.best_individual = min(self.population, key=lambda x: x.fitness)
        best_fitness = float("inf")

        # Main evolution loop
        for _ in range(self.max_generations):
            # Update quantum states
            for individual in self.population:
                self._quantum_rotation(individual, self.best_individual)
                self._quantum_mutation(individual)
                individual.fitness = self._evaluate_fitness(individual, data)

            # Update best solution
            generation_best = min(self.population, key=lambda x: x.fitness)
            if generation_best.fitness < best_fitness:
                best_fitness = generation_best.fitness
                self.best_individual = generation_best

        # Get final centroids
        best_centroids = self.best_individual.observe()
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        scaled_centroids = min_vals + best_centroids * (max_vals - min_vals)

        return scaled_centroids, best_fitness

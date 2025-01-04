import numpy as np
from typing import Tuple, List, Optional


class Individual:
    def __init__(self, centroids: np.ndarray):
        """
        Initialize an individual with cluster centroids.
        
        Args:
            centroids (np.ndarray): Initial cluster centroids
        """
        self.centroids = centroids.copy()
        self.fitness = float('inf')


class GeneticKMeans:
    def __init__(
        self,
        n_clusters: int,
        population_size: int = 20,
        max_generations: int = 100,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.01
    ):
        """
        Initialize Genetic Algorithm for k-means clustering.
        
        Args:
            n_clusters (int): Number of clusters
            population_size (int): Size of population
            max_generations (int): Maximum number of generations
            crossover_prob (float): Probability of crossover
            mutation_prob (float): Probability of mutation
        """
        self.n_clusters = n_clusters
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
    
    def _initialize_population(self, data: np.ndarray) -> None:
        """Initialize population with random centroids from data points."""
        n_samples = data.shape[0]
        self.population = []
        
        for _ in range(self.population_size):
            # Randomly select k points as initial centroids
            indices = np.random.choice(
                n_samples, self.n_clusters, replace=False
            )
            centroids = data[indices].copy()
            self.population.append(Individual(centroids))
    
    def _evaluate_fitness(
        self,
        individual: Individual,
        data: np.ndarray
    ) -> float:
        """Evaluate fitness using k-means objective function (SSE)."""
        distances = np.sqrt(((data[:, np.newaxis] - individual.centroids) ** 2)
                          .sum(axis=2))
        labels = np.argmin(distances, axis=1)
        
        # Calculate SSE
        sse = 0
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                cluster_points = data[labels == k]
                sse += np.sum((cluster_points - individual.centroids[k]) ** 2)
        
        return sse
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        return min(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform crossover between two parents."""
        if np.random.random() < self.crossover_prob:
            # Randomly select crossover point
            point = np.random.randint(1, self.n_clusters)
            # Create new centroids by combining parents
            new_centroids = np.vstack((
                parent1.centroids[:point],
                parent2.centroids[point:]
            ))
            return Individual(new_centroids)
        return Individual(parent1.centroids)
    
    def _mutation(self, individual: Individual, data: np.ndarray) -> None:
        """Apply mutation operator."""
        for i in range(self.n_clusters):
            if np.random.random() < self.mutation_prob:
                # Replace centroid with random data point
                idx = np.random.randint(len(data))
                individual.centroids[i] = data[idx]
    
    def fit(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit the genetic algorithm to the data.
        
        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple[np.ndarray, float]: Best centroids and final SSE
        """
        self._initialize_population(data)
        
        # Initial evaluation
        for individual in self.population:
            individual.fitness = self._evaluate_fitness(individual, data)
        
        self.best_individual = min(self.population, key=lambda x: x.fitness)
        best_fitness = self.best_individual.fitness
        
        # Main evolution loop
        for _ in range(self.max_generations):
            new_population = []
            
            # Create new population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                offspring = self._crossover(parent1, parent2)
                
                # Mutation
                self._mutation(offspring, data)
                
                # Evaluate new individual
                offspring.fitness = self._evaluate_fitness(offspring, data)
                new_population.append(offspring)
            
            # Update population
            self.population = new_population
            
            # Update best solution
            generation_best = min(self.population, key=lambda x: x.fitness)
            if generation_best.fitness < best_fitness:
                best_fitness = generation_best.fitness
                self.best_individual = generation_best
        
        return self.best_individual.centroids, best_fitness

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    title: str = "Clustering Results",
    save_path: Optional[str] = None
) -> None:
    """
    Plot clustering results with data points and centroids.
    
    Args:
        data (np.ndarray): Input data points
        labels (np.ndarray): Cluster labels for each point
        centroids (np.ndarray): Cluster centroids
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            c=[color],
            label=f'Cluster {label}',
            alpha=0.6
        )
    
    # Plot centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='black',
        marker='x',
        s=200,
        linewidths=3,
        label='Centroids'
    )
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_convergence(
    sse_history: list,
    title: str = "Convergence Plot",
    save_path: Optional[str] = None
) -> None:
    """
    Plot convergence history of the clustering algorithm.
    
    Args:
        sse_history (list): History of SSE values
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sse_history, 'b-', label='SSE')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Squared Errors')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_comparison(
    data: np.ndarray,
    kmeans_labels: np.ndarray,
    kmeans_centroids: np.ndarray,
    ga_labels: np.ndarray,
    ga_centroids: np.ndarray,
    qiga_labels: np.ndarray,
    qiga_centroids: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of different clustering algorithms.
    
    Args:
        data (np.ndarray): Input data points
        kmeans_labels (np.ndarray): K-means cluster labels
        kmeans_centroids (np.ndarray): K-means centroids
        ga_labels (np.ndarray): GA cluster labels
        ga_centroids (np.ndarray): GA centroids
        qiga_labels (np.ndarray): QIGA cluster labels
        qiga_centroids (np.ndarray): QIGA centroids
        save_path (Optional[str]): Path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot settings
    algorithms = ['K-means', 'Genetic Algorithm', 'QIGA']
    labels = [kmeans_labels, ga_labels, qiga_labels]
    centroids = [kmeans_centroids, ga_centroids, qiga_centroids]
    axes = [ax1, ax2, ax3]
    
    for ax, alg, lab, cent in zip(axes, algorithms, labels, centroids):
        # Plot data points
        unique_labels = np.unique(lab)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = lab == label
            ax.scatter(
                data[mask, 0],
                data[mask, 1],
                c=[color],
                label=f'Cluster {label}',
                alpha=0.6
            )
        
        # Plot centroids
        ax.scatter(
            cent[:, 0],
            cent[:, 1],
            c='black',
            marker='x',
            s=200,
            linewidths=3,
            label='Centroids'
        )
        
        ax.set_title(alg)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

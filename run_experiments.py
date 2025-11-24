"""
Run experiments from Private Geometric Median paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from private_geometric_median import (
    GeometricMedianProblem,
    private_geometric_median, 
    baseline_dpgd
)

sns.set_style("whitegrid")


def generate_data(n_samples, dimension):
    """Generate synthetic dataset with inliers and outliers."""
    n_inliers = int(0.9 * n_samples)
    n_outliers = n_samples - n_inliers
    radius = 100
    
    # Inliers: tightly clustered
    center_dir = np.random.randn(dimension)
    center = (radius / 2) * center_dir / np.linalg.norm(center_dir)
    inliers = np.random.multivariate_normal(
        center, 0.01 * np.eye(dimension), n_inliers
    )
    
    # Outliers: uniformly distributed
    directions = np.random.randn(n_outliers, dimension)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * np.random.rand(n_outliers) ** (1/dimension)
    outliers = directions * radii[:, np.newaxis]
    
    return np.vstack([inliers, outliers])


def run_comparison(data, max_radius, epsilon, n_trials=10):
    """Compare algorithms for given parameters."""
    problem = GeometricMedianProblem(data, max_radius)
    true_loss = problem.loss(problem.true_median)
    
    our_ratios = []
    baseline_ratios = []
    
    for _ in range(n_trials):
        # Our algorithm
        our_result = private_geometric_median(data, max_radius, epsilon)
        our_ratios.append(problem.loss(our_result) / true_loss)
        
        # Baseline
        baseline_result = baseline_dpgd(data, max_radius, epsilon)
        baseline_ratios.append(problem.loss(baseline_result) / true_loss)
    
    return (np.median(our_ratios), np.std(our_ratios) / np.sqrt(n_trials),
            np.median(baseline_ratios), np.std(baseline_ratios) / np.sqrt(n_trials))


def plot_results(n_samples=3000, dimension=200, epsilon=2.0):
    """Generate comparison plot."""
    print(f"\nRunning experiments: n={n_samples}, d={dimension}, ε={epsilon}")
    
    # Generate data once
    data = generate_data(n_samples, dimension)
    
    # Test different radius values
    radius_values = np.logspace(3, 10, 8)
    
    our_means, our_stds = [], []
    baseline_means, baseline_stds = [], []
    
    for radius in tqdm(radius_values):
        results = run_comparison(data, radius, epsilon, n_trials=10)
        our_means.append(results[0])
        our_stds.append(results[1])
        baseline_means.append(results[2])
        baseline_stds.append(results[3])
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(radius_values, our_means, yerr=our_stds, 
                label='Our Algorithm', linewidth=2, marker='o')
    plt.errorbar(radius_values, baseline_means, yerr=baseline_stds,
                label='DP-GD Baseline', linewidth=2, marker='s')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('A priori radius R', fontsize=14)
    plt.ylabel('Loss Ratio F(θ̂)/F(θ*)', fontsize=14)
    plt.title(f'd={dimension}, n={n_samples}, ε={epsilon}, δ=1/n', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results_eps_{epsilon}.pdf', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Run experiments for different privacy levels
    for epsilon in [2.0, 3.0]:
        plot_results(epsilon=epsilon)

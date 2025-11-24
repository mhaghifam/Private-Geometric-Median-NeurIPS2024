"""
Private Geometric Median - NeurIPS 2024
Haghifam, Steinke, Ullman

Single-file implementation for differentially private geometric median computation.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.distance import cdist
import torch
from geom_median.torch import compute_geometric_median


# ===== Privacy Conversion Utilities =====

def zcdp_to_dp(rho, delta):
    """Convert zCDP to (epsilon, delta)-DP."""
    return rho + np.sqrt(4 * rho * np.log(np.sqrt(np.pi * rho) / delta))


def dp_to_zcdp(epsilon, delta):
    """Convert (epsilon, delta)-DP to zCDP."""
    objective = lambda x: zcdp_to_dp(x, delta) - epsilon
    return fsolve(objective, x0=0.001)[0]


# ===== Core Classes and Functions =====

class GeometricMedianProblem:
    """Geometric median optimization problem."""
    
    def __init__(self, data, max_radius):
        """
        Args:
            data: (n, d) array of data points
            max_radius: A priori bound on data norm
        """
        data = np.array(data)
        assert len(data.shape) == 2, "Data must be 2D array"
        
        self.max_radius = max_radius
        self.n_samples, self.dimension = data.shape
        
        # Clip data to satisfy norm constraint
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        scale = np.minimum(max_radius / np.maximum(norms, 1e-10), 1.0)
        self.data = scale * data
        
        # Compute true geometric median
        self.true_median = self._compute_exact_median()
    
    def loss(self, theta):
        """Compute loss: (1/n) sum ||theta - xi||"""
        distances = np.linalg.norm(self.data - theta, axis=1)
        return np.mean(distances)
    
    def gradient(self, theta):
        """Compute gradient of loss."""
        diff = theta - self.data
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        normalized = diff / np.maximum(norms, 1e-10)
        return np.mean(normalized, axis=0)
    
    def _compute_exact_median(self):
        """Compute non-private geometric median."""
        torch_points = [torch.from_numpy(p).float() for p in self.data]
        result = compute_geometric_median(torch_points)
        return result.median.numpy()


def dp_gradient_descent(problem, init_point, iterations, privacy_budget, step_size, radius):
    """
    Differentially Private Gradient Descent.
    
    Args:
        problem: GeometricMedianProblem instance
        init_point: Starting point
        iterations: Number of iterations
        privacy_budget: Privacy budget (zCDP)
        step_size: Step size
        radius: Feasible set radius
        
    Returns:
        Average of all iterates
    """
    sensitivity = 2.0 / problem.n_samples
    rho_per_iter = privacy_budget / iterations
    
    theta = init_point.copy()
    avg_theta = init_point.copy()
    
    for t in range(iterations):
        # Add calibrated noise
        noise_scale = sensitivity / np.sqrt(2 * rho_per_iter)
        noise = np.random.normal(0, noise_scale, size=problem.dimension)
        
        # Gradient step
        theta = theta - step_size * (problem.gradient(theta) + noise)
        
        # Project to feasible set
        diff = theta - init_point
        norm_diff = np.linalg.norm(diff)
        if norm_diff > radius:
            theta = init_point + (diff / norm_diff) * radius
        
        # Update average
        avg_theta = (t / (t + 1)) * avg_theta + (1 / (t + 1)) * theta
    
    return avg_theta


# ===== Warm-up Algorithm =====

class WarmupAlgorithm:
    """Warm-up phase for finding good initialization."""
    
    def __init__(self, problem, discretization, privacy_budget, failure_prob):
        self.problem = problem
        self.discretization = discretization
        self.privacy_budget = privacy_budget
        self.failure_prob = failure_prob
        
        # Create radius grid
        max_exp = int(np.floor(np.log2(2 * problem.max_radius / discretization)))
        self.radius_grid = [discretization * (2 ** i) for i in range(max_exp + 1)]
        
        # Quantile parameter
        self.gamma = 3/4
        self.m = int(np.ceil(problem.n_samples * self.gamma))
    
    def above_threshold(self, queries):
        """Above threshold mechanism for private selection."""
        eps_at = np.sqrt(2 * self.privacy_budget / 2)
        num_queries = len(queries)
        
        # Compute noisy threshold
        threshold = self.m + (18 / eps_at) * np.log(2 * num_queries / self.failure_prob)
        noisy_threshold = threshold + np.random.laplace(0, 6 / eps_at)
        
        # Check queries with noise
        for idx, query in enumerate(queries):
            noisy_query = query + np.random.laplace(0, 12 / eps_at)
            if noisy_query >= noisy_threshold:
                return idx
        
        return np.random.randint(len(queries))
    
    def compute_queries(self):
        """Compute radius queries."""
        pairwise_dist = cdist(self.problem.data, self.problem.data, 'euclidean')
        queries = []
        
        for radius in self.radius_grid:
            counts = np.sum(pairwise_dist <= radius, axis=1)
            top_m = np.sort(counts)[-self.m:]
            queries.append(np.sum(top_m) / self.m)
        
        return np.array(queries)
    
    def estimate_radius(self):
        """Estimate quantile radius."""
        queries = self.compute_queries()
        idx = self.above_threshold(queries)
        return self.radius_grid[idx]
    
    def localize(self):
        """Find initialization point."""
        # Estimate radius
        est_radius = self.estimate_radius()
        print(f"Estimated radius: {est_radius:.4f}")
        
        # Iterative localization
        current_radius = self.problem.max_radius
        num_iter = max(1, int(np.ceil(np.log2(self.problem.max_radius / est_radius))))
        privacy_per_iter = (self.privacy_budget / 2) / num_iter
        
        current_center = np.zeros(self.problem.dimension)
        
        for _ in range(num_iter):
            step_size = current_radius * np.sqrt(
                2 * self.problem.dimension / 
                (3 * privacy_per_iter * self.problem.n_samples ** 2)
            )
            
            current_center = dp_gradient_descent(
                self.problem, current_center, 1000, 
                privacy_per_iter, step_size, current_radius
            )
            
            current_radius = 0.5 * current_radius + 12 * est_radius
        
        return current_center, est_radius


# ===== Main Algorithm =====

def private_geometric_median(data, max_radius, epsilon, delta=None, discretization=0.05):
    """
    Compute differentially private geometric median.
    
    Args:
        data: (n, d) array of data points
        max_radius: A priori bound on data norm
        epsilon: Privacy parameter
        delta: Privacy failure probability (default: 1/n)
        discretization: Grid discretization for radius search
        
    Returns:
        Private geometric median estimate
    """
    n = len(data)
    if delta is None:
        delta = 1.0 / n
    
    # Convert to zCDP
    rho = dp_to_zcdp(epsilon, delta)
    
    # Create problem
    problem = GeometricMedianProblem(data, max_radius)
    
    # Warm-up phase
    warmup = WarmupAlgorithm(problem, discretization, rho / 2, 0.05)
    init_point, est_radius = warmup.localize()
    
    # Fine-tuning phase
    fine_tune_radius = 25 * est_radius
    fine_tune_stepsize = 50 * est_radius * np.sqrt(
        problem.dimension / (6 * rho * problem.n_samples ** 2)
    )
    fine_tune_iterations = int(problem.n_samples ** 2 * rho / (256 * problem.dimension))
    
    result = dp_gradient_descent(
        problem, init_point, fine_tune_iterations,
        rho / 2, fine_tune_stepsize, fine_tune_radius
    )
    
    return result


# ===== Baseline Algorithm =====

def baseline_dpgd(data, max_radius, epsilon, delta=None):
    """Standard DP-GD baseline."""
    n = len(data)
    if delta is None:
        delta = 1.0 / n
    
    rho = dp_to_zcdp(epsilon, delta)
    problem = GeometricMedianProblem(data, max_radius)
    
    # Standard DP-GD parameters
    step_size = 2 * max_radius * np.sqrt(
        problem.dimension / (12 * rho * problem.n_samples ** 2)
    )
    iterations = int(problem.n_samples ** 2 * rho / (128 * problem.dimension))
    
    result = dp_gradient_descent(
        problem, np.zeros(problem.dimension), 
        iterations, rho, step_size, max_radius
    )
    
    return result

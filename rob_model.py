import numpy as np
import pandas as pd
from scipy.optimize import root
from joblib import Parallel, delayed

class RobModel:
    """
    Implementation of the Outbreak Reproduction Ratio (Rob) framework.
    This model quantifies community-specific epidemic risk in spatially 
    structured populations using multitype branching processes[cite: 6, 39].
    """

    def __init__(self, R_matrix=None):
        """
        Initialize the model with a Reproduction Operator (Matrix) R[cite: 41].
        R_ij represents the expected secondary infections in community i 
        caused by a single infected resident of community j[cite: 41, 182].
        """
        self.R = R_matrix
        self.n_nodes = R_matrix.shape[0] if R_matrix is not None else 0

    def solve_p_vector(self, R=None, tol=1e-12, max_iter=2000):
        """
        Solves the fixed-point equation for the epidemic probability vector p:
        p_i = 1 - exp(-sum_j R_ji * p_j)[cite: 51, 186].
        
        This vector represents the probability that an introduction in 
        community i triggers a major epidemic[cite: 47, 183].
        """
        mat = R if R is not None else self.R
        n = mat.shape[0]
        p = np.ones(n) * 0.5  # Initial seed for the iterative solver
        
        for _ in range(max_iter):
            p_old = p.copy()
            # Vectorized computation of the Poisson Generating Function [cite: 186]
            # p = 1 - G(1-p), where G(z) = exp(R^T * (z-1))
            p = 1 - np.exp(-mat.T @ p)
            
            if np.linalg.norm(p - p_old) < tol:
                return p
        return p

    def calculate_rob(self, R=None, p_vector=None):
        """
        Calculates the Outbreak Reproduction Ratio (Rob) for each community[cite: 32].
        Formula: Rob_i = -ln(1 - p_i) / p_i[cite: 82, 206].
        
        This metric accounts for local transmission and system-level 
        source-sink dynamics[cite: 33, 149].
        """
        mat = R if R is not None else self.R
        if p_vector is None:
            p_vector = self.solve_p_vector(mat)
            
        rob = np.zeros_like(p_vector)
        # Numerical mask to handle communities above the epidemic threshold [cite: 94]
        mask = p_vector > 1e-12 
        
        # Apply the analytical simplification derived in the paper [cite: 82, 206]
        rob[mask] = -np.log(1 - p_vector[mask]) / p_vector[mask]
        
        # Continuation: For subcritical nodes (p_i -> 0), Rob converges 
        # to the system's spectral radius (Reference R)[cite: 95, 235].
        if np.any(~mask):
            eigvals = np.linalg.eigvals(mat)
            r_ref = np.max(np.real(eigvals))
            rob[~mask] = r_ref
            
        return rob

    @staticmethod
    def simulate_branching_process(R, seed_idx, max_size, Tmax=500):
        """
        Simulates a single realization of a multitype branching process[cite: 236].
        
        Returns:
            - is_epidemic: Boolean, True if total size reaches max_size[cite: 240, 241].
            - gen1_infections: Total secondary infections from the index case[cite: 76, 181].
        """
        n = R.shape[0]
        # Current infected individuals per community
        active_cases = np.zeros(n)
        active_cases[seed_idx] = 1
        total_infections = 1
        
        # Generation 1: Track offspring specifically for Rob expectation definition [cite: 81]
        gen1_offspring = np.random.poisson(R @ active_cases)
        active_cases = gen1_offspring.copy()
        total_infections += active_cases.sum()
        
        # Secondary infection count from index case
        x_init = gen1_offspring.sum()
        
        # Propagation until extinction or threshold [cite: 240]
        for t in range(2, Tmax):
            if active_cases.sum() == 0:
                return False, x_init
            if total_infections >= max_size:
                return True, x_init
                
            # Each community's current cases generate new infections across the system [cite: 41, 186]
            active_cases = np.random.poisson(R @ active_cases)
            total_infections += active_cases.sum()
            
        return False, x_init

def run_monte_carlo_validation(R, seed_idx, max_size, n_runs=10000):
    """
    Validates Rob using the statistical expectation definition[cite: 81, 243]:
    Rob_i = E[Xi | epidemic] - E[Xi | extinction]
    """
    # Execute simulations in parallel for computational efficiency [cite: 28, 159]
    results = Parallel(n_jobs=-1)(
        delayed(RobModel.simulate_branching_process)(R, seed_idx, max_size) 
        for _ in range(n_runs)
    )
    
    is_epidemic, x_counts = zip(*results)
    is_epidemic = np.array(is_epidemic)
    x_counts = np.array(x_counts)
    
    p_epidemic = is_epidemic.mean()
    
    # Calculate empirical Rob based on the sample means of conditioned outcomes [cite: 81, 91]
    if p_epidemic > 0 and p_epidemic < 1:
        rob_est = x_counts[is_epidemic].mean() - x_counts[~is_epidemic].mean()
    else:
        rob_est = x_counts.mean() # Fallback for edge cases
        
    return p_epidemic, rob_est

# Rob: Outbreak Reproduction Ratio Framework

This repository provides the official implementation for the **Outbreak Reproduction Ratio ($R^{ob}$)**, a community-specific metric for epidemic risk assessment in spatially structured populations. 

## Overview
$R^{ob}$ integrates local transmission dynamics with mobility-driven "source-sink" effects. It identifies risk hotspots that traditional local ($R_{ii}$) or global ($R^{ref}$) metrics often overlook.

## Mathematical Core
The framework uses **multitype branching processes**. $R^{ob}$ is defined as the difference in expected secondary infections conditioned on the epidemic outcome:
$$R^{ob}_i = \mathbb{E}[X_i | \text{epidemic}] - \mathbb{E}[X_i | \text{extinction}]$$

Which simplifies to the elegant analytical form:
$$R^{ob}_i = -\frac{\ln(1 - p_i)}{p_i}$$

## Key Features
- **Theoretical Solver**: Compute $R^{ob}$ and epidemic probabilities from a reproduction operator (Next-Generation Matrix).
- **Stochastic Engine**: Validate metrics using parallelized Monte Carlo simulations of branching processes.
- **Real-world Application**: Validated against SARS-CoV-2 transmission chains in Canada and multi-country spatial datasets (13+ countries).

## Quick Start
```python
from rob_model import RobModel
import numpy as np

# 1. Initialize your Reproduction Operator Matrix R
# R_ij represents expected infections in i caused by a resident of j
R = np.load("your_matrix.npy") 

# 2. Calculate theoretical Rob values
model = RobModel()
p_vector = model.solve_p_vector(R)
rob_values = model.calculate_rob(R, p_vector)

# 3. Validation via Simulation
from rob_model import run_monte_carlo
p_epi, rob_sim = run_monte_carlo(R, seed_idx=0, max_size=1000)


## Citation
Wang, B., & Valdano, E. (2026). Redefining and estimating the early-phase reproduction ratio for epidemic outbreaks in spatially structured populations.


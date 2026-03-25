# R_ob_code
Here we offer all code of Redefining and estimating the early-phase reproduction ratio for  epidemic outbreaks in spatially structured populations
# Outbreak Reproduction Ratio ($R^{ob}$) Framework

[cite_start]This repository provides the official implementation for the **Outbreak Reproduction Ratio ($R^{ob}$)**, a community-specific metric for epidemic risk assessment in spatially structured populations[cite: 6].

## Overview
[cite_start]$R^{ob}$ integrates local transmission dynamics with mobility-driven "source-sink" effects[cite: 33]. [cite_start]It identifies risk hotspots that traditional local ($R_{ii}$) or global ($R^{ref}$) metrics often overlook[cite: 8, 72].

## Mathematical Core
[cite_start]The framework uses **multitype branching processes**[cite: 39, 46]. [cite_start]$R^{ob}$ is defined as[cite: 81]:
$$R^{ob}_i = \mathbb{E}[X_i | \text{epidemic}] - \mathbb{E}[X_i | \text{extinction}]$$
[cite_start]Which simplifies to the analytical form[cite: 82]:
$$R^{ob}_i = -\frac{\ln(1 - p_i)}{p_i}$$

## Features
- [cite_start]**Theoretical Solver**: Compute $R^{ob}$ and epidemic probabilities from a reproduction operator[cite: 41, 149].
- [cite_start]**Stochastic Engine**: Validate metrics using parallelized Monte Carlo simulations[cite: 236].
- [cite_start]**Real-world Application**: Applied to SARS-CoV-2 (Canada) and multi-country spatial datasets[cite: 37, 38].

## Quick Start
```python
from rob_model import RobModel
import numpy as np

# Load Reproduction Matrix R
R = np.load("contact_matrix.npy") 

# 1. Theoretical Calculation
rob_values = RobModel.calculate_rob(R)

# 2. Monte Carlo Validation
from rob_model import run_monte_carlo
p_epi, rob_sim = run_monte_carlo(R, seed_idx=0, max_size=1000)

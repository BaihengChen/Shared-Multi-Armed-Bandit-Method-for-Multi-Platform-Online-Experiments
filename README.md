# Chain Sharing UCB — Simulation Code

This repository contains the simulation code supporting the empirical results in Section 5.2 ("Simulation Study"-"Results Under Oracle Weights") of the paper *"Shared Multi-Armed Bandit Method for Multi-Platform Online Experiments."*

## Overview

The code implements the chain sharing UCB scheme and reproduces two sets of results:

1. **Oracle-weight simulations** (Section 5.1): Monte Carlo experiments under known arm gaps, evaluating regret and MSE win rates across $K \in \{2, 4, 7, 10\}$ platforms and three difficulty settings (Baseline, Easy, Hard).
2. **Realistic simulation** (Section 5.2): a four-platform advertising setting in which sharing weights are estimated online from a warm-start phase rather than assumed known, evaluating how closely the estimated-weight scheme recovers oracle performance.

## Repository Structure

### Core algorithm
| File | Description |
|---|---|
| `chain_ucb_simulation.py` | Core implementation of the chain sharing UCB algorithm for $K$ platforms. |
| `chain_sim_demo.py` | Minimal demo script illustrating a single run of the chain sharing scheme. |
| `realistic_simulation.py` | Implementation of the four-platform realistic simulation with warm-start weight estimation. |
| `format_results.py` | Utility functions for aggregating and formatting simulation output (win rate tables, summary statistics). |

### Simulation notebooks (oracle-weight experiments)
Organized by number of platforms ($K$) and difficulty setting. Each notebook runs the Monte Carlo simulation for one $(K, \text{setting})$ configuration and reports regret and MSE win rates.

| Platforms | Baseline | Easy | Hard |
|---|---|---|---|
| $K=2$  | `2_platform_Simulation_Baseline.ipynb`  | `2_platform_Simulation_Easy.ipynb`  | `2_platform_Simulation_Hard.ipynb` |
| $K=4$  | `4_platform_Simulation_Baseline.ipynb`  | `4_platform_Simulation_Easy.ipynb`  | `4_platform_Simulation_Hard.ipynb` |
| $K=7$  | `7_platform_Simulation_Baseline.ipynb`  | `7_platform_Simulation_Easy.ipynb`  | `7_platform_Simulation_Hard.ipynb` |
| $K=10$ | `10_platform_Simulation_Baseline.ipynb` | `10_platform_Simulation_Easy.ipynb` | `10_platform_Simulation_Hard.ipynb` |

### Realistic simulation
| File | Description |
|---|---|
| `Run_Simulation.ipynb` | Runs the four-platform realistic simulation (Section 5.2) using estimated, warm-start weights and compares against oracle performance. |

## Requirements

- Python 3.x
- Jupyter Notebook / JupyterLab
- Standard scientific Python stack (NumPy, pandas, SciPy, Matplotlib)

## Usage

1. Run the platform-specific notebooks (`*_platform_Simulation_*.ipynb`) to reproduce the oracle-weight regret and MSE win rate results in Table S2 and S3.
2. Run `Run_Simulation.ipynb` to reproduce the four-platform realistic simulation results in Section 5.2 and Table S4.
3. Use `format_results.py` to aggregate raw simulation output into the summary tables reported in the paper.

## Notes

- All reported results are averaged over $K_{\text{rep}} = 1000$ replications per configuration.
- Statistical significance of win rates is assessed via the Wilcoxon signed-rank test.

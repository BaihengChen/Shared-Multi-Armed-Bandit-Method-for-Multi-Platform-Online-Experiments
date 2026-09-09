# Chain Sharing UCB — Realistic Simulation (Section 5.3)

This repository contains the code supporting the realistic, estimated-weight simulation study in Section 5.3 of the paper *"Shared Multi-Armed Bandit Method for Multi-Platform Online Experiments."* It evaluates the chain sharing UCB scheme on a four-platform advertising setting where sharing weights are estimated online from a warm-start phase, rather than assumed known as in the oracle-weight simulations.

## Repository Structure

| File | Description |
|---|---|
| `chain_ucb_simulation.py` | Core implementation of the chain sharing UCB algorithm. |
| `chain_sim_demo.py` | Minimal demo script illustrating a single run of the chain sharing scheme. |
| `realistic_simulation.py` | Core simulation logic for the four-platform realistic setting, including warm-start weight estimation. |
| `realistic_simulation_full.ipynb` | Full notebook driving the realistic simulation end to end and reproducing the Section 5.3 results. |
| `format_results.py` | Utility functions for aggregating and formatting simulation output (win rate tables, summary statistics). |

## Overview

The realistic simulation is calibrated to four advertising platforms (Google Search, Facebook Feed, Instagram Stories, YouTube Pre-roll) with click-through rates set from realistic configurations. Sharing weights $(\alpha_i, \beta_i)$ are estimated from a brief warm-start phase using a pooled arm-2 baseline and isotonic projection, then held fixed throughout the UCB phase. Results are compared against oracle-weight performance to assess how much of the theoretical gain is retained when weights must be estimated rather than known.

## Requirements

- Python 3.x
- Jupyter Notebook / JupyterLab
- Standard scientific Python stack (NumPy, pandas, SciPy, Matplotlib)

## Usage

1. Run `realistic_simulation_full.ipynb` to reproduce the full Section 5.3 pipeline: warm-start weight estimation, chain sharing UCB simulation, and evaluation against the oracle baseline.
2. `chain_sim_demo.py` can be run independently for a quick illustration of the chain sharing algorithm on a single simulated instance.
3. Use `format_results.py` to aggregate raw simulation output into the summary tables reported in the paper (regret win rate, MSE win rate, mean absolute regret decrease by checkpoint).

## Notes

- All reported results are averaged over $K_{\text{rep}} = 1000$ replications.
- Statistical significance of win rates is assessed via the Wilcoxon signed-rank test.
- Sharing weights are fixed after the warm-start phase; see the paper's discussion of instability under continuous online weight updates.

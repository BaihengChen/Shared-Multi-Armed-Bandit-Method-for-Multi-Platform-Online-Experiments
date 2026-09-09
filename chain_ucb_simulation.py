#!/usr/bin/env python
# coding: utf-8

# ============================================================
# chain_ucb_simulation.py
#
# K-platform chain sharing UCB simulation.
# Platforms are ordered by gap: delta_0 > delta_1 > ... > delta_{K-1}
# Platform 0 is the anchor (pure UCB).
# Platform i (i >= 1) blends its own xbar with platform i-1's raw xbar.
# ============================================================

import numpy as np
from scipy.stats import wilcoxon


# ============================================================
# 1) Parameter generator
# ============================================================

def sample_params_chain(
    rng, K, delta_min, delta_max,
    mu2_range=(0.01, 0.10),
    offset_range=None,
    eps=0.005,
):
    """
    Sample environment parameters for K platforms.

    Gaps are drawn from Uniform(delta_min, delta_max) then sorted descending:
        delta[0] > delta[1] > ... > delta[K-1]

    Two modes for mu_i2 (suboptimal-arm base rate):

    Independent mode (offset_range=None):
        mu_i2 ~ Uniform(*mu2_range)  independently for each platform.

    Anchored mode (offset_range=(-h, h)):
        mu_base ~ Uniform(*mu2_range)            # one shared base rate
        mu_i2   = clip(mu_base + U(-h, h), eps, effective_hi)
        Simulates the same user population across platforms with small
        platform-specific deviations.

    mu_i1 = mu_i2 + delta[i]  (guaranteed < 1 by effective_hi constraint)

    Returns:
        list of K dicts, each with:
        - 'mu': {1: mu_i1, 2: mu_i2}
        - 'delta': float
    """
    deltas = np.sort(rng.uniform(delta_min, delta_max, size=K))[::-1]

    mu2_lo, mu2_hi = mu2_range
    effective_hi = min(mu2_hi, 1.0 - deltas[0] - eps)
    if effective_hi <= mu2_lo:
        raise ValueError(
            f"delta_max={deltas[0]:.4f} too large for mu2_range={mu2_range}; "
            f"no valid mu_i2 interval."
        )

    if offset_range is None:
        # Independent: each platform samples its own mu_i2
        mu2_vals = rng.uniform(mu2_lo, effective_hi, size=K)
    else:
        # Anchored: one shared base + small per-platform offsets
        mu_base = rng.uniform(mu2_lo, effective_hi)
        offsets = rng.uniform(offset_range[0], offset_range[1], size=K)
        mu2_vals = np.clip(mu_base + offsets, mu2_lo, effective_hi)

    params = []
    for i in range(K):
        mu2 = float(mu2_vals[i])
        mu1 = mu2 + deltas[i]
        params.append({"mu": {1: mu1, 2: mu2}, "delta": float(deltas[i])})

    return params


# ============================================================
# 2) Single paired replication
# ============================================================

def run_single_replication_chain(params, T, M0, C, rng):
    """
    Run one paired (baseline vs sharing) replication.

    Both modes use the same pre-generated reward table (paired design).

    Sharing formula for platform i (i >= 1):
        alpha_i = delta_{i-1} / (delta_{i-1} + delta_i)
        beta_i  = delta_i    / (delta_{i-1} + delta_i)
        mean_i(k) = alpha_i * xbar_i(k) + beta_i * xbar_{i-1}(k)

    Platform 0 always uses pure UCB (alpha=1, beta=0).

    Decision and update are synchronous: all platforms decide at time t
    using statistics from t-1, then all observe and update.

    Returns:
        dict with keys:
        - 'regret_base'  : list[K] cumulative regret per platform (baseline)
        - 'regret_share' : list[K] cumulative regret per platform (sharing)
        - 'means_base'   : list[K] of {1: float, 2: float} final mean estimates (baseline)
        - 'means_share'  : list[K] of {1: float, 2: float} final mean estimates (sharing)
    """
    K = len(params)

    # Pre-generate reward tables
    table = []
    for i in range(K):
        row = {}
        for k in [1, 2]:
            p = params[i]["mu"][k]
            row[k] = rng.binomial(n=1, p=p, size=T + M0 + 5)
        table.append(row)

    # Precompute sharing weights (platform 0 has alpha=1, beta=0)
    alphas = [1.0] * K
    betas = [0.0] * K
    for i in range(1, K):
        d_prev = params[i - 1]["delta"]
        d_curr = params[i]["delta"]
        alphas[i] = d_prev / (d_prev + d_curr)
        betas[i] = d_curr / (d_prev + d_curr)

    def run_ucb(mode):
        N = [{1: 0, 2: 0} for _ in range(K)]
        S = [{1: 0.0, 2: 0.0} for _ in range(K)]
        regret = [0.0] * K

        def xbar(i, k):
            return S[i][k] / N[i][k]

        # Warm start
        for i in range(K):
            for k in [1, 2]:
                for _ in range(M0):
                    y = table[i][k][N[i][k]]
                    N[i][k] += 1
                    S[i][k] += y

        for t in range(1, T + 1):
            # --- Phase 1: all platforms compute their action (synchronous) ---
            actions = []
            for i in range(K):
                if mode == "baseline" or i == 0:
                    mean = {k: xbar(i, k) for k in [1, 2]}
                else:
                    mean = {
                        k: alphas[i] * xbar(i, k) + betas[i] * xbar(i - 1, k)
                        for k in [1, 2]
                    }
                ucb = {k: mean[k] + C * np.sqrt(np.log(t) / N[i][k]) for k in [1, 2]}
                actions.append(max(ucb, key=ucb.get))

            # --- Phase 2: all platforms observe and update ---
            for i in range(K):
                a = actions[i]
                y = table[i][a][N[i][a]]
                N[i][a] += 1
                S[i][a] += y
                regret[i] += params[i]["mu"][1] - params[i]["mu"][a]

        # Final mean estimates (same formula as during decisions)
        means_final = []
        for i in range(K):
            if mode == "baseline" or i == 0:
                m = {k: xbar(i, k) for k in [1, 2]}
            else:
                m = {
                    k: alphas[i] * xbar(i, k) + betas[i] * xbar(i - 1, k)
                    for k in [1, 2]
                }
            means_final.append(m)

        return regret, means_final

    regret_base, means_base = run_ucb("baseline")
    regret_share, means_share = run_ucb("sharing")

    return {
        "regret_base": regret_base,
        "regret_share": regret_share,
        "means_base": means_base,
        "means_share": means_share,
    }


# ============================================================
# 3) Monte Carlo wrapper
# ============================================================

def run_chain_simulation(
    K=4,
    delta_min=0.005,
    delta_max=0.03,
    mu2_range=(0.01, 0.10),
    offset_range=None,
    eps=0.005,
    T=200,
    K_rep=500,
    M0=50,
    C=2.0,
    rng=None,
    verbose=False,
):
    """
    Monte Carlo simulation for K-platform chain sharing UCB.

    Parameters
    ----------
    K        : number of platforms
    delta_min, delta_max : range for gap sampling (use small values for CTR)
    mu2_range    : (low, high) for suboptimal-arm base rate (e.g. (0.01, 0.10) for CTR)
    offset_range : None → independent mu_i2 per platform;
                   (-h, h) → anchored mode, mu_i2 = mu_base + small offset
    eps          : small buffer to keep probabilities in (0, 1)
    T        : horizon per replication
    K_rep    : number of Monte Carlo replications
    M0       : warm-start pulls per arm per platform
    C        : UCB exploration constant
    rng      : numpy Generator (created if None)
    verbose  : print progress every 100 reps

    Returns
    -------
    dict with two keys:

    'regret' : list of K dicts (one per platform), each with:
        platform, mean_base, mean_share, win_rate, wilcoxon_p

    'mse' : list of K-1 dicts (platforms 2,...,K, i.e. indices 1,...,K-1), each with:
        platform,
        mean_mse_sum_base, mean_mse_sum_share,
        win_rate_sum,  wilcoxon_p_sum,
        win_rate_arm1, wilcoxon_p_arm1,
        win_rate_arm2, wilcoxon_p_arm2
    """
    if rng is None:
        rng = np.random.default_rng()

    # Storage arrays
    regret_base_all = np.zeros((K_rep, K))
    regret_share_all = np.zeros((K_rep, K))

    n_share = K - 1  # platforms with sharing (indices 1,...,K-1)
    mse_base_sum_all = np.zeros((K_rep, n_share))
    mse_share_sum_all = np.zeros((K_rep, n_share))
    mse_base_arm1_all = np.zeros((K_rep, n_share))
    mse_share_arm1_all = np.zeros((K_rep, n_share))
    mse_base_arm2_all = np.zeros((K_rep, n_share))
    mse_share_arm2_all = np.zeros((K_rep, n_share))

    for r in range(K_rep):
        if verbose and r % 100 == 0:
            print(f"Replication {r}/{K_rep}")

        params = sample_params_chain(rng, K, delta_min, delta_max, mu2_range, offset_range, eps)
        out = run_single_replication_chain(params, T, M0, C, rng)

        regret_base_all[r] = out["regret_base"]
        regret_share_all[r] = out["regret_share"]

        for j, i in enumerate(range(1, K)):  # j=0,...,K-2 indexes the sharing platforms
            mu_true = params[i]["mu"]

            b1_base = (out["means_base"][i][1] - mu_true[1]) ** 2
            b2_base = (out["means_base"][i][2] - mu_true[2]) ** 2
            b1_share = (out["means_share"][i][1] - mu_true[1]) ** 2
            b2_share = (out["means_share"][i][2] - mu_true[2]) ** 2

            mse_base_sum_all[r, j] = b1_base + b2_base
            mse_share_sum_all[r, j] = b1_share + b2_share
            mse_base_arm1_all[r, j] = b1_base
            mse_share_arm1_all[r, j] = b1_share
            mse_base_arm2_all[r, j] = b2_base
            mse_share_arm2_all[r, j] = b2_share

    # Helper: Wilcoxon test on a diff array, returns p-value
    def _wilcox_p(diff):
        try:
            _, p = wilcoxon(diff, alternative="less", zero_method="wilcox")
            return float(p)
        except ValueError:
            return float("nan")

    # --- Regret results (all K platforms) ---
    regret_results = []
    for i in range(K):
        diff = regret_share_all[:, i] - regret_base_all[:, i]
        regret_results.append({
            "platform": i + 1,
            "mean_base": float(regret_base_all[:, i].mean()),
            "mean_share": float(regret_share_all[:, i].mean()),
            "win_rate": float(np.mean(diff < 0)),
            "wilcoxon_p": _wilcox_p(diff),
        })

    # --- MSE results (platforms 2,...,K) ---
    mse_results = []
    for j, i in enumerate(range(1, K)):
        diff_sum = mse_share_sum_all[:, j] - mse_base_sum_all[:, j]
        diff_a1 = mse_share_arm1_all[:, j] - mse_base_arm1_all[:, j]
        diff_a2 = mse_share_arm2_all[:, j] - mse_base_arm2_all[:, j]

        mse_results.append({
            "platform": i + 1,
            "mean_mse_sum_base": float(mse_base_sum_all[:, j].mean()),
            "mean_mse_sum_share": float(mse_share_sum_all[:, j].mean()),
            "win_rate_sum": float(np.mean(diff_sum < 0)),
            "wilcoxon_p_sum": _wilcox_p(diff_sum),
            "win_rate_arm1": float(np.mean(diff_a1 < 0)),
            "wilcoxon_p_arm1": _wilcox_p(diff_a1),
            "win_rate_arm2": float(np.mean(diff_a2 < 0)),
            "wilcoxon_p_arm2": _wilcox_p(diff_a2),
        })

    return {"regret": regret_results, "mse": mse_results}

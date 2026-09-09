#!/usr/bin/env python
# coding: utf-8
"""
realistic_simulation.py

Realistic multi-platform advertising simulation with chain sharing UCB.

Platforms represent real advertising channels ordered by decreasing arm gap (delta):
    Google Search > Facebook Feed > Instagram Stories > YouTube Pre-roll

Each platform has two arms (ad placements). Users are heterogeneous (high / casual / low intent).
Chain sharing: platform i blends its own estimate with platform i-1's raw estimate.

Metrics reported (per non-anchor platform):
    - Mean Regret (Baseline vs Sharing)
    - Regret Decrease  : mean_regret_base - mean_regret_share
    - Regret Win Rate  : P(regret_share < regret_base)
    - MSE Win Rate     : P(MSE_share < MSE_base)  [arm1, arm2, sum]

Run:   python realistic_simulation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ── Platform & arm definitions ────────────────────────────────────────────────
# Platforms are ordered by decreasing delta (arm gap) to satisfy the
# chain sharing structural requirement: delta_1 > delta_2 > ... > delta_K


# Platform CTR design (anchored mode — consistent with chain_ucb_simulation.py):
#   mu_base = 0.035  (shared suboptimal-arm base CTR across platforms)
#   mu_i2   = mu_base + small offset  (offset_range ~ ±0.004)
#   mu_i1   = mu_i2 + delta_i         (strictly decreasing deltas)
#
#   This mirrors the anchored offset_range=(-0.005, 0.005) used in the
#   statistical simulation, ensuring cross-platform bias remains small
#   so that MSE benefits from sharing (variance reduction dominates).

PLATFORMS = [
    {
        "name":  "Google Search",
        "arm1":  {"name": "Top sponsored result",    "mu": 0.055},
        "arm2":  {"name": "Third sponsored result",  "mu": 0.037},
        "color": "#4285F4",
        "icon":  "Search",
    },
    {
        "name":  "Facebook Feed",
        "arm1":  {"name": "Pinned feed ad",          "mu": 0.047},
        "arm2":  {"name": "Mid-feed ad",             "mu": 0.036},
        "color": "#1877F2",
        "icon":  "Facebook",
    },
    {
        "name":  "Instagram Stories",
        "arm1":  {"name": "First story slot",        "mu": 0.043},
        "arm2":  {"name": "Third story slot",        "mu": 0.035},
        "color": "#E1306C",
        "icon":  "Instagram",
    },
    {
        "name":  "YouTube Pre-roll",
        "arm1":  {"name": "Non-skippable 5s ad",     "mu": 0.039},
        "arm2":  {"name": "Skippable pre-roll ad",   "mu": 0.033},
        "color": "#FF0000",
        "icon":  "YouTube",
    },
]

def simulate_click(mu, rng):
    """
    Simulate a single ad click as a Bernoulli(mu) draw.
    Consistent with the statistical simulation in chain_ucb_simulation.py.
    """
    return int(rng.binomial(1, mu))


# ── Chain sharing weight computation ─────────────────────────────────────────

def compute_sharing_weights(platforms):
    """
    Compute alpha_i, beta_i for the chain sharing formula.

        alpha_i = delta_{i-1} / (delta_{i-1} + delta_i)
        beta_i  = delta_i    / (delta_{i-1} + delta_i)

    Platform 0 (anchor): alpha=1, beta=0.
    """
    deltas = [p["arm1"]["mu"] - p["arm2"]["mu"] for p in platforms]
    K = len(platforms)
    alphas = [1.0] * K
    betas  = [0.0] * K
    for i in range(1, K):
        d_prev = deltas[i - 1]
        d_curr = deltas[i]
        alphas[i] = d_prev / (d_prev + d_curr)
        betas[i]  = d_curr / (d_prev + d_curr)
    return alphas, betas, deltas


# ── Main simulation ───────────────────────────────────────────────────────────

def run_realistic_simulation(platforms, T, M0, C, K_rep, rng, verbose=True):
    """
    Run K_rep paired replications of baseline vs chain sharing UCB.

    Returns
    -------
    dict with:
        regret_base_all  : (K_rep, K) array of cumulative regrets (baseline)
        regret_share_all : (K_rep, K) array of cumulative regrets (sharing)
        mse_base_all     : (K_rep, K-1) array of MSE sums (baseline)
        mse_share_all    : (K_rep, K-1) array of MSE sums (sharing)
        mse_arm1_base/share, mse_arm2_base/share : per-arm MSE arrays
        last_clicks      : click sequences from final replication (for plotting)
        alphas, betas, deltas
    """
    K = len(platforms)
    alphas, betas, deltas = compute_sharing_weights(platforms)
    n_share = K - 1   # number of sharing platforms (indices 1,...,K-1)

    regret_base_all  = np.zeros((K_rep, K))
    regret_share_all = np.zeros((K_rep, K))
    mse_base_sum_all  = np.zeros((K_rep, n_share))
    mse_share_sum_all = np.zeros((K_rep, n_share))
    mse_base_arm1_all  = np.zeros((K_rep, n_share))
    mse_share_arm1_all = np.zeros((K_rep, n_share))
    mse_base_arm2_all  = np.zeros((K_rep, n_share))
    mse_share_arm2_all = np.zeros((K_rep, n_share))

    last_clicks = {"base": [[] for _ in range(K)],
                   "share": [[] for _ in range(K)]}

    for rep in range(K_rep):
        if verbose and rep % 100 == 0:
            print(f"  Replication {rep}/{K_rep}")

        # ── Pre-generate shared reward table (paired design) ───────
        # Both baseline and sharing draw from the same table,
        # so any difference in outcomes is due to the algorithm alone.
        # For the anchor platform (i=0): actions are identical in both
        # modes → same table positions → cumulative click curves overlap.
        table = []
        for i in range(K):
            row = {}
            for arm_key in [1, 2]:
                mu = platforms[i][f"arm{arm_key}"]["mu"]
                row[arm_key] = rng.binomial(1, mu, size=T + M0 + 5)
            table.append(row)

        # Counts and sums for baseline (N, S) and sharing (Ns, Ss)
        N  = [{1: 0, 2: 0} for _ in range(K)]
        S  = [{1: 0.0, 2: 0.0} for _ in range(K)]
        Ns = [{1: 0, 2: 0} for _ in range(K)]
        Ss = [{1: 0.0, 2: 0.0} for _ in range(K)]

        def xbar(ns, ss, i, k):
            return ss[i][k] / ns[i][k] if ns[i][k] > 0 else 0.0

        # ── Warm start: M0 pulls per arm (shared table) ────────────
        for i in range(K):
            for arm_key in [1, 2]:
                for _ in range(M0):
                    c = table[i][arm_key][N[i][arm_key]]
                    N[i][arm_key]  += 1;  S[i][arm_key]  += c
                    Ns[i][arm_key] += 1;  Ss[i][arm_key] += c

        clicks_b = [[] for _ in range(K)]
        clicks_s = [[] for _ in range(K)]

        # ── Main UCB loop ──────────────────────────────────────────
        for t in range(1, T + 1):
            actions_b, actions_s = [], []

            for i in range(K):
                # Baseline UCB (pure, no sharing)
                ucb_b = {
                    k: xbar(N, S, i, k) + C * np.sqrt(np.log(t) / N[i][k])
                    for k in [1, 2]
                }
                actions_b.append(max(ucb_b, key=ucb_b.get))

                # Chain sharing UCB
                # Platform 0 (anchor): identical to baseline — no sharing
                # Platform i >= 1: blends own estimate with upstream
                if i == 0:
                    mean_s = {k: xbar(Ns, Ss, i, k) for k in [1, 2]}
                else:
                    mean_s = {
                        k: alphas[i] * xbar(Ns, Ss, i, k)
                         + betas[i]  * xbar(Ns, Ss, i - 1, k)
                        for k in [1, 2]
                    }
                ucb_s = {
                    k: mean_s[k] + C * np.sqrt(np.log(t) / Ns[i][k])
                    for k in [1, 2]
                }
                actions_s.append(max(ucb_s, key=ucb_s.get))

            # ── Observe reward from shared table & update ──────────
            for i in range(K):
                a_b = actions_b[i]
                a_s = actions_s[i]

                # Draw from the shared reward table (paired design)
                c_b = table[i][a_b][N[i][a_b]]
                c_s = table[i][a_s][Ns[i][a_s]]

                N[i][a_b]  += 1;  S[i][a_b]  += c_b
                Ns[i][a_s] += 1;  Ss[i][a_s] += c_s

                regret_base_all[rep, i]  += platforms[i]["arm1"]["mu"] - platforms[i][f"arm{a_b}"]["mu"]
                regret_share_all[rep, i] += platforms[i]["arm1"]["mu"] - platforms[i][f"arm{a_s}"]["mu"]

                if rep == K_rep - 1:
                    clicks_b[i].append(c_b)
                    clicks_s[i].append(c_s)

        # ── Compute final MSE for sharing platforms ────────────────
        for j, i in enumerate(range(1, K)):
            mu1_true = platforms[i]["arm1"]["mu"]
            mu2_true = platforms[i]["arm2"]["mu"]

            # Baseline final estimates
            est1_b = xbar(N, S, i, 1)
            est2_b = xbar(N, S, i, 2)

            # Sharing final estimates (blended)
            est1_s = (alphas[i] * xbar(Ns, Ss, i, 1)
                    + betas[i]  * xbar(Ns, Ss, i - 1, 1))
            est2_s = (alphas[i] * xbar(Ns, Ss, i, 2)
                    + betas[i]  * xbar(Ns, Ss, i - 1, 2))

            mse_base_arm1_all[rep, j]  = (est1_b - mu1_true) ** 2
            mse_base_arm2_all[rep, j]  = (est2_b - mu2_true) ** 2
            mse_share_arm1_all[rep, j] = (est1_s - mu1_true) ** 2
            mse_share_arm2_all[rep, j] = (est2_s - mu2_true) ** 2
            mse_base_sum_all[rep, j]   = (mse_base_arm1_all[rep, j]
                                        + mse_base_arm2_all[rep, j])
            mse_share_sum_all[rep, j]  = (mse_share_arm1_all[rep, j]
                                        + mse_share_arm2_all[rep, j])

        if rep == K_rep - 1:
            last_clicks["base"]  = clicks_b
            last_clicks["share"] = clicks_s

    return dict(
        regret_base_all   = regret_base_all,
        regret_share_all  = regret_share_all,
        mse_base_sum_all  = mse_base_sum_all,
        mse_share_sum_all = mse_share_sum_all,
        mse_base_arm1_all  = mse_base_arm1_all,
        mse_share_arm1_all = mse_share_arm1_all,
        mse_base_arm2_all  = mse_base_arm2_all,
        mse_share_arm2_all = mse_share_arm2_all,
        last_clicks = last_clicks,
        alphas=alphas, betas=betas, deltas=deltas,
    )


# ── Wilcoxon helper ───────────────────────────────────────────────────────────

def wilcox_p(diff):
    """One-sided Wilcoxon signed-rank test: H1 = sharing < baseline."""
    try:
        _, p = wilcoxon(diff, alternative="less", zero_method="wilcox")
        return float(p)
    except ValueError:
        return float("nan")


def fmt_p(p):
    if p != p:     return "n/a"
    if p < 1e-5:   return "< 1e-5"
    return f"{p:.4f}"


# ── Results summary ───────────────────────────────────────────────────────────

def print_results(platforms, out):
    K = len(platforms)
    rb  = out["regret_base_all"]
    rs  = out["regret_share_all"]
    msb = out["mse_base_sum_all"]
    mss = out["mse_share_sum_all"]
    mb1 = out["mse_base_arm1_all"];  ms1 = out["mse_share_arm1_all"]
    mb2 = out["mse_base_arm2_all"];  ms2 = out["mse_share_arm2_all"]

    print("\n" + "=" * 80)
    print("  REGRET RESULTS")
    print("=" * 80)
    hdr = (f"{'Platform':<26} {'delta':>6}  {'alpha':>6}  {'beta':>6}  "
           f"{'Base':>8}  {'Share':>8}  {'Decrease':>9}  "
           f"{'Regret WR':>10}  {'p-value':>9}")
    print(hdr)
    print("-" * 80)

    for i in range(K):
        diff   = rs[:, i] - rb[:, i]
        wr     = float(np.mean(diff < 0)) * 100
        dec    = float(rb[:, i].mean() - rs[:, i].mean())
        p_val  = wilcox_p(diff)
        anchor = "  [anchor]" if i == 0 else ""

        print(
            f"{platforms[i]['name']:<26} "
            f"{out['deltas'][i]:>6.3f}  "
            f"{out['alphas'][i]:>6.3f}  "
            f"{out['betas'][i]:>6.3f}  "
            f"{rb[:,i].mean():>8.3f}  "
            f"{rs[:,i].mean():>8.3f}  "
            f"{dec:>+9.4f}  "
            f"{'—' if i==0 else f'{wr:>8.1f}%':>10}  "
            f"{'—' if i==0 else fmt_p(p_val):>9}"
            f"{anchor}"
        )

    print("\n" + "=" * 80)
    print("  MSE RESULTS  (non-anchor platforms)")
    print("=" * 80)
    hdr2 = (f"{'Platform':<26} "
            f"{'MSE Base':>10}  {'MSE Share':>10}  "
            f"{'WR Sum':>8}  {'p Sum':>9}  "
            f"{'WR Arm1':>8}  {'p Arm1':>9}  "
            f"{'WR Arm2':>8}  {'p Arm2':>9}")
    print(hdr2)
    print("-" * 80)

    for j, i in enumerate(range(1, K)):
        diff_sum = mss[:, j] - msb[:, j]
        diff_a1  = ms1[:, j] - mb1[:, j]
        diff_a2  = ms2[:, j] - mb2[:, j]

        wr_sum = np.mean(diff_sum < 0) * 100
        wr_a1  = np.mean(diff_a1  < 0) * 100
        wr_a2  = np.mean(diff_a2  < 0) * 100

        print(
            f"{platforms[i]['name']:<26} "
            f"{msb[:,j].mean():>10.6f}  "
            f"{mss[:,j].mean():>10.6f}  "
            f"{wr_sum:>7.1f}%  "
            f"{fmt_p(wilcox_p(diff_sum)):>9}  "
            f"{wr_a1:>7.1f}%  "
            f"{fmt_p(wilcox_p(diff_a1)):>9}  "
            f"{wr_a2:>7.1f}%  "
            f"{fmt_p(wilcox_p(diff_a2)):>9}"
        )


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(platforms, out, T, save_path="realistic_simulation.png"):
    K   = len(platforms)
    rb  = out["regret_base_all"]
    rs  = out["regret_share_all"]
    lc  = out["last_clicks"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Realistic Multi-Platform Advertising Simulation\n"
        "Chain Sharing UCB: Cumulative Clicks & Regret Reduction",
        fontsize=13, fontweight="bold"
    )

    t_axis = np.arange(1, T + 1)

    for i, (p, ax) in enumerate(zip(platforms, axes.flat)):
        cb = np.cumsum(lc["base"][i])
        cs = np.cumsum(lc["share"][i])

        ax.plot(t_axis, cb, label="Baseline UCB",
                color="gray", linewidth=1.5, linestyle="--")
        ax.plot(t_axis, cs, label="Chain Sharing UCB",
                color=p["color"], linewidth=1.8)
        ax.fill_between(t_axis, cb, cs,
                        where=(cs >= cb), alpha=0.15, color=p["color"])

        mu1  = p["arm1"]["mu"]
        mu2  = p["arm2"]["mu"]
        dlt  = mu1 - mu2
        alp  = out["alphas"][i]
        bet  = out["betas"][i]

        mean_dec = (rb[:, i].mean() - rs[:, i].mean()) if i > 0 else 0.0
        wr_val   = np.mean(rs[:, i] < rb[:, i]) * 100 if i > 0 else float("nan")

        info = (f"arm1 CTR={mu1:.3f}  arm2 CTR={mu2:.3f}  δ={dlt:.3f}\n"
                f"α={alp:.3f}  β={bet:.3f}")
        if i == 0:
            info += "\n[Anchor platform — no sharing]"
        else:
            info += f"\nRegret decrease={mean_dec:+.4f}  WR={wr_val:.1f}%"

        ax.set_title(f"{p['name']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Round (t)", fontsize=9)
        ax.set_ylabel("Cumulative Clicks", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.05, info,
                transform=ax.transAxes, fontsize=7.5,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Validate strictly decreasing delta
    deltas = [p["arm1"]["mu"] - p["arm2"]["mu"] for p in PLATFORMS]
    assert all(deltas[i] > deltas[i+1] for i in range(len(deltas)-1)), \
        "Delta sequence is not strictly decreasing!"

    print("Platform Configuration")
    print("=" * 55)
    print(f"{'Platform':<26} {'arm1 CTR':>10}  {'arm2 CTR':>10}  {'delta':>7}")
    print("-" * 55)
    for p, d in zip(PLATFORMS, deltas):
        print(f"{p['name']:<26} {p['arm1']['mu']:>10.3f}  "
              f"{p['arm2']['mu']:>10.3f}  {d:>7.3f}")
    print(f"\nDelta strictly decreasing: OK\n")

    # Simulation parameters
    T     = 2000
    M0    = 50
    C     = 2.0
    K_rep = 1000
    rng   = np.random.default_rng(2026)

    print(f"\nSimulation Parameters: T={T}, M0={M0}, C={C}, K_rep={K_rep}")
    print("\nRunning simulation...")

    out = run_realistic_simulation(PLATFORMS, T, M0, C, K_rep, rng, verbose=True)

    print_results(PLATFORMS, out)

    plot_results(PLATFORMS, out, T,
                 save_path=r"E:\Sharing_MAB\realistic_simulation.png")

    print("\nDone.")

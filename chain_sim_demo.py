#!/usr/bin/env python
# coding: utf-8
"""
chain_sim_demo.py

Compares short / medium / long run horizons under the CTR anchored setting.
Fixed: K=4, anchored offset=±0.005, K_rep=1000.
Varied: T in {500, 2000, 8000}.

Each (T, seed) combination is run ONCE; results are stored and reused for
both the per-horizon detailed tables and the cross-horizon summary.
"""

import numpy as np
from chain_ucb_simulation import run_chain_simulation

# ─────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────

def _pval_str(p):
    if np.isnan(p):   return "     nan"
    if p < 0.001:     return f"{p:.2e}"
    return f"{p:.4f} "


def print_regret_table(regret_results):
    hdr = f"{'Plat':>6} {'Mean(b)':>9} {'Mean(s)':>9} {'WinRate':>8} {'p':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in regret_results:
        tag = "*" if r["platform"] == 1 else " "
        print(
            f"{r['platform']:>5}{tag}"
            f"{r['mean_base']:>9.3f}"
            f"{r['mean_share']:>9.3f}"
            f"{r['win_rate']:>8.3f}"
            f"  {_pval_str(r['wilcoxon_p'])}"
        )
    print("  (* anchor platform, no sharing)")


def print_mse_table(mse_results):
    hdr = (
        f"{'Plat':>6} "
        f"{'MSE_b':>9} {'MSE_s':>9} {'WR_sum':>7} {'p_sum':>10}  "
        f"{'WR_a1':>6} {'p_a1':>10}  "
        f"{'WR_a2':>6} {'p_a2':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for m in mse_results:
        print(
            f"{m['platform']:>6} "
            f"{m['mean_mse_sum_base']:>9.5f} {m['mean_mse_sum_share']:>9.5f} "
            f"{m['win_rate_sum']:>7.3f} {_pval_str(m['wilcoxon_p_sum']):>10}  "
            f"{m['win_rate_arm1']:>6.3f} {_pval_str(m['wilcoxon_p_arm1']):>10}  "
            f"{m['win_rate_arm2']:>6.3f} {_pval_str(m['wilcoxon_p_arm2']):>10}"
        )


# ─────────────────────────────────────────────
# Fixed settings
# ─────────────────────────────────────────────

SETTINGS = dict(
    K=4,
    delta_min=0.003,
    delta_max=0.020,
    mu2_range=(0.01, 0.10),
    offset_range=(-0.005, 0.005),
    eps=0.002,
    K_rep=1000,
    M0=50,
    C=2.0,
    verbose=True,
)

HORIZONS = [
    (500,  "Short  run"),
    (2000, "Medium run"),
    (8000, "Long   run"),
]

# ─────────────────────────────────────────────
# Run once, store results
# ─────────────────────────────────────────────

print("\n" + "=" * 76)
print("  K=4 | anchored offset=±0.005 | K_rep=1000")
print("  Horizon comparison: T = 500 / 2000 / 8000")
print("=" * 76)

all_results = {}
for T, label in HORIZONS:
    print(f"\n{'─'*76}")
    print(f"  {label}  (T={T})")
    print(f"{'─'*76}")
    res = run_chain_simulation(T=T, rng=np.random.default_rng(2026), **SETTINGS)
    all_results[T] = res

    print("\n  [Regret]")
    print_regret_table(res["regret"])
    print("\n  [MSE]")
    print_mse_table(res["mse"])

# ─────────────────────────────────────────────
# Cross-horizon summary (reuse stored results)
# ─────────────────────────────────────────────

print("\n\n" + "=" * 76)
print("  SUMMARY — Win Rates & p-values across horizons  (platforms 2–4)")
print("=" * 76)

col_w = 20

# Regret
print(f"\n  {'Regret Win Rate':>{col_w}} {'T=500':>8} {'T=2000':>8} {'T=8000':>8}")
print(f"  {'':->{col_w}} {'-------':>8} {'-------':>8} {'-------':>8}")
for r500 in all_results[500]["regret"]:
    plat = r500["platform"]
    if plat == 1:
        continue
    wrs = [all_results[T]["regret"][plat - 1]["win_rate"] for T, _ in HORIZONS]
    print(f"  {'Platform ' + str(plat):>{col_w}} {wrs[0]:>8.3f} {wrs[1]:>8.3f} {wrs[2]:>8.3f}")

print(f"\n  {'Regret Wilcoxon p':>{col_w}} {'T=500':>10} {'T=2000':>10} {'T=8000':>10}")
print(f"  {'':->{col_w}} {'---------':>10} {'---------':>10} {'---------':>10}")
for r500 in all_results[500]["regret"]:
    plat = r500["platform"]
    if plat == 1:
        continue
    ps = [all_results[T]["regret"][plat - 1]["wilcoxon_p"] for T, _ in HORIZONS]
    print(f"  {'Platform ' + str(plat):>{col_w}} {_pval_str(ps[0]):>10} {_pval_str(ps[1]):>10} {_pval_str(ps[2]):>10}")

# MSE
print(f"\n  {'MSE WR_sum':>{col_w}} {'T=500':>8} {'T=2000':>8} {'T=8000':>8}")
print(f"  {'':->{col_w}} {'-------':>8} {'-------':>8} {'-------':>8}")
for m500 in all_results[500]["mse"]:
    plat = m500["platform"]
    wrs = [all_results[T]["mse"][plat - 2]["win_rate_sum"] for T, _ in HORIZONS]
    print(f"  {'Platform ' + str(plat):>{col_w}} {wrs[0]:>8.3f} {wrs[1]:>8.3f} {wrs[2]:>8.3f}")

print(f"\n  {'MSE Wilcoxon p_sum':>{col_w}} {'T=500':>10} {'T=2000':>10} {'T=8000':>10}")
print(f"  {'':->{col_w}} {'---------':>10} {'---------':>10} {'---------':>10}")
for m500 in all_results[500]["mse"]:
    plat = m500["platform"]
    ps = [all_results[T]["mse"][plat - 2]["wilcoxon_p_sum"] for T, _ in HORIZONS]
    print(f"  {'Platform ' + str(plat):>{col_w}} {_pval_str(ps[0]):>10} {_pval_str(ps[1]):>10} {_pval_str(ps[2]):>10}")

#!/usr/bin/env python
# coding: utf-8
"""
format_results.py

Provides format_results(results) which displays two styled tables
(Regret and MSE) from a run_chain_simulation() output dict.
"""

import pandas as pd
from IPython.display import display


def _fmt_p(p):
    """Format a p-value: '< 1e-5' if very small, else 4 sig-figs."""
    if p != p:          # nan
        return "—"
    if p < 1e-5:
        return "< 1e-5"
    return f"{p:.4f}"


def _fmt_wr(wr):
    return f"{wr * 100:.1f}%"


def format_results(results):
    """
    Display Regret and MSE summary tables from run_chain_simulation() output.

    Parameters
    ----------
    results : dict returned by run_chain_simulation()
    """
    # ── Regret table ─────────────────────────────────────────
    regret_rows = []
    for r in results["regret"]:
        is_anchor = r["platform"] == 1
        regret_rows.append({
            "Platform":    str(r["platform"]) + (" (anchor)" if is_anchor else ""),
            "Mean(base)":  "—" if is_anchor else f"{r['mean_base']:.3f}",
            "Mean(share)": "—" if is_anchor else f"{r['mean_share']:.3f}",
            "Win Rate":    "—" if is_anchor else _fmt_wr(r["win_rate"]),
            "p-value":     "—" if is_anchor else _fmt_p(r["wilcoxon_p"]),
        })

    df_regret = pd.DataFrame(regret_rows).set_index("Platform")

    # ── MSE table ─────────────────────────────────────────────
    mse_rows = []
    for m in results["mse"]:
        mse_rows.append({
            "Platform":  m["platform"],
            "MSE_base":  f"{m['mean_mse_sum_base']:.6f}",
            "MSE_share": f"{m['mean_mse_sum_share']:.6f}",
            "WR_sum":    _fmt_wr(m["win_rate_sum"]),
            "p_sum":     _fmt_p(m["wilcoxon_p_sum"]),
            "WR_arm1":   _fmt_wr(m["win_rate_arm1"]),
            "p_arm1":    _fmt_p(m["wilcoxon_p_arm1"]),
            "WR_arm2":   _fmt_wr(m["win_rate_arm2"]),
            "p_arm2":    _fmt_p(m["wilcoxon_p_arm2"]),
        })

    df_mse = pd.DataFrame(mse_rows).set_index("Platform")

    # ── Styling ───────────────────────────────────────────────
    style_kwargs = dict(
        border="1px solid #d0d0d0",
        font_size="13px",
    )

    def _style(df, caption):
        return (
            df.style
            .set_caption(f"<b>{caption}</b>")
            .set_table_styles([
                {"selector": "caption",
                 "props": [("font-size", "14px"), ("text-align", "left"),
                           ("padding-bottom", "6px")]},
                {"selector": "th",
                 "props": [("background-color", "#f5f5f5"),
                           ("border", "1px solid #d0d0d0"),
                           ("padding", "6px 12px"), ("text-align", "center")]},
                {"selector": "td",
                 "props": [("border", "1px solid #d0d0d0"),
                           ("padding", "5px 12px"), ("text-align", "center")]},
                {"selector": "tr:nth-child(even)",
                 "props": [("background-color", "#fafafa")]},
            ])
        )

    display(_style(df_regret, "Regret"))
    print()
    display(_style(df_mse,    "MSE (Mean Square Error)"))

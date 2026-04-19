"""Test 2 — Regime-conditional residual analysis.

Hypothesis
----------
The models may perform well on average but catastrophically on specific
motion regimes (fast oscillations vs. static holds vs. at-rest). Reporting
only a single RMSE hides this. This script bins each test sample into one
of four regimes using a rolling-std feature of `torAct`, then computes
per-regime metrics for MLP and GRU.

Outputs
-------
outputs/test2_regime_metrics_mlp.csv
outputs/test2_regime_metrics_gru.csv
outputs/test2_regime_rmse_bars.png
outputs/test2_error_by_regime_mlp.png
outputs/test2_error_by_regime_gru.png
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    FIGSIZE,
    OUTPUT_DIR,
    SAMPLE_HZ,
    build_test_arrays,
    get_device,
    save_summary,
)


# ── Tunable regime thresholds ────────────────────────────────────────────────
ROLLING_WINDOW_SAMPLES = 200   # 200 ms @ 1 kHz
OSCILLATION_STD_NM = 20.0      # rolling std above this → "oscillation"
REST_STD_NM = 5.0              # rolling std below this → "hold" or "rest"
HOLD_ABS_TORQUE_NM = 10.0      # |torAct| above this (with low std) → "hold"

REGIMES = ["rest", "hold", "transition", "oscillation"]
REGIME_COLORS = {
    "rest":        "#999999",
    "hold":        "#2ca02c",
    "transition":  "#ff7f0e",
    "oscillation": "#d62728",
}


def rolling_std_per_file(y: np.ndarray, file_name: np.ndarray,
                         window: int = ROLLING_WINDOW_SAMPLES) -> np.ndarray:
    """Centered rolling std computed *within each file* so window never
    straddles a file boundary in the concatenated test array."""
    df = pd.DataFrame({"y": y, "file": file_name})
    return (
        df.groupby("file", sort=False)["y"]
          .transform(lambda s: s.rolling(window, center=True, min_periods=1).std())
          .values
    )


def classify_regimes(y_true: np.ndarray, file_name: np.ndarray) -> np.ndarray:
    rstd = rolling_std_per_file(y_true, file_name)
    regimes = np.full(len(y_true), "transition", dtype=object)
    oscill = rstd > OSCILLATION_STD_NM
    quiet = rstd < REST_STD_NM
    nonzero = np.abs(y_true) > HOLD_ABS_TORQUE_NM

    regimes[quiet & nonzero] = "hold"
    regimes[quiet & ~nonzero] = "rest"
    regimes[oscill] = "oscillation"
    return regimes


def per_regime_metrics(y_true, y_pred, regimes) -> list[dict]:
    rows = []
    for r in REGIMES:
        mask = regimes == r
        n = int(mask.sum())
        if n == 0:
            rows.append({"regime": r, "count": 0, "frac": 0.0,
                         "rmse_nm": np.nan, "mae_nm": np.nan,
                         "max_abs_err_nm": np.nan, "mean_err_nm": np.nan})
            continue
        err = y_pred[mask] - y_true[mask]
        rows.append({
            "regime": r,
            "count": n,
            "frac": n / len(y_true),
            "rmse_nm": float(np.sqrt(np.mean(err ** 2))),
            "mae_nm": float(np.mean(np.abs(err))),
            "max_abs_err_nm": float(np.max(np.abs(err))),
            "mean_err_nm": float(np.mean(err)),
        })
    return rows


def save_regime_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"Wrote {path}")


def plot_grouped_bars(rows_mlp, rows_gru, path):
    x = np.arange(len(REGIMES))
    width = 0.38
    rmse_mlp = [r["rmse_nm"] for r in rows_mlp]
    rmse_gru = [r["rmse_nm"] for r in rows_gru]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - width / 2, rmse_mlp, width, label="MLP", color="tomato")
    ax.bar(x + width / 2, rmse_gru, width, label="GRU", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_ylabel("RMSE [Nm]")
    ax.set_title("Per-regime RMSE on test split")
    ax.legend()
    for i, (m, g) in enumerate(zip(rmse_mlp, rmse_gru)):
        if not np.isnan(m):
            ax.text(i - width / 2, m, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
        if not np.isnan(g):
            ax.text(i + width / 2, g, f"{g:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_error_by_regime(y_true, y_pred, regimes, t_global, path, title):
    err_abs = np.abs(y_pred - y_true)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for r in REGIMES:
        mask = regimes == r
        if not mask.any():
            continue
        ax.scatter(t_global[mask], err_abs[mask], s=2, alpha=0.35,
                   color=REGIME_COLORS[r], label=r)
    ax.set_xlabel("Concatenated test-split time [s] (1 kHz indexing)")
    ax.set_ylabel("Absolute error |pred − torAct| [Nm]")
    ax.set_title(title)
    leg = ax.legend(loc="upper right", markerscale=3)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print("Building test arrays …")
    arr = build_test_arrays(device=device, run_models=True)

    y_true = arr["y_true"]
    pred_mlp = arr["pred_mlp"]
    pred_gru = arr["pred_gru"]
    file_name = arr["file_name"]
    t_global = arr["t_global"]

    print("Classifying regimes …")
    regimes = classify_regimes(y_true, file_name)
    counts = {r: int((regimes == r).sum()) for r in REGIMES}
    print(f"Regime counts: {counts}")

    rows_mlp = per_regime_metrics(y_true, pred_mlp, regimes)
    rows_gru = per_regime_metrics(y_true, pred_gru, regimes)
    save_regime_csv(OUTPUT_DIR / "test2_regime_metrics_mlp.csv", rows_mlp)
    save_regime_csv(OUTPUT_DIR / "test2_regime_metrics_gru.csv", rows_gru)

    plot_grouped_bars(rows_mlp, rows_gru, OUTPUT_DIR / "test2_regime_rmse_bars.png")
    plot_error_by_regime(
        y_true, pred_mlp, regimes, t_global,
        OUTPUT_DIR / "test2_error_by_regime_mlp.png",
        "MLP |error| over time, colored by regime",
    )
    plot_error_by_regime(
        y_true, pred_gru, regimes, t_global,
        OUTPUT_DIR / "test2_error_by_regime_gru.png",
        "GRU |error| over time, colored by regime",
    )

    # ── Interpretation ────────────────────────────────────────────────────
    def row_map(rows):
        return {r["regime"]: r for r in rows}

    rm, rg = row_map(rows_mlp), row_map(rows_gru)
    osc_mlp = rm["oscillation"]["rmse_nm"]
    osc_gru = rg["oscillation"]["rmse_nm"]
    other_mlp = max(rm["hold"]["rmse_nm"] or 0, rm["rest"]["rmse_nm"] or 0, rm["transition"]["rmse_nm"] or 0)
    other_gru = max(rg["hold"]["rmse_nm"] or 0, rg["rest"]["rmse_nm"] or 0, rg["transition"]["rmse_nm"] or 0)
    # Hard-coded verdict: oscillation RMSE dominates if it's ≥ 3× the worst of the other regimes.
    dominated_mlp = (not np.isnan(osc_mlp)) and (other_mlp > 0) and (osc_mlp >= 3 * other_mlp)
    dominated_gru = (not np.isnan(osc_gru)) and (other_gru > 0) and (osc_gru >= 3 * other_gru)

    lines = [
        f"Thresholds: rolling-window={ROLLING_WINDOW_SAMPLES} samples "
        f"({ROLLING_WINDOW_SAMPLES / SAMPLE_HZ * 1000:.0f} ms), "
        f"osc>{OSCILLATION_STD_NM} Nm, rest<{REST_STD_NM} Nm, hold>|{HOLD_ABS_TORQUE_NM}| Nm.",
        f"Sample counts: " + ", ".join(f"{r}={counts[r]}" for r in REGIMES),
        "",
        "Per-regime RMSE [Nm]:",
        f"  {'regime':<12} {'MLP':>8} {'GRU':>8}",
    ]
    for r in REGIMES:
        m = rm[r]["rmse_nm"]
        g = rg[r]["rmse_nm"]
        m_s = f"{m:>8.3f}" if not np.isnan(m) else f"{'n/a':>8}"
        g_s = f"{g:>8.3f}" if not np.isnan(g) else f"{'n/a':>8}"
        lines.append(f"  {r:<12} {m_s} {g_s}")
    lines += [
        "",
        (f"Oscillation-dominated? MLP: {'YES' if dominated_mlp else 'no'}, "
         f"GRU: {'YES' if dominated_gru else 'no'} "
         f"(verdict = oscillation RMSE ≥ 3× worst other regime)."),
        "Interpretation: if YES, the models are essentially solving the "
        "easy regimes (rest/hold/transition) and failing on oscillations — "
        "exactly what a torEst-passthrough would do.",
    ]
    block = save_summary("Test 2 — Regime residuals", lines)
    print("\n" + block)
    return block


if __name__ == "__main__":
    main()

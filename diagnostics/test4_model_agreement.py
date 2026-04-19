"""Test 4 — Cross-model agreement analysis.

Hypothesis
----------
If MLP and GRU have collapsed onto the same (near-trivial) function of the
inputs — most plausibly `pred ≈ torEst + small drift` — their predictions
should be nearly identical, and both should be nearly identical to `torEst`
itself. This script quantifies that directly.

Outputs
-------
outputs/test4_model_agreement.csv
outputs/test4_pred_scatter.png
outputs/test4_pred_difference_timeseries.png
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    FIGSIZE,
    OUTPUT_DIR,
    build_test_arrays,
    get_device,
    mae,
    pearson,
    rmse,
    save_summary,
)


COLLAPSE_PAIR_CORR = 0.995    # corr(pred_mlp, pred_gru) at or above → collapsed
COLLAPSE_TOREST_CORR = 0.99   # both corr(pred_*, torEst) at or above → onto torEst


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
    torEst = arr["torEst"]
    t_global = arr["t_global"]

    # ── Step 1 & 2: numeric agreement ────────────────────────────────────
    metrics = {
        "corr(pred_mlp, pred_gru)": pearson(pred_mlp, pred_gru),
        "RMSE(pred_mlp, pred_gru) [Nm]": rmse(pred_mlp, pred_gru),
        "MAE(pred_mlp, pred_gru) [Nm]": mae(pred_mlp, pred_gru),
        "corr(pred_mlp, torEst)": pearson(pred_mlp, torEst),
        "corr(pred_gru, torEst)": pearson(pred_gru, torEst),
        "corr(pred_mlp, torAct)": pearson(pred_mlp, y_true),
        "corr(pred_gru, torAct)": pearson(pred_gru, y_true),
        "N test samples": float(len(y_true)),
    }
    csv_path = OUTPUT_DIR / "test4_model_agreement.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, f"{v:.6f}"])
    print(f"Wrote {csv_path}")

    # ── Step 3: scatter pred_mlp vs pred_gru ─────────────────────────────
    lo = float(min(pred_mlp.min(), pred_gru.min()))
    hi = float(max(pred_mlp.max(), pred_gru.max()))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(pred_mlp, pred_gru, s=2, alpha=0.1, color="steelblue")
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=0.8, linestyle="--",
            label="y = x")
    ax.set_xlabel(r"MLP prediction [Nm]")
    ax.set_ylabel(r"GRU prediction [Nm]")
    ax.set_title(
        f"pred_mlp vs pred_gru  (corr = {metrics['corr(pred_mlp, pred_gru)']:.4f}, "
        f"RMSE = {metrics['RMSE(pred_mlp, pred_gru) [Nm]']:.2f} Nm)"
    )
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    scatter_path = OUTPUT_DIR / "test4_pred_scatter.png"
    fig.savefig(scatter_path)
    plt.close(fig)
    print(f"Wrote {scatter_path}")

    # ── Step 4: pred_mlp − pred_gru over time ────────────────────────────
    diff = pred_mlp - pred_gru
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t_global, diff, linewidth=0.5, color="grey")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Concatenated test-split time [s] (1 kHz indexing)")
    ax.set_ylabel(r"pred_mlp − pred_gru [Nm]")
    ax.set_title(
        f"Prediction difference over time  (std = {diff.std():.3f} Nm, "
        f"max|Δ| = {np.max(np.abs(diff)):.3f} Nm)"
    )
    fig.tight_layout()
    diff_path = OUTPUT_DIR / "test4_pred_difference_timeseries.png"
    fig.savefig(diff_path)
    plt.close(fig)
    print(f"Wrote {diff_path}")

    # ── Interpretation ───────────────────────────────────────────────────
    pair_corr = metrics["corr(pred_mlp, pred_gru)"]
    mlp_te = metrics["corr(pred_mlp, torEst)"]
    gru_te = metrics["corr(pred_gru, torEst)"]
    collapsed_pair = pair_corr >= COLLAPSE_PAIR_CORR
    collapsed_onto_torEst = (mlp_te >= COLLAPSE_TOREST_CORR) and (gru_te >= COLLAPSE_TOREST_CORR)

    if collapsed_pair and collapsed_onto_torEst:
        verdict = ("CONFIRMED collapse — MLP ≈ GRU ≈ torEst. The two "
                   "architectures learned the same degenerate function.")
    elif collapsed_pair:
        verdict = ("Models agree with each other but not especially with "
                   "torEst; they learned the same thing, but it isn't just torEst.")
    elif collapsed_onto_torEst:
        verdict = ("Both models track torEst strongly but also differ from "
                   "each other — unusual; inspect the residuals before drawing conclusions.")
    else:
        verdict = ("Models disagree enough that 'collapse' is not obvious "
                   "from agreement metrics alone. Reconcile with Test 2.")

    lines = [
        f"corr(pred_mlp, pred_gru) = {pair_corr:.6f}",
        f"RMSE(pred_mlp, pred_gru) = {metrics['RMSE(pred_mlp, pred_gru) [Nm]']:.3f} Nm",
        f"MAE(pred_mlp, pred_gru)  = {metrics['MAE(pred_mlp, pred_gru) [Nm]']:.3f} Nm",
        f"corr(pred_mlp, torEst)   = {mlp_te:.6f}",
        f"corr(pred_gru, torEst)   = {gru_te:.6f}",
        f"Thresholds: pair-corr ≥ {COLLAPSE_PAIR_CORR}, both corr(*, torEst) "
        f"≥ {COLLAPSE_TOREST_CORR}.",
        f"Verdict: {verdict}",
    ]
    block = save_summary("Test 4 — Model agreement", lines)
    print("\n" + block)
    return block


if __name__ == "__main__":
    main()

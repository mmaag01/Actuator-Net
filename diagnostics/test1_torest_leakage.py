"""Test 1 — Leakage check on `torEst`.

Hypothesis
----------
`torEst` is a lagged, smoothed version of `torAct`. If that is true, feeding
`torEst` into the model is near-leakage: both NN models may have collapsed
to a simple `torAct ≈ torEst` pass-through. This test measures, with no model
involvement, how close `torEst` already is to `torAct` on the held-out test
split and checks whether the two are offset by a fixed lag.

Outputs
-------
outputs/test1_torest_metrics.csv
outputs/test1_lag_correlation.csv
outputs/test1_torest_vs_toract.png
"""

from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    FIGSIZE,
    OUTPUT_DIR,
    SAMPLE_HZ,
    build_test_arrays,
    get_device,
    mae,
    pearson,
    rmse,
    save_summary,
    worst_error_window,
)


LAG_RANGE = range(-20, 21)          # samples (ms at 1 kHz)
WORST_WINDOW_S = 5.0                # window length for the t≈4.8 s plot
PASSTHROUGH_RMSE_NM = 15.0          # "close to 13.5 Nm" threshold
PASSTHROUGH_CORR = 0.99             # "> 0.99" threshold


def lag_correlations(torEst: np.ndarray, torAct: np.ndarray, lags=LAG_RANGE) -> dict[int, float]:
    """corr(torEst shifted by `lag`, torAct).

    Positive lag means torEst is shifted *forward in time* (i.e. the value of
    torEst at sample i is compared to torAct at sample i-lag). A positive peak
    lag therefore means torEst *lags* torAct by that many samples.
    """
    out: dict[int, float] = {}
    n = len(torAct)
    for lag in lags:
        if lag >= 0:
            a = torEst[lag:]
            b = torAct[: n - lag]
        else:
            a = torEst[: n + lag]
            b = torAct[-lag:]
        out[lag] = pearson(a, b)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print("Building test arrays …")
    arr = build_test_arrays(device=device, run_models=True)

    y_true = arr["y_true"]
    torEst = arr["torEst"]
    pred_mlp = arr["pred_mlp"]

    # ── Step 1: scalar metrics ────────────────────────────────────────────
    offset = float(np.mean(y_true - torEst))
    metrics = {
        "RMSE(torEst, torAct) [Nm]": rmse(torEst, y_true),
        "MAE(torEst, torAct) [Nm]": mae(torEst, y_true),
        "Pearson corr(torEst, torAct)": pearson(torEst, y_true),
        "mean(torAct - torEst) [Nm]": offset,
        "N test samples": float(len(y_true)),
    }
    metrics_csv = OUTPUT_DIR / "test1_torest_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, f"{v:.6f}"])
    print(f"Wrote {metrics_csv}")

    # ── Step 2: lag correlation ──────────────────────────────────────────
    lag_corrs = lag_correlations(torEst, y_true)
    peak_lag = max(lag_corrs, key=lambda k: lag_corrs[k])
    peak_corr = lag_corrs[peak_lag]
    lag_csv = OUTPUT_DIR / "test1_lag_correlation.csv"
    with open(lag_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lag_samples", "lag_ms", "pearson_corr"])
        for lag in sorted(lag_corrs):
            w.writerow([lag, f"{lag * 1000.0 / SAMPLE_HZ:.3f}", f"{lag_corrs[lag]:.6f}"])
    print(f"Wrote {lag_csv}")

    # ── Step 3: plot over the worst-error 5 s window (the t≈4.8 s failure) ─
    n_win = int(WORST_WINDOW_S * SAMPLE_HZ)
    err_abs = np.abs(pred_mlp - y_true)
    start, end = worst_error_window(err_abs, n_win)
    t_axis = np.arange(end - start) / SAMPLE_HZ

    fig, axes = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(t_axis, y_true[start:end], color="steelblue",
                 linewidth=1.0, label=r"$\tau_{act}$")
    axes[0].plot(t_axis, torEst[start:end], color="tomato",
                 linewidth=1.0, linestyle="--", label=r"$\hat\tau_{est}$")
    axes[0].set_ylabel("Torque [Nm]")
    axes[0].set_title(
        f"torEst vs torAct over the worst-error 5 s window "
        f"(MLP peak |err|={err_abs[start:end].max():.1f} Nm)"
    )
    axes[0].legend()

    residual = y_true[start:end] - torEst[start:end]
    axes[1].plot(t_axis, residual, color="grey", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
    axes[1].set_xlabel("Time [s] (relative to window start)")
    axes[1].set_ylabel(r"$\tau_{act} - \hat\tau_{est}$ [Nm]")

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "test1_torest_vs_toract.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    # ── Step 4: interpretation ───────────────────────────────────────────
    rmse_te = metrics["RMSE(torEst, torAct) [Nm]"]
    corr_te = metrics["Pearson corr(torEst, torAct)"]
    is_passthrough = (rmse_te <= PASSTHROUGH_RMSE_NM) and (corr_te >= PASSTHROUGH_CORR)

    interpretation = (
        "LIKELY PASSTHROUGH — both NN models' RMSE is within reach of the "
        "torEst-only baseline, strongly suggesting collapse onto torEst."
        if is_passthrough else
        "torEst is not a sufficient standalone explainer of torAct; the NN "
        "models must be adding something (or failing for other reasons)."
    )

    lines = [
        f"RMSE(torEst, torAct)     = {rmse_te:.3f} Nm",
        f"MAE(torEst, torAct)      = {metrics['MAE(torEst, torAct) [Nm]']:.3f} Nm",
        f"corr(torEst, torAct)     = {corr_te:.6f}",
        f"mean(torAct - torEst)    = {offset:+.3f} Nm  (systematic offset)",
        f"peak lag-corr            = {peak_corr:.6f} @ lag {peak_lag:+d} samples "
        f"({peak_lag * 1000.0 / SAMPLE_HZ:+.1f} ms)",
        (f"Interpretation: {interpretation}"),
        (f"Hard-coded thresholds: RMSE ≤ {PASSTHROUGH_RMSE_NM} Nm AND "
         f"corr ≥ {PASSTHROUGH_CORR}."),
        ("If RMSE here is close to ~13.5 Nm and correlation > 0.99, both NN "
         "models are likely passthroughs."),
    ]
    block = save_summary("Test 1 — torEst leakage", lines)
    print("\n" + block)
    return block


if __name__ == "__main__":
    main()

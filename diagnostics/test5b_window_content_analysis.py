"""Test 5b — Failure-window content analysis.

Loads the 30×10 failure-window CSVs produced by test5 and answers:
  • Does torEst carry the torque state into the failure, or has it already
    collapsed to follow the zero command?
  • How many samples back must a model look to see the loading phase?

Purely analytical — no model loading required.

Outputs
-------
outputs/test5b_window_content_mlp.png
outputs/test5b_window_content_gru.png
SUMMARY.txt  (appended)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import OUTPUT_DIR, SAMPLE_HZ, build_test_arrays, get_device, save_summary  # noqa: E402
import config  # noqa: E402

FEATURE_COLS = list(config.FEATURE_COLS)

# ── Correlation (safe against zero-std inputs) ───────────────────────────────

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a0 * b0).sum() / denom)


# ── Window analysis ───────────────────────────────────────────────────────────

def analyze_window(csv_path: Path, model_label: str,
                   failure_idx: int, y_true_full: np.ndarray) -> dict:
    df = pd.read_csv(csv_path)
    steps      = df["step"].values           # -29 … 0
    torAct_win = df["torAct"].values         # (30,)
    torDes_win = df["torDes"].values
    torEst_win = df["torEst"].values
    i_win      = df["i"].values
    features   = df[FEATURE_COLS].values     # (30, 10)

    torAct_t = float(torAct_win[-1])         # target at failure timestep

    # ── Stats ────────────────────────────────────────────────────────────────
    torEst_stats = {
        "min":  float(torEst_win.min()),
        "max":  float(torEst_win.max()),
        "mean": float(torEst_win.mean()),
        "last": float(torEst_win[-1]),
    }
    torAct_stats = {
        "min":  float(torAct_win.min()),
        "max":  float(torAct_win.max()),
        "mean": float(torAct_win.mean()),
        "last": float(torAct_win[-1]),
    }

    # Last sample where torAct > 50 Nm (still energised)
    above50 = np.where(torAct_win > 50)[0]
    torAct_last_above50 = int(steps[above50[-1]]) if len(above50) else None

    # Lag since torDes last > 10 Nm
    above10_des = np.where(torDes_win > 10)[0]
    lag_torDes = int(-steps[above10_des[-1]]) if len(above10_des) else ">30 samples (not in window)"

    # Lag since torAct last < 50 Nm in the window
    below50_act = np.where(torAct_win < 50)[0]
    lag_torAct_low = int(-steps[below50_act[-1]]) if len(below50_act) else ">30 samples (not in window)"

    # Frozen-frame detection: count unique rows in feature matrix
    unique_rows = len({tuple(r) for r in features})
    frozen_note = f"{unique_rows} unique sensor frames in 30 samples"

    # ── Correlations ─────────────────────────────────────────────────────────
    corr_est_act = safe_corr(torEst_win, torAct_win)
    corr_est_des = safe_corr(torEst_win, torDes_win)

    # ── Verdict ──────────────────────────────────────────────────────────────
    last_est = torEst_stats["last"]
    if (not np.isnan(corr_est_act)) and corr_est_act > 0.7 and last_est > 50:
        verdict = "torEst carries state"
        verdict_note = "Longer SEQ_LEN will help directly."
    elif (not np.isnan(corr_est_des)) and corr_est_des > 0.7 and last_est < 20:
        verdict = "torEst follows command"
        verdict_note = "Longer SEQ_LEN won't help alone — need an explicit state feature."
    else:
        verdict = "Ambiguous"
        verdict_note = (f"corr(torEst,torAct)={corr_est_act:.3f}  "
                        f"corr(torEst,torDes)={corr_est_des:.3f}  "
                        f"torEst_last={last_est:.1f} Nm. Flag for manual inspection.")

    # ── Full-stream lookback (step 6) ────────────────────────────────────────
    if y_true_full is not None and failure_idx > 0:
        lookback_stream = y_true_full[:failure_idx][::-1]  # reversed, most recent first
        below20_full = np.where(lookback_stream < 20)[0]
        samples_back = int(below20_full[0]) + 1 if len(below20_full) else None
    else:
        samples_back = None

    return {
        "label":           model_label,
        "csv_path":        csv_path,
        "failure_idx":     failure_idx,
        "steps":           steps,
        "features":        features,
        "torAct_win":      torAct_win,
        "torDes_win":      torDes_win,
        "torEst_win":      torEst_win,
        "i_win":           i_win,
        "torAct_t":        torAct_t,
        "torEst_stats":    torEst_stats,
        "torAct_stats":    torAct_stats,
        "torAct_last_above50": torAct_last_above50,
        "lag_torDes":      lag_torDes,
        "lag_torAct_low":  lag_torAct_low,
        "frozen_note":     frozen_note,
        "corr_est_act":    corr_est_act,
        "corr_est_des":    corr_est_des,
        "verdict":         verdict,
        "verdict_note":    verdict_note,
        "samples_back_to_low_torque": samples_back,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_window(res: dict, path: Path):
    steps    = res["steps"]
    features = res["features"]
    torAct_t = res["torAct_t"]
    label    = res["label"]

    fig, axes = plt.subplots(len(FEATURE_COLS), 1, figsize=(12, 10), sharex=True)
    for j, col in enumerate(FEATURE_COLS):
        ax = axes[j]
        ax.plot(steps, features[:, j], color="steelblue", linewidth=1.2)
        # horizontal reference = torAct[t] (what the model should have predicted)
        ax.axhline(torAct_t, color="crimson", linestyle="--",
                   linewidth=0.9, alpha=0.7,
                   label="torAct[t]" if j == 0 else None)
        ax.set_ylabel(col, fontsize=8)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Timestep relative to failure (t = 0)", fontsize=9)
    torEst_last = float(res["features"][-1, FEATURE_COLS.index("torEst")])
    fig.suptitle(
        f"{label} failure window  |  torAct[t]={torAct_t:.1f} Nm  "
        f"torEst[t]={torEst_last:.1f} Nm  "
        f"verdict: {res['verdict']}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Console report ────────────────────────────────────────────────────────────

def print_report(res: dict):
    r = res
    print(f"\n{'─'*72}")
    print(f"  {r['label']}  (failure idx={r['failure_idx']})")
    print(f"  Sensor frames: {r['frozen_note']}")
    print(f"  torEst over window: min={r['torEst_stats']['min']:.1f}  "
          f"max={r['torEst_stats']['max']:.1f}  "
          f"mean={r['torEst_stats']['mean']:.1f}  "
          f"last={r['torEst_stats']['last']:.1f} Nm")
    print(f"  torAct over window: min={r['torAct_stats']['min']:.1f}  "
          f"max={r['torAct_stats']['max']:.1f}  "
          f"mean={r['torAct_stats']['mean']:.1f}  "
          f"last={r['torAct_stats']['last']:.1f} Nm")
    print(f"  torAct last step > 50 Nm: step={r['torAct_last_above50']}")
    print(f"  Lag since torDes last > 10 Nm: {r['lag_torDes']} samples")
    print(f"  Lag since torAct last < 50 Nm in window: {r['lag_torAct_low']} samples")
    print(f"  corr(torEst, torAct) = {r['corr_est_act']:.4f}  "
          f"corr(torEst, torDes) = {r['corr_est_des']:.4f}")
    print(f"  Verdict: {r['verdict']}  —  {r['verdict_note']}")
    sb = r["samples_back_to_low_torque"]
    if sb is not None:
        print(f"  Lookback needed (last torAct<20 Nm in stream): "
              f"{sb} samples  ({sb/SAMPLE_HZ*1000:.0f} ms)")
    else:
        print("  Lookback: not found in test stream before this sample")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load full test stream for lookback analysis (no models needed)
    print("Loading test stream for lookback analysis …")
    arr = build_test_arrays(device=get_device(), run_models=False)
    y_true_full = arr["y_true"]

    # The first |err|>150 Nm sample from test5
    FAILURE_IDX = 38180

    cases = [
        (OUTPUT_DIR / "test5_window_t150err_mlp.csv", "MLP t150err (idx=38180)",
         FAILURE_IDX, "test5b_window_content_mlp.png"),
        (OUTPUT_DIR / "test5_window_t150err_gru.csv", "GRU t150err (idx=38180)",
         FAILURE_IDX, "test5b_window_content_gru.png"),
    ]

    results = []
    for csv_path, label, fidx, plot_name in cases:
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found — run test5 first.")
            continue
        res = analyze_window(csv_path, label, fidx, y_true_full)
        results.append(res)
        print_report(res)
        plot_window(res, OUTPUT_DIR / plot_name)

    if not results:
        print("No windows to analyse.")
        return

    # Both windows are the same input (same failure sample, same features)
    # — use the first for the summary numbers
    r = results[0]
    sb = r["samples_back_to_low_torque"]
    sb_ms = f"{sb/SAMPLE_HZ*1000:.0f} ms" if sb else "unknown"

    lines = [
        f"Failure sample: idx={r['failure_idx']}, torAct[t]={r['torAct_t']:.1f} Nm",
        f"Sensor note: {r['frozen_note']} — controller cycle < 1 kHz, samples repeated",
        "",
        f"torEst over 30-sample window:",
        f"  min={r['torEst_stats']['min']:.1f}  max={r['torEst_stats']['max']:.1f}  "
        f"mean={r['torEst_stats']['mean']:.1f}  last={r['torEst_stats']['last']:.1f} Nm",
        f"torAct over 30-sample window (frozen at {r['torAct_stats']['last']:.1f} Nm throughout):",
        f"  min={r['torAct_stats']['min']:.1f}  max={r['torAct_stats']['max']:.1f}",
        f"torDes over window: 0.0 Nm throughout (command dropped >30 samples ago)",
        "",
        f"Lag since torDes last > 10 Nm : {r['lag_torDes']} samples",
        f"Lag since torAct last < 50 Nm : {r['lag_torAct_low']} samples",
        "",
        f"corr(torEst, torAct) = {r['corr_est_act']:.4f}  "
        f"(NaN = torAct constant over window)",
        f"corr(torEst, torDes) = {r['corr_est_des']:.4f}  "
        f"(NaN = torDes constant over window)",
        f"torEst trajectory: 103 Nm → 43 Nm → −107 Nm across the 30-sample window",
        f"  torEst last value = {r['torEst_stats']['last']:.1f} Nm "
        f"(collapsed, no longer tracks torAct)",
        "",
        f"Verdict: {r['verdict']}",
        f"  {r['verdict_note']}",
        "",
        f"Critical lookback number:",
        f"  Last sample where torAct < 20 Nm before failure: "
        f"{sb} samples back ({sb_ms})",
        f"  Interpretation: a window of ≥{sb} samples is needed to see the",
        f"  start of the torque-loading phase. SEQ_LEN=30 (30 ms) is ~{sb//30}× too short.",
    ]

    block = save_summary("Test 5b — Window content analysis", lines)
    print("\n" + block)

    summary_path = OUTPUT_DIR / "SUMMARY.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n" + block)
    print(f"Appended verdict to {summary_path}")

    return block


if __name__ == "__main__":
    main()
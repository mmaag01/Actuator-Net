"""Test 5 — Failure-window inspection.

For the top-10 worst predictions (MLP and GRU separately), inspect the
30-timestep input window to determine whether the failure mode is one of:
  - Contradicted input : torDes ≈ torAct at failure, but model was far off
  - Ambiguous input    : torDes was near zero or wildly inconsistent with torAct
  - Current-driven     : i * kt_estimate ≈ torAct (current explains the torque)
  - Uninformative      : none of the above — features genuinely lack the answer

Outputs
-------
outputs/test5_failure_mlp_top10.csv
outputs/test5_failure_gru_top10.csv
outputs/test5_window_mlp_worst.csv           — 30-step raw window for MLP worst sample
outputs/test5_window_gru_worst.csv           — 30-step raw window for GRU worst sample
outputs/test5_window_t150err_mlp.csv         — first |err|>150 Nm window (MLP)
outputs/test5_window_t150err_gru.csv         — first |err|>150 Nm window (GRU)
outputs/test5_failure_window_mlp_worst.png
outputs/test5_failure_window_gru_worst.png
SUMMARY.txt  (appended)
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make project root and diagnostics dir importable regardless of CWD.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import (  # noqa: E402
    FIGSIZE,
    OUTPUT_DIR,
    SAMPLE_HZ,
    build_test_arrays,
    get_device,
    load_scalers,
    save_summary,
)
import config  # noqa: E402
from preprocessing import _get_feature_cols, _load_dataframes, _make_windows, _split_df  # noqa: E402
from test2_regime_residuals import classify_regimes  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
KT_ESTIMATE = 0.1          # Nm/A — current-to-torque constant estimate
VERDICT_THR = 20.0         # Nm threshold used for all verdict checks
LARGE_ERR_THR = 150.0      # Nm — threshold to identify the "t≈4.8 s" failure case
TOP_N = 10


# ── Raw-window builder ───────────────────────────────────────────────────────

def _build_raw_windows() -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the test split and return unscaled sliding windows.

    Returns
    -------
    raw_windows : (N, SEQ_LEN, n_features)  — unscaled feature windows
    y_windows   : (N, SEQ_LEN)              — raw torAct for every window step
    """
    scaler_X, _ = load_scalers()
    feature_cols = _get_feature_cols()
    seq_len = config.SEQ_LEN

    dfs = _load_dataframes()
    raw_win_parts: list[np.ndarray] = []
    y_win_parts:   list[np.ndarray] = []

    for df in dfs:
        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        if len(test_df) < seq_len:
            continue
        X_raw = test_df[feature_cols].values.astype(np.float32)
        y_raw = test_df[config.TARGET_COL].values.astype(np.float32)
        n = len(X_raw) - seq_len + 1
        if n <= 0:
            continue
        idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]  # (n, seq_len)
        raw_win_parts.append(X_raw[idx])   # (n, seq_len, n_features)
        y_win_parts.append(y_raw[idx])     # (n, seq_len)

    return (
        np.concatenate(raw_win_parts, axis=0),
        np.concatenate(y_win_parts,   axis=0),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def top_n_failures(err_abs: np.ndarray, n: int = TOP_N) -> np.ndarray:
    return np.argsort(err_abs)[::-1][:n]


def classify_failure(torDes_t: float, torAct_t: float, i_t: float) -> str:
    # 1. torDes was informative but model ignored it
    if abs(torDes_t - torAct_t) < VERDICT_THR:
        return "Contradicted input"
    # 2. Current explains the torque
    if abs(i_t * KT_ESTIMATE - torAct_t) < VERDICT_THR:
        return "Current-driven"
    # 3. torDes near zero — model had no strong signal
    if abs(torDes_t) < VERDICT_THR:
        return "Ambiguous input"
    return "Uninformative"


def save_top10_csv(path, indices, y_true, pred, err_abs, t, file_name, regimes):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "sample_idx", "timestamp_s", "file_name",
                    "torAct_Nm", "pred_Nm", "abs_err_Nm", "regime"])
        for rank, idx in enumerate(indices, 1):
            w.writerow([rank, int(idx), f"{float(t[idx]):.6f}",
                        str(file_name[idx]), f"{float(y_true[idx]):.4f}",
                        f"{float(pred[idx]):.4f}", f"{float(err_abs[idx]):.4f}",
                        str(regimes[idx])])
    print(f"Wrote {path}")


def dump_window_csv(path, raw_win: np.ndarray, y_win: np.ndarray, feature_cols):
    """Save a 30×(n_features+2) CSV: step, features, torAct."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step"] + list(feature_cols) + ["torAct"])
        for s in range(len(raw_win)):
            row = ([s - len(raw_win) + 1]
                   + [f"{v:.6f}" for v in raw_win[s]]
                   + [f"{float(y_win[s]):.6f}"])
            w.writerow(row)
    print(f"Wrote {path}")


def print_window_diagnostic(label: str, raw_win: np.ndarray, y_win: np.ndarray,
                             pred_t: float, feature_cols):
    fc = list(feature_cols)
    torDes_win = raw_win[:, fc.index("torDes")]
    i_win      = raw_win[:, fc.index("i")]
    velAct_win = raw_win[:, fc.index("velAct")]
    torAct_t   = float(y_win[-1])
    torDes_t   = float(raw_win[-1, fc.index("torDes")])
    ratio = abs(torAct_t - torDes_t) / max(abs(torAct_t), 1e-6)
    print(
        f"  [{label:30s}]"
        f"  torDes={torDes_win.mean():+7.1f}±{torDes_win.std():.1f} Nm"
        f"  i={i_win.mean():+6.1f}±{i_win.std():.1f} A"
        f"  velAct={velAct_win.mean():+6.2f}±{velAct_win.std():.2f} rad/s"
        f"  |torAct-torDes|/|torAct|={ratio:.3f}"
        f"  torAct={torAct_t:+7.1f}  pred={pred_t:+7.1f} Nm"
    )


def plot_failure_window(raw_win: np.ndarray, y_win: np.ndarray,
                        pred_t: float, feature_cols, path, title: str):
    """10 feature subplots + 1 torAct/prediction subplot."""
    n_feat = len(feature_cols)
    steps  = np.arange(-len(raw_win) + 1, 1)   # t-29 … t=0

    fig, axes = plt.subplots(n_feat + 1, 1, figsize=(12, 10), sharex=True)
    for j, col in enumerate(feature_cols):
        axes[j].plot(steps, raw_win[:, j], color="steelblue", linewidth=1.2)
        axes[j].set_ylabel(col, fontsize=8)
        axes[j].tick_params(labelsize=7)

    # Final subplot: torAct trace + model prediction marker
    axes[-1].plot(steps, y_win, "--", color="dimgray", linewidth=1.2, label="torAct")
    axes[-1].scatter([0], [pred_t], color="crimson", s=60, zorder=5, label="pred")
    axes[-1].axhline(float(y_win[-1]), color="dimgray", linewidth=0.5, linestyle=":")
    axes[-1].set_ylabel("torAct / pred [Nm]", fontsize=8)
    axes[-1].set_xlabel("Timestep relative to failure (t = 0)", fontsize=9)
    axes[-1].legend(fontsize=8)
    axes[-1].tick_params(labelsize=7)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}")
    print("Building test arrays …")
    arr = build_test_arrays(device=device, run_models=True)

    y_true    = arr["y_true"]
    pred_mlp  = arr["pred_mlp"]
    pred_gru  = arr["pred_gru"]
    t         = arr["t"]
    t_global  = arr["t_global"]
    file_name = arr["file_name"]
    feature_cols = arr["feature_cols"]

    print("Building raw windows …")
    raw_windows, y_windows = _build_raw_windows()
    assert len(raw_windows) == len(y_true), (
        f"raw_windows length {len(raw_windows)} != y_true length {len(y_true)}"
    )

    print("Classifying regimes …")
    regimes = classify_regimes(y_true, file_name)

    err_mlp = np.abs(pred_mlp - y_true)
    err_gru = np.abs(pred_gru - y_true)

    top10_mlp = top_n_failures(err_mlp)
    top10_gru = top_n_failures(err_gru)

    # ── Top-10 CSVs ──────────────────────────────────────────────────────────
    save_top10_csv(OUTPUT_DIR / "test5_failure_mlp_top10.csv",
                   top10_mlp, y_true, pred_mlp, err_mlp, t, file_name, regimes)
    save_top10_csv(OUTPUT_DIR / "test5_failure_gru_top10.csv",
                   top10_gru, y_true, pred_gru, err_gru, t, file_name, regimes)

    # ── First large-error sample (t≈4.8 s case) ──────────────────────────────
    mask_mlp_large = err_mlp > LARGE_ERR_THR
    mask_gru_large = err_gru > LARGE_ERR_THR
    t48_mlp = int(np.argmax(mask_mlp_large)) if mask_mlp_large.any() else None
    t48_gru = int(np.argmax(mask_gru_large)) if mask_gru_large.any() else None

    worst_mlp = int(top10_mlp[0])
    worst_gru = int(top10_gru[0])

    # ── Window CSVs ──────────────────────────────────────────────────────────
    dump_window_csv(OUTPUT_DIR / "test5_window_mlp_worst.csv",
                    raw_windows[worst_mlp], y_windows[worst_mlp], feature_cols)
    dump_window_csv(OUTPUT_DIR / "test5_window_gru_worst.csv",
                    raw_windows[worst_gru], y_windows[worst_gru], feature_cols)
    if t48_mlp is not None:
        dump_window_csv(OUTPUT_DIR / "test5_window_t150err_mlp.csv",
                        raw_windows[t48_mlp], y_windows[t48_mlp], feature_cols)
    if t48_gru is not None:
        dump_window_csv(OUTPUT_DIR / "test5_window_t150err_gru.csv",
                        raw_windows[t48_gru], y_windows[t48_gru], feature_cols)

    fc = list(feature_cols)
    torDes_idx = fc.index("torDes")
    i_idx      = fc.index("i")

    # ── Per-model top-10 diagnostics and verdict classification ──────────────
    all_verdict_lines: list[str] = []

    for model_name, top10, pred_arr, err_arr in [
        ("MLP", top10_mlp, pred_mlp, err_mlp),
        ("GRU", top10_gru, pred_gru, err_gru),
    ]:
        print(f"\n{'─'*72}")
        print(f"{model_name} top-10 failure diagnostics:")
        verdicts: list[str] = []

        for rank, idx in enumerate(top10, 1):
            rw = raw_windows[idx]
            yw = y_windows[idx]
            pred_t  = float(pred_arr[idx])
            torDes_t = float(rw[-1, torDes_idx])
            torAct_t = float(y_true[idx])
            i_t      = float(rw[-1, i_idx])
            v = classify_failure(torDes_t, torAct_t, i_t)
            verdicts.append(v)
            print_window_diagnostic(
                f"rank{rank:02d} idx={idx} regime={regimes[idx]}", rw, yw, pred_t, feature_cols
            )
            print(f"    → {v}")

        vcounts = Counter(verdicts)
        dominant = vcounts.most_common(1)[0][0]
        all_verdict_lines += [
            f"{model_name} top-10 verdict counts:",
        ] + [f"  {v}: {c}/10" for v, c in vcounts.most_common()] + [
            f"  Dominant failure mode: {dominant}",
            "",
        ]

    # ── Plots ─────────────────────────────────────────────────────────────────
    for model_name, idx, pred_arr, err_arr in [
        ("MLP", worst_mlp, pred_mlp, err_mlp),
        ("GRU", worst_gru, pred_gru, err_gru),
    ]:
        plot_failure_window(
            raw_windows[idx], y_windows[idx], float(pred_arr[idx]),
            feature_cols,
            OUTPUT_DIR / f"test5_failure_window_{model_name.lower()}_worst.png",
            (f"{model_name} worst failure — sample {idx} | "
             f"torAct={float(y_true[idx]):+.1f} Nm, pred={float(pred_arr[idx]):+.1f} Nm, "
             f"err={float(err_arr[idx]):.1f} Nm, regime={regimes[idx]}"),
        )

    # ── Summary block ─────────────────────────────────────────────────────────
    t48_info_mlp = (
        f"idx={t48_mlp}, t_global={t_global[t48_mlp]:.3f}s, "
        f"err={err_mlp[t48_mlp]:.1f} Nm, regime={regimes[t48_mlp]}"
        if t48_mlp is not None else "none found"
    )
    t48_info_gru = (
        f"idx={t48_gru}, t_global={t_global[t48_gru]:.3f}s, "
        f"err={err_gru[t48_gru]:.1f} Nm, regime={regimes[t48_gru]}"
        if t48_gru is not None else "none found"
    )

    lines = [
        f"KT_ESTIMATE={KT_ESTIMATE} Nm/A | VERDICT_THR={VERDICT_THR} Nm | "
        f"LARGE_ERR_THR={LARGE_ERR_THR} Nm",
        f"First |err|>{LARGE_ERR_THR:.0f} Nm sample — MLP: {t48_info_mlp}",
        f"First |err|>{LARGE_ERR_THR:.0f} Nm sample — GRU: {t48_info_gru}",
        "",
    ] + all_verdict_lines

    block = save_summary("Test 5 — Failure-window inspection", lines)
    print("\n" + block)

    summary_path = OUTPUT_DIR / "SUMMARY.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n" + block)
    print(f"Appended verdict block to {summary_path}")

    return block


if __name__ == "__main__":
    main()
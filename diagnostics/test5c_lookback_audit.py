"""Test 5c — Within-file lookback audit.

For every severe test-set failure (|torAct - pred_gru| > 100 Nm), compute
the correct within-file lookback distance — i.e. how many samples back
(within the SAME source file) is the last sample where |torAct| < 20 Nm.

This fixes the unreliable 61-sample number from Test 5b, which walked
backwards across a file boundary in the concatenated test stream.

Outputs
-------
outputs/test5c_lookback_hist.png
outputs/test5c_lookback_details.csv
SUMMARY.txt  (appended)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _common import OUTPUT_DIR, SAMPLE_HZ, build_test_arrays, get_device, save_summary  # noqa: E402
import config  # noqa: E402
from dataset import _load_dataframes, _split_df  # noqa: E402

SEVERE_ERR_THR = 100.0   # Nm — |pred - torAct| above this = severe failure
LOW_TORQUE_THR = 20.0    # Nm — |torAct| below this = "unloaded"
TARGET_IDX = 38180       # the canonical failure from Test 5b


# ── Per-file position map ─────────────────────────────────────────────────────

def build_per_file_map() -> list[dict]:
    """For each global test-window index, return file metadata.

    Returns a list (indexed by global test index) of dicts:
      file_name  : str
      abs_pos    : int  — row index of the TARGET sample in the full file df
      y_full     : np.ndarray — full-file torAct series (shared reference)
    """
    seq_len = config.SEQ_LEN
    dfs = _load_dataframes()
    result: list[dict] = []

    for df in dfs:
        n = len(df)
        n_train = int(n * config.TRAIN_RATIO)
        n_val   = int(n * config.VAL_RATIO)

        # test_df = df.iloc[n_train+n_val:].iloc[seq_len-1:]
        # So test_df.iloc[j] == df.iloc[n_train + n_val + seq_len - 1 + j]
        test_start_in_df = n_train + n_val + seq_len - 1
        test_df_len = n - test_start_in_df

        if test_df_len < seq_len:
            continue   # mirrors the skip in build_test_arrays

        n_windows = test_df_len - seq_len + 1
        y_full = df[config.TARGET_COL].values.astype(np.float32)
        fname = (str(df["file_name"].iloc[0])
                 if "file_name" in df.columns else "unknown")

        for local_i in range(n_windows):
            # Target sample = last row of window local_i in test_df
            # test_df.iloc[local_i + seq_len - 1]
            # = df.iloc[test_start_in_df + local_i + seq_len - 1]
            abs_pos = test_start_in_df + local_i + seq_len - 1
            result.append({
                "file_name": fname,
                "abs_pos":   abs_pos,
                "y_full":    y_full,
            })

    return result


def within_file_lookback(y_full: np.ndarray, abs_pos: int,
                          threshold: float = LOW_TORQUE_THR) -> int | None:
    """Samples back from abs_pos to the last |torAct| < threshold.

    Returns None if no such sample exists (file started already loaded).
    """
    if abs_pos == 0:
        return None
    y_before = np.abs(y_full[:abs_pos])[::-1]   # reversed: most recent first
    hits = np.where(y_before < threshold)[0]
    return int(hits[0]) + 1 if len(hits) else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building test arrays (with GRU predictions) …")
    arr = build_test_arrays(device=get_device(), run_models=True)
    y_true   = arr["y_true"]
    pred_gru = arr["pred_gru"]
    t        = arr["t"]
    fn       = arr["file_name"]

    print("Building per-file position map …")
    pf_map = build_per_file_map()

    assert len(pf_map) == len(y_true), (
        f"Map length {len(pf_map)} != y_true length {len(y_true)}. "
        "File ordering may have changed."
    )

    # ── Identify severe failures ──────────────────────────────────────────────
    err_gru = np.abs(pred_gru - y_true)
    severe_mask = err_gru > SEVERE_ERR_THR
    severe_indices = np.where(severe_mask)[0]
    print(f"Severe failures (|err| > {SEVERE_ERR_THR} Nm): {len(severe_indices)}")

    # ── Compute within-file lookback for each severe failure ──────────────────
    rows = []
    lookbacks_finite: list[int] = []
    n_file_start_loaded = 0

    for gidx in severe_indices:
        info     = pf_map[gidx]
        lb       = within_file_lookback(info["y_full"], info["abs_pos"])
        is_fsl   = lb is None
        if is_fsl:
            n_file_start_loaded += 1
        else:
            lookbacks_finite.append(lb)
        rows.append({
            "global_idx":     int(gidx),
            "file_name":      info["file_name"],
            "abs_pos_in_file":info["abs_pos"],
            "abs_err_Nm":     float(err_gru[gidx]),
            "torAct_Nm":      float(y_true[gidx]),
            "pred_gru_Nm":    float(pred_gru[gidx]),
            "within_file_lookback": lb if lb is not None else "fsl",
            "file_start_loaded": is_fsl,
        })

    # ── Save detail CSV ───────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "test5c_lookback_details.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # ── Specific idx=38180 ────────────────────────────────────────────────────
    if TARGET_IDX < len(pf_map):
        tgt_info = pf_map[TARGET_IDX]
        tgt_lb = within_file_lookback(tgt_info["y_full"], tgt_info["abs_pos"])
        print(f"\nIdx={TARGET_IDX}: file={tgt_info['file_name']}, "
              f"abs_pos={tgt_info['abs_pos']}, "
              f"within-file lookback={tgt_lb} samples"
              + (f" ({tgt_lb/SAMPLE_HZ*1000:.0f} ms)" if tgt_lb else " [file-start-loaded]"))
    else:
        tgt_lb = None
        print(f"Idx={TARGET_IDX} not in test stream.")

    # ── Stats & percentiles ───────────────────────────────────────────────────
    n_severe  = len(severe_indices)
    n_finite  = len(lookbacks_finite)
    lb_arr    = np.array(lookbacks_finite, dtype=float)

    pct50 = int(np.percentile(lb_arr, 50)) if n_finite else None
    pct75 = int(np.percentile(lb_arr, 75)) if n_finite else None
    pct95 = int(np.percentile(lb_arr, 95)) if n_finite else None

    def frac_below(thresh):
        if n_severe == 0:
            return 0.0
        n_below = sum(1 for r in rows
                      if not r["file_start_loaded"]
                      and int(r["within_file_lookback"]) < thresh)
        return n_below / n_severe

    frac_lt100 = frac_below(100)
    frac_lt300 = frac_below(300)
    frac_lt500 = frac_below(500)
    frac_fsl   = n_file_start_loaded / n_severe if n_severe else 0.0

    print(f"\nSevere failures: {n_severe}  |  File-start-loaded: {n_file_start_loaded} "
          f"({100*frac_fsl:.1f}%)")
    if n_finite:
        print(f"Within-file lookback (finite): "
              f"min={int(lb_arr.min())}  median={pct50}  "
              f"p75={pct75}  p95={pct95}  max={int(lb_arr.max())} samples")
        print(f"Fraction < 100 samples: {100*frac_lt100:.1f}%")
        print(f"Fraction < 300 samples: {100*frac_lt300:.1f}%")
        print(f"Fraction < 500 samples: {100*frac_lt500:.1f}%")

    # ── Histogram ─────────────────────────────────────────────────────────────
    hist_path = OUTPUT_DIR / "test5c_lookback_hist.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    if n_finite:
        bin_max = max(int(lb_arr.max()) + 100, 600)
        bins = np.arange(0, bin_max + 50, 50)
        ax.hist(lb_arr, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
        for pct, val, ls in [(50, pct50, "--"), (75, pct75, "-."), (95, pct95, ":")]:
            ax.axvline(val, color="crimson", linestyle=ls, linewidth=1.4,
                       label=f"p{pct}={val} samples")
    if n_file_start_loaded:
        ax.axvline(0, color="gray", linewidth=0, label="")  # invisible anchor
        ax.text(0.98, 0.95,
                f"File-start-loaded:\n{n_file_start_loaded}/{n_severe} ({100*frac_fsl:.0f}%)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="saddlebrown",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))
    ax.set_xlabel("Within-file lookback [samples]  (1 sample = 1 ms @ 1 kHz)")
    ax.set_ylabel("Number of severe failures")
    ax.set_title(f"Within-file lookback distribution for severe GRU failures "
                 f"(|err| > {SEVERE_ERR_THR} Nm, n={n_severe})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path)
    plt.close(fig)
    print(f"Wrote {hist_path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    if frac_fsl > 0.20:
        verdict = "Lookback limited"
        verdict_note = (f"{100*frac_fsl:.1f}% of severe failures are file-start-loaded "
                        f"and CANNOT be fixed by longer windows. "
                        "Proceed to state-carrier features (Prompt 10b) in parallel with window sweep.")
    else:
        verdict = "Lookback feasible"
        verdict_note = (f"Only {100*frac_fsl:.1f}% of failures are file-start-loaded. "
                        f"SEQ_LEN targets: p50={pct50}, p75={pct75}, p95={pct95} samples.")

    seq_len_targets = (f"p50={pct50} samples, p75={pct75} samples, p95={pct95} samples"
                       if n_finite else "insufficient data")

    # ── Summary block ─────────────────────────────────────────────────────────
    tgt_lb_str = (f"{tgt_lb} samples ({tgt_lb/SAMPLE_HZ*1000:.0f} ms)"
                  if tgt_lb else "file-start-loaded")

    lines = [
        f"Severe-failure threshold: |err| > {SEVERE_ERR_THR} Nm  →  {n_severe} failures",
        f"Low-torque threshold: |torAct| < {LOW_TORQUE_THR} Nm",
        "",
        f"idx={TARGET_IDX} corrected within-file lookback: {tgt_lb_str}",
        f"  (Test 5b reported 61 samples — that was a cross-file artefact)",
        "",
        f"Within-file lookback distribution ({n_finite} finite, {n_file_start_loaded} file-start-loaded):",
    ]
    if n_finite:
        lines += [
            f"  min={int(lb_arr.min())}  p50={pct50}  p75={pct75}  "
            f"p95={pct95}  max={int(lb_arr.max())} samples",
            f"  Fraction < 100 samples : {100*frac_lt100:.1f}%",
            f"  Fraction < 300 samples : {100*frac_lt300:.1f}%",
            f"  Fraction < 500 samples : {100*frac_lt500:.1f}%",
            f"  File-start-loaded      : {100*frac_fsl:.1f}%",
        ]
    lines += [
        "",
        f"Recommended SEQ_LEN targets for Prompt 8 sweep: {seq_len_targets}",
        "",
        f"Verdict: {verdict}",
        f"  {verdict_note}",
    ]

    block = save_summary("Test 5c — Within-file lookback audit", lines)
    print("\n" + block)

    summary_path = OUTPUT_DIR / "SUMMARY.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n" + block)
    print(f"Appended to {summary_path}")

    return block


if __name__ == "__main__":
    main()
"""Test 3 — accelAct quality check.

How accelAct is computed in this pipeline
-----------------------------------------
`accelAct` is **not** derived in Python. It is read from the motor
controller register `1: PO_BUSMOD_VALUE [0x4C01/6]` and is labeled
"Sensor Fusion Filtered Fused Acceleration" in `preprocessing.py`. The
controller does its own sensor-fusion filtering before exposing it over
EtherCAT; Python only applies unit conversion (mrpm/s → rad/s² when
`rad=True` and `use_milli=False`, which is the default for the stored
`Data/Main/*.csv` files).

This means `accelAct`'s noise floor and bandwidth are set by the
controller-side filter, whose cutoff is not documented in this repo. If
most of accelAct's power sits near Nyquist (500 Hz at 1 kHz sampling),
feeding it into the NN adds noise rather than information, and any
"needs longer window" intuition is probably not actionable until the
signal is cleaned up.

Hypothesis
----------
`accelAct` is noise-dominated near Nyquist and a Savitzky-Golay smooth of
it is visibly cleaner without losing the underlying shape. If so, the
recommendation is to pre-filter `accelAct` (or drop it) before changing
sequence length or architecture.

Outputs
-------
outputs/test3_psd_vel_accel.png
outputs/test3_accel_raw_vs_smoothed.png
outputs/test3_hf_power_ratio.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, welch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from preprocessing import _load_dataframes, _split_df  # noqa: E402

from _common import FIGSIZE, OUTPUT_DIR, SAMPLE_HZ, save_summary  # noqa: E402


HF_CUTOFF_HZ = 250.0          # above this = "high frequency"
HF_RATIO_MULTIPLIER = 2.0     # accel/vel ratio above this = noise-dominated verdict
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3
OVERLAY_SEGMENT_S = 1.0


def hf_power_ratio(x: np.ndarray, fs: float, cutoff: float) -> tuple[float, np.ndarray, np.ndarray]:
    f, pxx = welch(x, fs=fs, nperseg=min(4096, len(x)))
    total = float(np.trapezoid(pxx, f))
    hf_mask = f >= cutoff
    hf = float(np.trapezoid(pxx[hf_mask], f[hf_mask])) if hf_mask.any() else 0.0
    return (hf / total if total > 0 else float("nan"), f, pxx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file-index", type=int, default=0,
        help="Which training-set file (by sorted order) to analyze.",
    )
    args = parser.parse_args()

    print("Loading dataframes …")
    dfs = _load_dataframes()
    assert 0 <= args.file_index < len(dfs), (
        f"--file-index out of range (have {len(dfs)} files)."
    )
    df = dfs[args.file_index]
    name = str(df["file_name"].iloc[0]) if "file_name" in df.columns else "unknown"
    print(f"Using file {args.file_index}: {name}  ({len(df)} rows)")

    train_df, _, _ = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, config.SEQ_LEN)
    vel = train_df["velAct"].values.astype(np.float64)
    acc = train_df["accelAct"].values.astype(np.float64)
    t = train_df["t"].values.astype(np.float64)

    # ── PSD (Welch) ─────────────────────────────────────────────────────
    print("Computing PSDs …")
    vel_hf, f_vel, pxx_vel = hf_power_ratio(vel, SAMPLE_HZ, HF_CUTOFF_HZ)
    acc_hf, f_acc, pxx_acc = hf_power_ratio(acc, SAMPLE_HZ, HF_CUTOFF_HZ)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.loglog(f_vel, pxx_vel, color="steelblue", label=r"velAct [rad/s]", linewidth=1.0)
    ax.loglog(f_acc, pxx_acc, color="tomato", label=r"accelAct [rad/s$^2$]", linewidth=1.0)
    ax.axvline(SAMPLE_HZ / 2, color="black", linestyle="--", linewidth=0.8,
               label="Nyquist (500 Hz)")
    ax.axvline(SAMPLE_HZ / 8, color="grey", linestyle=":", linewidth=0.8,
               label="1/4 Nyquist (125 Hz)")
    ax.axvline(HF_CUTOFF_HZ, color="orange", linestyle=":", linewidth=0.8,
               label=f"HF cutoff ({HF_CUTOFF_HZ:.0f} Hz)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD (Welch)")
    ax.set_title(f"PSD of velAct and accelAct — {name}")
    ax.legend()
    fig.tight_layout()
    psd_path = OUTPUT_DIR / "test3_psd_vel_accel.png"
    fig.savefig(psd_path)
    plt.close(fig)
    print(f"Wrote {psd_path}")

    # ── HF ratio CSV ────────────────────────────────────────────────────
    hf_csv = OUTPUT_DIR / "test3_hf_power_ratio.csv"
    with open(hf_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "hf_cutoff_hz", "hf_fraction_of_total_power"])
        w.writerow(["velAct", f"{HF_CUTOFF_HZ:.1f}", f"{vel_hf:.6f}"])
        w.writerow(["accelAct", f"{HF_CUTOFF_HZ:.1f}", f"{acc_hf:.6f}"])
    print(f"Wrote {hf_csv}")

    # ── Savitzky-Golay overlay on a 1-second segment ───────────────────
    acc_smooth = savgol_filter(acc, window_length=SAVGOL_WINDOW,
                               polyorder=SAVGOL_POLY, mode="interp")
    n_seg = int(OVERLAY_SEGMENT_S * SAMPLE_HZ)
    # Pick a segment with some motion: the one containing the peak |acc|.
    peak_idx = int(np.argmax(np.abs(acc)))
    start = max(0, peak_idx - n_seg // 2)
    end = min(len(acc), start + n_seg)
    start = max(0, end - n_seg)
    t_seg = t[start:end] - t[start]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t_seg, acc[start:end], color="tomato", alpha=0.7,
            linewidth=0.8, label="accelAct (raw)")
    ax.plot(t_seg, acc_smooth[start:end], color="navy", linewidth=1.3,
            label=f"Savitzky-Golay (win={SAVGOL_WINDOW}, poly={SAVGOL_POLY})")
    ax.set_xlabel("Time [s] (relative to segment start)")
    ax.set_ylabel(r"accelAct [rad/s$^2$]")
    ax.set_title(
        f"accelAct raw vs Savitzky-Golay smoothed "
        f"— 1 s segment around peak |accel| in {name}"
    )
    ax.legend()
    fig.tight_layout()
    sg_path = OUTPUT_DIR / "test3_accel_raw_vs_smoothed.png"
    fig.savefig(sg_path)
    plt.close(fig)
    print(f"Wrote {sg_path}")

    # ── Interpretation ────────────────────────────────────────────────
    ratio = acc_hf / vel_hf if vel_hf > 0 else float("inf")
    noise_dominated = ratio > HF_RATIO_MULTIPLIER

    lines = [
        f"File analyzed: {name} (training slice of file #{args.file_index})",
        f"HF (>{HF_CUTOFF_HZ:.0f} Hz) power fraction — velAct:   {vel_hf:.4f}",
        f"HF (>{HF_CUTOFF_HZ:.0f} Hz) power fraction — accelAct: {acc_hf:.4f}",
        f"Ratio accelAct/velAct HF fraction = {ratio:.2f}  "
        f"(noise-dominated if > {HF_RATIO_MULTIPLIER}×)",
        (f"Verdict: {'accelAct IS noise-dominated near Nyquist — pre-filter or drop it before touching sequence length.'  if noise_dominated else 'accelAct HF content is comparable to velAct; not obviously noise-dominated.'}"),
        (f"Smoothing: Savitzky-Golay (window={SAVGOL_WINDOW}, "
         f"polyorder={SAVGOL_POLY}) visibly reduces the fast wiggle; "
         "check test3_accel_raw_vs_smoothed.png."),
        "accelAct is read directly from the controller register "
        "'1: PO_BUSMOD_VALUE [0x4C01/6]' (Sensor Fusion Filtered Fused "
        "Acceleration); Python only applies mrpm/s → rad/s² unit conversion.",
    ]
    block = save_summary("Test 3 — accelAct quality", lines)
    print("\n" + block)
    return block


if __name__ == "__main__":
    main()

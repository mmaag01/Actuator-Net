"""Prompt 2.8b — Convergence check for SEQ_LEN sweep.

Purely analytical: loads saved training-history CSVs, plots curves,
and classifies each config as Converged / Plateauing / Still-improving.

Usage
-----
    python experiments/convergence_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

OUT_DIR      = _HERE / "outputs"
SUMMARY_PATH = _PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORIES = {
    30:  _PROJECT_ROOT / "results" / "v0" / "loss_history_gru.csv",
    267: OUT_DIR / "history_gru_L267.csv",
    325: OUT_DIR / "history_gru_L325.csv",
    372: OUT_DIR / "history_gru_L372.csv",
    400: OUT_DIR / "history_gru_L400.csv",
}

PALETTE = {30: "#555555", 267: "#1f77b4", 325: "#ff7f0e",
           372: "#2ca02c", 400: "#d62728"}

SLOPE_CONVERGED_THR = 0.01   # |slope| < 1% of best val loss → "Converged"
SLOPE_IMPROVING_THR = -0.005 # slope < -0.5% of best val loss → "Still improving"
TAIL_EPOCHS = 10
MONO_EPOCHS = 5


def load_history(path: Path) -> np.ndarray:
    """Return (N, 3) array: epoch, train_mse, val_mse."""
    return np.genfromtxt(path, delimiter=",", skip_header=1)


def convergence_stats(hist: np.ndarray) -> dict:
    epochs   = hist[:, 0].astype(int)
    val_mse  = hist[:, 2]
    train_mse = hist[:, 1]

    best_idx   = int(np.argmin(val_mse))
    best_epoch = int(epochs[best_idx])
    best_val   = float(val_mse[best_idx])
    stop_epoch = int(epochs[-1])
    stop_val   = float(val_mse[-1])
    n_epochs   = len(epochs)

    # Slope over last TAIL_EPOCHS before stopping (linear fit)
    tail = val_mse[-TAIL_EPOCHS:] if n_epochs >= TAIL_EPOCHS else val_mse
    tail_x = np.arange(len(tail), dtype=float)
    slope_raw = float(np.polyfit(tail_x, tail, 1)[0])  # MSE/epoch
    # Normalise by best val loss for threshold comparison
    slope_norm = slope_raw / max(best_val, 1e-12)

    # Monotonically decreasing over last MONO_EPOCHS?
    mono_tail = val_mse[-MONO_EPOCHS:] if n_epochs >= MONO_EPOCHS else val_mse
    mono_dec = bool(np.all(np.diff(mono_tail) < 0))

    # Verdict
    if abs(slope_norm) < SLOPE_CONVERGED_THR:
        verdict = "Converged"
    elif slope_norm < SLOPE_IMPROVING_THR:
        verdict = "Still improving at stop"
    else:
        verdict = "Plateauing"

    return {
        "best_epoch":  best_epoch,
        "stop_epoch":  stop_epoch,
        "best_val":    best_val,
        "stop_val":    stop_val,
        "slope_raw":   slope_raw,
        "slope_norm":  slope_norm,
        "mono_dec":    mono_dec,
        "verdict":     verdict,
        "epochs":      epochs,
        "val_mse":     val_mse,
        "train_mse":   train_mse,
    }


def plot_curves(stats: dict[int, dict], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, use_log in zip(axes, [False, True]):
        for sl, st in sorted(stats.items()):
            epochs = st["epochs"]
            color  = PALETTE[sl]
            ax.plot(epochs, st["train_mse"], color=color, linewidth=0.9,
                    linestyle=":", alpha=0.6)
            ax.plot(epochs, st["val_mse"], color=color, linewidth=1.4,
                    label=f"SEQ={sl} (best ep {st['best_epoch']}, {st['verdict'][:4]})")
            # Mark best epoch
            ax.axvline(st["best_epoch"], color=color, linewidth=0.7,
                       linestyle="--", alpha=0.5)
            # Mark stop epoch
            be = st["best_epoch"]
            ax.scatter([be], [st["best_val"]], color=color, s=40, zorder=5)

        if use_log:
            ax.set_yscale("log")
            ax.set_title("Val loss (log y-axis) — dotted=train, solid=val")
        else:
            ax.set_title("Val loss — dotted=train, solid=val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (scaled)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("GRU convergence — SEQ_LEN sweep", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    stats: dict[int, dict] = {}
    for sl, path in sorted(HISTORIES.items()):
        if not path.exists():
            print(f"WARNING: {path} not found — skipping SEQ_LEN={sl}")
            continue
        hist = load_history(path)
        stats[sl] = convergence_stats(hist)

    # ── Console report ────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  {'SEQ':>6}  {'BestEp':>7}  {'StopEp':>7}  "
          f"{'BestVal':>10}  {'StopVal':>10}  "
          f"{'Slope(norm)':>12}  {'MonoDec':>8}  Verdict")
    print(f"{'─'*80}")
    for sl, st in sorted(stats.items()):
        print(
            f"  {sl:>6}  {st['best_epoch']:>7}  {st['stop_epoch']:>7}  "
            f"{st['best_val']:>10.6f}  {st['stop_val']:>10.6f}  "
            f"{st['slope_norm']:>+12.4f}  "
            f"{'yes' if st['mono_dec'] else 'no':>8}  "
            f"{st['verdict']}"
        )
    print(f"{'═'*80}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_curves(stats, OUT_DIR / "seq_len_sweep_convergence.png")

    # ── Interpretation ────────────────────────────────────────────────────────
    verdicts = {sl: st["verdict"] for sl, st in stats.items()}
    all_converged = all(v == "Converged" for v in verdicts.values())
    any_improving = any(v == "Still improving at stop" for v in verdicts.values())
    improving_seqs = [sl for sl, v in verdicts.items() if v == "Still improving at stop"]

    if all_converged:
        interp = ("All configs converged. Prompt 8 result stands: longer windows "
                  "genuinely do not help on this setup. Proceed to Prompt 10b "
                  "(state-carrier features).")
    elif any_improving:
        interp = (
            f"SEQ_LEN ∈ {improving_seqs} verdict 'Still improving at stop'. "
            "Re-run those configs with patience=50 and LR schedule scaled to window length "
            "before concluding. This is a 1-day retrain."
        )
    else:
        lines = [f"SEQ_LEN={sl}: {v}" for sl, v in sorted(verdicts.items())]
        interp = "Mixed convergence: " + "; ".join(lines) + "."

    print(f"\nInterpretation: {interp}")

    # ── Summary block ─────────────────────────────────────────────────────────
    verdict_rows = "\n".join(
        f"  SEQ={sl:>3}: best_ep={st['best_epoch']:>3}, stop_ep={st['stop_epoch']:>3}, "
        f"slope_norm={st['slope_norm']:>+.4f}, mono_dec={st['mono_dec']}, "
        f"verdict={st['verdict']}"
        for sl, st in sorted(stats.items())
    )
    block = (
        f"\n{'='*72}\n"
        f"[Prompt 2.8b — Convergence check]\n"
        f"{'='*72}\n"
        f"Purpose: Verify whether longer-SEQ configs truly converged or were "
        f"cut short by the patience=20 early-stopping rule.\n\n"
        f"Per-config verdicts (slope threshold: |norm_slope|<{SLOPE_CONVERGED_THR:.2f} → "
        f"Converged, slope<{SLOPE_IMPROVING_THR:.3f} → Still improving):\n"
        f"{verdict_rows}\n\n"
        f"Interpretation: {interp}\n"
    )
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"Appended to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

"""Test 6 — Training-set regime distribution.

Quantifies how much of the training split the model ever saw of each regime
(rest / hold / transition / oscillation) and whether sustained holds
(>= 500 ms continuous) are effectively absent.

Outputs
-------
outputs/test6_regime_comparison.csv        — train vs test per-regime fractions
outputs/test6_per_file_regimes_train.csv   — per-file regime breakdown (training)
outputs/test6_sustained_hold_runs.csv      — each sustained-hold run (train + test)
outputs/test6_hold_runlength_hist.png      — histogram of run lengths
outputs/test6_stacked_bar_regimes.png      — per-file regime stacked bar (training)
SUMMARY.txt  (appended)
"""

from __future__ import annotations

import csv
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

from _common import (  # noqa: E402
    OUTPUT_DIR,
    SAMPLE_HZ,
    build_test_arrays,
    get_device,
    save_summary,
)
import config  # noqa: E402
from preprocessing import _get_feature_cols, _load_dataframes, _split_df  # noqa: E402
from test2_regime_residuals import (  # noqa: E402
    REGIME_COLORS,
    REGIMES,
    classify_regimes,
)

SUSTAINED_HOLD_MIN_SAMPLES = 500   # 500 ms @ 1 kHz
HOLD_DEFICIENT_FRAC = 0.05         # < 5 % hold → deficient
HOLD_DEFICIENT_RUNS = 20           # < 20 sustained runs → deficient


# ── Data builders ────────────────────────────────────────────────────────────

def build_train_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return (y_train, file_name_train) — raw training rows, all timesteps."""
    seq_len = config.SEQ_LEN
    dfs = _load_dataframes()
    y_parts:    list[np.ndarray] = []
    file_parts: list[np.ndarray] = []

    for df in dfs:
        train_df, _, _ = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        y_parts.append(train_df[config.TARGET_COL].values.astype(np.float32))
        if "file_name" in train_df.columns:
            file_parts.append(train_df["file_name"].values.astype(str))
        else:
            file_parts.append(np.full(len(train_df), "unknown", dtype=object))

    return np.concatenate(y_parts), np.concatenate(file_parts)


# ── Sustained-hold run detection ─────────────────────────────────────────────

def find_sustained_hold_runs(regimes: np.ndarray, file_name: np.ndarray,
                              min_samples: int = SUSTAINED_HOLD_MIN_SAMPLES) -> list[dict]:
    """Return metadata for each contiguous hold run >= min_samples within a file."""
    runs: list[dict] = []
    n, i = len(regimes), 0
    while i < n:
        if regimes[i] == "hold":
            j = i + 1
            while j < n and regimes[j] == "hold" and file_name[j] == file_name[i]:
                j += 1
            length = j - i
            if length >= min_samples:
                runs.append({
                    "file":       str(file_name[i]),
                    "start_idx":  int(i),
                    "length_samples": int(length),
                    "duration_s": length / SAMPLE_HZ,
                })
            i = j
        else:
            i += 1
    return runs


# ── Regime stats helpers ──────────────────────────────────────────────────────

def regime_counts(regimes: np.ndarray) -> dict[str, int]:
    return {r: int((regimes == r).sum()) for r in REGIMES}


def per_file_regime_counts(regimes: np.ndarray,
                            file_name: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"regime": regimes, "file": file_name})
    counts = (df.groupby(["file", "regime"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=REGIMES, fill_value=0))
    counts["total"] = counts.sum(axis=1)
    counts["hold_frac"] = counts["hold"] / counts["total"].clip(lower=1)
    return counts.reset_index()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_runlength_hist(train_runs: list[dict], test_runs: list[dict], path):
    bin_edges_s = [0.5, 1.0, 2.0, 5.0, 10.0, 9999.0]
    labels = ["0.5–1 s", "1–2 s", "2–5 s", "5–10 s", ">10 s"]

    def _hist(runs):
        durations = [r["duration_s"] for r in runs]
        counts, _ = np.histogram(durations, bins=bin_edges_s)
        return counts

    train_h = _hist(train_runs)
    test_h  = _hist(test_runs)
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, train_h, w, label="train", color="steelblue")
    ax.bar(x + w / 2, test_h,  w, label="test",  color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Run length")
    ax.set_ylabel("Number of sustained-hold runs")
    ax.set_title(f"Sustained-hold run-length distribution (≥{SUSTAINED_HOLD_MIN_SAMPLES} ms)")
    ax.legend()
    for i, (tr, te) in enumerate(zip(train_h, test_h)):
        if tr: ax.text(i - w / 2, tr, str(tr), ha="center", va="bottom", fontsize=8)
        if te: ax.text(i + w / 2, te, str(te), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_stacked_bar(per_file_df: pd.DataFrame, path):
    """Stacked bar chart: one bar per file, stacked by regime, training data."""
    files = per_file_df["file"].values

    # Shorten labels for readability
    short = [Path(f).name if f != "unknown" else f for f in files]
    short = [s[:28] + "…" if len(s) > 29 else s for s in short]

    x = np.arange(len(files))
    fig, ax = plt.subplots(figsize=(max(12, len(files) * 0.7), 6))

    bottom = np.zeros(len(files))
    for regime in REGIMES:
        if regime not in per_file_df.columns:
            continue
        vals = per_file_df[regime].values.astype(float)
        ax.bar(x, vals, bottom=bottom, label=regime,
               color=REGIME_COLORS[regime], edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Samples")
    ax.set_title("Training-set regime composition per source file")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


# ── CSV savers ────────────────────────────────────────────────────────────────

def save_comparison_csv(path, train_counts, test_counts):
    total_tr = sum(train_counts.values())
    total_te = sum(test_counts.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["regime", "train_count", "train_frac",
                    "test_count",  "test_frac"])
        for r in REGIMES:
            tr = train_counts.get(r, 0)
            te = test_counts.get(r, 0)
            w.writerow([r, tr, f"{tr/max(total_tr,1):.6f}",
                           te, f"{te/max(total_te,1):.6f}"])
    print(f"Wrote {path}")


def save_runs_csv(path, train_runs, test_runs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "start_idx",
                    "length_samples", "duration_s"])
        for r in train_runs:
            w.writerow(["train", r["file"], r["start_idx"],
                        r["length_samples"], f"{r['duration_s']:.3f}"])
        for r in test_runs:
            w.writerow(["test",  r["file"], r["start_idx"],
                        r["length_samples"], f"{r['duration_s']:.3f}"])
    print(f"Wrote {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building training arrays …")
    y_train, fn_train = build_train_arrays()

    print("Building test arrays …")
    arr = build_test_arrays(device=get_device(), run_models=False)
    y_test = arr["y_true"]
    fn_test = arr["file_name"]

    print("Classifying regimes …")
    reg_train = classify_regimes(y_train, fn_train)
    reg_test  = classify_regimes(y_test,  fn_test)

    # ── Per-regime counts ────────────────────────────────────────────────────
    tr_counts = regime_counts(reg_train)
    te_counts = regime_counts(reg_test)
    total_tr  = sum(tr_counts.values())
    total_te  = sum(te_counts.values())

    save_comparison_csv(OUTPUT_DIR / "test6_regime_comparison.csv",
                        tr_counts, te_counts)

    # ── Sustained-hold runs ──────────────────────────────────────────────────
    print("Finding sustained-hold runs …")
    train_runs = find_sustained_hold_runs(reg_train, fn_train)
    test_runs  = find_sustained_hold_runs(reg_test,  fn_test)

    save_runs_csv(OUTPUT_DIR / "test6_sustained_hold_runs.csv",
                  train_runs, test_runs)

    # ── Per-file breakdown ───────────────────────────────────────────────────
    pf_df = per_file_regime_counts(reg_train, fn_train)

    # Add sustained-hold run count per file
    from collections import Counter
    run_counts_per_file = Counter(r["file"] for r in train_runs)
    pf_df["sustained_hold_runs"] = pf_df["file"].map(
        lambda f: run_counts_per_file.get(f, 0)
    )
    pf_df.to_csv(OUTPUT_DIR / "test6_per_file_regimes_train.csv",
                 index=False, encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'test6_per_file_regimes_train.csv'}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_runlength_hist(train_runs, test_runs,
                        OUTPUT_DIR / "test6_hold_runlength_hist.png")
    plot_stacked_bar(pf_df, OUTPUT_DIR / "test6_stacked_bar_regimes.png")

    # ── Run-length stats ─────────────────────────────────────────────────────
    def run_stats(runs):
        if not runs:
            return {"n": 0, "total_s": 0.0,
                    "min_s": float("nan"), "med_s": float("nan"),
                    "max_s": float("nan")}
        durs = [r["duration_s"] for r in runs]
        return {"n": len(durs), "total_s": sum(durs),
                "min_s": min(durs), "med_s": float(np.median(durs)),
                "max_s": max(durs)}

    tr_stats = run_stats(train_runs)
    te_stats = run_stats(test_runs)

    # ── Verdict ──────────────────────────────────────────────────────────────
    train_hold_frac = tr_counts.get("hold", 0) / max(total_tr, 1)
    hold_deficient  = (train_hold_frac < HOLD_DEFICIENT_FRAC
                       and tr_stats["n"] < HOLD_DEFICIENT_RUNS)
    verdict = ("Hold-deficient training set"
               if hold_deficient else "Hold-sufficient training set")

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"{'regime':<14} {'train count':>12} {'train %':>9}  "
          f"{'test count':>12} {'test %':>9}")
    for r in REGIMES:
        tr = tr_counts.get(r, 0)
        te = te_counts.get(r, 0)
        print(f"  {r:<12} {tr:>12,}  {100*tr/max(total_tr,1):>8.2f}%  "
              f"{te:>12,}  {100*te/max(total_te,1):>8.2f}%")
    print(f"  {'TOTAL':<12} {total_tr:>12,}             {total_te:>12,}")

    print(f"\nSustained-hold runs (≥{SUSTAINED_HOLD_MIN_SAMPLES} ms):")
    print(f"  Train: {tr_stats['n']} runs, total={tr_stats['total_s']:.1f}s,"
          f" min={tr_stats['min_s']:.2f}s, median={tr_stats['med_s']:.2f}s,"
          f" max={tr_stats['max_s']:.2f}s")
    print(f"  Test:  {te_stats['n']} runs, total={te_stats['total_s']:.1f}s,"
          f" min={te_stats['min_s']:.2f}s, median={te_stats['med_s']:.2f}s,"
          f" max={te_stats['max_s']:.2f}s")

    # ── Summary block ─────────────────────────────────────────────────────────
    lines = [
        f"Training rows: {total_tr:,}   Test rows: {total_te:,}",
        f"Regime split (train | test):",
    ]
    for r in REGIMES:
        tr = tr_counts.get(r, 0)
        te = te_counts.get(r, 0)
        lines.append(
            f"  {r:<14} train={tr:>7,} ({100*tr/max(total_tr,1):.2f}%)  "
            f"test={te:>6,} ({100*te/max(total_te,1):.2f}%)"
        )
    lines += [
        "",
        f"Sustained-hold runs (>={SUSTAINED_HOLD_MIN_SAMPLES} ms):",
        (f"  Train: {tr_stats['n']} runs, {tr_stats['total_s']:.1f}s total  "
         f"[min={tr_stats['min_s']:.2f}s  med={tr_stats['med_s']:.2f}s  "
         f"max={tr_stats['max_s']:.2f}s]"),
        (f"  Test:  {te_stats['n']} runs, {te_stats['total_s']:.1f}s total  "
         f"[min={te_stats['min_s']:.2f}s  med={te_stats['med_s']:.2f}s  "
         f"max={te_stats['max_s']:.2f}s]"),
        "",
        (f"Thresholds: hold_frac<{HOLD_DEFICIENT_FRAC:.0%} AND "
         f"sustained_runs<{HOLD_DEFICIENT_RUNS}"),
        f"Verdict: {verdict}",
    ]

    block = save_summary("Test 6 — Training-set regime distribution", lines)
    print("\n" + block)

    summary_path = OUTPUT_DIR / "SUMMARY.txt"
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n" + block)
    print(f"Appended verdict block to {summary_path}")

    return block


if __name__ == "__main__":
    main()
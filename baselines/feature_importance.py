"""Permutation feature importance analysis for MLP and GRU models.

Loads a trained checkpoint and computes per-feature RMSE increase (in Nm) when
that feature is shuffled across the test set. Outputs rankings, plots, and
summary markdown.

Usage
-----
    python baselines/feature_importance.py --model gru
    python baselines/feature_importance.py --model mlp
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing import get_dataloaders, permutation_importance
from models import ActuatorGRU, WindowedMLP

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_TXT      = PROJECT_ROOT / "diagnostics" / "outputs" / "SUMMARY.txt"
SUMMARY_CONC_TXT = PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute permutation feature importance for trained models"
    )
    parser.add_argument("--model", choices=["mlp", "gru"], required=True,
                        help="Model architecture")
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Number of shuffle repeats per feature")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model_type: str, device: torch.device):
    """Load best checkpoint for model_type."""
    ckpt_path = config.CHECKPOINT_DIR / f"best_model_{model_type}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    if model_type == "mlp":
        model = WindowedMLP(
            seq_len=cfg["seq_len"],
            n_features=cfg["n_features"],
            hidden_size=cfg["hidden_size"],
            n_layers=cfg["n_layers"],
        )
    else:
        model = ActuatorGRU(
            n_features=cfg["n_features"],
            hidden_size=cfg["hidden_size"],
            n_layers=cfg["n_layers"],
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def compute_importance(model_type: str, device: torch.device, n_repeats: int = 5):
    """Compute permutation importance on test set."""
    print(f"\n[1] Loading checkpoint for {model_type.upper()} …")
    model = load_checkpoint(model_type, device)

    print(f"[2] Loading data …")
    train_loader, val_loader, test_loader, scaler_X, scaler_y, feature_names = (
        get_dataloaders(batch_size=256, save_scalers=False)
    )

    print(f"[3] Computing permutation importance ({n_repeats} repeats per feature) …")
    importances = permutation_importance(
        model, test_loader, scaler_y, feature_names, device, n_repeats=n_repeats
    )

    # Sort by importance (descending)
    ranked = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked, feature_names, importances


def save_csv(ranked, path):
    """Save feature importance table."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "rmse_increase_nm"])
        for rank, (feat, imp) in enumerate(ranked, 1):
            w.writerow([rank, feat, f"{imp:.4f}"])
    print(f"Wrote {path}")


def plot_importance(ranked, model_type: str, path):
    """Plot top-10 features."""
    top_n = min(10, len(ranked))
    feats = [f for f, _ in ranked[:top_n]]
    imps  = [abs(v) for _, v in ranked[:top_n]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(top_n)[::-1], imps, color="steelblue")
    ax.set_yticks(range(top_n)[::-1])
    ax.set_yticklabels(feats)
    ax.set_xlabel("RMSE increase (Nm) when feature shuffled")
    ax.set_title(f"Permutation Feature Importance — {model_type.upper()}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def write_md_summary(model_type: str, ranked, feature_names, path):
    """Write markdown summary."""
    lines = [
        f"# Permutation Feature Importance — {model_type.upper()}",
        "",
        "## Method",
        "",
        "For each feature, shuffle its values across all test samples and timesteps,",
        "recompute model RMSE (in Nm), and record the increase vs. baseline.",
        "Average over 5 random shuffles.",
        "",
        "## Results (Top-15 Features)",
        "",
        "| Rank | Feature | RMSE Increase (Nm) |",
        "|------|---------|-------------------|",
    ]
    for rank, (feat, imp) in enumerate(ranked[:15], 1):
        lines.append(f"| {rank} | {feat} | {imp:.4f} |")

    lines += [
        "",
        f"## Verdict",
        "",
        f"**{len(ranked)} features analyzed.** " +
        f"Top 3 drivers: {', '.join(f[0] for f in ranked[:3])}. " +
        f"Range: {abs(ranked[0][1]):.4f} to {abs(ranked[-1][1]):.4f} Nm.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def append_summary_txt(model_type: str, ranked):
    """Append to SUMMARY.txt."""
    lines = [
        "",
        "=" * 72,
        f"[Permutation Feature Importance — {model_type.upper()}]",
        "=" * 72,
        "",
        "Feature                          | RMSE Increase (Nm)",
        "-" * 50,
    ]
    for feat, imp in ranked[:10]:
        lines.append(f"{feat:<30} | {imp:>8.4f}")
    lines.append("")
    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended to {SUMMARY_TXT}")


def append_summary_conc(model_type: str, ranked):
    """Append to SUMMARY+CONCLUSION.txt."""
    top_3 = ", ".join(f[0] for f in ranked[:3])
    block = (
        "\n"
        "=" * 72 + "\n"
        f"[Permutation Feature Importance — {model_type.upper()}]\n"
        "=" * 72 + "\n"
        "\n"
        "Shuffles each feature across test set, measures RMSE increase (Nm).\n"
        f"Top 3 drivers: {top_3}.\n"
        f"Importance range: {abs(ranked[0][1]):.4f} to {abs(ranked[-1][1]):.4f} Nm.\n"
        "\n"
    )
    with open(SUMMARY_CONC_TXT, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"Appended to {SUMMARY_CONC_TXT}")


def main():
    args = parse_args()
    device = get_device()

    print("=" * 60)
    print(f"Permutation Feature Importance — {args.model.upper()}")
    print("=" * 60)

    ranked, feature_names, importances = compute_importance(
        args.model, device, n_repeats=args.n_repeats
    )

    print(f"\n[4] Top-10 features by |RMSE increase|:")
    for rank, (feat, imp) in enumerate(ranked[:10], 1):
        print(f"  {rank:2d}. {feat:<30} {imp:>+8.4f} Nm")

    # Save outputs
    csv_path = OUTPUT_DIR / f"feature_importance_{args.model}.csv"
    plot_path = OUTPUT_DIR / f"feature_importance_{args.model}.png"
    md_path = OUTPUT_DIR / f"feature_importance_{args.model}.md"

    save_csv(ranked, csv_path)
    plot_importance(ranked, args.model, plot_path)
    write_md_summary(args.model, ranked, feature_names, md_path)
    append_summary_txt(args.model, ranked)
    append_summary_conc(args.model, ranked)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

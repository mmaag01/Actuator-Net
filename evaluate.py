"""Evaluation script — loads the best checkpoint(s) and reports test-set metrics.

Usage
-----
    python evaluate.py --model gru          # single-model evaluation
    python evaluate.py --model mlp
    python evaluate.py --ensemble           # MLP + GRU averaged ensemble
"""

import argparse
import csv
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from dataset import get_dataloaders
from models import ActuatorGRU, WindowedMLP


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate actuator torque model")
    parser.add_argument("--model", choices=["mlp", "gru"], default=None,
                        help="Model type to evaluate (required unless --ensemble)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Average MLP and GRU predictions (both checkpoints must exist)")
    args = parser.parse_args()
    if not args.ensemble and args.model is None:
        parser.error("Specify --model mlp|gru or use --ensemble.")
    return args


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_checkpoint(model_type: str, device: torch.device):
    """Load a named checkpoint and reconstruct the model from its saved config."""
    ckpt_path = config.CHECKPOINT_DIR / f"best_model_{model_type}.pt"
    assert ckpt_path.exists(), (
        f"No checkpoint found at {ckpt_path}. Run: python train.py --model {model_type}"
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    if model_type == "mlp":
        model = WindowedMLP(
            seq_len=cfg["seq_len"],
            n_features=cfg["n_features"],
            hidden_size=cfg.get("hidden_size", config.MLP_HIDDEN_SIZE),
            n_layers=cfg.get("n_layers",    config.MLP_N_LAYERS),
        )
    else:
        model = ActuatorGRU(
            n_features=cfg["n_features"],
            hidden_size=cfg.get("hidden_size", config.GRU_HIDDEN_SIZE),
            n_layers=cfg.get("n_layers",    config.GRU_N_LAYERS),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["epoch"], ckpt["val_loss"]


def _load_scalers():
    sx = config.CHECKPOINT_DIR / "scaler_X.pkl"
    sy = config.CHECKPOINT_DIR / "scaler_y.pkl"
    assert sx.exists() and sy.exists(), (
        "Scalers not found in checkpoints/. Run train.py first."
    )
    return joblib.load(sx), joblib.load(sy)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(preds).ravel(), np.concatenate(targets).ravel()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "RMSE [Nm]":          float(np.sqrt(np.mean(err ** 2))),
        "MAE [Nm]":           float(np.mean(np.abs(err))),
        "Max Abs Error [Nm]": float(np.max(np.abs(err))),
    }


def _print_metrics_table(rows: dict[str, dict]):
    """Print a side-by-side metrics table.
    rows: {label: {metric_name: value}}
    """
    labels  = list(rows.keys())
    metrics = list(next(iter(rows.values())).keys())
    col_w   = 16

    header = f"{'Metric':<25s}" + "".join(f"{l:>{col_w}s}" for l in labels)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for m in metrics:
        line = f"  {m:<23s}" + "".join(f"{rows[l][m]:>{col_w}.4f}" for l in labels)
        print(line)
    print("─" * len(header))


# ── Single-model plots ────────────────────────────────────────────────────────

def _worst_window(errors_abs, n_samples):
    """Return (start, end) of the 5 s window with highest mean |error|."""
    conv  = np.convolve(errors_abs, np.ones(n_samples) / n_samples, mode="valid")
    start = int(np.argmax(conv))
    return start, start + n_samples


def plot_timeseries(y_true, y_pred, hz=1000, window_s=5.0, save_path=None):
    n = int(window_s * hz)
    start, end = _worst_window(np.abs(y_pred - y_true), n)
    t = np.arange(n) / hz

    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t, y_true[start:end], label="Measured $\\tau_{act}$",
                 linewidth=1.0, color="steelblue")
    axes[0].plot(t, y_pred[start:end], label="Predicted $\\tau_{act}$",
                 linewidth=1.0, linestyle="--", color="tomato")
    axes[0].set_ylabel("Joint Torque [Nm]")
    axes[0].set_title("Predicted vs. Measured Joint Torque (highest-error 5 s window)")
    axes[0].legend()
    axes[1].plot(t, y_pred[start:end] - y_true[start:end],
                 linewidth=0.8, color="grey")
    axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Error [Nm]")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Time-series plot saved to {save_path}")
    plt.close(fig)


def plot_error_hist(y_true, y_pred, save_path=None):
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=120, edgecolor="none", color="steelblue", alpha=0.8)
    ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="zero error")
    ax.axvline(errors.mean(), color="orange", linewidth=1.2,
               label=f"mean = {errors.mean():.4f} Nm")
    ax.set_xlabel("Prediction Error [Nm]")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution on Test Set")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Error histogram saved to {save_path}")
    plt.close(fig)


# ── Ensemble plots ────────────────────────────────────────────────────────────

def plot_timeseries_ensemble(y_true, y_mlp, y_gru, y_ens,
                             hz=1000, window_s=5.0, save_path=None):
    n = int(window_s * hz)
    # Choose worst window based on ensemble error
    start, end = _worst_window(np.abs(y_ens - y_true), n)
    t = np.arange(n) / hz

    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t, y_true[start:end], label="Measured $\\tau_{act}$",
                 linewidth=1.2, color="steelblue")
    axes[0].plot(t, y_mlp[start:end],  label="MLP",
                 linewidth=0.9, linestyle=":", color="tomato")
    axes[0].plot(t, y_gru[start:end],  label="GRU",
                 linewidth=0.9, linestyle="--", color="seagreen")
    axes[0].plot(t, y_ens[start:end],  label="Ensemble",
                 linewidth=1.2, linestyle="-", color="darkorange")
    axes[0].set_ylabel("Joint Torque [Nm]")
    axes[0].set_title("Ensemble — Predicted vs. Measured (highest-error 5 s window)")
    axes[0].legend()

    axes[1].plot(t, y_mlp[start:end] - y_true[start:end],
                 linewidth=0.8, color="tomato",    label="MLP")
    axes[1].plot(t, y_gru[start:end] - y_true[start:end],
                 linewidth=0.8, color="seagreen",  label="GRU")
    axes[1].plot(t, y_ens[start:end] - y_true[start:end],
                 linewidth=0.8, color="darkorange", label="Ensemble")
    axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Error [Nm]")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Ensemble time-series plot saved to {save_path}")
    plt.close(fig)


def plot_error_hist_ensemble(y_true, y_mlp, y_gru, y_ens, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    palette = [("MLP", y_mlp, "tomato"),
               ("GRU", y_gru, "seagreen"),
               ("Ensemble", y_ens, "darkorange")]
    for ax, (label, y_pred, color) in zip(axes, palette):
        errors = y_pred - y_true
        ax.hist(errors, bins=120, edgecolor="none", color=color, alpha=0.8)
        ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
        ax.axvline(errors.mean(), color="navy", linewidth=1.0,
                   label=f"mean = {errors.mean():.4f} Nm")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Error Distribution")
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Prediction Error [Nm]")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Ensemble error histogram saved to {save_path}")
    plt.close(fig)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inverse(scaler_y, arr):
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).ravel()


def _save_metrics_csv(path, rows: dict[str, dict]):
    labels  = list(rows.keys())
    metrics = list(next(iter(rows.values())).keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + labels)
        for m in metrics:
            writer.writerow([m] + [f"{rows[l][m]:.6f}" for l in labels])
    print(f"Metrics saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    scaler_X, scaler_y = _load_scalers()
    print("Loading test data …")
    _, _, test_loader, _, _ = get_dataloaders(
        save_scalers=False, scaler_X=scaler_X, scaler_y=scaler_y
    )
    config.RESULTS_DIR.mkdir(exist_ok=True)

    if args.ensemble:
        # ── Ensemble path ─────────────────────────────────────────────────────
        mlp_model, mlp_epoch, mlp_val = load_checkpoint("mlp", device)
        gru_model, gru_epoch, gru_val = load_checkpoint("gru", device)
        print(f"Loaded MLP checkpoint (epoch {mlp_epoch}, val MSE {mlp_val:.6f})")
        print(f"Loaded GRU checkpoint (epoch {gru_epoch}, val MSE {gru_val:.6f})")

        print("Running inference …")
        y_mlp_sc, y_true_sc = run_inference(mlp_model, test_loader, device)
        y_gru_sc, _         = run_inference(gru_model, test_loader, device)
        y_ens_sc            = (y_mlp_sc + y_gru_sc) / 2.0

        y_true = _inverse(scaler_y, y_true_sc)
        y_mlp  = _inverse(scaler_y, y_mlp_sc)
        y_gru  = _inverse(scaler_y, y_gru_sc)
        y_ens  = _inverse(scaler_y, y_ens_sc)

        rows = {
            "MLP":      compute_metrics(y_true, y_mlp),
            "GRU":      compute_metrics(y_true, y_gru),
            "Ensemble": compute_metrics(y_true, y_ens),
        }
        _print_metrics_table(rows)
        _save_metrics_csv(config.RESULTS_DIR / "metrics_ensemble.csv", rows)

        plot_timeseries_ensemble(
            y_true, y_mlp, y_gru, y_ens,
            save_path=config.RESULTS_DIR / "timeseries_ensemble.png",
        )
        plot_error_hist_ensemble(
            y_true, y_mlp, y_gru, y_ens,
            save_path=config.RESULTS_DIR / "error_hist_ensemble.png",
        )

    else:
        # ── Single-model path ─────────────────────────────────────────────────
        model, best_epoch, best_val = load_checkpoint(args.model, device)
        print(f"Loaded {args.model.upper()} checkpoint  "
              f"(epoch {best_epoch}, val MSE {best_val:.6f})")

        print("Running inference …")
        y_pred_sc, y_true_sc = run_inference(model, test_loader, device)
        y_pred = _inverse(scaler_y, y_pred_sc)
        y_true = _inverse(scaler_y, y_true_sc)

        metrics = compute_metrics(y_true, y_pred)
        _print_metrics_table({args.model.upper(): metrics})
        _save_metrics_csv(
            config.RESULTS_DIR / f"metrics_{args.model}.csv",
            {args.model.upper(): metrics},
        )

        plot_timeseries(
            y_true, y_pred,
            save_path=config.RESULTS_DIR / f"timeseries_{args.model}.png",
        )
        plot_error_hist(
            y_true, y_pred,
            save_path=config.RESULTS_DIR / f"error_hist_{args.model}.png",
        )


if __name__ == "__main__":
    main()

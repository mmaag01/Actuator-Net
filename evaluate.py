"""Evaluation script — loads the best checkpoint(s) and reports test-set metrics.

Usage
-----
    python evaluate.py --model gru
    python evaluate.py --model mlp
    python evaluate.py --model wh
    python evaluate.py --ensemble           # MLP + GRU + WH averaged
    python evaluate.py --ensemble "mlp gru"   # any two-model combination
"""

import argparse
import csv
import sys
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch # type: ignore

import config
from preprocessing import get_dataloaders
from models import ActuatorGRU, WindowedMLP, WienerHammersteinNet


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate actuator torque model")
    parser.add_argument("--model", choices=["mlp", "gru", "wh"], default=None,
                        help="Model type to evaluate (required unless --ensemble)")
    parser.add_argument("--ensemble",
                        choices=["all", "mlp-gru", "gru-mlp",
                                 "mlp-wh", "wh-mlp",
                                 "gru-wh", "wh-gru"],
                        default=None,
                        help="Ensemble two or all three models (average predictions)")
    args = parser.parse_args()
    if args.ensemble is None and args.model is None:
        parser.error("Specify --model mlp|gru|wh or --ensemble <combo>.")
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
    elif model_type == "wh":
        model = WienerHammersteinNet(
            n_features=cfg["n_features"],
            n_channels=cfg.get("n_channels", config.WH_CHANNELS),
            mlp_hidden=cfg.get("mlp_hidden", config.WH_MLP_HIDDEN),
            stable=cfg.get("stable",     config.WH_STABLE),
        )
    else:
        model = ActuatorGRU(
            n_features=cfg["n_features"],
            hidden_size=cfg.get("hidden_size", config.GRU_HIDDEN_SIZE),
            n_layers=cfg.get("n_layers",    config.GRU_N_LAYERS),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["epoch"], ckpt["val_loss"], cfg


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

# Fixed style per model name — ensemble always gets darkorange/solid.
_MODEL_STYLE = {
    "mlp": ("tomato",       ":"),
    "gru": ("seagreen",     "--"),
    "wh":  ("mediumpurple", "-."),
}


def plot_timeseries_ensemble(y_true, preds: dict[str, np.ndarray], y_ens,
                             hz=1000, window_s=5.0, save_path=None):
    """preds: {model_name: y_pred_nm} for each individual model."""
    n = int(window_s * hz)
    start, end = _worst_window(np.abs(y_ens - y_true), n)
    t = np.arange(n) / hz

    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t, y_true[start:end], label="Measured $\\tau_{act}$",
                 linewidth=1.2, color="steelblue")
    for name, yp in preds.items():
        color, ls = _MODEL_STYLE.get(name, ("grey", "--"))
        axes[0].plot(t, yp[start:end], label=name.upper(),
                     linewidth=0.9, linestyle=ls, color=color)
    axes[0].plot(t, y_ens[start:end], label="Ensemble",
                 linewidth=1.2, linestyle="-", color="darkorange")
    axes[0].set_ylabel("Joint Torque [Nm]")
    axes[0].set_title("Ensemble — Predicted vs. Measured (highest-error 5 s window)")
    axes[0].legend()

    for name, yp in preds.items():
        color, ls = _MODEL_STYLE.get(name, ("grey", "--"))
        axes[1].plot(t, yp[start:end] - y_true[start:end],
                     linewidth=0.8, color=color, label=name.upper())
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


def plot_error_hist_ensemble(y_true, preds: dict[str, np.ndarray], y_ens,
                             save_path=None):
    """preds: {model_name: y_pred_nm} for each individual model."""
    entries = list(preds.items()) + [("ensemble", y_ens)]
    n_plots = len(entries)
    fig, axes = plt.subplots(n_plots, 1, figsize=(9, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, (name, yp) in zip(axes, entries):
        color = "darkorange" if name == "ensemble" else _MODEL_STYLE.get(name, ("grey",))[0]
        label = "Ensemble" if name == "ensemble" else name.upper()
        errors = yp - y_true
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


def _ensemble_model_names(ensemble_str: str) -> list[str]:
    """Parse '--ensemble' value into an ordered, deduplicated list of model names."""
    if ensemble_str == "all":
        return ["mlp", "gru", "wh"]

    # e.g. "gru mlp" → ["gru", "mlp"]; keep the user's order but deduplicate
    seen, names = set(), []
    for n in ensemble_str.split("-"):
        if n not in seen:
            seen.add(n)
            names.append(n)
    return names


# ── Evaluations.md logging ────────────────────────────────────────────────────

_MD_PATH = config.PROJECT_ROOT / "Evaluations.md"

_TABLE_HEADER = (
    "| Date       | Model      | RMSE    | MAE    | Max Err. "
    "| Ep. | Val MSE  "
    "| Main/Crashes/Other | tStep/TMS/PMS/PLC   "
    "| Smooth | Scaler | Excluded Features    | Hyperparameters |\n"
    "|------------|------------|---------|--------|----------"
    "|-----|----------"
    "|--------------------|---------------------"
    "|--------|--------|----------------------|-----------------|\n"
)

# Map INCLUDE_* config flags → the feature name they gate
_INCLUDE_MAP = [
    ("INCLUDE_I2T",      "i2t"),
    ("INCLUDE_curr",     "i"),
    ("INCLUDE_torKdEst", "torKdEst"),
    ("INCLUDE_torEst",   "torEst"),
    ("INCLUDE_kd",       "kd"),
    ("INCLUDE_posDes",   "posDes"),
    ("INCLUDE_accelAct", "accelAct"),
    ("INCLUDE_t",        "t"),
]


def _get_excluded_features() -> str:
    excluded = [feat for attr, feat in _INCLUDE_MAP if not getattr(config, attr)]
    return ", ".join(excluded) if excluded else "none"


def _format_hyperparams(model_type: str, cfg: dict) -> str:
    if model_type == "mlp":
        return f"{cfg.get('hidden_size')},{cfg.get('n_layers')}"
    if model_type == "wh":
        return f"{cfg.get('n_channels')},{cfg.get('mlp_hidden')},{config.WH_NA},{cfg.get('stable')}"
    return (f"{cfg.get('hidden_size')}, {cfg.get('n_layers')}, {config.GRU_DROPOUT}")


def _append_evaluations_md(model: str, hyperparams: str, epoch_str: str, val_mse_str: str, metrics: dict,) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")[6:]
    rmse   = f"{metrics['RMSE [Nm]']:.4f}"
    mae    = f"{metrics['MAE [Nm]']:.4f}"
    maxerr = f"{metrics['Max Abs Error [Nm]']:.4f}"
    blank = (10-len(model))*" "

    row = (
        f"| {date_str} | {model}{blank} | {rmse} | {mae} | {maxerr} "
        f"| {epoch_str}  | {val_mse_str} "
        f"| {config.USE_MAIN}/{config.USE_CRASHES}/{config.USE_OTHER}   "
        f"| {config.USE_tStep}/{config.USE_TMS}/{config.USE_PMS}/{config.USE_PLC} "
        f"| {config.SMOOTH_ACCEL}   | {config.SCALER_TYPE} "
        f"| {_get_excluded_features()} | {hyperparams} |\n"
    )

    if _MD_PATH.exists():
        content = _MD_PATH.read_text(encoding="utf-8")
    else:
        content = ""

    if "| Date       |" in content:
        content = content.rstrip("\n") + "\n" + row
    else:
        content = content.rstrip("\n")
        if content:
            content += "\n\n"
        content += "## Evaluation Results\n\n" + _TABLE_HEADER + row

    _MD_PATH.write_text(content, encoding="utf-8")
    print(f"Results appended to {_MD_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    scaler_X, scaler_y = _load_scalers()
    _, _, test_loader, _, _, feature_names = get_dataloaders(
        save_scalers=False, scaler_X=scaler_X, scaler_y=scaler_y
    )
    print(f"Loading test data with {feature_names} .…")
    config.RESULTS_DIR.mkdir(exist_ok=True)

    if args.ensemble is not None:
        # ── Ensemble path ─────────────────────────────────────────────────────
        model_names = _ensemble_model_names(args.ensemble)
        models = {}
        ckpt_cfgs: dict[str, dict] = {}
        for name in model_names:
            m, epoch, val, cfg = load_checkpoint(name, device)
            models[name] = m
            ckpt_cfgs[name] = cfg
            ckpt_cfgs[name]["_epoch"]   = epoch
            ckpt_cfgs[name]["_val_mse"] = val
            print(f"Loaded {name.upper()} checkpoint (epoch {epoch}, val MSE {val:.6f})")

        print("Running inference …")
        scaled_preds: dict[str, np.ndarray] = {}
        y_true_sc = None
        for name, m in models.items():
            p, y_true_sc = run_inference(m, test_loader, device)
            scaled_preds[name] = p
        y_ens_sc = np.mean(list(scaled_preds.values()), axis=0)

        y_true = _inverse(scaler_y, y_true_sc)
        preds  = {name: _inverse(scaler_y, p) for name, p in scaled_preds.items()}
        y_ens  = _inverse(scaler_y, y_ens_sc)

        rows = {name.upper(): compute_metrics(y_true, p) for name, p in preds.items()}
        rows["Ensemble"] = compute_metrics(y_true, y_ens)
        _print_metrics_table(rows)

        suffix = "_".join(model_names)
        _save_metrics_csv(config.RESULTS_DIR / f"metrics_ensemble_{suffix}.csv", rows)

        plot_timeseries_ensemble(
            y_true, preds, y_ens,
            save_path=config.RESULTS_DIR / f"timeseries_ensemble_{suffix}.png",
        )
        plot_error_hist_ensemble(
            y_true, preds, y_ens,
            save_path=config.RESULTS_DIR / f"error_hist_ensemble_{suffix}.png",
        )

        _append_evaluations_md(
            model  = "/".join(n.upper() for n in model_names),
            hyperparams  = "/".join(f"{n.upper()}({_format_hyperparams(n, ckpt_cfgs[n])})"for n in model_names),
            #epoch_str    = "/".join(f"{n.upper()}:{ckpt_cfgs[n]['_epoch']}" for n in model_names),
            #val_mse_str  = "/".join(f"{n.upper()}:{ckpt_cfgs[n]['_val_mse']:.6f}" for n in model_names),
            epoch_str    = "--",
            val_mse_str  = 8*"-",
            metrics      = rows["Ensemble"],
        )

    else:
        # ── Single-model path ─────────────────────────────────────────────────
        model, best_epoch, best_val, ckpt_cfg = load_checkpoint(args.model, device)
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

        _append_evaluations_md(
            model  = args.model.upper(),
            hyperparams  = _format_hyperparams(args.model, ckpt_cfg),
            epoch_str    = str(best_epoch),
            val_mse_str  = f"{best_val:.6f}",
            metrics      = metrics,
        )


if __name__ == "__main__":
    main()

"""Prompt 5 — accelAct Savitzky-Golay smoothing (v2 models).

Trains MLP and GRU on SG-smoothed accelAct (config.SMOOTH_ACCEL=True),
saves checkpoints/scalers to checkpoints/v2/, evaluates on:
  - Test 2: per-regime RMSE comparison vs v1
  - Test 3: HF power ratio on smoothed signal (target < 2×)
Writes evaluate_v2/v1_vs_v2_comparison.md and appends to
prompts/SUMMARY+CONCLUSION.txt.

Usage
-----
    python experiments/accel_smoothing.py [--force]
    --force  Overwrite existing v2 checkpoints (retrains from scratch).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

assert config.SMOOTH_ACCEL, (
    "Set SMOOTH_ACCEL=True in config.py before running this script."
)

# ── Path overrides for v2 ─────────────────────────────────────────────────────
V1_CKPT_DIR    = config.PROJECT_ROOT / "checkpoints"
V1_RESULTS_DIR = config.PROJECT_ROOT / "results" / "v1"
V1_DIAG_DIR    = config.PROJECT_ROOT / "diagnostics" / "outputs"

V2_CKPT_DIR    = config.PROJECT_ROOT / "checkpoints" / "v2"
V2_RESULTS_DIR = config.PROJECT_ROOT / "evaluate_v2"

V2_CKPT_DIR.mkdir(parents=True, exist_ok=True)
V2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Patch config so build_datasets() saves scalers to v2 dir
config.CHECKPOINT_DIR = V2_CKPT_DIR
config.RESULTS_DIR    = V2_RESULTS_DIR

from dataset import (  # noqa: E402  (must follow config patch)
    _get_feature_cols,
    _load_dataframes,
    _make_windows,
    _split_df,
    build_datasets,
    ActuatorDataset,
)
from models import ActuatorGRU, WindowedMLP  # noqa: E402

SAMPLE_HZ     = 1000.0
HF_CUTOFF_HZ  = 250.0
HF_RATIO_MULTIPLIER = 2.0

ROLLING_WINDOW_SAMPLES = 200
OSCILLATION_STD_NM     = 20.0
REST_STD_NM            = 5.0
HOLD_ABS_TORQUE_NM     = 10.0
REGIMES = ["rest", "hold", "transition", "oscillation"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="Retrain from scratch even if checkpoints exist.")
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Training ──────────────────────────────────────────────────────────────────

def build_model(model_type: str, n_features: int) -> nn.Module:
    if model_type == "mlp":
        return WindowedMLP(
            seq_len=config.SEQ_LEN,
            n_features=n_features,
            hidden_size=config.MLP_HIDDEN_SIZE,
            n_layers=config.MLP_N_LAYERS,
        )
    return ActuatorGRU(
        n_features=n_features,
        hidden_size=config.GRU_HIDDEN_SIZE,
        n_layers=config.GRU_N_LAYERS,
        dropout=config.GRU_DROPOUT,
    )


def train_one(model_type: str, train_ds, val_ds, device, force: bool) -> nn.Module:
    ckpt_path = V2_CKPT_DIR / f"best_model_{model_type}.pt"
    if ckpt_path.exists() and not force:
        print(f"  [{model_type.upper()}] Checkpoint found — skipping training.")
        return _load_model(model_type, ckpt_path, device)

    from torch.utils.data import DataLoader as TLoader
    train_loader = TLoader(train_ds, batch_size=config.BATCH_SIZE,
                           shuffle=True, drop_last=True)
    val_loader   = TLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)

    n_features = config.N_FEATURES
    model = build_model(model_type, n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                 weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.MAX_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.ONE_CYCLE_MAX_LR, total_steps=total_steps,
    )

    best_val, patience_ctr = float("inf"), 0
    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                val_loss += criterion(model(X.to(device)), y.to(device)).item() * len(X)
        val_loss /= len(val_ds)

        improved = val_loss < best_val
        tag = " [saved]" if improved else f" (pat {patience_ctr+1}/{config.PATIENCE})"
        print(f"    Ep {epoch:4d} | val MSE {val_loss:.6f}{tag}")

        if improved:
            best_val = val_loss
            patience_ctr = 0
            hidden = config.MLP_HIDDEN_SIZE if model_type == "mlp" else config.GRU_HIDDEN_SIZE
            n_lay  = config.MLP_N_LAYERS    if model_type == "mlp" else config.GRU_N_LAYERS
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss, "model_type": model_type,
                "config": {
                    "seq_len": config.SEQ_LEN, "n_features": n_features,
                    "hidden_size": hidden, "n_layers": n_lay,
                },
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= config.PATIENCE:
                print(f"    Early stopping at epoch {epoch}.")
                break

    print(f"  [{model_type.upper()}] Best val MSE: {best_val:.6f}")
    return _load_model(model_type, ckpt_path, device)


def _load_model(model_type: str, ckpt_path: Path, device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    if model_type == "mlp":
        m = WindowedMLP(seq_len=cfg["seq_len"], n_features=cfg["n_features"],
                        hidden_size=cfg["hidden_size"], n_layers=cfg["n_layers"])
    else:
        m = ActuatorGRU(n_features=cfg["n_features"], hidden_size=cfg["hidden_size"],
                        n_layers=cfg["n_layers"])
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval()


# ── Inference ─────────────────────────────────────────────────────────────────

def batched_infer(model: nn.Module, X: np.ndarray, device,
                  batch: int = 2048) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            e = min(s + batch, len(X))
            xb = torch.from_numpy(X[s:e]).float().to(device)
            out[s:e] = model(xb).cpu().numpy().ravel()
    return out


# ── Test-set window builder ───────────────────────────────────────────────────

def build_test_windows(dfs, scaler_X, scaler_y):
    """Return aligned test arrays using v2 scalers."""
    feature_cols = _get_feature_cols()
    seq_len = config.SEQ_LEN

    y_true_parts, torEst_parts, t_parts, file_parts, Xw_parts = [], [], [], [], []
    for df in dfs:
        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        if len(test_df) < seq_len:
            continue
        X_raw = test_df[feature_cols].values.astype(np.float32)
        y_raw = test_df[config.TARGET_COL].values.astype(np.float32)
        Xs    = scaler_X.transform(X_raw).astype(np.float32)
        Xw, _ = _make_windows(Xs, np.zeros(len(Xs), np.float32), seq_len)
        tgt   = slice(seq_len - 1, len(test_df))
        y_true_parts.append(y_raw[tgt])
        torEst_parts.append(test_df["torEst"].values.astype(np.float32)[tgt])
        t_parts.append(test_df["t"].values.astype(np.float32)[tgt])
        name = test_df["file_name"].values[tgt].astype(str) if "file_name" in test_df else (
            np.full(len(Xw), "unknown", object))
        file_parts.append(name)
        Xw_parts.append(Xw)

    y_true   = np.concatenate(y_true_parts)
    torEst   = np.concatenate(torEst_parts)
    t        = np.concatenate(t_parts)
    file_name = np.concatenate(file_parts)
    X_windows = np.concatenate(Xw_parts)
    return y_true, torEst, t, file_name, X_windows


# ── Regime analysis (Test 2) ──────────────────────────────────────────────────

def rolling_std_per_file(y, file_name, window=ROLLING_WINDOW_SAMPLES):
    df = pd.DataFrame({"y": y, "file": file_name})
    return (df.groupby("file", sort=False)["y"]
              .transform(lambda s: s.rolling(window, center=True, min_periods=1).std())
              .values)


def classify_regimes(y_true, file_name):
    rstd = rolling_std_per_file(y_true, file_name)
    regimes = np.full(len(y_true), "transition", dtype=object)
    osc   = rstd > OSCILLATION_STD_NM
    quiet = rstd < REST_STD_NM
    nz    = np.abs(y_true) > HOLD_ABS_TORQUE_NM
    regimes[quiet & nz]   = "hold"
    regimes[quiet & ~nz]  = "rest"
    regimes[osc]          = "oscillation"
    return regimes


def per_regime_metrics(y_true, y_pred, regimes):
    rows = {}
    for r in REGIMES:
        mask = regimes == r
        n = int(mask.sum())
        if n == 0:
            rows[r] = {"count": 0, "rmse": float("nan")}
        else:
            err = y_pred[mask] - y_true[mask]
            rows[r] = {"count": n, "rmse": float(np.sqrt(np.mean(err ** 2)))}
    return rows


def overall_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


# ── HF power ratio (Test 3) ───────────────────────────────────────────────────

def hf_power_ratio(x, fs, cutoff):
    f, pxx = welch(x, fs=fs, nperseg=min(4096, len(x)))
    total   = float(np.trapezoid(pxx, f))
    hf_mask = f >= cutoff
    hf      = float(np.trapezoid(pxx[hf_mask], f[hf_mask])) if hf_mask.any() else 0.0
    return hf / total if total > 0 else float("nan")


# ── Load v1 metrics from disk ─────────────────────────────────────────────────

def load_v1_metrics():
    """Read v1 overall RMSE and per-regime RMSE from saved CSVs."""
    def read_overall(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["metric"] == "RMSE [Nm]":
                    return float(list(row.values())[1])
        raise ValueError(f"RMSE not found in {path}")

    def read_regime(path):
        result = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                result[row["regime"]] = float(row["rmse_nm"])
        return result

    mlp_rmse = read_overall(V1_RESULTS_DIR / "metrics_mlp.csv")
    gru_rmse = read_overall(V1_RESULTS_DIR / "metrics_gru.csv")
    mlp_reg  = read_regime(V1_DIAG_DIR / "test2_regime_metrics_mlp.csv")
    gru_reg  = read_regime(V1_DIAG_DIR / "test2_regime_metrics_gru.csv")
    return mlp_rmse, gru_rmse, mlp_reg, gru_reg


# ── Regime plot ───────────────────────────────────────────────────────────────

def plot_regime_bars(v1_mlp, v1_gru, v2_mlp, v2_gru, path):
    x = np.arange(len(REGIMES))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*w, [v1_mlp[r]["rmse"] for r in REGIMES], w,
           label="MLP v1", color="tomato",   alpha=0.7)
    ax.bar(x - 0.5*w, [v2_mlp[r]["rmse"] for r in REGIMES], w,
           label="MLP v2", color="tomato")
    ax.bar(x + 0.5*w, [v1_gru[r]["rmse"] for r in REGIMES], w,
           label="GRU v1", color="seagreen", alpha=0.7)
    ax.bar(x + 1.5*w, [v2_gru[r]["rmse"] for r in REGIMES], w,
           label="GRU v2", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(REGIMES)
    ax.set_ylabel("RMSE [Nm]")
    ax.set_title("Per-regime RMSE: v1 (raw accelAct) vs v2 (SG-smoothed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_comparison_md(
    v1_mlp_rmse, v2_mlp_rmse, v1_gru_rmse, v2_gru_rmse,
    v1_mlp_reg, v2_mlp_reg, v1_gru_reg, v2_gru_reg,
    vel_hf, acc_hf_raw, acc_hf_smooth,
    verdict: str,
    path: Path,
):
    def delta(v1, v2):
        return f"{v2 - v1:+.2f} Nm ({(v2 - v1)/v1*100:+.1f}%)"

    lines = [
        "# v1 vs v2 Comparison: accelAct Savitzky-Golay Smoothing",
        "",
        "## Configuration",
        f"- `SMOOTH_ACCEL = True`",
        f"- `SG_WINDOW = {config.SG_WINDOW}`, `SG_POLYORDER = {config.SG_POLYORDER}`",
        f"- Smoothing applied per-file before windowing (`mode='interp'`)",
        "",
        "## Overall RMSE",
        "",
        "| Model | v1 RMSE | v2 RMSE | Δ |",
        "|-------|---------|---------|---|",
        f"| MLP   | {v1_mlp_rmse:.2f} Nm | {v2_mlp_rmse:.2f} Nm | {delta(v1_mlp_rmse, v2_mlp_rmse)} |",
        f"| GRU   | {v1_gru_rmse:.2f} Nm | {v2_gru_rmse:.2f} Nm | {delta(v1_gru_rmse, v2_gru_rmse)} |",
        "",
        "## Per-regime RMSE",
        "",
        "| Model | Version | rest | hold | transition | oscillation |",
        "|-------|---------|------|------|------------|-------------|",
    ]
    for mt, v1r, v2r in [("MLP", v1_mlp_reg, v2_mlp_reg), ("GRU", v1_gru_reg, v2_gru_reg)]:
        def f(d, r):
            v = d[r]["rmse"]
            return f"{v:.2f}" if not np.isnan(v) else "n/a"
        lines.append(f"| {mt} | v1 | {f(v1r,'rest')} | {f(v1r,'hold')} | {f(v1r,'transition')} | {f(v1r,'oscillation')} |")
        lines.append(f"| {mt} | v2 | {f(v2r,'rest')} | {f(v2r,'hold')} | {f(v2r,'transition')} | {f(v2r,'oscillation')} |")
    lines += [
        "",
        "## Test 3 Re-run: HF Power Ratio (>250 Hz)",
        "",
        f"| Signal   | HF fraction | Ratio accel/vel |",
        f"|----------|-------------|-----------------|",
        f"| velAct   | {vel_hf:.4f}      | —               |",
        f"| accelAct (raw)    | {acc_hf_raw:.4f}      | {acc_hf_raw/vel_hf:.1f}× |",
        f"| accelAct (v2 SG) | {acc_hf_smooth:.4f}      | {acc_hf_smooth/vel_hf:.1f}× |",
        "",
        f"Threshold: {HF_RATIO_MULTIPLIER}×.  "
        f"{'PASS' if acc_hf_smooth/vel_hf < HF_RATIO_MULTIPLIER else 'FAIL'}: "
        f"SG smoothing reduced the HF ratio from {acc_hf_raw/vel_hf:.1f}× to "
        f"{acc_hf_smooth/vel_hf:.1f}×.",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "### Interpretation",
        "- Smoothing **helped** if overall RMSE improves >5% AND osc RMSE improves.",
        "- Smoothing **neutral** if overall RMSE change <5% in either direction.",
        "- Smoothing **hurt** if overall RMSE degrades >5%.",
        "",
        "## Thesis-Ready Description",
        "",
        (
            f"The EPOS4 motor controller exposes joint acceleration as the "
            f"'Sensor Fusion Filtered Fused Acceleration' value from register 0x4C01/6, "
            f"but spectral analysis (Welch PSD, Test 3) revealed that the built-in "
            f"controller filter is insufficient at 1 kHz logging: the high-frequency "
            f"(>250 Hz) power fraction of `accelAct` was {acc_hf_raw/vel_hf:.0f}× "
            f"that of `velAct`, indicating the signal is noise-dominated near Nyquist. "
            f"To mitigate this, a Savitzky-Golay pre-filter (window={config.SG_WINDOW} samples, "
            f"polynomial order {config.SG_POLYORDER}, `mode='interp'`) was applied to "
            f"`accelAct` on a per-file basis before sliding-window construction, "
            f"reducing the HF power ratio from {acc_hf_raw/vel_hf:.0f}× to "
            f"{acc_hf_smooth/vel_hf:.1f}× (threshold: {HF_RATIO_MULTIPLIER}×). "
            f"Models retrained on the smoothed signal (v2) achieved overall RMSE of "
            f"{v2_mlp_rmse:.2f} Nm (MLP) and {v2_gru_rmse:.2f} Nm (GRU), "
            f"compared with {v1_mlp_rmse:.2f} Nm and {v1_gru_rmse:.2f} Nm for v1."
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


# ── SUMMARY+CONCLUSION append ─────────────────────────────────────────────────

def append_summary(verdict: str, v1_mlp_rmse, v2_mlp_rmse,
                   v1_gru_rmse, v2_gru_rmse, acc_hf_raw, acc_hf_smooth, vel_hf):
    summary_path = PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"
    block = (
        "\n"
        "========================================================================\n"
        "[Prompt 5 — Acceleration Smoothing]\n"
        "========================================================================\n"
        "Purpose: Reduce accelAct HF noise (>250 Hz power fraction 542× velAct,\n"
        "measured in Test 3 from register 0x4C01/6) via Savitzky-Golay smoothing\n"
        f"(window={config.SG_WINDOW}, polyorder={config.SG_POLYORDER}) applied per-file before windowing.\n"
        "\n"
        f"HF ratio: {acc_hf_raw/vel_hf:.1f}× → {acc_hf_smooth/vel_hf:.1f}× after smoothing "
        f"({'PASS' if acc_hf_smooth/vel_hf < HF_RATIO_MULTIPLIER else 'FAIL'}, threshold={HF_RATIO_MULTIPLIER}×).\n"
        f"Overall RMSE: MLP {v1_mlp_rmse:.2f}→{v2_mlp_rmse:.2f} Nm, "
        f"GRU {v1_gru_rmse:.2f}→{v2_gru_rmse:.2f} Nm.\n"
        "\n"
        f"Verdict: {verdict}\n"
        f"SMOOTH_ACCEL defaults to {'True (smoothing retained)' if 'helped' in verdict.lower() or 'neutral' in verdict.lower() else 'False (smoothing reverted)'}.\n"
    )
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"Appended to {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    print("=" * 60)
    print("Prompt 5 — Acceleration Smoothing (v2)")
    print(f"SMOOTH_ACCEL={config.SMOOTH_ACCEL}, SG_WINDOW={config.SG_WINDOW}, "
          f"SG_POLYORDER={config.SG_POLYORDER}")
    print(f"Checkpoints → {V2_CKPT_DIR}")
    print(f"Results     → {V2_RESULTS_DIR}")
    print("=" * 60)

    # ── 1. Build datasets (smoothed) ─────────────────────────────────────────
    print("\n[1] Loading and preprocessing data (SMOOTH_ACCEL=True) …")
    train_ds, val_ds, test_ds, scaler_X, scaler_y = build_datasets(save_scalers=True)
    print(f"    Train windows: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # ── 2. Train MLP ─────────────────────────────────────────────────────────
    print("\n[2] Training MLP v2 …")
    mlp = train_one("mlp", train_ds, val_ds, device, args.force)

    # ── 3. Train GRU ─────────────────────────────────────────────────────────
    print("\n[3] Training GRU v2 …")
    gru = train_one("gru", train_ds, val_ds, device, args.force)

    # ── 4. Build test windows ─────────────────────────────────────────────────
    print("\n[4] Building test windows …")
    dfs = _load_dataframes()
    y_true, torEst, t, file_name, X_windows = build_test_windows(
        dfs, scaler_X, scaler_y
    )

    # ── 5. Inference ──────────────────────────────────────────────────────────
    print("[5] Running inference …")
    pred_mlp_sc = batched_infer(mlp, X_windows, device)
    pred_gru_sc = batched_infer(gru, X_windows, device)
    pred_mlp = scaler_y.inverse_transform(pred_mlp_sc.reshape(-1, 1)).ravel()
    pred_gru = scaler_y.inverse_transform(pred_gru_sc.reshape(-1, 1)).ravel()

    v2_mlp_rmse = overall_rmse(y_true, pred_mlp)
    v2_gru_rmse = overall_rmse(y_true, pred_gru)
    print(f"    v2 MLP RMSE: {v2_mlp_rmse:.4f} Nm")
    print(f"    v2 GRU RMSE: {v2_gru_rmse:.4f} Nm")

    # Save v2 overall metrics CSVs
    for model_type, rmse_val, pred in [
        ("mlp", v2_mlp_rmse, pred_mlp), ("gru", v2_gru_rmse, pred_gru)
    ]:
        err = pred - y_true
        rows_m = {
            "RMSE [Nm]":          rmse_val,
            "MAE [Nm]":           float(np.mean(np.abs(err))),
            "Max Abs Error [Nm]": float(np.max(np.abs(err))),
        }
        mpath = V2_RESULTS_DIR / f"metrics_{model_type}.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", model_type.upper()])
            for k, v in rows_m.items():
                w.writerow([k, f"{v:.6f}"])
        print(f"    Saved {mpath}")

    # ── 6. Regime analysis (Test 2) ───────────────────────────────────────────
    print("\n[6] Classifying regimes …")
    regimes = classify_regimes(y_true, file_name)
    counts = {r: int((regimes == r).sum()) for r in REGIMES}
    print(f"    Regime counts: {counts}")

    v2_mlp_reg = per_regime_metrics(y_true, pred_mlp, regimes)
    v2_gru_reg = per_regime_metrics(y_true, pred_gru, regimes)

    for model_type, reg in [("mlp", v2_mlp_reg), ("gru", v2_gru_reg)]:
        rpath = V2_RESULTS_DIR / f"test2_regime_metrics_{model_type}.csv"
        with open(rpath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["regime", "count", "rmse_nm"])
            for r in REGIMES:
                w.writerow([r, reg[r]["count"],
                             f"{reg[r]['rmse']:.6f}" if not np.isnan(reg[r]["rmse"]) else "nan"])
        print(f"    Saved {rpath}")

    # ── 7. Load v1 metrics ────────────────────────────────────────────────────
    print("\n[7] Loading v1 metrics …")
    v1_mlp_rmse, v1_gru_rmse, v1_mlp_reg_raw, v1_gru_reg_raw = load_v1_metrics()
    v1_mlp_reg = {r: {"rmse": v1_mlp_reg_raw[r], "count": 0} for r in REGIMES}
    v1_gru_reg = {r: {"rmse": v1_gru_reg_raw[r], "count": 0} for r in REGIMES}
    print(f"    v1 MLP RMSE: {v1_mlp_rmse:.4f} Nm   v1 GRU RMSE: {v1_gru_rmse:.4f} Nm")

    # ── 8. Comparison table (terminal) ────────────────────────────────────────
    print("\n[8] Comparison table:")
    hdr = f"{'Model':<6} | {'v1 RMSE':>8} | {'v2 RMSE':>8} | {'Δ RMSE':>10} | {'v1 hold':>8} | {'v2 hold':>8} | {'v1 osc':>8} | {'v2 osc':>8}"
    print(hdr)
    print("-" * len(hdr))
    for mt, v1r, v2r, v1reg, v2reg in [
        ("MLP", v1_mlp_rmse, v2_mlp_rmse, v1_mlp_reg, v2_mlp_reg),
        ("GRU", v1_gru_rmse, v2_gru_rmse, v1_gru_reg, v2_gru_reg),
    ]:
        d = v2r - v1r
        pct = d / v1r * 100
        print(
            f"{mt:<6} | {v1r:>8.2f} | {v2r:>8.2f} | {d:>+6.2f} ({pct:>+5.1f}%) "
            f"| {v1reg['hold']['rmse']:>8.2f} | {v2reg['hold']['rmse']:>8.2f} "
            f"| {v1reg['oscillation']['rmse']:>8.2f} | {v2reg['oscillation']['rmse']:>8.2f}"
        )

    # ── 9. Test 3 re-run: HF power ratio on smoothed signal ──────────────────
    print("\n[9] Test 3 re-run: HF power ratio …")
    # Load raw (unsmoothed) accelAct for comparison: temporarily disable smoothing
    config.SMOOTH_ACCEL = False
    dfs_raw = _load_dataframes()
    config.SMOOTH_ACCEL = True

    df0_raw    = dfs_raw[0]
    df0_smooth = dfs[0]   # already smoothed (from step 4 load)

    train_raw, _, _ = _split_df(df0_raw,    config.TRAIN_RATIO, config.VAL_RATIO, config.SEQ_LEN)
    train_sm,  _, _ = _split_df(df0_smooth, config.TRAIN_RATIO, config.VAL_RATIO, config.SEQ_LEN)

    vel_arr       = train_raw["velAct"].values.astype(np.float64)
    acc_raw_arr   = train_raw["accelAct"].values.astype(np.float64)
    acc_sm_arr    = train_sm["accelAct"].values.astype(np.float64)

    vel_hf      = hf_power_ratio(vel_arr,     SAMPLE_HZ, HF_CUTOFF_HZ)
    acc_hf_raw  = hf_power_ratio(acc_raw_arr, SAMPLE_HZ, HF_CUTOFF_HZ)
    acc_hf_sm   = hf_power_ratio(acc_sm_arr,  SAMPLE_HZ, HF_CUTOFF_HZ)

    ratio_raw = acc_hf_raw / vel_hf if vel_hf > 0 else float("inf")
    ratio_sm  = acc_hf_sm  / vel_hf if vel_hf > 0 else float("inf")
    print(f"    velAct HF fraction:            {vel_hf:.4f}")
    print(f"    accelAct (raw)  HF fraction:   {acc_hf_raw:.4f}  ratio={ratio_raw:.1f}×")
    print(f"    accelAct (SG)   HF fraction:   {acc_hf_sm:.4f}  ratio={ratio_sm:.1f}×")
    hf_pass = ratio_sm < HF_RATIO_MULTIPLIER
    print(f"    {'PASS' if hf_pass else 'FAIL'}: ratio {'<' if hf_pass else '>='} {HF_RATIO_MULTIPLIER}×")

    # Save Test 3 HF CSV for v2
    hf_csv = V2_RESULTS_DIR / "test3_hf_power_ratio.csv"
    with open(hf_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "hf_cutoff_hz", "hf_fraction", "ratio_vs_velAct"])
        w.writerow(["velAct",          HF_CUTOFF_HZ, f"{vel_hf:.6f}",     "1.00"])
        w.writerow(["accelAct (raw)",  HF_CUTOFF_HZ, f"{acc_hf_raw:.6f}", f"{ratio_raw:.2f}"])
        w.writerow(["accelAct (v2 SG)", HF_CUTOFF_HZ, f"{acc_hf_sm:.6f}", f"{ratio_sm:.2f}"])
    print(f"    Saved {hf_csv}")

    # ── 10. Verdict ───────────────────────────────────────────────────────────
    mlp_delta_pct = (v2_mlp_rmse - v1_mlp_rmse) / v1_mlp_rmse * 100
    gru_delta_pct = (v2_gru_rmse - v1_gru_rmse) / v1_gru_rmse * 100
    mlp_osc_imp   = v2_mlp_reg["oscillation"]["rmse"] < v1_mlp_reg["oscillation"]["rmse"]
    gru_osc_imp   = v2_gru_reg["oscillation"]["rmse"] < v1_gru_reg["oscillation"]["rmse"]
    both_improved = (mlp_delta_pct < -5) and (gru_delta_pct < -5) and mlp_osc_imp and gru_osc_imp
    either_degraded = (mlp_delta_pct > 5) or (gru_delta_pct > 5)

    if both_improved:
        verdict = "Smoothing HELPED (overall RMSE improved >5% and osc RMSE improved for both models)"
    elif either_degraded:
        verdict = "Smoothing HURT (overall RMSE degraded >5% — HF accelAct content carried useful signal)"
    else:
        verdict = "Smoothing NEUTRAL (overall RMSE change <5% in either direction)"

    print(f"\n[Verdict] {verdict}")

    # ── 11. Regime bar plot ───────────────────────────────────────────────────
    plot_path = V2_RESULTS_DIR / "test2_regime_rmse_bars.png"
    plot_regime_bars(v1_mlp_reg, v1_gru_reg, v2_mlp_reg, v2_gru_reg, plot_path)

    # ── 12. Write comparison markdown ─────────────────────────────────────────
    md_path = V2_RESULTS_DIR / "v1_vs_v2_comparison.md"
    write_comparison_md(
        v1_mlp_rmse, v2_mlp_rmse, v1_gru_rmse, v2_gru_rmse,
        v1_mlp_reg, v2_mlp_reg, v1_gru_reg, v2_gru_reg,
        vel_hf, acc_hf_raw, acc_hf_sm,
        verdict, md_path,
    )

    # ── 13. Append to SUMMARY+CONCLUSION.txt ─────────────────────────────────
    append_summary(
        verdict, v1_mlp_rmse, v2_mlp_rmse,
        v1_gru_rmse, v2_gru_rmse, acc_hf_raw, acc_hf_sm, vel_hf,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

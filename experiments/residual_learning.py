"""Prompt 10 — Residual learning (torAct - torEst).

Trains MLP and GRU to predict the residual torAct-torEst, then reconstructs
predicted torque as residual_pred + torEst at inference. Evaluates whether
this framing reduces RMSE vs v1, especially on PLC and PMS profiles.

Outputs
-------
experiments/outputs/residual_comparison.md
experiments/outputs/residual_timeseries.png
experiments/outputs/residual_histogram.png
experiments/outputs/residual_profile_bar.png  (if asymmetric benefit)
prompts/SUMMARY+CONCLUSION.txt  (appended)

Checkpoints: checkpoints/residual/
Scalers:     checkpoints/residual/scaler_{X,y}.pkl

Usage
-----
    python experiments/residual_learning.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing import _get_feature_cols, _load_dataframes, _make_windows, _split_df
from models import ActuatorGRU, WindowedMLP

RESIDUAL_CKPT_DIR = config.PROJECT_ROOT / "checkpoints" / "residual"
OUTPUTS_DIR       = Path(__file__).resolve().parent / "outputs"
SUMMARY_CONC      = config.PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"

RESIDUAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = ["PLC", "PMS", "TMS", "tStep"]
PROFILE_COLORS = {"PLC": "steelblue", "PMS": "seagreen",
                  "TMS": "darkorange", "tStep": "tomato"}
PROFILE_FLAGS = {
    "PLC":   {"USE_PLC": True,  "USE_PMS": False, "USE_TMS": False, "USE_tStep": False},
    "PMS":   {"USE_PLC": False, "USE_PMS": True,  "USE_TMS": False, "USE_tStep": False},
    "TMS":   {"USE_PLC": False, "USE_PMS": False, "USE_TMS": True,  "USE_tStep": False},
    "tStep": {"USE_PLC": False, "USE_PMS": False, "USE_TMS": False, "USE_tStep": True},
}

V1_FEATURE_COLS = [
    'torDes', 'posDes', 'velDes', 'posAct', 'velAct',
    'accelAct', 'i', 'torEst', 'posErr', 'velErr',
]
V1_RMSE = {
    "mlp": {"overall": 13.53, "mae": 4.13, "max_err": 223.5,
            "hold": 64.99, "osc": 12.92,
            "tStep": 1.73, "TMS": 4.73, "PMS": 8.18, "PLC": 51.31},
    "gru": {"overall": 13.66, "mae": 4.04, "max_err": 225.4,
            "hold": 65.97, "osc": 12.99, "err_38180": 215.8, "mean_severe": 147.3,
            "tStep": 0.37, "TMS": 4.56, "PMS": 8.08, "PLC": 67.25},
}

SAMPLE_HZ = 1000.0


# ── CLI / device ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="Retrain from scratch even if checkpoints exist.")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_all_dfs(smooth_accel: bool | None = None) -> list:
    """Load all-profile DFs with optional SMOOTH_ACCEL override."""
    patch = {}
    if smooth_accel is not None:
        patch["SMOOTH_ACCEL"] = smooth_accel
    saved = {k: getattr(config, k) for k in patch}
    for k, v in patch.items():
        setattr(config, k, v)
    try:
        return _load_dataframes()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def load_profile_dfs(profile: str, smooth_accel: bool | None = None) -> list:
    flags = dict(PROFILE_FLAGS[profile])
    if smooth_accel is not None:
        flags["SMOOTH_ACCEL"] = smooth_accel
    saved = {k: getattr(config, k) for k in flags}
    for k, v in flags.items():
        setattr(config, k, v)
    try:
        return _load_dataframes()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def build_train_arrays(dfs, feat_cols):
    """Return (X_parts, residual_y_parts, torEst_parts) for training split."""
    seq_len = config.SEQ_LEN
    Xp, yp, tep = [], [], []
    for df in dfs:
        train_df, _, _ = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        Xp.append(train_df[feat_cols].values.astype(np.float32))
        residual = (train_df[config.TARGET_COL] - train_df['torEst']).values.astype(np.float32)
        yp.append(residual)
        tep.append(train_df['torEst'].values.astype(np.float32))
    return Xp, yp, tep


def build_val_arrays(dfs, feat_cols):
    seq_len = config.SEQ_LEN
    Xp, yp = [], []
    for df in dfs:
        _, val_df, _ = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        if len(val_df) < seq_len:
            continue
        Xp.append(val_df[feat_cols].values.astype(np.float32))
        residual = (val_df[config.TARGET_COL] - val_df['torEst']).values.astype(np.float32)
        yp.append(residual)
    return Xp, yp


def build_test_with_torest(dfs, feat_cols, sx, sy):
    """Return (y_true, torEst_arr, X_wins, file_name) for test split.

    y_true  : torAct in Nm (reconstruction target, not residual)
    torEst_arr: torEst at each prediction timestep, aligned with y_true
    X_wins  : (N, seq_len, n_features), scaled
    file_name: for per-profile stratification
    """
    seq_len = config.SEQ_LEN
    y_parts, te_parts, Xw_parts, fn_parts = [], [], [], []
    for df in dfs:
        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        if len(test_df) < seq_len:
            continue
        X_raw  = test_df[feat_cols].values.astype(np.float32)
        y_raw  = test_df[config.TARGET_COL].values.astype(np.float32)
        te_raw = test_df['torEst'].values.astype(np.float32)
        Xs = sx.transform(X_raw).astype(np.float32)
        Xw, _ = _make_windows(Xs, np.zeros(len(Xs), np.float32), seq_len)
        tgt = slice(seq_len - 1, len(test_df))
        y_parts.append(y_raw[tgt])
        te_parts.append(te_raw[tgt])
        Xw_parts.append(Xw)
        fn = (test_df["file_name"].values[tgt].astype(str)
              if "file_name" in test_df.columns else
              np.full(len(Xw), "unknown", object))
        fn_parts.append(fn)
    return (np.concatenate(y_parts), np.concatenate(te_parts),
            np.concatenate(Xw_parts), np.concatenate(fn_parts))


def make_windows_scaled(X_parts, y_parts, sx, sy):
    """Scale and build sliding windows; return concatenated arrays."""
    seq_len = config.SEQ_LEN
    Xw_list, yw_list = [], []
    for X, y in zip(X_parts, y_parts):
        if len(X) < seq_len:
            continue
        Xs = sx.transform(X).astype(np.float32)
        ys = sy.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
        Xw, yw = _make_windows(Xs, ys, seq_len)
        if len(Xw):
            Xw_list.append(Xw)
            yw_list.append(yw)
    return np.concatenate(Xw_list), np.concatenate(yw_list)


# ── Model helpers ─────────────────────────────────────────────────────────────

def build_model(arch: str, n_features: int) -> nn.Module:
    if arch == "mlp":
        return WindowedMLP(seq_len=config.SEQ_LEN, n_features=n_features,
                           hidden_size=config.MLP_HIDDEN_SIZE,
                           n_layers=config.MLP_N_LAYERS)
    return ActuatorGRU(n_features=n_features, hidden_size=config.GRU_HIDDEN_SIZE,
                       n_layers=config.GRU_N_LAYERS, dropout=config.GRU_DROPOUT)


def save_ckpt(model, arch, n_features, epoch, val_loss, path):
    hidden = config.MLP_HIDDEN_SIZE if arch == "mlp" else config.GRU_HIDDEN_SIZE
    n_lay  = config.MLP_N_LAYERS    if arch == "mlp" else config.GRU_N_LAYERS
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "val_loss": val_loss, "model_type": arch,
        "config": {"seq_len": config.SEQ_LEN, "n_features": n_features,
                   "hidden_size": hidden, "n_layers": n_lay},
    }, path)


def load_ckpt(path, arch, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    if arch == "mlp":
        m = WindowedMLP(seq_len=cfg["seq_len"], n_features=cfg["n_features"],
                        hidden_size=cfg["hidden_size"], n_layers=cfg["n_layers"])
    else:
        m = ActuatorGRU(n_features=cfg["n_features"], hidden_size=cfg["hidden_size"],
                        n_layers=cfg["n_layers"])
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval(), ckpt["epoch"], ckpt["val_loss"]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(arch: str, Xw_tr: np.ndarray, yw_tr: np.ndarray,
                Xw_va: np.ndarray, yw_va: np.ndarray,
                n_features: int, device, ckpt_path: Path,
                force: bool) -> tuple[nn.Module, list, int, float]:
    if ckpt_path.exists() and not force:
        print(f"  [{arch.upper()}] checkpoint found — skipping training.")
        model, ep, val = load_ckpt(ckpt_path, arch, device)
        return model, [], ep, val

    from torch.utils.data import DataLoader as TDL, TensorDataset
    tr_ds = TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(yw_tr).unsqueeze(1))
    va_ds = TensorDataset(torch.from_numpy(Xw_va), torch.from_numpy(yw_va).unsqueeze(1))
    tr_ld = TDL(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    va_ld = TDL(va_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    model = build_model(arch, n_features).to(device)
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=config.LR,
                              weight_decay=config.WEIGHT_DECAY)
    total_steps = len(tr_ld) * config.MAX_EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.ONE_CYCLE_MAX_LR, total_steps=total_steps)

    best_val, pat_ctr, best_ep = float("inf"), 0, 1
    history = []

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            opt.step()
            sched.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(tr_ds)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for Xb, yb in va_ld:
                vl_loss += crit(model(Xb.to(device)), yb.to(device)).item() * len(Xb)
        vl_loss /= len(va_ds)
        history.append((epoch, tr_loss, vl_loss))

        improved = vl_loss < best_val
        tag = " [saved]" if improved else f" (pat {pat_ctr+1}/{config.PATIENCE})"
        print(f"    Ep {epoch:4d} | tr {tr_loss:.5f} | val {vl_loss:.5f}{tag}")

        if improved:
            best_val, pat_ctr, best_ep = vl_loss, 0, epoch
            save_ckpt(model, arch, n_features, epoch, vl_loss, ckpt_path)
        else:
            pat_ctr += 1
            if pat_ctr >= config.PATIENCE:
                print(f"    Early stop at epoch {epoch}.")
                break

    print(f"  [{arch.upper()}] best val MSE={best_val:.6f} @ ep {best_ep}")
    model, best_ep, best_val = load_ckpt(ckpt_path, arch, device)
    return model, history, best_ep, best_val


# ── Inference ─────────────────────────────────────────────────────────────────

def batched_infer(model, X: np.ndarray, device, batch=2048) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            e = min(s + batch, len(X))
            xb = torch.from_numpy(X[s:e]).float().to(device)
            out[s:e] = model(xb).cpu().numpy().ravel()
    return out


def reconstruct_torque(model, X_wins, torEst_arr, sy, device) -> np.ndarray:
    pred_sc  = batched_infer(model, X_wins, device)
    residual = sy.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    return residual + torEst_arr


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(model, Xw_tr: np.ndarray, yw_tr_scaled: np.ndarray,
                 sy: StandardScaler, device, label: str):
    """Assert RMSE_recon ≈ sy.scale_[0] * RMSE_scaled (within 1%)."""
    pred_sc  = batched_infer(model, Xw_tr, device)
    rmse_sc  = float(np.sqrt(np.mean((pred_sc - yw_tr_scaled) ** 2)))
    expected = float(sy.scale_[0]) * rmse_sc
    residual_pred = sy.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    true_residual = sy.inverse_transform(yw_tr_scaled.reshape(-1, 1)).ravel()
    rmse_recon    = float(np.sqrt(np.mean((residual_pred - true_residual) ** 2)))
    rel_err = abs(rmse_recon / expected - 1.0) if expected > 0 else 0.0
    print(f"  [{label}] Sanity check: RMSE_scaled={rmse_sc:.6f}, "
          f"expected={expected:.4f} Nm, actual={rmse_recon:.4f} Nm, "
          f"rel_err={rel_err*100:.4f}%")
    assert rel_err < 0.01, (
        f"{label} reconstruction sanity check FAILED: rel_err={rel_err:.4f} > 1%"
    )
    print(f"  [{label}] Sanity check PASSED.")


# ── Residual distribution stats ───────────────────────────────────────────────

def compute_residual_stats(residuals: np.ndarray) -> dict:
    return {
        "mean":     float(np.mean(residuals)),
        "std":      float(np.std(residuals)),
        "skew":     float(scipy.stats.skew(residuals)),
        "kurtosis": float(scipy.stats.kurtosis(residuals)),
        "min":      float(np.min(residuals)),
        "max":      float(np.max(residuals)),
    }


# ── v1 baseline evaluation (combined test set) ────────────────────────────────

def eval_v1_combined(feat_cols, sx_res, sy_res, device) -> dict:
    """Evaluate v1 checkpoints on the combined test set (SMOOTH_ACCEL=False).
    Returns dict with predictions and metrics for both architectures."""
    v1_dir  = config.PROJECT_ROOT / "checkpoints"
    sx_path = v1_dir / "scaler_X.pkl"
    sy_path = v1_dir / "scaler_y.pkl"

    if not all((v1_dir / f"best_model_{a}.pt").exists() for a in ["mlp", "gru"]):
        print("  v1 checkpoints missing — v1 baseline from spec only.")
        return {}
    if not (sx_path.exists() and sy_path.exists()):
        print("  v1 scalers missing — v1 baseline from spec only.")
        return {}

    sx_v1 = joblib.load(sx_path)
    sy_v1 = joblib.load(sy_path)
    if sx_v1.n_features_in_ != len(V1_FEATURE_COLS):
        print(f"  v1 scaler has {sx_v1.n_features_in_} features != "
              f"{len(V1_FEATURE_COLS)} expected — using spec.")
        return {}

    dfs_v1 = load_all_dfs(smooth_accel=False)
    y_true_v1, torEst_v1, Xw_v1, fn_v1 = build_test_with_torest(
        dfs_v1, V1_FEATURE_COLS, sx_v1, sy_v1)

    result = {}
    for arch in ["mlp", "gru"]:
        ckpt = load_ckpt(v1_dir / f"best_model_{arch}.pt", arch, device)[0]
        pred_sc_v1 = batched_infer(ckpt, Xw_v1, device)
        pred_v1    = sy_v1.inverse_transform(pred_sc_v1.reshape(-1, 1)).ravel()
        result[arch] = {
            "pred":    pred_v1,
            "y_true":  y_true_v1,
            "torEst":  torEst_v1,
            "fn":      fn_v1,
            "rmse":    rmse(y_true_v1, pred_v1),
            "mae":     mae(y_true_v1, pred_v1),
            "max_err": float(np.max(np.abs(y_true_v1 - pred_v1))),
        }
        print(f"  v1 {arch.upper()} overall RMSE={result[arch]['rmse']:.4f} Nm")
    return result


def per_profile_rmse(y_true, pred, fn_arr) -> dict:
    """Stratify by profile prefix in file_name and return RMSE per profile."""
    out = {}
    for prof in PROFILES:
        mask = np.array([prof in fn for fn in fn_arr])
        if mask.sum() == 0:
            out[prof] = float("nan")
        else:
            out[prof] = rmse(y_true[mask], pred[mask])
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_timeseries(y_true, torEst, pred_v1_gru, pred_res_gru, path: Path,
                   idx_start: int = 38150, idx_end: int = 38250):
    n = len(y_true)
    if idx_end <= n:
        s, e = idx_start, idx_end
    else:
        s = max(0, n - 100)
        e = n
        print(f"  Time-series: only {n} test samples — plotting [{s}:{e}] instead.")

    t = np.arange(e - s) / SAMPLE_HZ * 1000  # ms

    fig, axes = plt.subplots(2, 1, figsize=(13, 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(t, y_true[s:e],        label="torAct (truth)",   color="steelblue", lw=1.4)
    ax.plot(t, torEst[s:e],        label="torEst (baseline)", color="grey",      lw=1.0, ls=":")
    if pred_v1_gru is not None:
        nv = len(pred_v1_gru)
        if e <= nv:
            ax.plot(t, pred_v1_gru[s:e],  label="v1 GRU",      color="darkorange", lw=1.0, ls="--")
    ax.plot(t, pred_res_gru[s:e],  label="residual GRU",  color="tomato",     lw=1.2)
    ax.set_ylabel("Torque [Nm]")
    ax.set_title(f"Test samples {s}–{e}: torAct, torEst, v1 GRU, residual GRU")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    if pred_v1_gru is not None and e <= len(pred_v1_gru):
        ax2.plot(t, pred_v1_gru[s:e] - y_true[s:e],
                 label="v1 GRU err",    color="darkorange", lw=0.9, ls="--")
    ax2.plot(t, pred_res_gru[s:e] - y_true[s:e],
             label="residual GRU err", color="tomato",     lw=0.9)
    ax2.plot(t, torEst[s:e] - y_true[s:e],
             label="torEst err",       color="grey",       lw=0.9, ls=":")
    ax2.axhline(0, color="black", lw=0.6, ls="--")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Error [Nm]")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_histogram(residuals: np.ndarray, stats: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=200, color="steelblue", alpha=0.75, density=True)
    ax.axvline(stats["mean"], color="red",    lw=1.2, ls="--", label=f"mean={stats['mean']:.2f}")
    ax.axvline(stats["mean"] + stats["std"], color="orange", lw=1.0, ls=":",
               label=f"±1σ={stats['std']:.2f}")
    ax.axvline(stats["mean"] - stats["std"], color="orange", lw=1.0, ls=":")
    ax.set_xlabel("torAct − torEst  [Nm]")
    ax.set_ylabel("Density")
    ax.set_title(f"Training-set residual distribution  "
                 f"(skew={stats['skew']:.2f}, kurt={stats['kurtosis']:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_profile_bar(v1_pp, res_pp, path: Path):
    """Bar chart: v1 vs residual RMSE per profile."""
    x = np.arange(len(PROFILES))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    v1_vals  = [v1_pp.get(p, float("nan")) for p in PROFILES]
    res_vals = [res_pp.get(p, float("nan")) for p in PROFILES]
    bars1 = ax.bar(x - w/2, v1_vals,  w, label="v1 GRU",      color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + w/2, res_vals, w, label="residual GRU", color="tomato",    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(PROFILES)
    ax.set_ylabel("RMSE [Nm]")
    ax.set_title("Per-profile RMSE: v1 GRU vs residual GRU")
    ax.legend()
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha="center", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(stats, overall_metrics, pp_rmse, corr, verdict, path: Path):
    def f(v): return f"{v:.4f}" if not np.isnan(float(v)) else "n/a"
    def pct(new, old):
        if np.isnan(float(new)) or np.isnan(float(old)) or float(old) == 0:
            return "n/a"
        return f"{(float(new) - float(old)) / float(old) * 100:+.1f}%"

    v1m = V1_RMSE["mlp"]
    v1g = V1_RMSE["gru"]
    om  = overall_metrics  # [arch] -> dict with rmse, mae, max_err

    rm_mlp = om.get("mlp", {})
    rm_gru = om.get("gru", {})

    lines = [
        "# Prompt 10 — Residual Learning (torAct − torEst)",
        "",
        "## Overall (combined test set)",
        "",
        "| Metric | v1 MLP | res MLP | Δ% | v1 GRU | res GRU | Δ% |",
        "|---|---|---|---|---|---|---|",
        f"| Overall RMSE | {v1m['overall']} | {f(rm_mlp.get('rmse', float('nan')))} "
        f"| {pct(rm_mlp.get('rmse', float('nan')), v1m['overall'])} "
        f"| {v1g['overall']} | {f(rm_gru.get('rmse', float('nan')))} "
        f"| {pct(rm_gru.get('rmse', float('nan')), v1g['overall'])} |",
        f"| MAE | {v1m['mae']} | {f(rm_mlp.get('mae', float('nan')))} "
        f"| {pct(rm_mlp.get('mae', float('nan')), v1m['mae'])} "
        f"| {v1g['mae']} | {f(rm_gru.get('mae', float('nan')))} "
        f"| {pct(rm_gru.get('mae', float('nan')), v1g['mae'])} |",
        f"| Max |err| | {v1m['max_err']} | {f(rm_mlp.get('max_err', float('nan')))} "
        f"| {pct(rm_mlp.get('max_err', float('nan')), v1m['max_err'])} "
        f"| {v1g['max_err']} | {f(rm_gru.get('max_err', float('nan')))} "
        f"| {pct(rm_gru.get('max_err', float('nan')), v1g['max_err'])} |",
        f"| Hold RMSE | {v1m['hold']} | n/a | n/a | {v1g['hold']} "
        f"| {f(rm_gru.get('hold_rmse', float('nan')))} "
        f"| {pct(rm_gru.get('hold_rmse', float('nan')), v1g['hold'])} |",
        f"| Err idx=38180 | — | — | — | {v1g['err_38180']} "
        f"| {f(rm_gru.get('err_38180', float('nan')))} "
        f"| {pct(rm_gru.get('err_38180', float('nan')), v1g['err_38180'])} |",
        f"| Mean severe |err| | — | — | — | {v1g['mean_severe']} "
        f"| {f(rm_gru.get('mean_severe', float('nan')))} "
        f"| {pct(rm_gru.get('mean_severe', float('nan')), v1g['mean_severe'])} |",
        f"| corr(pred_mlp, pred_gru) | 0.999 | {f(corr)} | - | - | - | - |",
        "",
        "## Per-profile RMSE [Nm]",
        "",
        "| Profile | Control | v1 MLP | res MLP | Δ% | v1 GRU | res GRU | Δ% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    profile_control = {"tStep": "torque", "TMS": "torque", "PMS": "velocity", "PLC": "velocity"}
    for prof in PROFILES:
        ctrl = profile_control[prof]
        v1mp = v1m.get(prof, float("nan"))
        v1gp = v1g.get(prof, float("nan"))
        rm   = pp_rmse.get("mlp", {}).get(prof, float("nan"))
        rg   = pp_rmse.get("gru", {}).get(prof, float("nan"))
        lines.append(
            f"| {prof} | {ctrl} | {f(v1mp)} | {f(rm)} | {pct(rm, v1mp)} "
            f"| {f(v1gp)} | {f(rg)} | {pct(rg, v1gp)} |"
        )
    lines += [
        "",
        "## Residual target statistics (training set)",
        "",
        "| Statistic | Value |",
        "|---|---|",
    ]
    for k, v in stats.items():
        lines.append(f"| {k} | {v:.4f} |")

    flag_text = ""
    if abs(stats["skew"]) > 2.0:
        flag_text += f"⚠ High skew ({stats['skew']:.2f}): MSE may be biased toward tails. "
    if abs(stats["kurtosis"]) > 10.0:
        flag_text += f"⚠ High kurtosis ({stats['kurtosis']:.2f}): heavy tails may warrant Huber loss."
    if flag_text:
        lines += ["", f"**{flag_text}**"]

    lines += ["", "## Interpretation", "", f"**{verdict}**", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def append_summary(pp_rmse_gru, verdict):
    v1g_overall = V1_RMSE["gru"]["overall"]
    res_overall = pp_rmse_gru.get("overall", float("nan"))
    block = (
        "\n"
        "========================================================================\n"
        "[Prompt 10 — Residual learning (torAct − torEst)]\n"
        "========================================================================\n"
        "Purpose: Test whether training on the residual torAct-torEst and\n"
        "reconstructing at inference reduces RMSE vs v1 direct prediction.\n"
        "Motivated by torEst RMSE=19.8 Nm (Test 1) as a partial predictor.\n"
        "\n"
        f"v1 GRU overall RMSE: {v1g_overall:.2f} Nm\n"
        f"Residual GRU overall RMSE: "
        + (f"{res_overall:.2f} Nm\n" if not np.isnan(res_overall) else "n/a\n")
        + f"Verdict: {verdict}\n"
    )
    with open(SUMMARY_CONC, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"Appended to {SUMMARY_CONC}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    feat_cols  = _get_feature_cols()
    n_features = len(feat_cols)
    seq_len    = config.SEQ_LEN

    print("=" * 65)
    print("Prompt 10 — Residual learning")
    print(f"Device    : {device}")
    print(f"Features  : {n_features}  {feat_cols}")
    print(f"SEQ_LEN   : {seq_len}  |  SMOOTH_ACCEL: {config.SMOOTH_ACCEL}")
    print(f"Checkpoint: {RESIDUAL_CKPT_DIR}")
    print("=" * 65)

    # ── 1. Load all data ──────────────────────────────────────────────────────
    print("\n[1] Loading data …")
    dfs_all = load_all_dfs()
    print(f"  Loaded {len(dfs_all)} files.")

    # ── 2. Build train/val arrays with residual targets ───────────────────────
    print("\n[2] Building train/val arrays (residual targets) …")
    tr_Xp, tr_yp, _ = build_train_arrays(dfs_all, feat_cols)
    va_Xp, va_yp    = build_val_arrays(dfs_all, feat_cols)

    # Fit scalers on training residuals
    X_tr_all = np.concatenate(tr_Xp)
    y_tr_all = np.concatenate(tr_yp)  # training residuals (Nm)
    sx = StandardScaler().fit(X_tr_all)
    sy = StandardScaler().fit(y_tr_all.reshape(-1, 1))
    joblib.dump(sx, RESIDUAL_CKPT_DIR / "scaler_X.pkl")
    joblib.dump(sy, RESIDUAL_CKPT_DIR / "scaler_y.pkl")
    print(f"  Train samples: {len(X_tr_all):,}  "
          f"| sy.scale_={sy.scale_[0]:.4f} Nm  sy.mean_={sy.mean_[0]:.4f} Nm")

    # ── 3. Residual distribution stats ────────────────────────────────────────
    print("\n[3] Residual distribution statistics …")
    res_stats = compute_residual_stats(y_tr_all)
    for k, v in res_stats.items():
        print(f"  {k:12s}: {v:.4f}")
    if abs(res_stats["skew"]) > 2.0:
        print("  ⚠ High skew — consider Huber loss as follow-up.")
    if abs(res_stats["kurtosis"]) > 10.0:
        print("  ⚠ High kurtosis — heavy tails detected.")

    # Plot residual histogram
    plot_histogram(y_tr_all, res_stats, OUTPUTS_DIR / "residual_histogram.png")

    # ── 4. Build windowed datasets ────────────────────────────────────────────
    print("\n[4] Building windowed arrays …")
    Xw_tr, yw_tr = make_windows_scaled(tr_Xp, tr_yp, sx, sy)
    Xw_va, yw_va = make_windows_scaled(va_Xp, va_yp, sx, sy)
    print(f"  Train windows: {len(Xw_tr):,}  Val windows: {len(Xw_va):,}")

    # ── 5. Train MLP and GRU ──────────────────────────────────────────────────
    models = {}
    histories = {}
    for arch in ["mlp", "gru"]:
        print(f"\n[5] Training {arch.upper()} (residual target) …")
        ckpt_path = RESIDUAL_CKPT_DIR / f"best_model_{arch}.pt"
        model, hist, best_ep, best_val = train_model(
            arch, Xw_tr, yw_tr, Xw_va, yw_va, n_features, device, ckpt_path, args.force)
        models[arch]    = model
        histories[arch] = hist

    # ── 6. Sanity check ───────────────────────────────────────────────────────
    print("\n[6] Sanity check on training data …")
    for arch in ["mlp", "gru"]:
        sanity_check(models[arch], Xw_tr, yw_tr, sy, device, arch.upper())

    # ── 7. Build test set with torEst ─────────────────────────────────────────
    print("\n[7] Building test set …")
    y_true, torEst_arr, Xw_te, fn_arr = build_test_with_torest(dfs_all, feat_cols, sx, sy)
    print(f"  Test windows: {len(y_true):,}")

    # ── 8. Reconstruct test predictions ──────────────────────────────────────
    print("\n[8] Reconstructing test predictions …")
    pred_mlp = reconstruct_torque(models["mlp"], Xw_te, torEst_arr, sy, device)
    pred_gru = reconstruct_torque(models["gru"], Xw_te, torEst_arr, sy, device)

    overall_mlp = {
        "rmse":    rmse(y_true, pred_mlp),
        "mae":     mae(y_true, pred_mlp),
        "max_err": float(np.max(np.abs(y_true - pred_mlp))),
    }
    overall_gru = {
        "rmse":    rmse(y_true, pred_gru),
        "mae":     mae(y_true, pred_gru),
        "max_err": float(np.max(np.abs(y_true - pred_gru))),
    }

    # Error at idx=38180
    for arch, pred in [("mlp", pred_mlp), ("gru", pred_gru)]:
        err_38180 = float(abs(pred[38180] - y_true[38180])) if 38180 < len(y_true) else float("nan")
        (overall_mlp if arch == "mlp" else overall_gru)["err_38180"] = err_38180

    # Mean severe |error| (> 100 Nm)
    for arch, pred in [("mlp", pred_mlp), ("gru", pred_gru)]:
        errs = np.abs(pred - y_true)
        severe = errs[errs > 100.0]
        mean_sev = float(np.mean(severe)) if len(severe) > 0 else float("nan")
        (overall_mlp if arch == "mlp" else overall_gru)["mean_severe"] = mean_sev

    print(f"\n  MLP overall: RMSE={overall_mlp['rmse']:.4f} "
          f"MAE={overall_mlp['mae']:.4f} Max={overall_mlp['max_err']:.2f}")
    print(f"  GRU overall: RMSE={overall_gru['rmse']:.4f} "
          f"MAE={overall_gru['mae']:.4f} Max={overall_gru['max_err']:.2f}")

    # ── 9. Per-profile RMSE ───────────────────────────────────────────────────
    print("\n[9] Per-profile RMSE …")
    pp_rmse_mlp = per_profile_rmse(y_true, pred_mlp, fn_arr)
    pp_rmse_gru = per_profile_rmse(y_true, pred_gru, fn_arr)
    for prof in PROFILES:
        print(f"  {prof:6s}: MLP={pp_rmse_mlp[prof]:.4f}  GRU={pp_rmse_gru[prof]:.4f} Nm")

    # ── 10. v1 baseline evaluation ────────────────────────────────────────────
    print("\n[10] v1 baseline evaluation …")
    v1_results = eval_v1_combined(feat_cols, sx, sy, device)
    pred_v1_gru_arr = None
    if v1_results:
        pred_v1_gru_arr = v1_results["gru"]["pred"]
        y_v1 = v1_results["gru"]["y_true"]
        torEst_v1 = v1_results["gru"]["torEst"]
        fn_v1 = v1_results["gru"]["fn"]
        pp_v1_gru = per_profile_rmse(y_v1, pred_v1_gru_arr, fn_v1)
        for prof in PROFILES:
            print(f"  v1 GRU {prof:6s}: {pp_v1_gru[prof]:.4f} Nm")

    # ── 11. Architecture convergence ──────────────────────────────────────────
    corr = float(np.corrcoef(pred_mlp, pred_gru)[0, 1])
    print(f"\n[11] corr(pred_mlp, pred_gru) = {corr:.4f}")
    if corr < 0.995:
        print("  ⚠ corr < 0.995: residual framing caused architecture divergence.")
    else:
        print("  Architectures converged (corr ≥ 0.995).")

    # ── 12. Time-series plot ──────────────────────────────────────────────────
    print("\n[12] Time-series plot …")
    plot_timeseries(
        y_true, torEst_arr, pred_v1_gru_arr, pred_gru,
        OUTPUTS_DIR / "residual_timeseries.png",
    )

    # ── 13. Per-profile bar chart ─────────────────────────────────────────────
    v1g_pp = {p: V1_RMSE["gru"].get(p, float("nan")) for p in PROFILES}
    plot_profile_bar(v1g_pp, pp_rmse_gru, OUTPUTS_DIR / "residual_profile_bar.png")

    # ── 14. Verdict ───────────────────────────────────────────────────────────
    torque_profiles  = {"tStep", "TMS"}
    velocity_profiles = {"PMS", "PLC"}

    def delta_pct(new, old):
        if np.isnan(float(new)) or np.isnan(float(old)) or float(old) == 0:
            return float("nan")
        return (float(new) - float(old)) / float(old) * 100

    gru_overall_delta = delta_pct(overall_gru["rmse"], V1_RMSE["gru"]["overall"])
    mlp_overall_delta = delta_pct(overall_mlp["rmse"], V1_RMSE["mlp"]["overall"])

    tq_imp  = all(delta_pct(pp_rmse_gru.get(p, float("nan")), V1_RMSE["gru"].get(p, float("nan"))) < -20
                  for p in torque_profiles)
    vel_imp = all(delta_pct(pp_rmse_gru.get(p, float("nan")), V1_RMSE["gru"].get(p, float("nan"))) < -20
                  for p in velocity_profiles)

    both_improve = gru_overall_delta < -10 and mlp_overall_delta < -10

    if both_improve:
        verdict = (f"Residual framing broadly helps: GRU RMSE {V1_RMSE['gru']['overall']:.2f} → "
                   f"{overall_gru['rmse']:.2f} Nm ({gru_overall_delta:+.1f}%), "
                   f"MLP {V1_RMSE['mlp']['overall']:.2f} → {overall_mlp['rmse']:.2f} Nm "
                   f"({mlp_overall_delta:+.1f}%).")
    elif tq_imp and not vel_imp:
        verdict = ("Residual helps torque-controlled only (tStep/TMS >20% improvement), "
                   "velocity-controlled (PMS/PLC) flat or worse. torEst clean residual "
                   "on kd=0 profiles; not on velocity-controlled.")
    elif vel_imp and not tq_imp:
        verdict = ("Residual helps velocity-controlled only (PMS/PLC >20% improvement), "
                   "torque-controlled flat. torEst carries useful state during ringing "
                   "that v1 failed to exploit.")
    elif not tq_imp and not vel_imp and abs(gru_overall_delta) < 5:
        verdict = ("Residual neutral (<5% change): v1 models already implicitly learn "
                   "the residual; explicit framing adds no signal.")
    elif gru_overall_delta > 5 or mlp_overall_delta > 5:
        worst = max(pp_rmse_gru, key=lambda p: pp_rmse_gru.get(p, 0))
        verdict = (f"Residual hurts: GRU RMSE {gru_overall_delta:+.1f}%, "
                   f"MLP {mlp_overall_delta:+.1f}%. "
                   f"Worst profile: {worst} (RMSE={pp_rmse_gru[worst]:.2f} Nm).")
    else:
        verdict = ("Residual partially helps: mixed results across profiles. "
                   f"GRU overall {gru_overall_delta:+.1f}%, MLP {mlp_overall_delta:+.1f}%.")

    print(f"\n[Verdict] {verdict}")

    # ── 15. Write report ──────────────────────────────────────────────────────
    overall_metrics = {"mlp": overall_mlp, "gru": overall_gru}
    write_report(
        res_stats, overall_metrics,
        {"mlp": pp_rmse_mlp, "gru": pp_rmse_gru},
        corr, verdict,
        OUTPUTS_DIR / "residual_comparison.md",
    )

    # ── 16. Append to SUMMARY+CONCLUSION.txt ──────────────────────────────────
    pp_rmse_gru["overall"] = overall_gru["rmse"]
    append_summary(pp_rmse_gru, verdict)

    print("\nDone.")


if __name__ == "__main__":
    main()

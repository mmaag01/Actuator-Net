"""Prompt 2.10b — Parts B, C, D: engineered state-carrier feature sweep.

Trains MLP and GRU at SEQ_LEN=30 with each of 5 engineered features added
independently, then all five together (12 runs total).  Also runs
ablation-from-full inference and optional Part D (winning feature + SEQ_LEN=267).

Usage
-----
    python experiments/state_features_sweep.py
    python experiments/state_features_sweep.py --dry-run   # skip training
    python experiments/state_features_sweep.py --skip-part-d
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_DIAG = _PROJECT_ROOT / "diagnostics"
for _p in (_PROJECT_ROOT, _DIAG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config
from dataset import (
    ActuatorDataset,
    _get_feature_cols,
    _load_dataframes,
    _make_windows,
    _split_df,
)
from feature_engineering import (
    ENGINEERED_FEATURE_COLS,
    assert_feature_causality,
    compute_features,
)
from models import ActuatorGRU, WindowedMLP
from test2_regime_residuals import REGIMES, classify_regimes

# ── Paths ──────────────────────────────────────────────────────────────────────

CKPT_DIR     = _PROJECT_ROOT / "checkpoints" / "state_features_v2"
OUT_DIR      = _HERE / "outputs"
SEVERE_CSV   = _PROJECT_ROOT / "diagnostics" / "outputs" / "test5c_lookback_details.csv"
SUMMARY_PATH = _PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

SEQ_LEN          = 30
CANONICAL_FILE   = "PLC_0.50-10_oldLim_exp"
CANONICAL_ABS    = 10936

BASE_FEAT_COLS   = list(config.FEATURE_COLS)   # original 10 features

FEATURE_CONFIGS: dict[str, list[str]] = {
    "baseline":            [],
    "+torDes_max":         ["torDes_max_abs_500ms"],
    "+torAct_max_lag":     ["torAct_max_abs_500ms_lag100"],
    "+posAct_range":       ["posAct_range_500ms"],
    "+rotorAccelEst":      ["rotorAccelEstimate"],
    "+rotorAccelEst_max":  ["rotorAccelEstimate_max_abs_500ms"],
    "+all_five":           ENGINEERED_FEATURE_COLS,
}

# v1 baseline metrics (GRU from Prompt 8 / v1 results; MLP from v1)
V1_GRU_BASELINE = dict(
    overall_rmse=13.66, hold_rmse=65.97, osc_rmse=12.99,
    err_canonical=215.8, severe_mean_err=147.33,
)
V1_MLP_BASELINE = dict(
    overall_rmse=None, hold_rmse=None, osc_rmse=None,
    err_canonical=None, severe_mean_err=None,
)

PLOT_LO, PLOT_HI = 38150, 38251


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_split_data(dfs: list[pd.DataFrame], feat_cols: list[str], seq_len: int = SEQ_LEN):
    """Split all DFs and return parts + per-file metadata.

    Returns
    -------
    tr_X, tr_y, va_X, va_y, te_X, te_y : lists of np.ndarray (unscaled)
    y_true_test      : (N,) unscaled torAct for the test split (aligned to windows)
    test_meta        : list of {file_name, abs_pos} dicts, one per test window
    file_name_test   : (N,) string array aligned to test windows
    """
    tr_X, tr_y = [], []
    va_X, va_y = [], []
    te_X_parts, te_y_parts = [], []
    test_meta: list[dict] = []

    for df in dfs:
        n = len(df)
        n_train = int(n * config.TRAIN_RATIO)
        n_val   = int(n * config.VAL_RATIO)
        fname   = str(df["file_name"].iloc[0]) if "file_name" in df.columns else "unknown"

        train_df, val_df, test_df = _split_df(
            df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)

        for df_part, Xl, yl in [(train_df, tr_X, tr_y), (val_df, va_X, va_y)]:
            if len(df_part) < seq_len:
                continue
            Xl.append(df_part[feat_cols].values.astype(np.float32))
            yl.append(df_part[config.TARGET_COL].values.astype(np.float32))

        if len(test_df) < seq_len:
            continue
        X_raw_te = test_df[feat_cols].values.astype(np.float32)
        y_raw_te = test_df[config.TARGET_COL].values.astype(np.float32)
        te_X_parts.append(X_raw_te)
        te_y_parts.append(y_raw_te)

        # Build abs_pos map (same formula as seq_len_sweep.py / test5c)
        test_start_in_df = n_train + n_val + seq_len - 1
        n_windows = len(test_df) - seq_len + 1
        abs_positions = test_start_in_df + np.arange(n_windows) + (seq_len - 1)
        for ap in abs_positions:
            test_meta.append({"file_name": fname, "abs_pos": int(ap)})

    y_true_test = np.concatenate(
        [y[seq_len - 1:] for y in te_y_parts if len(y) >= seq_len], axis=0
    ).astype(np.float32)
    file_name_test = np.array([m["file_name"] for m in test_meta], dtype=object)

    return (tr_X, tr_y, va_X, va_y, te_X_parts, te_y_parts,
            y_true_test, test_meta, file_name_test)


def build_datasets(tr_X, tr_y, va_X, va_y, te_X_parts, te_y_parts, seq_len=SEQ_LEN):
    """Fit scalers and build ActuatorDatasets + raw test windows."""
    X_train_all = np.concatenate(tr_X, axis=0)
    y_train_all = np.concatenate(tr_y, axis=0)
    scaler_X = StandardScaler().fit(X_train_all)
    scaler_y = StandardScaler().fit(y_train_all.reshape(-1, 1))

    def _ds(X_parts, y_parts):
        Xw_list, yw_list = [], []
        for X, y in zip(X_parts, y_parts):
            if len(X) < seq_len:
                continue
            Xs = scaler_X.transform(X).astype(np.float32)
            ys = scaler_y.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
            Xw, yw = _make_windows(Xs, ys, seq_len)
            if len(Xw):
                Xw_list.append(Xw)
                yw_list.append(yw)
        return ActuatorDataset(np.concatenate(Xw_list), np.concatenate(yw_list))

    train_ds = _ds(tr_X, tr_y)
    val_ds   = _ds(va_X, va_y)

    # Test windows — also keep raw for inference
    te_Xw, te_yw = [], []
    for X, y in zip(te_X_parts, te_y_parts):
        if len(X) < seq_len:
            continue
        Xs = scaler_X.transform(X).astype(np.float32)
        ys = scaler_y.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
        Xw, yw = _make_windows(Xs, ys, seq_len)
        if len(Xw):
            te_Xw.append(Xw)
            te_yw.append(yw)

    X_test_windows = np.concatenate(te_Xw, axis=0)
    test_ds = ActuatorDataset(X_test_windows, np.concatenate(te_yw, axis=0))

    return train_ds, val_ds, test_ds, X_test_windows, scaler_X, scaler_y


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(model_type: str, n_features: int,
              train_ds, val_ds, ckpt_path: Path, device):
    """Train one model. Returns (model, history, best_epoch, mean_epoch_t, vram_gb)."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    if model_type == "mlp":
        model = WindowedMLP(
            seq_len=SEQ_LEN, n_features=n_features,
            hidden_size=config.MLP_HIDDEN_SIZE, n_layers=config.MLP_N_LAYERS,
        )
    else:
        model = ActuatorGRU(
            n_features=n_features, hidden_size=config.GRU_HIDDEN_SIZE,
            n_layers=config.GRU_N_LAYERS, dropout=config.GRU_DROPOUT,
        )
    model.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.MAX_EPOCHS
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.ONE_CYCLE_MAX_LR, total_steps=total_steps)

    best_val, patience_ctr, best_epoch = float("inf"), 0, 0
    history, epoch_times = [], []

    print(f"\n{'═'*60}")
    print(f"  {model_type.upper()}  n_features={n_features}  "
          f"({len(train_ds):,} train / {len(val_ds):,} val)")
    print(f"{'═'*60}")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item() * len(X)
        val_loss /= len(val_ds)

        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        history.append((epoch, train_loss, val_loss))

        improved = val_loss < best_val
        tag = " [saved]" if improved else f" (pat {patience_ctr+1}/{config.PATIENCE})"
        print(f"  Epoch {epoch:4d} | train {train_loss:.6f} | "
              f"val {val_loss:.6f}{tag} | {dt:.1f}s")

        if improved:
            best_val, patience_ctr, best_epoch = val_loss, 0, epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "model_type": model_type,
                "config": {
                    "seq_len":     SEQ_LEN,
                    "n_features":  n_features,
                    "hidden_size": (config.MLP_HIDDEN_SIZE if model_type == "mlp"
                                    else config.GRU_HIDDEN_SIZE),
                    "n_layers":    (config.MLP_N_LAYERS if model_type == "mlp"
                                    else config.GRU_N_LAYERS),
                },
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    vram_gb = (torch.cuda.max_memory_allocated(device) / 1e9
               if device.type == "cuda" else None)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, history, best_epoch, float(np.mean(epoch_times)), vram_gb


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, X_windows: np.ndarray, scaler_y, device,
                  batch: int = 2048) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X_windows), batch):
            xb = torch.from_numpy(X_windows[s: s + batch]).float().to(device)
            preds.append(model(xb).cpu().numpy().ravel())
    pred_sc = np.concatenate(preds)
    return scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()


# ── Metrics ───────────────────────────────────────────────────────────────────

def load_severe_targets() -> list[dict]:
    if not SEVERE_CSV.exists():
        print(f"WARNING: {SEVERE_CSV} not found — severe-failure analysis skipped.")
        return []
    targets = []
    with open(SEVERE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            targets.append({
                "file_name": row["file_name"],
                "abs_pos":   int(row["abs_pos_in_file"]),
            })
    return targets


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    file_name_test: np.ndarray,
                    test_meta: list[dict],
                    severe_targets: list[dict]) -> dict:
    err = np.abs(y_pred - y_true)
    overall_rmse = float(np.sqrt(np.mean(err ** 2)))
    mae          = float(np.mean(err))
    max_err      = float(err.max())

    regimes = classify_regimes(y_true, file_name_test)
    regime_rmse = {}
    for r in REGIMES:
        mask = regimes == r
        if mask.sum() == 0:
            regime_rmse[r] = float("nan")
        else:
            regime_rmse[r] = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))

    # Canonical failure error
    canonical_err = None
    for i, m in enumerate(test_meta):
        if CANONICAL_FILE in m["file_name"] and m["abs_pos"] == CANONICAL_ABS:
            canonical_err = float(abs(y_pred[i] - y_true[i]))
            break

    # Severe-failure metrics
    if severe_targets:
        lookup = {(m["file_name"], m["abs_pos"]): i for i, m in enumerate(test_meta)}
        errs = []
        for tgt in severe_targets:
            gidx = lookup.get((tgt["file_name"], tgt["abs_pos"]))
            if gidx is not None:
                errs.append(float(abs(y_pred[gidx] - y_true[gidx])))
        if errs:
            errs_arr = np.array(errs)
            severe = {
                "severe_mean_err":   float(errs_arr.mean()),
                "severe_frac_lt100": float((errs_arr < 100).mean()),
                "severe_frac_lt50":  float((errs_arr < 50).mean()),
                "severe_n_found":    len(errs),
            }
        else:
            severe = {"severe_mean_err": float("nan"), "severe_frac_lt100": float("nan"),
                      "severe_frac_lt50": float("nan"), "severe_n_found": 0}
    else:
        severe = {"severe_mean_err": float("nan"), "severe_frac_lt100": float("nan"),
                  "severe_frac_lt50": float("nan"), "severe_n_found": 0}

    return {
        "overall_rmse":  overall_rmse,
        "mae":           mae,
        "max_abs_err":   max_err,
        "hold_rmse":     regime_rmse.get("hold", float("nan")),
        "osc_rmse":      regime_rmse.get("oscillation", float("nan")),
        "trans_rmse":    regime_rmse.get("transition", float("nan")),
        "rest_rmse":     regime_rmse.get("rest", float("nan")),
        "err_canonical": canonical_err,
        **severe,
    }


# ── Ablation-from-full ────────────────────────────────────────────────────────

def run_ablation(model, X_test_windows: np.ndarray, scaler_y, device,
                 feat_cols: list[str], base_n: int) -> dict[str, np.ndarray]:
    """Zero each engineered feature column after scaling, re-run inference.

    Returns a dict mapping feature_name → y_pred when that feature is zeroed.
    Only operates on the engineered feature indices (beyond base_n).
    """
    results = {}
    eng_indices = {col: feat_cols.index(col)
                   for col in ENGINEERED_FEATURE_COLS if col in feat_cols}

    for col, col_idx in eng_indices.items():
        X_ablated = X_test_windows.copy()
        X_ablated[:, :, col_idx] = 0.0   # zero the feature across all timesteps
        results[col] = run_inference(model, X_ablated, scaler_y, device)

    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_pred_vs_truth(y_true: np.ndarray,
                       pred_v1_gru: np.ndarray | None,
                       pred_best_single: np.ndarray | None,
                       pred_all_five: np.ndarray | None,
                       best_single_name: str,
                       out_path: Path):
    lo, hi = PLOT_LO, PLOT_HI
    if hi > len(y_true):
        hi = len(y_true)
    steps = np.arange(lo, hi)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(steps, y_true[lo:hi], color="black", linewidth=1.8,
            label="torAct (truth)", zorder=5)
    if pred_v1_gru is not None and len(pred_v1_gru) >= hi:
        ax.plot(steps, pred_v1_gru[lo:hi], color="#999999", linewidth=1.2,
                linestyle="--", label="v1 GRU (baseline)")
    if pred_best_single is not None and len(pred_best_single) >= hi:
        ax.plot(steps, pred_best_single[lo:hi], color="#1f77b4", linewidth=1.3,
                linestyle="--", label=f"GRU {best_single_name}")
    if pred_all_five is not None and len(pred_all_five) >= hi:
        ax.plot(steps, pred_all_five[lo:hi], color="#2ca02c", linewidth=1.3,
                linestyle="--", label="GRU +all_five")
    ax.axvline(38180, color="crimson", linewidth=0.8, linestyle=":",
               label="idx=38180 (canonical failure)")
    ax.set_xlabel("Global test-stream index")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("Predicted vs truth — samples 38150–38250")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_severe_err_bar(rows: list[dict], out_path: Path):
    """Bar chart of mean severe-failure |error| per config (GRU only)."""
    gru_rows = [r for r in rows if r["model"] == "gru" and
                not np.isnan(r.get("severe_mean_err", float("nan")))]
    if not gru_rows:
        return

    names = [r["config"] for r in gru_rows]
    vals  = [r["severe_mean_err"] for r in gru_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#999999" if n == "baseline" else "#1f77b4" if "+all_five" not in n
              else "#2ca02c" for n in names]
    ax.barh(names, vals, color=colors)
    ax.axvline(147.33, color="black", linewidth=0.8, linestyle="--",
               label="v1 GRU baseline (147.33 Nm)")
    ax.set_xlabel("Mean |error| on 234 severe failures [Nm]")
    ax.set_title("Severe-failure mean error — GRU configs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── Interpretation ────────────────────────────────────────────────────────────

def interpret(rows: list[dict], v1_severe: float = 147.33) -> str:
    gru_rows = {r["config"]: r for r in rows if r["model"] == "gru"}
    msgs = []

    # Check each single-feature config
    single_configs = [n for n in FEATURE_CONFIGS if n not in ("baseline", "+all_five")]
    best_single, best_sev_drop = None, 0.0

    for cfg in single_configs:
        r = gru_rows.get(cfg)
        if r is None:
            continue
        sev = r.get("severe_mean_err")
        if sev is None or np.isnan(sev):
            continue
        drop = (v1_severe - sev) / max(v1_severe, 1e-6)
        can_err = r.get("err_canonical")
        can_drop = ((215.8 - can_err) / 215.8) if can_err else 0.0
        if drop > best_sev_drop or (best_single is None and can_drop > 0):
            best_sev_drop = drop
            best_single = cfg

    if best_single:
        r = gru_rows[best_single]
        sev = r["severe_mean_err"]
        can_err = r.get("err_canonical", float("nan"))
        if best_sev_drop > 0.50 or ((215.8 - can_err) / 215.8 > 0.70 if can_err else False):
            msgs.append(
                f"**Feature resolves backlash ringing**: '{best_single}' alone reduces "
                f"severe-failure mean |err| by {100*best_sev_drop:.0f}%."
            )
        elif best_sev_drop > 0.30:
            msgs.append(
                f"**Magnitude features partially help**: best single feature '{best_single}' "
                f"reduces severe-failure mean |err| by {100*best_sev_drop:.0f}%."
            )
        else:
            msgs.append(
                f"**No single feature recovers failures**: best single feature '{best_single}' "
                f"reduces severe-failure mean |err| by only {100*best_sev_drop:.0f}%. "
                "Joint-side-only telemetry may be fundamentally insufficient — "
                "motor-side signals must be added to hardware logging."
            )

    all_r = gru_rows.get("+all_five")
    if all_r and best_single and gru_rows.get(best_single):
        all_sev = all_r.get("severe_mean_err", float("nan"))
        best_sev = gru_rows[best_single].get("severe_mean_err", float("nan"))
        if not np.isnan(all_sev) and not np.isnan(best_sev) and best_sev > 0:
            comb_gain = (best_sev - all_sev) / best_sev
            if comb_gain > 0.20:
                msgs.append(
                    f"**Combination wins cleanly**: all-five outperforms best single by "
                    f"{100*comb_gain:.0f}% on severe failures."
                )

    return "\n\n".join(msgs) if msgs else "Results pending full run."


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown(rows: list[dict], ablation: dict, interp: str, out_path: Path):
    def _f(v, fmt=".2f"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:{fmt}}"

    lines = [
        "# State-Carrier Feature Sweep — Prompt 2.10b",
        "",
        "## Part C — Training results (SEQ_LEN=30)",
        "",
        "| Config | Model | Overall RMSE | Hold RMSE | Osc RMSE | "
        "Err idx=38180 | Mean severe |err| | Severe < 100 Nm | Severe < 50 Nm |",
        "|---|---|---|---|---|---|---|---|---|",
        "| baseline (v1, GRU) | GRU | 13.66 | 65.97 | 12.99 | 215.8 | 147.33 | 0.0% | 0.0% |",
    ]
    for r in rows:
        if r["config"] == "baseline":
            continue
        pct = lambda v: f"{100*v:.1f}%" if v is not None and not np.isnan(v) else "—"
        lines.append(
            f"| {r['config']} | {r['model'].upper()} "
            f"| {_f(r['overall_rmse'])} "
            f"| {_f(r['hold_rmse'])} "
            f"| {_f(r['osc_rmse'])} "
            f"| {_f(r['err_canonical'])} "
            f"| {_f(r['severe_mean_err'])} "
            f"| {pct(r.get('severe_frac_lt100'))} "
            f"| {pct(r.get('severe_frac_lt50'))} |"
        )

    lines += [
        "",
        "## Ablation-from-full (GRU +all_five, inference only)",
        "",
        "Each feature column is zeroed after scaling. "
        "Larger drop = model relied on that feature.",
        "",
        "| Feature zeroed | Overall RMSE | Err idx=38180 | Mean severe |err| |",
        "|---|---|---|---|",
    ]
    for feat, abl_row in ablation.items():
        lines.append(
            f"| {feat} "
            f"| {_f(abl_row.get('overall_rmse'))} "
            f"| {_f(abl_row.get('err_canonical'))} "
            f"| {_f(abl_row.get('severe_mean_err'))} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        interp,
        "",
        "## Outputs",
        "- `state_features_pred_vs_truth.png` — time-series at idx=38180",
        "- `state_features_severe_err_bar.png` — severe-failure error per config",
        f"- Checkpoints: `checkpoints/state_features_v2/`",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


# ── v1 baseline inference ─────────────────────────────────────────────────────

def get_v1_baseline(device, test_meta, y_true_test, file_name_test, severe_targets):
    """Load v1 checkpoints and compute baseline metrics."""
    v1_scaler_X = joblib.load(config.CHECKPOINT_DIR / "scaler_X.pkl")
    v1_scaler_y = joblib.load(config.CHECKPOINT_DIR / "scaler_y.pkl")

    # Build v1 test windows (baseline 10 features, seq_len=30)
    dfs = _load_dataframes()
    base_feat_cols = _get_feature_cols()
    te_Xw_parts = []
    for df in dfs:
        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, SEQ_LEN)
        if len(test_df) < SEQ_LEN:
            continue
        Xs = v1_scaler_X.transform(test_df[base_feat_cols].values.astype(np.float32)).astype(np.float32)
        Xw, _ = _make_windows(Xs, np.zeros(len(Xs), dtype=np.float32), SEQ_LEN)
        if len(Xw):
            te_Xw_parts.append(Xw)
    X_windows_v1 = np.concatenate(te_Xw_parts, axis=0)

    results = {}
    for model_type in ("mlp", "gru"):
        ckpt_path = config.CHECKPOINT_DIR / f"best_model_{model_type}.pt"
        if not ckpt_path.exists():
            print(f"WARNING: v1 {model_type.upper()} checkpoint not found, skipping baseline.")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg  = ckpt["config"]
        if model_type == "mlp":
            m = WindowedMLP(seq_len=cfg["seq_len"], n_features=cfg["n_features"],
                            hidden_size=cfg.get("hidden_size", config.MLP_HIDDEN_SIZE),
                            n_layers=cfg.get("n_layers", config.MLP_N_LAYERS))
        else:
            m = ActuatorGRU(n_features=cfg["n_features"],
                            hidden_size=cfg.get("hidden_size", config.GRU_HIDDEN_SIZE),
                            n_layers=cfg.get("n_layers", config.GRU_N_LAYERS))
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device).eval()
        y_pred = run_inference(m, X_windows_v1, v1_scaler_y, device)
        metrics = compute_metrics(y_true_test, y_pred, file_name_test, test_meta, severe_targets)
        results[model_type] = {"y_pred": y_pred, "metrics": metrics}
        print(f"  v1 {model_type.upper()}: RMSE={metrics['overall_rmse']:.2f} Nm  "
              f"Canonical={metrics['err_canonical']}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",      action="store_true",
                        help="Skip training; load existing checkpoints.")
    parser.add_argument("--skip-part-d", action="store_true",
                        help="Skip Part D (winning feature + SEQ_LEN=267).")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    severe_targets = load_severe_targets()
    print(f"Loaded {len(severe_targets)} severe-failure targets.")

    print("\nLoading and feature-engineering data …")
    dfs_raw = _load_dataframes()
    dfs = [compute_features(df) for df in dfs_raw]

    print("Running causality assertion …")
    assert_feature_causality(dfs)

    # Build shared metadata (abs_pos map) using baseline feature cols
    base_feat_cols = _get_feature_cols()   # original 10 features
    (tr_X_base, tr_y_base, va_X_base, va_y_base, te_X_base, te_y_base,
     y_true_test, test_meta, file_name_test) = build_split_data(dfs, base_feat_cols)

    print("\nComputing v1 baseline metrics …")
    v1_baseline = get_v1_baseline(device, test_meta, y_true_test,
                                   file_name_test, severe_targets)

    rows: list[dict] = []
    pred_store: dict[str, np.ndarray] = {}   # config+model → y_pred (GRU only)
    all_five_model_gru = None
    all_five_feat_cols = None
    all_five_Xw        = None
    all_five_scaler_y  = None

    for cfg_name, extra_feats in FEATURE_CONFIGS.items():
        feat_cols = base_feat_cols + extra_feats
        n_feat    = len(feat_cols)
        print(f"\n{'─'*60}")
        print(f"  Config: {cfg_name}  (n_features={n_feat})")

        (tr_X, tr_y, va_X, va_y, te_X_parts, te_y_parts,
         _, _, _) = build_split_data(dfs, feat_cols)

        train_ds, val_ds, test_ds, X_test_windows, scaler_X, scaler_y = build_datasets(
            tr_X, tr_y, va_X, va_y, te_X_parts, te_y_parts)

        for model_type in ("mlp", "gru"):
            key = f"{cfg_name}_{model_type}"
            ckpt_path = CKPT_DIR / f"{cfg_name.lstrip('+').replace(' ', '_')}_{model_type}.pt"

            if cfg_name == "baseline":
                # Use v1 metrics directly, no retraining
                v1_m = v1_baseline.get(model_type)
                if v1_m is None:
                    continue
                metrics = v1_m["metrics"]
                if model_type == "gru":
                    pred_store[key] = v1_m["y_pred"]
                row = {"config": cfg_name, "model": model_type,
                       "best_epoch": "v1", "time_s": "—", **metrics}
                rows.append(row)
                print(f"  {model_type.upper()} baseline: "
                      f"RMSE={metrics['overall_rmse']:.2f} Nm")
                continue

            if args.dry_run and ckpt_path.exists():
                print(f"  [dry-run] Loading {ckpt_path.name} …")
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                cfg  = ckpt["config"]
                if model_type == "mlp":
                    model = WindowedMLP(
                        seq_len=cfg["seq_len"], n_features=cfg["n_features"],
                        hidden_size=cfg.get("hidden_size", config.MLP_HIDDEN_SIZE),
                        n_layers=cfg.get("n_layers", config.MLP_N_LAYERS))
                else:
                    model = ActuatorGRU(
                        n_features=cfg["n_features"],
                        hidden_size=cfg.get("hidden_size", config.GRU_HIDDEN_SIZE),
                        n_layers=cfg.get("n_layers", config.GRU_N_LAYERS))
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(device).eval()
                best_epoch = ckpt["epoch"]
                mean_t = None
            else:
                model, hist, best_epoch, mean_t, vram_gb = train_one(
                    model_type, n_feat, train_ds, val_ds, ckpt_path, device)
                hist_path = OUT_DIR / f"history_{cfg_name.lstrip('+').replace(' ','_')}_{model_type}.csv"
                with open(hist_path, "w", newline="") as f:
                    csv.writer(f).writerows([["epoch", "train_mse", "val_mse"]] + list(hist))

            y_pred = run_inference(model, X_test_windows, scaler_y, device)
            metrics = compute_metrics(y_true_test, y_pred, file_name_test,
                                      test_meta, severe_targets)

            row = {"config": cfg_name, "model": model_type,
                   "best_epoch": best_epoch, "time_s": mean_t, **metrics}
            rows.append(row)

            if model_type == "gru":
                pred_store[cfg_name] = y_pred
                if cfg_name == "+all_five":
                    all_five_model_gru = model
                    all_five_feat_cols = feat_cols
                    all_five_Xw        = X_test_windows
                    all_five_scaler_y  = scaler_y

            print(f"  {model_type.upper()} {cfg_name}: "
                  f"RMSE={metrics['overall_rmse']:.2f} Nm  "
                  f"Hold={metrics['hold_rmse']:.2f}  "
                  f"Canonical err={metrics.get('err_canonical')}")

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print(f"  {'Config':20s}  {'Model':5s}  {'RMSE':8s}  {'Hold':8s}  "
          f"{'Osc':8s}  {'Err@38180':10s}  {'SevMean':9s}")
    print(f"{'─'*100}")
    for r in rows:
        def _f(v):
            return f"{v:8.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "      —  "
        print(f"  {r['config']:20s}  {r['model']:5s}  "
              f"{_f(r['overall_rmse'])}  {_f(r['hold_rmse'])}  "
              f"{_f(r['osc_rmse'])}  {_f(r.get('err_canonical', float('nan'))):>10}  "
              f"{_f(r.get('severe_mean_err', float('nan'))):>9}")
    print(f"{'═'*100}")

    # ── Ablation-from-full ────────────────────────────────────────────────────
    ablation_results: dict[str, dict] = {}
    if all_five_model_gru is not None:
        print("\nRunning ablation-from-full (GRU +all_five) …")
        base_n = len(base_feat_cols)
        abl_preds = run_ablation(
            all_five_model_gru, all_five_Xw, all_five_scaler_y, device,
            all_five_feat_cols, base_n)
        for feat, y_pred_abl in abl_preds.items():
            m = compute_metrics(y_true_test, y_pred_abl, file_name_test,
                                test_meta, severe_targets)
            ablation_results[feat] = m
            print(f"  zero '{feat}': RMSE={m['overall_rmse']:.2f}  "
                  f"Canonical={m.get('err_canonical')}")

    # ── Part D: winning feature + SEQ_LEN=267 ────────────────────────────────
    if not args.skip_part_d:
        v1_sev = 147.33
        v1_can = 215.8
        best_cfg, best_imp_type = None, None

        gru_rows = {r["config"]: r for r in rows if r["model"] == "gru"
                    and r["config"] not in ("baseline", "+all_five")}
        for cfg, r in gru_rows.items():
            can_err = r.get("err_canonical")
            sev_err = r.get("severe_mean_err")
            if can_err is not None and not np.isnan(can_err):
                if (v1_can - can_err) / v1_can > 0.70:
                    print(f"\nPart D trigger: '{cfg}' reduces canonical err by "
                          f"{100*(v1_can - can_err)/v1_can:.0f}% (>70%)")
                    best_cfg = cfg
                    best_imp_type = "canonical"
                    break
            if sev_err is not None and not np.isnan(sev_err):
                if (v1_sev - sev_err) / v1_sev > 0.50:
                    print(f"\nPart D trigger: '{cfg}' reduces severe-failure mean by "
                          f"{100*(v1_sev - sev_err)/v1_sev:.0f}% (>50%)")
                    best_cfg = cfg
                    best_imp_type = "severe"
                    break

        if best_cfg:
            print(f"\nRunning Part D: {best_cfg} + SEQ_LEN=267 …")
            extra_feats_d = FEATURE_CONFIGS[best_cfg]
            feat_cols_d   = base_feat_cols + extra_feats_d
            (tr_X_d, tr_y_d, va_X_d, va_y_d, te_X_d, te_y_d,
             y_true_d, test_meta_d, fn_d) = build_split_data(dfs, feat_cols_d, seq_len=267)
            train_ds_d, val_ds_d, test_ds_d, Xw_d, scaler_X_d, scaler_y_d = build_datasets(
                tr_X_d, tr_y_d, va_X_d, va_y_d, te_X_d, te_y_d, seq_len=267)
            ckpt_d = CKPT_DIR / f"partD_{best_cfg.lstrip('+').replace(' ','_')}_L267_gru.pt"
            n_feat_d = len(feat_cols_d)

            # Temporarily monkey-patch config.SEQ_LEN for model construction
            _orig_seq = config.SEQ_LEN
            config.SEQ_LEN = 267   # used by train_one for scheduler total_steps via loader
            model_d, hist_d, ep_d, t_d, vram_d = train_one(
                "gru", n_feat_d, train_ds_d, val_ds_d, ckpt_d, device)
            config.SEQ_LEN = _orig_seq

            y_pred_d = run_inference(model_d, Xw_d, scaler_y_d, device)
            m_d = compute_metrics(y_true_d, y_pred_d, fn_d, test_meta_d, severe_targets)
            print(f"\nPart D result: RMSE={m_d['overall_rmse']:.2f} Nm  "
                  f"Canonical err={m_d.get('err_canonical')}  "
                  f"Severe mean={m_d.get('severe_mean_err'):.1f} Nm")

            # Compare
            base_gru_r = next((r for r in rows
                               if r["config"] == best_cfg and r["model"] == "gru"), None)
            if base_gru_r:
                sev_d = m_d.get("severe_mean_err", float("nan"))
                sev_s = base_gru_r.get("severe_mean_err", float("nan"))
                if not np.isnan(sev_d) and not np.isnan(sev_s) and sev_s > 0:
                    gain = (sev_s - sev_d) / sev_s
                    tag = "length+feature compound" if gain > 0.30 else "no compounding"
                    print(f"Part D vs best-single: severe mean {sev_s:.1f} → {sev_d:.1f} Nm "
                          f"({100*gain:.0f}% gain) → {tag}")
        else:
            print("\nPart D: no single feature met the threshold — skipping SEQ_LEN=267 retrain.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    v1_gru_pred = (v1_baseline["gru"]["y_pred"]
                   if "gru" in v1_baseline else None)

    gru_sev = {cfg: r.get("severe_mean_err", float("nan"))
               for r in rows if r["model"] == "gru"
               for cfg in [r["config"]] if cfg not in ("baseline", "+all_five")}
    best_sgl_cfg = min(gru_sev, key=lambda k: gru_sev[k]) if gru_sev else None
    pred_best_single = pred_store.get(best_sgl_cfg) if best_sgl_cfg else None

    plot_pred_vs_truth(
        y_true_test, v1_gru_pred, pred_best_single,
        pred_store.get("+all_five"),
        best_sgl_cfg or "—",
        OUT_DIR / "state_features_pred_vs_truth.png",
    )
    plot_severe_err_bar(rows, OUT_DIR / "state_features_severe_err_bar.png")

    # ── Markdown report ───────────────────────────────────────────────────────
    interp = interpret(rows)
    write_markdown(rows, ablation_results, interp,
                   OUT_DIR / "state_features_sweep.md")

    # ── Append to SUMMARY+CONCLUSION.txt ─────────────────────────────────────
    gru_rows_d = {r["config"]: r for r in rows if r["model"] == "gru"}
    all5 = gru_rows_d.get("+all_five", {})

    block = (
        f"\n{'='*72}\n"
        f"[Prompt 2.10b — State-carrier feature sweep]\n"
        f"{'='*72}\n"
        f"Purpose: Test whether 5 engineered state-carrier features fix the "
        f"ambiguous-input failure mode (idx=38180, torDes=0, torAct=+106.6 Nm).\n\n"
        f"Feature configs tested: {list(FEATURE_CONFIGS.keys())}\n"
        f"Models: MLP + GRU × 6 configs = 12 training runs\n\n"
        f"GRU +all_five: RMSE={all5.get('overall_rmse', float('nan')):.2f} Nm  "
        f"Canonical err={all5.get('err_canonical', float('nan')):.1f} Nm  "
        f"Severe mean={all5.get('severe_mean_err', float('nan')):.1f} Nm\n\n"
        f"{interp}\n"
    )
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"\nAppended to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

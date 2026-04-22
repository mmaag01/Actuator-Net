"""SEQ_LEN sweep — GRU only.

Trains GRU at SEQ_LEN ∈ {267, 325, 372, 400} with v1 scalers and identical
hyperparameters. Evaluates on the v1 test split.

Usage
-----
    python experiments/seq_len_sweep.py
    python experiments/seq_len_sweep.py --dry-run   # skip training, re-plot only
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_DIAG = _PROJECT_ROOT / "diagnostics"
for _p in (_PROJECT_ROOT, _DIAG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config
from dataset import (
    ActuatorDataset, _get_feature_cols, _load_dataframes,
    _make_windows, _split_df,
)
from models import ActuatorGRU
from test2_regime_residuals import REGIMES, classify_regimes

# ── Paths & constants ─────────────────────────────────────────────────────────
SWEEP_SEQ_LENS    = [267, 325, 372, 400]
SEVERE_ERR_THR    = 100.0       # Nm
CANONICAL_FILE    = "PLC_0.50-10_oldLim_exp"
CANONICAL_ABS_POS = 10936       # row in that file for idx=38180 in v1

CKPT_DIR     = _PROJECT_ROOT / "checkpoints" / "seq_sweep"
OUT_DIR      = _HERE / "outputs"
V1_HIST_PATH = _PROJECT_ROOT / "results" / "v0" / "loss_history_gru.csv"
SEVERE_CSV   = _PROJECT_ROOT / "diagnostics" / "outputs" / "test5c_lookback_details.csv"
SUMMARY_PATH = _PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# v1 baseline (SEQ_LEN=30) numbers from the prompt table
V1_ROW = dict(
    seq_len=30, best_epoch=29, time_per_epoch=None, vram_gb=None,
    overall_rmse=13.66, mae=4.04, max_abs_err=225.4,
    hold_rmse=65.97, osc_rmse=12.99, trans_rmse=11.67, rest_rmse=3.80,
    err_canonical=215.8,
    severe_mean_err=None, severe_frac_lt100=0.0, severe_frac_lt50=0.0,
)

PALETTE = {30: "#555555", 267: "#1f77b4", 325: "#ff7f0e",
           372: "#2ca02c", 400: "#d62728"}


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Dataset builder (monkey-patches config.SEQ_LEN) ──────────────────────────

def _build_test_meta(seq_len: int):
    """Return (y_true, file_name, pf_map) for the test split at this seq_len.

    pf_map[i] = {"file_name": str, "abs_pos": int}
    abs_pos is the row index in the ORIGINAL (full) file df for the target sample.
    """
    dfs = _load_dataframes()
    y_parts, fn_parts, pf_parts = [], [], []

    for df in dfs:
        n = len(df)
        n_train = int(n * config.TRAIN_RATIO)
        n_val   = int(n * config.VAL_RATIO)
        test_start = n_train + n_val + seq_len - 1   # start of test_df in df
        test_df_len = n - test_start
        if test_df_len < seq_len:
            continue

        fname  = str(df["file_name"].iloc[0]) if "file_name" in df.columns else "unknown"
        y_full = df[config.TARGET_COL].values.astype(np.float32)
        fn_all = (df["file_name"].values.astype(str)
                  if "file_name" in df.columns
                  else np.full(n, fname, dtype=object))

        n_windows = test_df_len - seq_len + 1
        abs_positions = test_start + np.arange(n_windows) + (seq_len - 1)
        y_parts.append(y_full[abs_positions])
        fn_parts.extend([fname] * n_windows)
        pf_parts.extend({"file_name": fname, "abs_pos": int(ap)}
                         for ap in abs_positions)

    return (np.concatenate(y_parts).astype(np.float32),
            np.array(fn_parts, dtype=object),
            pf_parts)


def build_datasets_for(seq_len: int, scaler_X, scaler_y):
    """Build ActuatorDataset objects with the given seq_len and pre-fitted scalers."""
    feature_cols = _get_feature_cols()
    dfs = _load_dataframes()

    tr_X, tr_y = [], []
    va_X, va_y = [], []
    te_X, te_y = [], []

    for df in dfs:
        train_df, val_df, test_df = _split_df(
            df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)

        for df_part, Xlist, ylist in [(train_df, tr_X, tr_y),
                                       (val_df,   va_X, va_y),
                                       (test_df,  te_X, te_y)]:
            if len(df_part) < seq_len:
                continue
            X_raw = df_part[feature_cols].values.astype(np.float32)
            y_raw = df_part[config.TARGET_COL].values.astype(np.float32)
            Xs = scaler_X.transform(X_raw).astype(np.float32)
            ys = scaler_y.transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32)
            Xw, yw = _make_windows(Xs, ys, seq_len)
            if len(Xw):
                Xlist.append(Xw)
                ylist.append(yw)

    def _ds(Xl, yl):
        return ActuatorDataset(np.concatenate(Xl), np.concatenate(yl))

    return _ds(tr_X, tr_y), _ds(va_X, va_y), _ds(te_X, te_y)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, dataset, scaler_y, device, batch=2048) -> np.ndarray:
    """Return raw (unscaled) predictions aligned to dataset order."""
    loader = DataLoader(dataset, batch_size=batch, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for X, _ in loader:
            preds.append(model(X.to(device)).cpu().numpy().ravel())
    pred_sc = np.concatenate(preds)
    return scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(seq_len: int, train_ds, val_ds, device):
    """Train GRU with v1 hyperparameters. Returns (model, history, best_epoch, vram_gb)."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    model = ActuatorGRU(
        n_features=config.N_FEATURES,
        hidden_size=config.GRU_HIDDEN_SIZE,
        n_layers=config.GRU_N_LAYERS,
        dropout=config.GRU_DROPOUT,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.MAX_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.ONE_CYCLE_MAX_LR, total_steps=total_steps)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_val, patience_ctr = float("inf"), 0
    best_epoch = 0
    history = []
    epoch_times = []

    ckpt_path = CKPT_DIR / f"gru_L{seq_len}.pt"

    print(f"\n{'═'*60}")
    print(f"  Training GRU  SEQ_LEN={seq_len}  "
          f"({len(train_ds):,} train / {len(val_ds):,} val windows)")
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
                "config": {
                    "seq_len":     seq_len,
                    "n_features":  config.N_FEATURES,
                    "hidden_size": config.GRU_HIDDEN_SIZE,
                    "n_layers":    config.GRU_N_LAYERS,
                },
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    vram_gb = (torch.cuda.max_memory_allocated(device) / 1e9
               if device.type == "cuda" else None)

    # Reload best weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean_epoch_t = float(np.mean(epoch_times))
    print(f"  Best epoch={best_epoch}  val_MSE={best_val:.6f}  "
          f"mean_epoch={mean_epoch_t:.1f}s"
          + (f"  VRAM={vram_gb:.2f}GB" if vram_gb else ""))

    return model, history, best_epoch, mean_epoch_t, vram_gb


def load_ckpt_gru(seq_len: int, device) -> ActuatorGRU:
    path = CKPT_DIR / f"gru_L{seq_len}.pt"
    assert path.exists(), f"Checkpoint missing: {path}"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = ActuatorGRU(
        n_features=cfg["n_features"],
        hidden_size=cfg.get("hidden_size", config.GRU_HIDDEN_SIZE),
        n_layers=cfg.get("n_layers",    config.GRU_N_LAYERS),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    best_epoch = ckpt["epoch"]
    return model, best_epoch


# ── Metrics ───────────────────────────────────────────────────────────────────

def per_regime_rmse(y_true, y_pred, regimes) -> dict[str, float]:
    out = {}
    for r in REGIMES:
        mask = regimes == r
        if mask.sum() == 0:
            out[r] = float("nan")
        else:
            out[r] = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
    return out


def find_canonical_err(y_true, y_pred, pf_map) -> float | None:
    for i, entry in enumerate(pf_map):
        if (CANONICAL_FILE in entry["file_name"]
                and entry["abs_pos"] == CANONICAL_ABS_POS):
            return float(abs(y_pred[i] - y_true[i]))
    return None


def load_severe_targets() -> list[dict]:
    """Load the 234 severe-failure (file_name, abs_pos) pairs from Test 5c."""
    if not SEVERE_CSV.exists():
        print(f"WARNING: {SEVERE_CSV} not found — severe-failure analysis skipped.")
        return []
    targets = []
    with open(SEVERE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            targets.append({
                "file_name": row["file_name"],
                "abs_pos":   int(row["abs_pos_in_file"]),
                "v1_err":    float(row["abs_err_Nm"]),
            })
    return targets


def severe_metrics(y_true, y_pred, pf_map, severe_targets) -> dict:
    if not severe_targets:
        return {"severe_mean_err": float("nan"),
                "severe_frac_lt100": float("nan"),
                "severe_frac_lt50":  float("nan"),
                "n_found": 0}

    # Build fast lookup: (file_name, abs_pos) → global_idx
    lookup = {(e["file_name"], e["abs_pos"]): i for i, e in enumerate(pf_map)}

    errs = []
    for tgt in severe_targets:
        key = (tgt["file_name"], tgt["abs_pos"])
        gidx = lookup.get(key)
        if gidx is None:
            continue
        errs.append(abs(float(y_pred[gidx]) - float(y_true[gidx])))

    if not errs:
        return {"severe_mean_err": float("nan"),
                "severe_frac_lt100": float("nan"),
                "severe_frac_lt50":  float("nan"),
                "n_found": 0}

    errs = np.array(errs)
    return {
        "severe_mean_err":    float(errs.mean()),
        "severe_frac_lt100":  float((errs < 100).mean()),
        "severe_frac_lt50":   float((errs <  50).mean()),
        "n_found":            len(errs),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: dict[int, list], path):
    """5 val-loss curves: v1 baseline + 4 sweep configs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # v1 baseline
    if V1_HIST_PATH.exists():
        v1 = np.genfromtxt(V1_HIST_PATH, delimiter=",", skip_header=1)
        ax.plot(v1[:, 0], v1[:, 2], color=PALETTE[30], linewidth=1.5,
                linestyle="--", label="v1 SEQ_LEN=30")

    for sl, hist in sorted(histories.items()):
        arr = np.array(hist)
        ax.plot(arr[:, 0], arr[:, 2], color=PALETTE[sl], linewidth=1.4,
                label=f"SEQ_LEN={sl}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (scaled)")
    ax.set_title("GRU training curves — SEQ_LEN sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_severe_err_vs_seqlen(rows: list[dict], path):
    seq_lens = [r["seq_len"] for r in rows if r.get("severe_mean_err") is not None]
    mean_errs = [r["severe_mean_err"] for r in rows if r.get("severe_mean_err") is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE.get(sl, "gray") for sl in seq_lens]
    ax.plot(seq_lens, mean_errs, "o-", color="steelblue", linewidth=1.5, markersize=7)
    for sl, me, c in zip(seq_lens, mean_errs, colors):
        ax.annotate(f"{me:.1f} Nm", (sl, me), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("SEQ_LEN [samples]")
    ax.set_ylabel("Mean |error| on 234 severe failures [Nm]")
    ax.set_title("Mean severe-failure error vs window length (GRU)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_pred_vs_truth(all_preds: dict[int, np.ndarray], y_true_v1: np.ndarray,
                       pf_maps: dict[int, list], path):
    """Plot torAct + predictions over the v1 samples 38150-38250."""
    lo, hi = 38150, 38251
    steps = np.arange(lo, hi)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, y_true_v1[lo:hi], color="black", linewidth=1.8,
            label="torAct (truth)", zorder=5)

    # For v1 (SEQ_LEN=30): direct indexing
    if 30 in all_preds:
        ax.plot(steps, all_preds[30][lo:hi], color=PALETTE[30], linewidth=1.2,
                linestyle="--", label="GRU v1 SEQ_LEN=30")

    # For each new SEQ_LEN: find the global indices that correspond to the
    # same target abs_pos values as steps lo..hi in the v1 map
    v1_pf = pf_maps.get(30)
    if v1_pf:
        for sl, pred_arr in sorted(all_preds.items()):
            if sl == 30:
                continue
            sl_pf = pf_maps.get(sl)
            if sl_pf is None:
                continue
            lookup = {(e["file_name"], e["abs_pos"]): gi
                      for gi, e in enumerate(sl_pf)}
            mapped_preds = []
            for v1_gi in range(lo, hi):
                if v1_gi >= len(v1_pf):
                    mapped_preds.append(float("nan"))
                    continue
                key = (v1_pf[v1_gi]["file_name"], v1_pf[v1_gi]["abs_pos"])
                new_gi = lookup.get(key)
                if new_gi is not None:
                    mapped_preds.append(float(pred_arr[new_gi]))
                else:
                    mapped_preds.append(float("nan"))
            ax.plot(steps, mapped_preds, color=PALETTE[sl], linewidth=1.2,
                    linestyle="--", label=f"GRU SEQ_LEN={sl}")

    ax.axvline(38180, color="crimson", linewidth=0.8, linestyle=":",
               label="idx=38180 (canonical failure)")
    ax.set_xlabel("Global test-stream index")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title("Predicted vs truth around the canonical failure (samples 38150–38250)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown(rows: list[dict], path: Path, interpretation: str,
                   v1_severe_mean_err: float):
    n_severe = 234
    lines = [
        "# SEQ_LEN Sweep — GRU Results",
        "",
        "## Table 1 — Overall and per-regime metrics",
        "",
        "| SEQ_LEN | Time/epoch [s] | VRAM [GB] | Best epoch | "
        "Overall RMSE | MAE | Max\\|err\\| | Hold RMSE | Osc RMSE | "
        "Trans RMSE | Rest RMSE | Err at idx=38180 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        def _f(v, fmt=".2f"):
            return f"{v:{fmt}}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"
        lines.append(
            f"| {r['seq_len']} {'(v1)' if r['seq_len']==30 else ''} "
            f"| {_f(r['time_per_epoch'])} "
            f"| {_f(r['vram_gb'])} "
            f"| {_f(r['best_epoch'], 'd') if r['best_epoch'] else '—'} "
            f"| {_f(r['overall_rmse'])} "
            f"| {_f(r['mae'])} "
            f"| {_f(r['max_abs_err'])} "
            f"| {_f(r['hold_rmse'])} "
            f"| {_f(r['osc_rmse'])} "
            f"| {_f(r['trans_rmse'])} "
            f"| {_f(r['rest_rmse'])} "
            f"| {_f(r['err_canonical'])} |"
        )
    lines += [
        "",
        f"## Table 2 — Severe-failure analysis (n={n_severe} samples from Test 5c)",
        "",
        "| SEQ_LEN | Mean \\|err\\| [Nm] | Frac < 100 Nm | Frac < 50 Nm | "
        "Samples found |",
        "|---|---|---|---|---|",
    ]
    # v1 baseline severe row
    lines.append(
        f"| 30 (v1) | {v1_severe_mean_err:.1f} | 0.0% | 0.0% | {n_severe} |"
    )
    for r in rows:
        if r["seq_len"] == 30:
            continue
        me  = r.get("severe_mean_err")
        f1  = r.get("severe_frac_lt100")
        f5  = r.get("severe_frac_lt50")
        nf  = r.get("n_found", "—")
        def pct(v):
            return f"{100*v:.1f}%" if v is not None and not np.isnan(v) else "—"
        me_s = f"{me:.1f}" if me is not None and not np.isnan(me) else "—"
        lines.append(f"| {r['seq_len']} | {me_s} | {pct(f1)} | {pct(f5)} | {nf} |")

    lines += [
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Outputs",
        f"- Training curves: `experiments/outputs/seq_len_sweep_training_curves.png`",
        f"- Severe-failure error vs SEQ_LEN: "
        f"`experiments/outputs/seq_len_sweep_severe_err_vs_seqlen.png`",
        f"- Prediction vs truth (38150–38250): "
        f"`experiments/outputs/seq_len_sweep_pred_vs_truth.png`",
        f"- Checkpoints: `checkpoints/seq_sweep/gru_L{{N}}.pt`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


# ── Interpretation logic ──────────────────────────────────────────────────────

def interpret(rows: list[dict], v1_severe_mean_err: float) -> str:
    # Find SEQ_LEN=372 row
    r372 = next((r for r in rows if r["seq_len"] == 372), None)
    r400 = next((r for r in rows if r["seq_len"] == 400), None)
    r325 = next((r for r in rows if r["seq_len"] == 325), None)

    msgs = []

    if r372 and v1_severe_mean_err > 0:
        me372 = r372.get("severe_mean_err")
        rmse_drop = (13.66 - r372.get("overall_rmse", 13.66)) / 13.66
        if me372 and not np.isnan(me372):
            sev_drop = (v1_severe_mean_err - me372) / v1_severe_mean_err
            if sev_drop > 0.70 and rmse_drop > 0.20:
                msgs.append("**Longer windows resolve ringing broadly** "
                             f"(SEQ_LEN=372 reduces severe-failure |err| by "
                             f"{100*sev_drop:.0f}%, overall RMSE by {100*rmse_drop:.0f}%).")
            elif 0.30 <= sev_drop <= 0.70 or 0.10 <= rmse_drop <= 0.20:
                msgs.append("**Longer windows partially resolve ringing** "
                             f"(SEQ_LEN=372: severe-failure drop {100*sev_drop:.0f}%, "
                             f"RMSE drop {100*rmse_drop:.0f}%).")
            elif sev_drop < 0.30:
                msgs.append("**Length alone insufficient** "
                             f"(SEQ_LEN=372 reduces severe-failure |err| by only "
                             f"{100*sev_drop:.0f}%). "
                             "State-carrier features (Prompt 10b) are required.")

    if r372 and r400:
        r372_rmse = r372.get("overall_rmse", float("inf"))
        r400_rmse = r400.get("overall_rmse", float("inf"))
        if abs(r372_rmse - r400_rmse) / max(r372_rmse, 1e-6) < 0.02:
            if r325:
                r325_rmse = r325.get("overall_rmse", float("inf"))
                if abs(r325_rmse - r372_rmse) / max(r325_rmse, 1e-6) < 0.02:
                    msgs.append("**Diminishing returns at p75** "
                                 f"(SEQ_LEN=325 and 372 within 2% RMSE). "
                                 "SEQ_LEN=325 is the efficient choice going forward.")
        if r400_rmse > r372_rmse * 1.00:
            msgs.append("**Saturation + overfit at SEQ_LEN=400** "
                         f"(RMSE worsens from {r372_rmse:.2f} at 372 to "
                         f"{r400_rmse:.2f} at 400).")

    return "\n\n".join(msgs) if msgs else "Results pending."


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip training; load existing checkpoints.")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    print("Loading v1 scalers …")
    scaler_X = joblib.load(config.CHECKPOINT_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(config.CHECKPOINT_DIR / "scaler_y.pkl")

    severe_targets = load_severe_targets()
    if severe_targets:
        v1_severe_mean_err = float(np.mean([t["v1_err"] for t in severe_targets]))
        print(f"Loaded {len(severe_targets)} severe-failure targets "
              f"(v1 mean |err| = {v1_severe_mean_err:.1f} Nm)")
    else:
        v1_severe_mean_err = float("nan")

    histories:  dict[int, list]      = {}
    all_preds:  dict[int, np.ndarray] = {}
    all_ytrue:  dict[int, np.ndarray] = {}
    pf_maps:    dict[int, list]       = {}
    rows: list[dict] = []

    for seq_len in SWEEP_SEQ_LENS:
        print(f"\n{'─'*60}")
        print(f"  SEQ_LEN = {seq_len}")

        # Build metadata + y_true for this seq_len
        y_true, file_name, pf_map = _build_test_meta(seq_len)
        all_ytrue[seq_len] = y_true
        pf_maps[seq_len]   = pf_map

        # Build datasets
        print(f"  Building datasets …")
        train_ds, val_ds, test_ds = build_datasets_for(seq_len, scaler_X, scaler_y)

        if args.dry_run:
            model, best_epoch = load_ckpt_gru(seq_len, device)
            hist, mean_epoch_t, vram_gb = [], None, None
        else:
            model, hist, best_epoch, mean_epoch_t, vram_gb = train_one(
                seq_len, train_ds, val_ds, device)
            histories[seq_len] = hist
            # Save history
            hist_path = OUT_DIR / f"history_gru_L{seq_len}.csv"
            with open(hist_path, "w", newline="") as f:
                csv.writer(f).writerows(
                    [["epoch", "train_mse", "val_mse"]] + list(hist))

        print(f"  Running test inference …")
        y_pred = run_inference(model, test_ds, scaler_y, device)
        all_preds[seq_len] = y_pred

        err_abs = np.abs(y_pred - y_true)
        overall_rmse = float(np.sqrt(np.mean(err_abs ** 2)))
        mae          = float(np.mean(err_abs))
        max_err      = float(err_abs.max())

        # Regime classification
        regimes = classify_regimes(y_true, file_name)
        rm = per_regime_rmse(y_true, y_pred, regimes)

        canonical_err = find_canonical_err(y_true, y_pred, pf_map)
        sev           = severe_metrics(y_true, y_pred, pf_map, severe_targets)

        row = dict(
            seq_len=seq_len,
            best_epoch=best_epoch,
            time_per_epoch=mean_epoch_t,
            vram_gb=vram_gb,
            overall_rmse=overall_rmse,
            mae=mae,
            max_abs_err=max_err,
            hold_rmse=rm.get("hold", float("nan")),
            osc_rmse=rm.get("oscillation", float("nan")),
            trans_rmse=rm.get("transition", float("nan")),
            rest_rmse=rm.get("rest", float("nan")),
            err_canonical=canonical_err,
            **sev,
        )
        rows.append(row)

        print(f"  RMSE={overall_rmse:.2f} Nm  Hold={rm.get('hold',float('nan')):.2f}"
              f"  Osc={rm.get('oscillation',float('nan')):.2f}"
              f"  Canonical err={canonical_err}")

    # ── Also get v1 predictions for the pred-vs-truth plot ───────────────────
    print("\nLoading v1 GRU for comparison plot …")
    from _common import build_test_arrays as _bta
    v1_arr = _bta(device=device, run_models=True)
    all_preds[30]  = v1_arr["pred_gru"]
    all_ytrue[30]  = v1_arr["y_true"]

    # Build v1 per_file_map (SEQ_LEN=30)
    _, _, pf_maps[30] = _build_test_meta(30)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if histories:
        plot_training_curves(histories,
                             OUT_DIR / "seq_len_sweep_training_curves.png")

    # Build full rows list including v1 baseline for severe-err plot
    all_rows = [V1_ROW.copy()] + rows
    # Patch v1 severe_mean_err into V1_ROW
    all_rows[0]["severe_mean_err"] = v1_severe_mean_err

    plot_severe_err_vs_seqlen(all_rows,
                              OUT_DIR / "seq_len_sweep_severe_err_vs_seqlen.png")
    plot_pred_vs_truth(all_preds, all_ytrue[30], pf_maps,
                       OUT_DIR / "seq_len_sweep_pred_vs_truth.png")

    # ── Markdown report ───────────────────────────────────────────────────────
    interp = interpret(rows, v1_severe_mean_err)
    write_markdown(all_rows, OUT_DIR / "seq_len_sweep.md", interp,
                   v1_severe_mean_err)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  {'SEQ':>6}  {'RMSE':>8}  {'MAE':>7}  {'Hold':>7}  "
          f"{'Osc':>7}  {'Err38180':>10}  {'SevMean':>9}")
    print(f"{'─'*80}")
    for r in all_rows:
        def _f(v):
            return f"{v:7.2f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "    —  "
        print(f"  {r['seq_len']:>6}  {_f(r['overall_rmse'])}  "
              f"{_f(r['mae'])}  {_f(r['hold_rmse'])}  "
              f"{_f(r['osc_rmse'])}  {_f(r['err_canonical']):>10}  "
              f"{_f(r['severe_mean_err']):>9}")
    print(f"{'═'*80}")

    # ── Append to SUMMARY+CONCLUSION.txt ─────────────────────────────────────
    summary_block = (
        f"\n{'='*72}\n"
        f"[Prompt 2.8 — SEQ_LEN sweep (GRU)]\n"
        f"{'='*72}\n"
        f"Purpose: Quantify how much longer sequence windows reduce catastrophic errors.\n"
        f"SEQ_LEN tested: {[r['seq_len'] for r in rows]}\n"
        f"v1 severe-failure mean |err|: {v1_severe_mean_err:.1f} Nm  "
        f"(234 samples, all > 100 Nm)\n\n"
        f"{interp}\n"
    )
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(summary_block)
    print(f"\nAppended to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

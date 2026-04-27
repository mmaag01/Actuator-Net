"""PLC training-quality diagnostic (companion to Prompt 13).

The per-profile PLC GRU reported best val MSE = 1.013 (normalized) with
val/train ratio = 344 and test RMSE = 40.36 Nm.  This script determines
whether the model is genuinely trained or essentially outputs the mean.

Steps
-----
1. Replay PLC GRU training to capture the full loss curve (saved to CSV so
   subsequent runs skip retraining).
2. Load existing PLC_gru.pt checkpoint; evaluate on PLC test split.
3. Compute three naive baselines on the PLC test split:
       zero mean:    predict 0
       train mean:   predict training-set mean of torAct
       torEst pass:  predict test-sample torEst directly (no learning)
4. Compare GRU RMSE vs all baselines.
5. Plot 5-second (5000-sample) prediction overlay (worst window).
6. Report train/val/test torAct distribution statistics.
7. Write verdict and append to SUMMARY+CONCLUSION.txt.

Outputs
-------
experiments/outputs/plc_check_training_curve.png
experiments/outputs/plc_check_5sec_overlay.png
experiments/outputs/plc_check_distributions.png
experiments/outputs/plc_check_history.csv        (training history cache)
experiments/outputs/plc_check_report.md
prompts/SUMMARY+CONCLUSION.txt  (appended)

Usage
-----
    python experiments/plc_training_check.py [--force]
    --force  Re-run GRU training even if history CSV already exists.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing import _get_feature_cols, _load_dataframes, _make_windows, _split_df
from models import ActuatorGRU, WindowedMLP

PER_PROF_DIR = config.PROJECT_ROOT / "checkpoints" / "per_profile"
OUTPUTS_DIR  = Path(__file__).resolve().parent / "outputs"
SUMMARY_CONC = config.PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_CSV   = OUTPUTS_DIR / "plc_check_history.csv"
SAMPLE_HZ     = 1000.0
SEG_LEN       = 5000   # 5-second overlay segment

PROFILE_FLAGS_PLC = {
    "USE_PLC": True, "USE_PMS": False, "USE_TMS": False, "USE_tStep": False
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true",
                   help="Re-run GRU training even if history CSV exists.")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_plc_dfs():
    saved = {k: getattr(config, k) for k in PROFILE_FLAGS_PLC}
    for k, v in PROFILE_FLAGS_PLC.items():
        setattr(config, k, v)
    try:
        return _load_dataframes()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_gru_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    m = ActuatorGRU(n_features=cfg["n_features"], hidden_size=cfg["hidden_size"],
                    n_layers=cfg["n_layers"])
    m.load_state_dict(ckpt["model_state_dict"])
    return m.to(device).eval(), ckpt["epoch"], ckpt["val_loss"]


def batched_infer(model, X: np.ndarray, device, batch=2048) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            e = min(s + batch, len(X))
            xb = torch.from_numpy(X[s:e]).float().to(device)
            out[s:e] = model(xb).cpu().numpy().ravel()
    return out


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


# ── Training curve capture ────────────────────────────────────────────────────

def run_plc_gru_training(feat_cols, n_features, device, save_path: Path) -> list:
    """Re-run PLC GRU training with fixed seed; return epoch history list."""
    from torch.utils.data import DataLoader as TDL, TensorDataset
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    dfs = load_plc_dfs()
    seq_len = config.SEQ_LEN

    tr_Xp, tr_yp, va_Xp, va_yp = [], [], [], []
    for df in dfs:
        train_df, val_df, _ = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        tr_Xp.append(train_df[feat_cols].values.astype(np.float32))
        tr_yp.append(train_df[config.TARGET_COL].values.astype(np.float32))
        if len(val_df) >= seq_len:
            va_Xp.append(val_df[feat_cols].values.astype(np.float32))
            va_yp.append(val_df[config.TARGET_COL].values.astype(np.float32))

    X_tr = np.concatenate(tr_Xp)
    y_tr = np.concatenate(tr_yp)
    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))

    def make_windows(Xp, yp):
        Xw_l, yw_l = [], []
        for X, y in zip(Xp, yp):
            if len(X) < seq_len:
                continue
            Xs = sx.transform(X).astype(np.float32)
            ys = sy.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
            Xw, yw = _make_windows(Xs, ys, seq_len)
            if len(Xw):
                Xw_l.append(Xw); yw_l.append(yw)
        return np.concatenate(Xw_l), np.concatenate(yw_l)

    Xw_tr, yw_tr = make_windows(tr_Xp, tr_yp)
    Xw_va, yw_va = make_windows(va_Xp, va_yp)

    tr_ds = TensorDataset(torch.from_numpy(Xw_tr), torch.from_numpy(yw_tr).unsqueeze(1))
    va_ds = TensorDataset(torch.from_numpy(Xw_va), torch.from_numpy(yw_va).unsqueeze(1))
    tr_ld = TDL(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    va_ld = TDL(va_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"  Train windows: {len(Xw_tr):,}  Val windows: {len(Xw_va):,}")

    model = ActuatorGRU(n_features=n_features, hidden_size=config.GRU_HIDDEN_SIZE,
                        n_layers=config.GRU_N_LAYERS, dropout=config.GRU_DROPOUT).to(device)
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
            opt.step(); sched.step()
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
        tag = " ★" if improved else f" ({pat_ctr+1}/{config.PATIENCE})"
        print(f"    Ep {epoch:4d} | tr {tr_loss:.5f} | val {vl_loss:.5f}{tag}")

        if improved:
            best_val, pat_ctr, best_ep = vl_loss, 0, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": vl_loss, "model_type": "gru",
                        "config": {"seq_len": config.SEQ_LEN, "n_features": n_features,
                                   "hidden_size": config.GRU_HIDDEN_SIZE,
                                   "n_layers": config.GRU_N_LAYERS}}, save_path)
        else:
            pat_ctr += 1
            if pat_ctr >= config.PATIENCE:
                print(f"    Early stop at epoch {epoch}. Best val={best_val:.6f} @ ep {best_ep}")
                break

    print(f"  GRU training complete: best val MSE={best_val:.6f} @ epoch {best_ep}")
    return history, best_ep, best_val


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curve(history: list, best_ep: int, best_val: float, path: Path):
    epochs   = [h[0] for h in history]
    tr_loss  = [h[1] for h in history]
    val_loss = [h[2] for h in history]
    early_ep = epochs[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Full-scale view
    ax = axes[0]
    ax.plot(epochs, tr_loss,  color="steelblue", lw=1.2, label="Train MSE")
    ax.plot(epochs, val_loss, color="tomato",    lw=1.2, label="Val MSE")
    ax.axvline(best_ep, color="green", ls="--", lw=1.0,
               label=f"Best ep={best_ep} (val={best_val:.4f})")
    ax.axvline(early_ep, color="grey", ls=":", lw=1.0,
               label=f"Early stop ep={early_ep}")
    ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.6, label="MSE=1.0 (predict mean)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (normalized)")
    ax.set_title("PLC GRU — training curve (full scale)")
    ax.legend(fontsize=8)

    # Zoomed: train loss only (val never drops so zoom would just show flat line)
    ax2 = axes[1]
    ax2.plot(epochs, tr_loss, color="steelblue", lw=1.2, label="Train MSE")
    val_min = min(val_loss)
    ax2.axhline(val_min, color="tomato", ls="--", lw=1.0,
                label=f"Val min={val_min:.4f}")
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE (log scale)")
    ax2.set_title("PLC GRU — train MSE (log scale)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_overlay(y_true, pred_gru, torEst, train_mean, path: Path):
    n = len(y_true)
    if n < SEG_LEN:
        s, e = 0, n
    else:
        # worst SEG_LEN-sample window by mean |gru_error|
        err = np.abs(pred_gru - y_true)
        conv = np.convolve(err, np.ones(SEG_LEN) / SEG_LEN, mode="valid")
        s = int(np.argmax(conv))
        e = s + SEG_LEN
    t = np.arange(e - s) / SAMPLE_HZ  # seconds

    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(t, y_true[s:e],    label="torAct (truth)",        color="steelblue", lw=1.2)
    ax.plot(t, pred_gru[s:e],  label="per-profile GRU",       color="tomato",    lw=1.0)
    ax.plot(t, torEst[s:e],    label="torEst (passthrough)",   color="seagreen",  lw=0.9, ls="--")
    ax.axhline(train_mean, color="grey", ls=":", lw=0.9,
               label=f"train mean = {train_mean:.1f} Nm")
    ax.set_ylabel("Torque [Nm]")
    ax.set_title(f"PLC test — worst {SEG_LEN/SAMPLE_HZ:.0f}-second window "
                 f"(samples {s}–{e})")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(t, pred_gru[s:e] - y_true[s:e],
             label="GRU error", color="tomato", lw=0.9)
    ax2.plot(t, torEst[s:e] - y_true[s:e],
             label="torEst error", color="seagreen", lw=0.9, ls="--")
    ax2.axhline(0, color="black", lw=0.6, ls="--")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Error [Nm]")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_distributions(split_stats: dict, path: Path):
    splits = ["train", "val", "test"]
    colors = {"train": "steelblue", "val": "darkorange", "test": "seagreen"}
    metrics = ["mean", "std", "p05", "p95"]

    x = np.arange(len(metrics))
    w = 0.25
    offsets = [-w, 0, w]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart of statistics
    ax = axes[0]
    for i, split in enumerate(splits):
        vals = [split_stats[split][m] for m in metrics]
        ax.bar(x + offsets[i], vals, w, label=split, color=colors[split], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("torAct [Nm]")
    ax.set_title("PLC torAct: train / val / test distribution stats")
    ax.legend()

    # Distribution overlap (histograms)
    ax2 = axes[1]
    for split in splits:
        vals = split_stats[split]["raw"]
        ax2.hist(vals, bins=150, density=True, alpha=0.5,
                 color=colors[split], label=split)
    ax2.set_xlabel("torAct [Nm]")
    ax2.set_ylabel("Density")
    ax2.set_title("PLC torAct density: train / val / test")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(history, best_ep, best_val, baselines, gru_rmse,
                 split_stats, verdict, path: Path):
    val_ever_below_05 = any(h[2] < 0.5 for h in history)
    val_min = min(h[2] for h in history)
    early_ep = history[-1][0]
    last_tr  = history[-1][1]

    lines = [
        "# PLC Training Quality Diagnostic",
        "",
        "## Training curve summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Best epoch | {best_ep} |",
        f"| Early-stop epoch | {early_ep} |",
        f"| Best val MSE (normalized) | {best_val:.6f} |",
        f"| Val MSE ever < 0.5 | {'Yes' if val_ever_below_05 else 'No'} |",
        f"| Val MSE minimum | {val_min:.6f} |",
        f"| Train MSE at early stop | {last_tr:.6f} |",
        f"| val / train ratio | {best_val/last_tr:.2f} |",
        "",
        "## PLC test-set baselines vs GRU",
        "",
        "| Model | RMSE [Nm] | vs GRU Δ% |",
        "|---|---|---|",
    ]
    for name, base_rmse in baselines.items():
        delta = (gru_rmse - base_rmse) / base_rmse * 100
        lines.append(f"| {name} | {base_rmse:.4f} | {delta:+.1f}% |")
    lines.append(f"| **per-profile GRU** | **{gru_rmse:.4f}** | — |")

    lines += [
        "",
        "## Train / Val / Test torAct distribution",
        "",
        "| Split | mean | std | p05 | p95 | min | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in ["train", "val", "test"]:
        s = split_stats[split]
        lines.append(
            f"| {split} | {s['mean']:.2f} | {s['std']:.2f} | "
            f"{s['p05']:.2f} | {s['p95']:.2f} | {s['min']:.2f} | {s['max']:.2f} |"
        )

    lines += ["", "## Interpretation", "", f"**{verdict}**"]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def append_summary(verdict, gru_rmse, baselines):
    block = (
        "\n"
        "========================================================================\n"
        "[PLC Training Quality Diagnostic — Prompt 13 companion]\n"
        "========================================================================\n"
        "Purpose: Determine whether per-profile PLC GRU (val MSE=1.013, val/train\n"
        "ratio=344, test RMSE=40.36 Nm) is genuinely trained or merely outputs\n"
        "the training-set mean.\n"
        "\n"
        f"Per-profile GRU test RMSE: {gru_rmse:.4f} Nm\n"
    )
    for name, r in baselines.items():
        block += f"  {name}: {r:.4f} Nm\n"
    block += f"Verdict: {verdict}\n"
    with open(SUMMARY_CONC, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"Appended to {SUMMARY_CONC}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    feat_cols  = _get_feature_cols()
    n_features = len(feat_cols)
    seq_len    = config.SEQ_LEN

    print("=" * 65)
    print("PLC training quality diagnostic")
    print(f"Device   : {device}")
    print(f"Features : {n_features}  SEQ_LEN={seq_len}  SMOOTH_ACCEL={config.SMOOTH_ACCEL}")
    print("=" * 65)

    # ── 1. Training curve ─────────────────────────────────────────────────────
    diag_ckpt = PER_PROF_DIR / "PLC_gru_diag.pt"

    if HISTORY_CSV.exists() and not args.force:
        print(f"\n[1] Loading cached training history from {HISTORY_CSV}")
        history = []
        with open(HISTORY_CSV, newline="") as f:
            for row in csv.DictReader(f):
                history.append((int(row["epoch"]),
                                 float(row["train_mse"]),
                                 float(row["val_mse"])))
        # Load checkpoint metadata
        ckpt_data = torch.load(PER_PROF_DIR / "PLC_gru.pt",
                               map_location="cpu", weights_only=False)
        best_ep  = ckpt_data["epoch"]
        best_val = ckpt_data["val_loss"]
        print(f"  Loaded {len(history)} epochs. best_ep={best_ep} best_val={best_val:.6f}")
    else:
        print("\n[1] Running PLC GRU training to capture history …")
        history, best_ep, best_val = run_plc_gru_training(
            feat_cols, n_features, device, diag_ckpt)
        with open(HISTORY_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_mse", "val_mse"])
            w.writerows(history)
        print(f"  History saved to {HISTORY_CSV}")

    plot_training_curve(history, best_ep, best_val,
                        OUTPUTS_DIR / "plc_check_training_curve.png")

    # ── 2. Load PLC data and scalers ──────────────────────────────────────────
    print("\n[2] Loading PLC data and scalers …")
    dfs = load_plc_dfs()
    sx  = joblib.load(PER_PROF_DIR / "PLC_scaler_X.pkl")
    sy  = joblib.load(PER_PROF_DIR / "PLC_scaler_y.pkl")

    # Distribution stats per split (raw torAct, before windowing)
    tr_act_raw, va_act_raw, te_act_raw = [], [], []
    tr_est_raw, te_est_raw             = [], []
    for df in dfs:
        train_df, val_df, test_df = _split_df(
            df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        tr_act_raw.append(train_df[config.TARGET_COL].values)
        tr_est_raw.append(train_df["torEst"].values)
        va_act_raw.append(val_df[config.TARGET_COL].values)
        te_act_raw.append(test_df[config.TARGET_COL].values)
        te_est_raw.append(test_df["torEst"].values)

    tr_act = np.concatenate(tr_act_raw)
    va_act = np.concatenate(va_act_raw)
    te_act = np.concatenate(te_act_raw)
    te_est = np.concatenate(te_est_raw)
    train_mean_tor = float(np.mean(tr_act))

    def dist_stats(arr):
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "min": float(np.min(arr)), "max": float(np.max(arr)),
                "p05": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)), "raw": arr}

    split_stats = {
        "train": dist_stats(tr_act),
        "val":   dist_stats(va_act),
        "test":  dist_stats(te_act),
    }

    print("\n  torAct distribution per split:")
    for split in ["train", "val", "test"]:
        s = split_stats[split]
        print(f"  {split:6s}: mean={s['mean']:7.2f}  std={s['std']:6.2f}  "
              f"[{s['min']:.1f}, {s['max']:.1f}]  p05={s['p05']:.1f} p95={s['p95']:.1f}")

    plot_distributions(split_stats, OUTPUTS_DIR / "plc_check_distributions.png")

    # ── 3. Build test windows ─────────────────────────────────────────────────
    print("\n[3] Building PLC test windows …")
    te_Xw_l, te_yw_l, te_est_l = [], [], []
    for df in dfs:
        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        if len(test_df) < seq_len:
            continue
        X_raw  = test_df[feat_cols].values.astype(np.float32)
        y_raw  = test_df[config.TARGET_COL].values.astype(np.float32)
        te_raw = test_df["torEst"].values.astype(np.float32)
        Xs = sx.transform(X_raw).astype(np.float32)
        Xw, _ = _make_windows(Xs, np.zeros(len(Xs), np.float32), seq_len)
        tgt = slice(seq_len - 1, len(test_df))
        te_Xw_l.append(Xw)
        te_yw_l.append(y_raw[tgt])
        te_est_l.append(te_raw[tgt])

    Xw_te   = np.concatenate(te_Xw_l)
    y_te    = np.concatenate(te_yw_l)
    est_te  = np.concatenate(te_est_l)
    print(f"  Test windows: {len(y_te):,}")

    # ── 4. GRU inference ──────────────────────────────────────────────────────
    print("\n[4] Running GRU inference …")
    gru_ckpt_path = PER_PROF_DIR / "PLC_gru.pt"
    gru_model, ckpt_ep, ckpt_val = load_gru_ckpt(gru_ckpt_path, device)
    print(f"  Loaded checkpoint: epoch={ckpt_ep} val_loss={ckpt_val:.6f}")

    pred_sc  = batched_infer(gru_model, Xw_te, device)
    pred_gru = sy.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
    gru_rmse = rmse(y_te, pred_gru)
    print(f"  GRU test RMSE = {gru_rmse:.4f} Nm")

    # ── 5. Baselines ──────────────────────────────────────────────────────────
    print("\n[5] Computing baselines …")
    baselines = {
        "predict zero":       rmse(y_te, np.zeros_like(y_te)),
        "predict train mean": rmse(y_te, np.full_like(y_te, train_mean_tor)),
        "predict torEst":     rmse(y_te, est_te),
    }
    print(f"  Baselines on PLC test set:")
    for name, r in baselines.items():
        beats = "GRU better" if gru_rmse < r else "GRU WORSE"
        delta_pct = (gru_rmse - r) / r * 100
        print(f"    {name:22s}: {r:.4f} Nm  ({beats}, {delta_pct:+.1f}%)")

    # ── 6. 5-second overlay plot ──────────────────────────────────────────────
    print("\n[6] 5-second overlay plot (worst window) …")
    plot_overlay(y_te, pred_gru, est_te, train_mean_tor,
                 OUTPUTS_DIR / "plc_check_5sec_overlay.png")

    # ── 7. Verdict ────────────────────────────────────────────────────────────
    base_mean_rmse = baselines["predict train mean"]
    base_test_rmse = baselines["predict torEst"]
    val_ever_below = any(h[2] < 0.5 for h in history)
    std_ratio = split_stats["val"]["std"] / split_stats["train"]["std"] if split_stats["train"]["std"] > 0 else float("nan")

    beats_mean  = gru_rmse < base_mean_rmse * 0.80   # >20% better than mean
    beats_torest = gru_rmse < base_test_rmse * 0.80  # >20% better than torEst passthrough

    if not val_ever_below and std_ratio > 1.5:
        verdict = (
            f"Model trained, val set pathological: val MSE never dropped below 0.5 "
            f"(min={min(h[2] for h in history):.4f}) but val torAct std "
            f"({split_stats['val']['std']:.2f} Nm) is {std_ratio:.1f}× train std "
            f"({split_stats['train']['std']:.2f} Nm). "
            f"The 40.36 Nm test number is real but val loss was not a useful stopping "
            f"criterion — early stopping fired at epoch {history[-1][0]} on pathological val loss."
        )
    elif not val_ever_below and std_ratio <= 1.5:
        verdict = (
            f"Model undertrained: val MSE never dropped below 0.5 "
            f"(min={min(h[2] for h in history):.4f}) AND train/val distributions are comparable "
            f"(val std {split_stats['val']['std']:.2f} Nm = {std_ratio:.1f}× train std). "
            f"Test RMSE {gru_rmse:.2f} Nm ≈ predict-mean ({base_mean_rmse:.2f} Nm). "
            f"The 40.36 Nm result is not informative — the model is near random-init performance."
        )
    elif beats_mean and beats_torest:
        verdict = (
            f"Model trained, learned something useful: GRU ({gru_rmse:.2f} Nm) beats "
            f"predict-mean ({base_mean_rmse:.2f} Nm) by >{(1 - gru_rmse/base_mean_rmse)*100:.0f}% "
            f"and torEst passthrough ({base_test_rmse:.2f} Nm) by >{(1 - gru_rmse/base_test_rmse)*100:.0f}%."
        )
    else:
        verdict = (
            f"Model barely trained: GRU ({gru_rmse:.2f} Nm) does not beat all baselines "
            f"by >20%. predict-mean={base_mean_rmse:.2f} torEst={base_test_rmse:.2f}. "
            f"PLC loss landscape may be degenerate at current hyperparameters."
        )

    print(f"\n[Verdict] {verdict}")

    # ── 8. Write report ───────────────────────────────────────────────────────
    write_report(history, best_ep, best_val, baselines, gru_rmse,
                 split_stats, verdict,
                 OUTPUTS_DIR / "plc_check_report.md")

    # ── 9. Append to SUMMARY+CONCLUSION.txt ──────────────────────────────────
    append_summary(verdict, gru_rmse, baselines)

    print("\nDone.")


if __name__ == "__main__":
    main()

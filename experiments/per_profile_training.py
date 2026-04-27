"""Prompt 13 — Per-profile model training.

Trains MLP and GRU from scratch on each profile type independently (PLC, PMS,
TMS, tStep) to isolate whether mixed-profile training (cause 1) is responsible
for the v1 PLC RMSE gap vs. literature.

Outputs
-------
experiments/outputs/per_profile_training.md
experiments/outputs/per_profile_training_curves.png
experiments/outputs/per_profile_cross_heatmap.png
experiments/outputs/per_profile_plc_comparison.png
prompts/SUMMARY+CONCLUSION.txt (appended)

Checkpoints: checkpoints/per_profile/{profile}_{arch}.pt
Scalers:     checkpoints/per_profile/{profile}_scaler_{X,y}.pkl

Usage
-----
    python experiments/per_profile_training.py [--force]
    --force  Retrain from scratch even if checkpoints exist.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from preprocessing import _get_feature_cols, _load_dataframes, _make_windows, _split_df
from models import ActuatorGRU, WindowedMLP

OUTPUTS_DIR   = Path(__file__).resolve().parent / "outputs"
PER_PROF_DIR  = config.PROJECT_ROOT / "checkpoints" / "per_profile"
SUMMARY_CONC  = config.PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PER_PROF_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = ["PLC", "PMS", "TMS", "tStep"]
ARCHS    = ["mlp", "gru"]

PROFILE_COLORS = {"PLC": "steelblue", "PMS": "seagreen",
                  "TMS": "darkorange", "tStep": "tomato"}
ARCH_LS = {"mlp": "--", "gru": "-"}

# v1 per-profile RMSE from results/v1/Metrics.md (used as fallback if checkpoint missing)
V1_RMSE_SPEC = {
    "mlp": {"tStep": 1.734929, "TMS": 4.731154, "PMS": 8.177079, "PLC": 51.310169},
    "gru": {"tStep": 0.365783, "TMS": 4.555592, "PMS": 8.079439, "PLC": 67.246284},
}

# Original v1 feature columns (for loading v1 checkpoint correctly)
V1_FEATURE_COLS = [
    'torDes', 'posDes', 'velDes', 'posAct', 'velAct',
    'accelAct', 'i', 'torEst', 'posErr', 'velErr',
]

# Per-profile config flag overrides
PROFILE_FLAGS = {
    "PLC":   {"USE_PLC": True,  "USE_PMS": False, "USE_TMS": False, "USE_tStep": False},
    "PMS":   {"USE_PLC": False, "USE_PMS": True,  "USE_TMS": False, "USE_tStep": False},
    "TMS":   {"USE_PLC": False, "USE_PMS": False, "USE_TMS": True,  "USE_tStep": False},
    "tStep": {"USE_PLC": False, "USE_PMS": False, "USE_TMS": False, "USE_tStep": True},
}

SAMPLE_HZ = 1000.0


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


# ── Profile-specific data loading ─────────────────────────────────────────────

def load_profile_dfs(profile: str, smooth_accel: bool | None = None) -> list:
    """Load DFs for one profile only, temporarily overriding config flags."""
    flags = dict(PROFILE_FLAGS[profile])
    if smooth_accel is not None:
        flags["SMOOTH_ACCEL"] = smooth_accel
    saved = {k: getattr(config, k) for k in flags}
    for k, v in flags.items():
        setattr(config, k, v)
    try:
        dfs = _load_dataframes()
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
    return dfs


# ── Split and window building ─────────────────────────────────────────────────

def build_split_arrays(dfs, feat_cols):
    """Return (train_X_parts, train_y_parts, val_X_parts, val_y_parts,
    test_X_parts, test_y_parts)."""
    seq_len = config.SEQ_LEN
    tr_X, tr_y, va_X, va_y, te_X, te_y = [], [], [], [], [], []
    for df in dfs:
        train_df, val_df, test_df = _split_df(
            df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len
        )
        tr_X.append(train_df[feat_cols].values.astype(np.float32))
        tr_y.append(train_df[config.TARGET_COL].values.astype(np.float32))
        va_X.append(val_df[feat_cols].values.astype(np.float32))
        va_y.append(val_df[config.TARGET_COL].values.astype(np.float32))
        te_X.append(test_df[feat_cols].values.astype(np.float32))
        te_y.append(test_df[config.TARGET_COL].values.astype(np.float32))
    return tr_X, tr_y, va_X, va_y, te_X, te_y


def fit_scaler(X_parts, y_parts):
    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)
    sx = StandardScaler().fit(X_all)
    sy = StandardScaler().fit(y_all.reshape(-1, 1))
    return sx, sy


def build_windows(X_parts, y_parts, sx, sy):
    """Scale and window; return concatenated (X_wins, y_wins) arrays."""
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


def build_test_raw(dfs, feat_cols, sx, sy):
    """Return (y_true, X_windows, file_name) aligned test arrays."""
    seq_len = config.SEQ_LEN
    y_parts, Xw_parts, fn_parts = [], [], []
    for df in dfs:
        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        if len(test_df) < seq_len:
            continue
        X_raw = test_df[feat_cols].values.astype(np.float32)
        y_raw = test_df[config.TARGET_COL].values.astype(np.float32)
        Xs    = sx.transform(X_raw).astype(np.float32)
        Xw, _ = _make_windows(Xs, np.zeros(len(Xs), np.float32), seq_len)
        tgt   = slice(seq_len - 1, len(test_df))
        y_parts.append(y_raw[tgt])
        Xw_parts.append(Xw)
        fn = (test_df["file_name"].values[tgt].astype(str)
              if "file_name" in test_df else
              np.full(len(Xw), "unknown", object))
        fn_parts.append(fn)
    return (np.concatenate(y_parts), np.concatenate(Xw_parts),
            np.concatenate(fn_parts))


# ── Model building ────────────────────────────────────────────────────────────

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

def train_one(arch: str, train_Xw: np.ndarray, train_yw: np.ndarray,
              val_Xw: np.ndarray, val_yw: np.ndarray,
              n_features: int, device, ckpt_path: Path,
              force: bool, profile: str) -> tuple[nn.Module, list, int, float]:
    """Train one model. Returns (model, history, best_epoch, best_val)."""
    if ckpt_path.exists() and not force:
        print(f"    [{arch.upper()}] checkpoint found — skipping training.")
        model, best_ep, best_val = load_ckpt(ckpt_path, arch, device)
        return model, [], best_ep, best_val

    from torch.utils.data import DataLoader as TDL, TensorDataset
    train_ds = TensorDataset(torch.from_numpy(train_Xw), torch.from_numpy(train_yw).unsqueeze(1))
    val_ds   = TensorDataset(torch.from_numpy(val_Xw),   torch.from_numpy(val_yw).unsqueeze(1))
    train_ld = TDL(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_ld   = TDL(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)

    model = build_model(arch, n_features).to(device)
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=config.LR,
                              weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_ld) * config.MAX_EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.ONE_CYCLE_MAX_LR, total_steps=total_steps
    )

    best_val, pat_ctr = float("inf"), 0
    best_ep  = 1
    history  = []   # [(epoch, train_mse, val_mse)]
    last_tr  = float("nan")

    for epoch in range(1, config.MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_ld:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            opt.step()
            sched.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(train_ds)
        last_tr  = tr_loss

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_ld:
                vl_loss += crit(model(Xb.to(device)), yb.to(device)).item() * len(Xb)
        vl_loss /= len(val_ds)
        history.append((epoch, tr_loss, vl_loss))

        improved = vl_loss < best_val
        tag = " [saved]" if improved else f" (pat {pat_ctr+1}/{config.PATIENCE})"
        print(f"      Ep {epoch:4d} | tr {tr_loss:.5f} | val {vl_loss:.5f}{tag}")

        if improved:
            best_val, pat_ctr, best_ep = vl_loss, 0, epoch
            save_ckpt(model, arch, n_features, epoch, vl_loss, ckpt_path)
        else:
            pat_ctr += 1
            if pat_ctr >= config.PATIENCE:
                print(f"      Early stop at epoch {epoch}.")
                break

    print(f"    [{arch.upper()}] {profile}: best val MSE={best_val:.6f} @ ep {best_ep}  "
          f"val/train ratio={best_val/last_tr:.2f}")
    model, best_ep, best_val = load_ckpt(ckpt_path, arch, device)
    return model, history, best_ep, best_val


# ── Inference ─────────────────────────────────────────────────────────────────

def batched_infer(model, X: np.ndarray, device, batch=2048) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            e  = min(s + batch, len(X))
            xb = torch.from_numpy(X[s:e]).float().to(device)
            out[s:e] = model(xb).cpu().numpy().ravel()
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def predict(model, X_windows, sy, device) -> np.ndarray:
    pred_sc = batched_infer(model, X_windows, device)
    return sy.inverse_transform(pred_sc.reshape(-1, 1)).ravel()


# ── v1 baseline evaluation ────────────────────────────────────────────────────

def eval_v1_per_profile(device, feat_cols: list) -> dict:
    """Evaluate v1 checkpoints on per-profile test splits.

    Uses the same feat_cols and SMOOTH_ACCEL setting as v1 training so that
    the v1 scaler is applied to correctly-ordered, correctly-preprocessed data.
    """
    v1_dir = config.PROJECT_ROOT / "checkpoints"
    sx_path = v1_dir / "scaler_X.pkl"
    sy_path = v1_dir / "scaler_y.pkl"
    mlp_ckpt_path = v1_dir / "best_model_mlp.pt"
    gru_ckpt_path = v1_dir / "best_model_gru.pt"

    if not all(p.exists() for p in [sx_path, sy_path, mlp_ckpt_path, gru_ckpt_path]):
        print("  v1 checkpoints/scalers not found — using spec numbers from Metrics.md")
        return V1_RMSE_SPEC

    sx = joblib.load(sx_path)
    sy = joblib.load(sy_path)

    if sx.n_features_in_ != len(feat_cols):
        print(f"  v1 scaler n_features={sx.n_features_in_} != {len(feat_cols)} "
              "— using spec numbers")
        return V1_RMSE_SPEC

    mlp_v1, _, _ = load_ckpt(mlp_ckpt_path, "mlp", device)
    gru_v1, _, _ = load_ckpt(gru_ckpt_path, "gru", device)

    result = {"mlp": {}, "gru": {}}
    for profile in PROFILES:
        dfs = load_profile_dfs(profile)
        y_true, X_wins, _ = build_test_raw(dfs, feat_cols, sx, sy)
        if len(y_true) == 0:
            result["mlp"][profile] = float("nan")
            result["gru"][profile] = float("nan")
            continue
        pred_mlp = predict(mlp_v1, X_wins, sy, device)
        pred_gru = predict(gru_v1, X_wins, sy, device)
        result["mlp"][profile] = rmse(y_true, pred_mlp)
        result["gru"][profile] = rmse(y_true, pred_gru)
        print(f"  v1 {profile:6s}: MLP={result['mlp'][profile]:.3f}  GRU={result['gru'][profile]:.3f} Nm")
    return result


# ── Training curves plot ──────────────────────────────────────────────────────

def plot_training_curves(all_history: dict, path: Path):
    """all_history[profile][arch] = list of (epoch, train_mse, val_mse)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, arch in zip(axes, ARCHS):
        for prof in PROFILES:
            hist = all_history.get(prof, {}).get(arch, [])
            if not hist:
                continue
            epochs  = [h[0] for h in hist]
            val_mse = [h[2] for h in hist]
            ax.plot(epochs, val_mse, color=PROFILE_COLORS[prof],
                    label=prof, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val MSE (scaled)")
        ax.set_title(f"{arch.upper()} — per-profile training curves")
        ax.legend()
    fig.suptitle("Per-profile Training Curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Cross-profile heatmap ─────────────────────────────────────────────────────

def plot_cross_heatmap(cross_rmse: dict, path: Path):
    """cross_rmse[train_profile][test_profile] = RMSE (GRU)."""
    n = len(PROFILES)
    mat = np.full((n, n), np.nan)
    for i, tp in enumerate(PROFILES):
        for j, ep in enumerate(PROFILES):
            if tp in cross_rmse and ep in cross_rmse[tp]:
                mat[i, j] = cross_rmse[tp][ep]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log1p(mat), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(PROFILES, rotation=30, ha="right")
    ax.set_yticklabels(PROFILES)
    ax.set_xlabel("Test profile")
    ax.set_ylabel("Trained on profile")
    ax.set_title("GRU Cross-profile RMSE [Nm] (log scale)")
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if np.log1p(v) > np.log1p(mat[~np.isnan(mat)].mean()) else "black")
    plt.colorbar(im, ax=ax, label="log(1+RMSE)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ── PLC before/after time-series plot ────────────────────────────────────────

def plot_plc_comparison(y_true, pred_v1_gru, pred_pp_gru, path: Path):
    """Plot worst-100-sample window comparing v1 vs per-profile PLC GRU."""
    err_pp = np.abs(pred_pp_gru - y_true)
    # Find worst 100-sample window by per-profile error
    conv  = np.convolve(err_pp, np.ones(100) / 100, mode="valid")
    start = int(np.argmax(conv))
    end   = start + 100
    t     = np.arange(100) / SAMPLE_HZ * 1000  # ms

    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                              gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t, y_true[start:end], label="torAct (truth)", color="steelblue", lw=1.2)
    axes[0].plot(t, pred_v1_gru[start:end], label="v1 GRU (mixed)", color="grey",
                  lw=1.0, linestyle="--")
    axes[0].plot(t, pred_pp_gru[start:end], label="per-profile PLC GRU",
                  color="tomato", lw=1.2, linestyle="-")
    axes[0].set_ylabel("Torque [Nm]")
    axes[0].set_title("PLC test: v1 GRU vs per-profile PLC GRU (worst 100-sample window)")
    axes[0].legend()

    axes[1].plot(t, pred_v1_gru[start:end] - y_true[start:end],
                  color="grey",  lw=0.9, label="v1 GRU error")
    axes[1].plot(t, pred_pp_gru[start:end] - y_true[start:end],
                  color="tomato", lw=0.9, label="per-profile error")
    axes[1].axhline(0, color="black", lw=0.6, linestyle="--")
    axes[1].set_xlabel("Time within window [ms]")
    axes[1].set_ylabel("Error [Nm]")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_report(v1_rmse, pp_rmse, cross_rmse, dyn, verdict, path: Path):
    """v1_rmse[arch][profile], pp_rmse[arch][profile],
    cross_rmse[train_prof][test_prof] (GRU),
    dyn[profile] = {train_samples, best_ep, best_val, last_tr}."""
    def fmt(d, arch, prof):
        v = d.get(arch, {}).get(prof, float("nan"))
        return f"{v:.2f}" if not np.isnan(v) else "n/a"

    lines = [
        "# Prompt 13 — Per-profile Model Training",
        "",
        "## Primary: matched training/test RMSE [Nm]",
        "",
        "| Profile | v1 MLP | per-profile MLP | Δ MLP | v1 GRU | per-profile GRU | Δ GRU |",
        "|---------|--------|-----------------|-------|--------|-----------------|-------|",
    ]
    for prof in PROFILES:
        v1m  = v1_rmse["mlp"].get(prof, float("nan"))
        ppm  = pp_rmse["mlp"].get(prof, float("nan"))
        v1g  = v1_rmse["gru"].get(prof, float("nan"))
        ppg  = pp_rmse["gru"].get(prof, float("nan"))
        dm   = (ppm - v1m) if not (np.isnan(ppm) or np.isnan(v1m)) else float("nan")
        dg   = (ppg - v1g) if not (np.isnan(ppg) or np.isnan(v1g)) else float("nan")
        def f(v): return f"{v:.2f}" if not np.isnan(v) else "n/a"
        def fd(v): return f"{v:+.2f}" if not np.isnan(v) else "n/a"
        lines.append(f"| {prof} | {f(v1m)} | {f(ppm)} | {fd(dm)} "
                     f"| {f(v1g)} | {f(ppg)} | {fd(dg)} |")
    lines += [
        "",
        "## Cross-profile generalization (GRU, RMSE [Nm])",
        "",
        "Rows = trained on, columns = tested on. Diagonal = matched.",
        "",
        "| Trained↓ / Test→ | " + " | ".join(PROFILES) + " |",
        "|---|" + "|".join(["---"] * len(PROFILES)) + "|",
    ]
    for tp in PROFILES:
        row = [f"**{tp}**"]
        for ep in PROFILES:
            v = cross_rmse.get(tp, {}).get(ep, float("nan"))
            cell = f"{v:.2f}" if not np.isnan(v) else "n/a"
            if tp == ep:
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Training dynamics (GRU)",
        "",
        "| Profile | Train windows | Best epoch | Best val MSE | val/train ratio |",
        "|---------|--------------|------------|--------------|-----------------|",
    ]
    for prof in PROFILES:
        d = dyn.get(prof, {})
        n   = d.get("train_samples", "?")
        ep  = d.get("best_ep", "?")
        bv  = d.get("best_val", float("nan"))
        lt  = d.get("last_tr", float("nan"))
        rat = bv / lt if (not np.isnan(bv) and not np.isnan(lt) and lt > 0) else float("nan")
        lines.append(f"| {prof} | {n} | {ep} | "
                     f"{'n/a' if np.isnan(bv) else f'{bv:.6f}'} | "
                     f"{'n/a' if np.isnan(rat) else f'{rat:.2f}'} |")
    lines += [
        "",
        "## Interpretation",
        "",
        f"**{verdict}**",
        "",
        "- Per-profile PLC GRU closes gap: if RMSE < 5 Nm → cause (1) dominates.",
        "- Partially closes: 5–30 Nm → dilution real but not full story.",
        "- Does not close: >30 Nm → PLC data intrinsically harder; causes (2)–(5).",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


# ── SUMMARY+CONCLUSION append ─────────────────────────────────────────────────

def append_summary(pp_rmse, v1_rmse, verdict):
    plc_v1  = v1_rmse["gru"].get("PLC", float("nan"))
    plc_pp  = pp_rmse["gru"].get("PLC", float("nan"))
    block = (
        "\n"
        "========================================================================\n"
        "[Prompt 13 — Per-profile model training]\n"
        "========================================================================\n"
        "Purpose: Isolate whether mixed-profile training (cause 1) explains the\n"
        "v1 PLC GRU RMSE gap (67.25 Nm) vs. literature sub-1 Nm benchmarks.\n"
        "\n"
        f"v1 GRU PLC RMSE: {plc_v1:.2f} Nm\n"
        f"Per-profile PLC GRU RMSE: {plc_pp:.2f} Nm\n"
        f"Verdict: {verdict}\n"
        "\n"
        "Implication: "
    )
    if "closes" in verdict.lower() and "partially" not in verdict.lower():
        block += ("Mixed-profile training was the primary cause. Per-profile models "
                  "are required for literature-level PLC RMSE.\n")
    elif "partially" in verdict.lower():
        block += ("Mixed-profile training is a contributing factor but not the full "
                  "explanation. Investigate causes (2)–(5).\n")
    else:
        block += ("PLC data is intrinsically difficult regardless of training-set "
                  "composition. Pivot to causes (2)–(5): feature set, excitation "
                  "range, control topology, or window/rate mismatches.\n")
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
    print("=" * 60)
    print("Prompt 13 — Per-profile model training")
    print(f"Device: {device}")
    print(f"Features ({n_features}): {feat_cols}")
    print(f"SEQ_LEN={config.SEQ_LEN}  SMOOTH_ACCEL={config.SMOOTH_ACCEL}")
    print("=" * 60)

    # ── 1. Evaluate v1 baseline per-profile ───────────────────────────────────
    print("\n[1] Evaluating v1 per-profile baseline …")
    v1_rmse = eval_v1_per_profile(device, feat_cols)

    # ── 2. Train per-profile models ───────────────────────────────────────────
    pp_models   = {}   # [profile][arch] = model
    pp_scalers  = {}   # [profile] = (sx, sy)
    pp_rmse     = {"mlp": {}, "gru": {}}
    cross_dfs   = {}   # [profile] = (dfs, sx, sy) for cross-eval
    dynamics    = {}   # [profile] = {train_samples, best_ep, best_val, last_tr}
    all_history = {}   # [profile][arch] = history

    for profile in PROFILES:
        print(f"\n{'='*50}")
        print(f"[{profile}] Loading profile data …")
        dfs = load_profile_dfs(profile)

        print(f"  Building splits …")
        tr_X, tr_y, va_X, va_y, te_X, te_y = build_split_arrays(dfs, feat_cols)

        sx, sy = fit_scaler(tr_X, tr_y)
        joblib.dump(sx, PER_PROF_DIR / f"{profile}_scaler_X.pkl")
        joblib.dump(sy, PER_PROF_DIR / f"{profile}_scaler_y.pkl")
        pp_scalers[profile]  = (sx, sy)
        cross_dfs[profile]   = (dfs, sx, sy)

        Xw_tr, yw_tr = build_windows(tr_X, tr_y, sx, sy)
        Xw_va, yw_va = build_windows(va_X, va_y, sx, sy)
        y_te_arr, Xw_te, fn_te = build_test_raw(dfs, feat_cols, sx, sy)

        n_tr = len(Xw_tr)
        print(f"  Train windows: {n_tr:,}  Val: {len(Xw_va):,}  Test: {len(Xw_te):,}")

        pp_models[profile]  = {}
        all_history[profile] = {}
        dynamics[profile]   = {"train_samples": n_tr}

        for arch in ARCHS:
            ckpt_path = PER_PROF_DIR / f"{profile}_{arch}.pt"
            print(f"\n  Training {arch.upper()} for {profile} …")
            model, hist, best_ep, best_val = train_one(
                arch, Xw_tr, yw_tr, Xw_va, yw_va,
                n_features, device, ckpt_path, args.force, profile
            )
            pp_models[profile][arch] = model
            all_history[profile][arch] = hist

            # Own-profile RMSE
            pred = predict(model, Xw_te, sy, device)
            own_rmse = rmse(y_te_arr, pred)
            pp_rmse[arch][profile] = own_rmse
            print(f"  [{arch.upper()}] {profile} own-profile RMSE: {own_rmse:.4f} Nm")

            if arch == "gru":
                # Save dynamics (GRU only, as per table requirement)
                last_tr = hist[-1][1] if hist else float("nan")
                dynamics[profile]["best_ep"]      = best_ep
                dynamics[profile]["best_val"]      = best_val
                dynamics[profile]["last_tr"]       = last_tr

    # ── 3. Cross-profile evaluation (GRU only) ────────────────────────────────
    print("\n[3] Cross-profile evaluation (GRU) …")
    cross_rmse = {}
    for train_prof in PROFILES:
        gru_model = pp_models[train_prof]["gru"]
        _, sx_tr, sy_tr = cross_dfs[train_prof]
        cross_rmse[train_prof] = {}
        for test_prof in PROFILES:
            test_dfs, _, _ = cross_dfs[test_prof]
            # Evaluate trained model on test_prof's test split using train_prof's scaler
            y_te, Xw_te_cross, _ = build_test_raw(test_dfs, feat_cols, sx_tr, sy_tr)
            if len(y_te) == 0:
                cross_rmse[train_prof][test_prof] = float("nan")
                continue
            pred = predict(gru_model, Xw_te_cross, sy_tr, device)
            r = rmse(y_te, pred)
            cross_rmse[train_prof][test_prof] = r
        row_str = "  ".join(f"{p}={cross_rmse[train_prof][p]:.2f}" for p in PROFILES)
        print(f"  trained {train_prof:6s}: {row_str}")

    # ── 4. PLC before/after plot ──────────────────────────────────────────────
    print("\n[4] PLC before/after comparison plot …")
    plc_dfs = load_profile_dfs("PLC")
    sx_v1   = joblib.load(config.PROJECT_ROOT / "checkpoints" / "scaler_X.pkl") \
               if (config.PROJECT_ROOT / "checkpoints" / "scaler_X.pkl").exists() else None
    sy_v1   = joblib.load(config.PROJECT_ROOT / "checkpoints" / "scaler_y.pkl") \
               if (config.PROJECT_ROOT / "checkpoints" / "scaler_y.pkl").exists() else None

    if sx_v1 is not None and sx_v1.n_features_in_ == len(feat_cols):
        try:
            gru_v1, _, _ = load_ckpt(config.PROJECT_ROOT / "checkpoints" / "best_model_gru.pt",
                                      "gru", device)
            y_plc_v1, Xw_plc_v1, _ = build_test_raw(plc_dfs, feat_cols, sx_v1, sy_v1)
            pred_v1_plc = predict(gru_v1, Xw_plc_v1, sy_v1, device)

            # Per-profile PLC predictions on same test set (use pp scaler + smoothed data)
            plc_dfs_sm = load_profile_dfs("PLC")
            sx_pp, sy_pp = pp_scalers["PLC"]
            y_plc_pp, Xw_plc_pp, _ = build_test_raw(plc_dfs_sm, feat_cols, sx_pp, sy_pp)
            pred_pp_plc = predict(pp_models["PLC"]["gru"], Xw_plc_pp, sy_pp, device)

            # Use shorter of the two for the plot (in case they differ by 1 due to seq alignment)
            n_common = min(len(y_plc_v1), len(y_plc_pp))
            plot_plc_comparison(
                y_plc_v1[:n_common], pred_v1_plc[:n_common], pred_pp_plc[:n_common],
                OUTPUTS_DIR / "per_profile_plc_comparison.png",
            )
        except Exception as e:
            print(f"  PLC comparison plot skipped: {e}")
    else:
        print("  v1 scalers unavailable — PLC comparison plot skipped.")

    # ── 5. Training curves plot ───────────────────────────────────────────────
    print("\n[5] Plotting training curves …")
    plot_training_curves(all_history, OUTPUTS_DIR / "per_profile_training_curves.png")

    # ── 6. Cross-profile heatmap ──────────────────────────────────────────────
    print("[6] Plotting cross-profile heatmap …")
    plot_cross_heatmap(cross_rmse, OUTPUTS_DIR / "per_profile_cross_heatmap.png")

    # ── 7. Verdict ────────────────────────────────────────────────────────────
    plc_pp_gru = pp_rmse["gru"].get("PLC", float("nan"))
    if np.isnan(plc_pp_gru):
        verdict = "Per-profile PLC GRU RMSE unavailable"
    elif plc_pp_gru < 5.0:
        verdict = ("Per-profile training CLOSES the gap (PLC GRU RMSE = "
                   f"{plc_pp_gru:.2f} Nm < 5 Nm): cause (1) mixed-profile "
                   "training is dominant.")
    elif plc_pp_gru < 30.0:
        verdict = ("Per-profile training PARTIALLY closes the gap (PLC GRU RMSE = "
                   f"{plc_pp_gru:.2f} Nm, 5–30 Nm): dilution is real but not the "
                   "full story — causes (2)–(5) contribute.")
    else:
        verdict = ("Per-profile training does NOT close the gap (PLC GRU RMSE = "
                   f"{plc_pp_gru:.2f} Nm > 30 Nm): PLC data is intrinsically "
                   "harder. Pivot to causes (2)–(5).")

    print(f"\n[Verdict] {verdict}")

    # Print primary table to terminal
    print("\nPrimary table:")
    print(f"{'Profile':<10} {'v1 MLP':>8} {'pp MLP':>8} {'v1 GRU':>8} {'pp GRU':>8}")
    print("-" * 42)
    for prof in PROFILES:
        def f(d, arch): return f"{d[arch].get(prof, float('nan')):>8.2f}"
        print(f"{prof:<10} {f(v1_rmse,'mlp')} {f(pp_rmse,'mlp')} "
              f"{f(v1_rmse,'gru')} {f(pp_rmse,'gru')}")

    # ── 8. Write report ───────────────────────────────────────────────────────
    write_report(
        v1_rmse, pp_rmse, cross_rmse, dynamics, verdict,
        OUTPUTS_DIR / "per_profile_training.md",
    )

    # ── 9. Append to SUMMARY+CONCLUSION.txt ──────────────────────────────────
    append_summary(pp_rmse, v1_rmse, verdict)

    print("\nDone.")


if __name__ == "__main__":
    main()

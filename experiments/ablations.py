"""Prompt 4 — Feature ablation study.

Trains 3 ablated configs (B, C, D) × 2 models = 6 new runs.
Config A (v1 baseline) is loaded from existing checkpoints — no retraining.

Configs
-------
  A — all 10 features (v1 baseline — loaded, not retrained)
  B — drop torDes   (9 features)
  C — drop i        (9 features)
  D — drop torDes + i (8 features)

Resume support: if a checkpoint already exists for a config×model pair,
training is skipped so the script can be restarted cleanly.

Usage
-----
    python experiments/ablations.py
    python experiments/ablations.py --force   # retrain even if checkpoints exist
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
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
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
from preprocessing import (
    ActuatorDataset,
    _get_feature_cols,
    _load_dataframes,
    _make_windows,
    _split_df,
)
from models import ActuatorGRU, WindowedMLP
from test2_regime_residuals import REGIMES, classify_regimes

# ── Paths ──────────────────────────────────────────────────────────────────────

CKPT_DIR     = _PROJECT_ROOT / "checkpoints" / "ablations"
OUT_DIR      = _HERE / "outputs"
SUMMARY_PATH = _PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature configs ───────────────────────────────────────────────────────────

V1_FEAT_COLS = list(config.FEATURE_COLS)   # all 10 features

ABLATION_CONFIGS: dict[str, list[str]] = {
    "A": V1_FEAT_COLS,
    "B": [f for f in V1_FEAT_COLS if f != "torDes"],
    "C": [f for f in V1_FEAT_COLS if f != "i"],
    "D": [f for f in V1_FEAT_COLS if f not in ("torDes", "i")],
}

CONFIG_LABELS = {
    "A": "none (v1 baseline)",
    "B": "torDes",
    "C": "i",
    "D": "torDes, i",
}

PALETTE = {"A": "#999999", "B": "#1f77b4", "C": "#ff7f0e", "D": "#2ca02c"}


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_datasets_for(feat_cols: list[str], seq_len: int = config.SEQ_LEN):
    """Return (train_ds, val_ds, test_ds, X_test_windows, y_true_test,
               file_name_test, scaler_X, scaler_y)."""
    dfs = _load_dataframes()

    tr_X, tr_y = [], []
    va_X, va_y = [], []
    te_X, te_y = [], []
    fn_parts: list[str] = []

    for df in dfs:
        train_df, val_df, test_df = _split_df(
            df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        fname = str(df["file_name"].iloc[0]) if "file_name" in df.columns else "unknown"

        for df_part, Xl, yl in [(train_df, tr_X, tr_y), (val_df, va_X, va_y)]:
            if len(df_part) < seq_len:
                continue
            Xl.append(df_part[feat_cols].values.astype(np.float32))
            yl.append(df_part[config.TARGET_COL].values.astype(np.float32))

        if len(test_df) < seq_len:
            continue
        te_X.append(test_df[feat_cols].values.astype(np.float32))
        te_y.append(test_df[config.TARGET_COL].values.astype(np.float32))
        n_windows = len(test_df) - seq_len + 1
        fn_parts.extend([fname] * n_windows)

    # Fit scalers on training data
    scaler_X = StandardScaler().fit(np.concatenate(tr_X, axis=0))
    scaler_y = StandardScaler().fit(np.concatenate(tr_y).reshape(-1, 1))

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

    # Test windows
    te_Xw, te_yw = [], []
    for X, y in zip(te_X, te_y):
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

    y_true_test = np.concatenate(
        [y[seq_len - 1:] for y in te_y if len(y) >= seq_len]
    ).astype(np.float32)
    file_name_test = np.array(fn_parts, dtype=object)

    return (train_ds, val_ds, test_ds,
            X_test_windows, y_true_test, file_name_test,
            scaler_X, scaler_y)


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(model_type: str, n_features: int,
              train_ds, val_ds, ckpt_path: Path, device):
    """Train one model to completion. Returns (model, history, best_epoch)."""
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    if model_type == "mlp":
        model = WindowedMLP(
            seq_len=config.SEQ_LEN, n_features=n_features,
            hidden_size=config.MLP_HIDDEN_SIZE, n_layers=config.MLP_N_LAYERS,
        )
    else:
        model = ActuatorGRU(
            n_features=n_features, hidden_size=config.GRU_HIDDEN_SIZE,
            n_layers=config.GRU_N_LAYERS, dropout=config.GRU_DROPOUT,
        )
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.ONE_CYCLE_MAX_LR,
        total_steps=len(train_loader) * config.MAX_EPOCHS)

    best_val, patience_ctr, best_epoch = float("inf"), 0, 0
    history, epoch_times = [], []

    print(f"\n{'═'*60}")
    print(f"  {model_type.upper()}  n_features={n_features}  "
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
                "model_type": model_type,
                "config": {
                    "seq_len":     config.SEQ_LEN,
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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, history, best_epoch, float(np.mean(epoch_times))


def load_ckpt(model_type: str, n_features: int, ckpt_path: Path, device):
    """Load a saved checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    if model_type == "mlp":
        model = WindowedMLP(
            seq_len=cfg["seq_len"], n_features=cfg["n_features"],
            hidden_size=cfg.get("hidden_size", config.MLP_HIDDEN_SIZE),
            n_layers=cfg.get("n_layers", config.MLP_N_LAYERS),
        )
    else:
        model = ActuatorGRU(
            n_features=cfg["n_features"],
            hidden_size=cfg.get("hidden_size", config.GRU_HIDDEN_SIZE),
            n_layers=cfg.get("n_layers", config.GRU_N_LAYERS),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["epoch"]


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

def per_regime_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                    file_name: np.ndarray) -> dict[str, float]:
    regimes = classify_regimes(y_true, file_name)
    out = {}
    for r in REGIMES:
        mask = regimes == r
        if mask.sum() == 0:
            out[r] = float("nan")
        else:
            out[r] = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    file_name: np.ndarray,
                    ref_pred: np.ndarray | None = None) -> dict:
    err = np.abs(y_pred - y_true)
    rm  = per_regime_rmse(y_true, y_pred, file_name)
    spearman_vs_A = (float(spearmanr(y_pred, ref_pred).statistic)
                     if ref_pred is not None else None)
    return {
        "overall_rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae":          float(np.mean(err)),
        "max_abs_err":  float(err.max()),
        "hold_rmse":    rm.get("hold", float("nan")),
        "osc_rmse":     rm.get("oscillation", float("nan")),
        "trans_rmse":   rm.get("transition", float("nan")),
        "rest_rmse":    rm.get("rest", float("nan")),
        "spearman_vs_A": spearman_vs_A,
    }


# ── v1 Config A baseline ──────────────────────────────────────────────────────

def get_config_a(device, y_true_test: np.ndarray, file_name_test: np.ndarray) -> dict:
    """Load v1 MLP and GRU checkpoints and compute Config A metrics."""
    v1_sx = joblib.load(config.CHECKPOINT_DIR / "scaler_X.pkl")
    v1_sy = joblib.load(config.CHECKPOINT_DIR / "scaler_y.pkl")

    # Rebuild test windows for v1 (10 features, SEQ_LEN=30)
    dfs = _load_dataframes()
    base_cols = _get_feature_cols()
    te_Xw_parts = []
    for df in dfs:
        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, config.SEQ_LEN)
        if len(test_df) < config.SEQ_LEN:
            continue
        Xs = v1_sx.transform(test_df[base_cols].values.astype(np.float32)).astype(np.float32)
        Xw, _ = _make_windows(Xs, np.zeros(len(Xs), dtype=np.float32), config.SEQ_LEN)
        if len(Xw):
            te_Xw_parts.append(Xw)
    X_v1 = np.concatenate(te_Xw_parts, axis=0)

    result = {}
    for model_type in ("mlp", "gru"):
        ckpt_path = config.CHECKPOINT_DIR / f"best_model_{model_type}.pt"
        if not ckpt_path.exists():
            print(f"WARNING: v1 {model_type.upper()} checkpoint not found.")
            continue
        model, _ = load_ckpt(model_type, len(base_cols), ckpt_path, device)
        y_pred = run_inference(model, X_v1, v1_sy, device)
        metrics = compute_metrics(y_true_test, y_pred, file_name_test, ref_pred=None)
        metrics["spearman_vs_A"] = 1.0   # comparing A to itself
        result[model_type] = {"y_pred": y_pred, "metrics": metrics, "history": []}
        print(f"  Config A {model_type.upper()}: "
              f"RMSE={metrics['overall_rmse']:.2f} Nm  "
              f"Hold={metrics['hold_rmse']:.2f}  Osc={metrics['osc_rmse']:.2f}")
    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: dict[tuple[str, str], list], out_path: Path):
    """Faceted by model (2 panels), coloured by config."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, model_type in zip(axes, ("mlp", "gru")):
        for cfg in ("B", "C", "D"):
            key = (cfg, model_type)
            hist = histories.get(key)
            if not hist:
                continue
            arr = np.array(hist)
            ax.plot(arr[:, 0], arr[:, 2], color=PALETTE[cfg], linewidth=1.4,
                    label=f"Config {cfg} (drop {CONFIG_LABELS[cfg]})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE (scaled)")
        ax.set_title(f"{model_type.upper()} — ablation training curves")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Feature ablation — validation loss curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── Interpretation ────────────────────────────────────────────────────────────

def interpret(rows: dict[tuple[str, str], dict]) -> str:
    msgs = []

    def _pct_change(cfg, model, metric, ref_val):
        r = rows.get((cfg, model), {}).get(metric)
        if r is None or (isinstance(r, float) and np.isnan(r)):
            return None
        return (r - ref_val) / max(abs(ref_val), 1e-6)

    for model_type in ("mlp", "gru"):
        a = rows.get(("A", model_type), {})
        ref_hold  = a.get("hold_rmse",    float("nan"))
        ref_osc   = a.get("osc_rmse",     float("nan"))
        ref_rmse  = a.get("overall_rmse", float("nan"))
        if any(np.isnan(v) for v in (ref_hold, ref_osc, ref_rmse)):
            continue

        b_hold = _pct_change("B", model_type, "hold_rmse", ref_hold)
        c_osc  = _pct_change("C", model_type, "osc_rmse",  ref_osc)
        d_rmse = _pct_change("D", model_type, "overall_rmse", ref_rmse)

        model_msgs = []

        if b_hold is not None and b_hold > 0.30:
            model_msgs.append(
                f"**torDes is load-bearing** ({model_type.upper()}: "
                f"dropping torDes increases hold RMSE by {100*b_hold:.0f}%). "
                "The NN is learning the feed-forward error model — torDes is the primary signal."
            )
        elif b_hold is not None:
            model_msgs.append(
                f"torDes drop has minor hold impact ({model_type.upper()}: "
                f"{100*b_hold:+.0f}% on hold RMSE)."
            )

        if c_osc is not None and c_osc > 0.30:
            model_msgs.append(
                f"**i is load-bearing** ({model_type.upper()}: "
                f"dropping i increases osc RMSE by {100*c_osc:.0f}%). "
                "The NN is performing sensor fusion — using current to estimate torque."
            )
        elif c_osc is not None:
            model_msgs.append(
                f"i drop has minor osc impact ({model_type.upper()}: "
                f"{100*c_osc:+.0f}% on osc RMSE)."
            )

        if d_rmse is not None and abs(d_rmse) < 0.10:
            model_msgs.append(
                f"**Proprioception alone sufficient** ({model_type.upper()}: "
                f"Config D within {100*abs(d_rmse):.1f}% of Config A overall). "
                "This is a surprising result: the 8-feature purely proprioceptive "
                "config matches the full model — the NN may have learned a pose-based "
                "torque estimator that does not rely on the command or current signals."
            )

        msgs.extend(model_msgs)

    if not msgs:
        return "Results pending — run the full ablation."
    return "\n\n".join(msgs)


def thesis_recommendation(rows: dict[tuple[str, str], dict]) -> str:
    """Two-sentence thesis narrative recommendation."""
    a_gru        = rows.get(("A", "gru"), {})
    ref_hold_gru = a_gru.get("hold_rmse",    float("nan"))
    ref_osc_gru  = a_gru.get("osc_rmse",     float("nan"))

    b_hold = rows.get(("B", "gru"), {}).get("hold_rmse")
    c_osc  = rows.get(("C", "gru"), {}).get("osc_rmse")

    tordes_load = (b_hold is not None and not np.isnan(b_hold)
                   and not np.isnan(ref_hold_gru)
                   and (b_hold - ref_hold_gru) / max(ref_hold_gru, 1e-6) > 0.30)
    i_load      = (c_osc is not None  and not np.isnan(c_osc)
                   and not np.isnan(ref_osc_gru)
                   and (c_osc  - ref_osc_gru)  / max(ref_osc_gru, 1e-6)  > 0.30)

    if tordes_load and i_load:
        return (
            "Both torDes and i are load-bearing: torDes drives hold-regime accuracy while i "
            "drives oscillation-regime accuracy, indicating the model jointly acts as a "
            "feed-forward error compensator and a torque-from-current estimator. "
            "The thesis should present the NN as a multi-signal fusion model, not a "
            "single-mechanism predictor."
        )
    elif tordes_load:
        return (
            "torDes is the dominant signal: the feed-forward command carries the bulk of "
            "hold-regime information, so the NN is primarily learning the error on top of "
            "the open-loop command. "
            "The thesis should emphasize that the NN refines the feed-forward, with i "
            "as a secondary contributor that does not independently drive accuracy."
        )
    elif i_load:
        return (
            "i is the dominant signal: motor current provides the primary cue for "
            "oscillation-regime torque, meaning the NN is essentially performing "
            "sensor fusion of a current-based torque estimate. "
            "The thesis should frame this as current-based estimation augmented by "
            "position/velocity context from the remaining features."
        )
    else:
        return (
            "Neither torDes nor i alone is strongly load-bearing: the model appears to "
            "distribute its reliance across many features, suggesting the 8-feature "
            "proprioceptive subset is nearly equivalent. "
            "The thesis should note the robustness of the model to individual feature "
            "removal and investigate whether further ablation (e.g., torEst) reveals "
            "a single dominant signal."
        )


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown(rows: dict[tuple[str, str], dict], interp: str,
                   rec: str, out_path: Path):
    def _f(v, fmt=".2f"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:{fmt}}"

    lines = [
        "# Feature Ablation Study — Prompt 4",
        "",
        "## Results table",
        "",
        "| Config | Features dropped | MLP overall RMSE | MLP hold RMSE | MLP osc RMSE "
        "| MLP ρ_A | GRU overall RMSE | GRU hold RMSE | GRU osc RMSE | GRU ρ_A |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg in ("A", "B", "C", "D"):
        mlp = rows.get((cfg, "mlp"), {})
        gru = rows.get((cfg, "gru"), {})
        lines.append(
            f"| {cfg} | {CONFIG_LABELS[cfg]} "
            f"| {_f(mlp.get('overall_rmse'))} "
            f"| {_f(mlp.get('hold_rmse'))} "
            f"| {_f(mlp.get('osc_rmse'))} "
            f"| {_f(mlp.get('spearman_vs_A'), '.4f')} "
            f"| {_f(gru.get('overall_rmse'))} "
            f"| {_f(gru.get('hold_rmse'))} "
            f"| {_f(gru.get('osc_rmse'))} "
            f"| {_f(gru.get('spearman_vs_A'), '.4f')} |"
        )
    lines += [
        "",
        "ρ_A = Spearman rank correlation between this config's predictions and Config A's predictions.",
        "",
        "## Interpretation",
        "",
        interp,
        "",
        "## Thesis narrative recommendation",
        "",
        rec,
        "",
        "## Outputs",
        "- `ablations_training_curves.png` — val loss curves (faceted by model)",
        "- Checkpoints: `checkpoints/ablations/{config}_{model}.pt`",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if checkpoint already exists.")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Shared test-set ground truth (built once from v1 feature set)
    print("\nBuilding shared test metadata …")
    (_, _, _, _, y_true_test, file_name_test,
     _, _) = build_datasets_for(V1_FEAT_COLS)

    print("\nLoading Config A (v1 baseline) …")
    config_a = get_config_a(device, y_true_test, file_name_test)
    ref_preds = {mt: config_a[mt]["y_pred"] for mt in config_a}

    rows: dict[tuple[str, str], dict] = {}
    for mt, m in config_a.items():
        rows[("A", mt)] = m["metrics"]

    histories: dict[tuple[str, str], list] = {}

    for cfg in ("B", "C", "D"):
        feat_cols = ABLATION_CONFIGS[cfg]
        print(f"\n{'─'*60}")
        print(f"  Config {cfg}  —  drop: {CONFIG_LABELS[cfg]}")
        print(f"  Features ({len(feat_cols)}): {feat_cols}")

        print("  Building datasets …")
        (train_ds, val_ds, _, X_test_windows, _, _,
         _, scaler_y) = build_datasets_for(feat_cols)

        for model_type in ("mlp", "gru"):
            ckpt_path = CKPT_DIR / f"{cfg}_{model_type}.pt"
            hist_path = OUT_DIR / f"ablation_{cfg}_{model_type}_history.csv"

            if ckpt_path.exists() and not args.force:
                print(f"  [resume] {ckpt_path.name} already exists — skipping training.")
                model, best_epoch = load_ckpt(model_type, len(feat_cols), ckpt_path, device)
                # Load saved history if available
                if hist_path.exists():
                    hist = list(csv.reader(open(hist_path)))
                    histories[(cfg, model_type)] = [
                        (int(r[0]), float(r[1]), float(r[2])) for r in hist[1:]
                    ]
            else:
                model, hist, best_epoch, mean_t = train_one(
                    model_type, len(feat_cols), train_ds, val_ds, ckpt_path, device)
                histories[(cfg, model_type)] = hist
                with open(hist_path, "w", newline="") as f:
                    csv.writer(f).writerows(
                        [["epoch", "train_mse", "val_mse"]] + list(hist))
                print(f"  Best epoch={best_epoch}  mean_epoch={mean_t:.1f}s")

            y_pred = run_inference(model, X_test_windows, scaler_y, device)
            metrics = compute_metrics(
                y_true_test, y_pred, file_name_test,
                ref_pred=ref_preds.get(model_type))
            rows[(cfg, model_type)] = metrics

            print(f"  {model_type.upper()} Config {cfg}: "
                  f"RMSE={metrics['overall_rmse']:.2f}  "
                  f"Hold={metrics['hold_rmse']:.2f}  "
                  f"Osc={metrics['osc_rmse']:.2f}  "
                  f"ρ_A={metrics['spearman_vs_A']:.4f}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  {'':3s}  {'Model':5s}  {'RMSE':8s}  {'Hold':8s}  "
          f"{'Osc':8s}  {'Trans':8s}  {'Rest':8s}  {'ρ_A':8s}")
    print(f"{'─'*90}")
    for cfg in ("A", "B", "C", "D"):
        for mt in ("mlp", "gru"):
            m = rows.get((cfg, mt))
            if m is None:
                continue
            def _f(v):
                return f"{v:8.2f}" if v is not None and not np.isnan(v) else "      —  "
            rho = m.get("spearman_vs_A")
            rho_s = f"{rho:8.4f}" if rho is not None else "      —  "
            print(f"  {cfg}    {mt:5s}  "
                  f"{_f(m['overall_rmse'])}  {_f(m['hold_rmse'])}  "
                  f"{_f(m['osc_rmse'])}  {_f(m['trans_rmse'])}  "
                  f"{_f(m['rest_rmse'])}  {rho_s}")
    print(f"{'═'*90}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(histories, OUT_DIR / "ablations_training_curves.png")

    # ── Markdown ──────────────────────────────────────────────────────────────
    interp = interpret(rows)
    rec    = thesis_recommendation(rows)
    write_markdown(rows, interp, rec,
                   OUT_DIR / "ablations_summary.md")

    # ── Append to SUMMARY+CONCLUSION.txt ─────────────────────────────────────
    gru_a = rows.get(("A", "gru"), {})
    gru_b = rows.get(("B", "gru"), {})
    gru_c = rows.get(("C", "gru"), {})
    gru_d = rows.get(("D", "gru"), {})

    def _r(d, k):
        v = d.get(k, float("nan"))
        return f"{v:.2f}" if not np.isnan(v) else "—"

    block = (
        f"\n{'='*72}\n"
        f"[Prompt 4 — Feature ablation (torDes, i)]\n"
        f"{'='*72}\n"
        f"Purpose: Identify which features (torDes, i) the MLP/GRU actually rely on "
        f"by dropping them one at a time and measuring regime-specific RMSE degradation.\n\n"
        f"GRU results (RMSE / hold / osc):\n"
        f"  Config A (baseline):        {_r(gru_a,'overall_rmse')} / "
        f"{_r(gru_a,'hold_rmse')} / {_r(gru_a,'osc_rmse')}\n"
        f"  Config B (no torDes):       {_r(gru_b,'overall_rmse')} / "
        f"{_r(gru_b,'hold_rmse')} / {_r(gru_b,'osc_rmse')}\n"
        f"  Config C (no i):            {_r(gru_c,'overall_rmse')} / "
        f"{_r(gru_c,'hold_rmse')} / {_r(gru_c,'osc_rmse')}\n"
        f"  Config D (no torDes, no i): {_r(gru_d,'overall_rmse')} / "
        f"{_r(gru_d,'hold_rmse')} / {_r(gru_d,'osc_rmse')}\n\n"
        f"{interp}\n\n"
        f"Thesis recommendation: {rec}\n"
    )
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"\nAppended to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

"""Prompt 16 — 8-feature evaluation: multi-seed robustness, per-profile/regime.

8-feature set: {torDes, posDes, velDes, posAct, posErr, velErr, velAct, accelAct}
(Hwangbo-style kinematic features — no torque telemetry, no current measurement).

Usage
-----
    python experiments/prompt16_8feature_eval.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader as TorchDataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Patch config to 8-feature set BEFORE importing data modules ───────────────
import config

config.INCLUDE_torEst   = False
config.INCLUDE_curr     = False
config.INCLUDE_I2T      = False
config.INCLUDE_kd       = False
config.INCLUDE_torKdEst = False
config.INCLUDE_t        = False
config.INCLUDE_posDes   = True
config.INCLUDE_accelAct = True

from preprocessing import (   # noqa: E402
    _get_feature_cols, _load_dataframes, _split_df, _make_windows,
    build_datasets,
)
from models import ActuatorGRU, WindowedMLP  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

SEEDS      = [42, 123, 7]
MAX_EPOCHS = config.MAX_EPOCHS
PATIENCE   = config.PATIENCE
BATCH_SIZE = config.BATCH_SIZE
SEQ_LEN    = config.SEQ_LEN

OUTPUT_DIR       = PROJECT_ROOT / "baselines" / "outputs"
SUMMARY_TXT      = PROJECT_ROOT / "diagnostics" / "outputs" / "SUMMARY.txt"
SUMMARY_CONC_TXT = PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# v1 reference numbers (10-feature: includes torEst + i)
V1_PROFILE = {
    "tStep": {"MLP": 1.73,  "GRU": 0.37},
    "TMS":   {"MLP": 4.73,  "GRU": 4.56},
    "PMS":   {"MLP": 8.18,  "GRU": 8.08},
    "PLC":   {"MLP": 51.31, "GRU": 67.25},
}
V1_REGIME = {
    "rest":        {"GRU": 3.80},
    "hold":        {"GRU": 65.97},
    "transition":  {"GRU": 11.66},
    "oscillation": {"GRU": 12.99},
}
V1_OVERALL = {"MLP": 13.53, "GRU": 13.66}

PROFILES = ["tStep", "TMS", "PMS", "PLC"]
REGIMES  = ["rest", "hold", "transition", "oscillation"]

ROLLING_WINDOW = 200
OSC_STD  = 20.0
REST_STD = 5.0
HOLD_ABS = 10.0


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))


def detect_profile(fname: str) -> str:
    u = str(fname).upper()
    if "PLC"   in u: return "PLC"
    if "PMS"   in u: return "PMS"
    if "TMS"   in u: return "TMS"
    if "TSTEP" in u or "T_STEP" in u or "STEP" in u: return "tStep"
    return "unknown"


def classify_regimes(y_true: np.ndarray, file_labels: np.ndarray) -> np.ndarray:
    df = pd.DataFrame({"y": y_true, "file": file_labels})
    rstd = (df.groupby("file", sort=False)["y"]
              .transform(lambda s: s.rolling(ROLLING_WINDOW, center=True, min_periods=1).std())
              .values)
    regimes = np.full(len(y_true), "transition", dtype=object)
    osc   = rstd > OSC_STD
    quiet = rstd < REST_STD
    nz    = np.abs(y_true) > HOLD_ABS
    regimes[quiet & nz]  = "hold"
    regimes[quiet & ~nz] = "rest"
    regimes[osc]         = "oscillation"
    return regimes


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_set(scaler_X, scaler_y):
    """Build windowed test set with per-window profile and file labels.

    Returns
    -------
    X_test  : (N, seq_len, n_features)  scaled, float32
    y_nm    : (N,)  torAct in Nm (unscaled)
    tordes  : (N,)  torDes in Nm (unscaled, aligned with labels)
    profiles: (N,)  string array  'PLC' | 'PMS' | 'TMS' | 'tStep'
    files   : (N,)  string array  file_name per window
    """
    feature_cols = _get_feature_cols()
    tordes_idx   = feature_cols.index("torDes")
    dfs          = _load_dataframes()

    X_wins, y_wins, tordes_wins, prof_wins, file_wins = [], [], [], [], []

    for df in dfs:
        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, SEQ_LEN)
        if len(test_df) < SEQ_LEN:
            continue

        fname = (test_df["file_name"].values.astype(str)
                 if "file_name" in test_df.columns
                 else np.full(len(test_df), "unknown"))

        X_raw = test_df[feature_cols].values.astype(np.float64)
        y_raw = test_df[config.TARGET_COL].values.astype(np.float64)
        prof  = np.array([detect_profile(f) for f in fname])

        Xs = scaler_X.transform(X_raw).astype(np.float32)
        Xw, _ = _make_windows(Xs, y_raw.astype(np.float32), SEQ_LEN)

        # Labels aligned to last timestep of each window
        y_win      = y_raw[SEQ_LEN - 1:]
        tordes_win = X_raw[SEQ_LEN - 1:, tordes_idx]
        prof_win   = prof[SEQ_LEN - 1:]
        file_win   = fname[SEQ_LEN - 1:]

        if len(Xw):
            X_wins.append(Xw)
            y_wins.append(y_win)
            tordes_wins.append(tordes_win)
            prof_wins.append(prof_win)
            file_wins.append(file_win)

    return (
        np.concatenate(X_wins),
        np.concatenate(y_wins),
        np.concatenate(tordes_wins),
        np.concatenate(prof_wins),
        np.concatenate(file_wins),
    )


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_type: str, n_features: int) -> nn.Module:
    if model_type == "mlp":
        return WindowedMLP(
            seq_len=SEQ_LEN, n_features=n_features,
            hidden_size=config.MLP_HIDDEN_SIZE, n_layers=config.MLP_N_LAYERS,
        )
    return ActuatorGRU(
        n_features=n_features,
        hidden_size=config.GRU_HIDDEN_SIZE,
        n_layers=config.GRU_N_LAYERS,
        dropout=config.GRU_DROPOUT,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one(model_type: str, seed: int, train_ds, val_ds, n_features: int, device):
    """Train one model with a given seed. Returns (model, best_val_mse, best_epoch, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen          = torch.Generator().manual_seed(seed)
    train_loader = TorchDataLoader(train_ds, batch_size=BATCH_SIZE,
                                   shuffle=True, drop_last=True, generator=gen)
    val_loader   = TorchDataLoader(val_ds,   batch_size=512, shuffle=False)

    model     = build_model(model_type, n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.ONE_CYCLE_MAX_LR,
        total_steps=len(train_loader) * MAX_EPOCHS,
    )

    best_val, best_epoch, best_state = float("inf"), 1, None
    patience_ctr = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        tr_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * len(X)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                vl_loss += criterion(model(X), y).item() * len(X)
        vl_loss /= len(val_loader.dataset)

        history.append((epoch, tr_loss, vl_loss))
        improved = vl_loss < best_val
        marker   = " [saved]" if improved else f" (patience {patience_ctr + 1}/{config.PATIENCE})"
        print(f"Epoch {epoch:4d} | train MSE {tr_loss:.6f} | val MSE {vl_loss:.6f}{marker} | duration {(time.perf_counter()-t_epoch_start):.3f} s")
        if vl_loss < best_val:
            best_val   = vl_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_val, best_epoch, history


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_nm(model, X_np: np.ndarray, scaler_y, device, batch_size: int = 512):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[s:s + batch_size]).float().to(device)
            preds.append(model(xb).cpu().numpy())
    p_scaled = np.concatenate(preds).ravel()
    return scaler_y.inverse_transform(p_scaled.reshape(-1, 1)).ravel()


# ── Multi-seed runner ─────────────────────────────────────────────────────────

class RunResult:
    __slots__ = ("seed", "model", "rmse", "mae", "best_val", "best_epoch", "history")

    def __init__(self, seed, model, rmse_, mae_, best_val, best_epoch, history):
        self.seed       = seed
        self.model      = model
        self.rmse       = rmse_
        self.mae        = mae_
        self.best_val   = best_val
        self.best_epoch = best_epoch
        self.history    = history


def run_seeds(model_type: str, train_ds, val_ds,
              X_test: np.ndarray, y_nm: np.ndarray,
              scaler_y, n_features: int, device) -> list[RunResult]:
    results = []
    for seed in SEEDS:
        print(f"    seed {seed:4d} … ", end="", flush=True)
        model, best_val, best_epoch, history = train_one(
            model_type, seed, train_ds, val_ds, n_features, device
        )
        y_pred = predict_nm(model, X_test, scaler_y, device)
        r, m   = rmse(y_nm, y_pred), mae(y_nm, y_pred)
        _, tr, vl = history[best_epoch - 1]
        print(f"epoch={best_epoch:3d}  val_MSE={best_val:.6f}  "
              f"RMSE={r:.3f} Nm  MAE={m:.3f} Nm  v/t={vl/tr:.3f}")
        results.append(RunResult(seed, model, r, m, best_val, best_epoch, history))
    return results


def median_run(results: list[RunResult]) -> RunResult:
    return sorted(results, key=lambda r: r.rmse)[len(results) // 2]


# ── Per-profile / per-regime breakdown ────────────────────────────────────────

def per_profile(model, X_test, y_nm, profiles, scaler_y, device) -> dict:
    y_pred = predict_nm(model, X_test, scaler_y, device)
    return {p: (rmse(y_nm[profiles == p], y_pred[profiles == p])
                if (profiles == p).sum() > 0 else float("nan"))
            for p in PROFILES}


def per_regime(model, X_test, y_nm, file_labels, scaler_y, device) -> dict:
    y_pred  = predict_nm(model, X_test, scaler_y, device)
    regimes = classify_regimes(y_nm, file_labels)
    return {r: (rmse(y_nm[regimes == r], y_pred[regimes == r])
                if (regimes == r).sum() > 0 else float("nan"))
            for r in REGIMES}


# ── Baselines ─────────────────────────────────────────────────────────────────

def compute_baselines(y_nm: np.ndarray, tordes_nm: np.ndarray):
    return rmse(y_nm, np.zeros_like(y_nm)), rmse(y_nm, tordes_nm)


# ── Verdict ───────────────────────────────────────────────────────────────────

def compute_verdict(mlp_results, gru_results, tordes_rmse,
                    pp_mlp, pp_gru) -> str:
    mlp_rmses = [r.rmse for r in mlp_results]
    gru_rmses = [r.rmse for r in gru_results]
    mlp_std   = float(np.std(mlp_rmses))
    gru_std   = float(np.std(gru_rmses))

    parts = []

    # Stability
    if max(mlp_std, gru_std) < 2.0:
        parts.append(f"Result is ROBUST (MLP std={mlp_std:.2f}, GRU std={gru_std:.2f} Nm < 2 Nm).")
    elif max(mlp_std, gru_std) > 5.0:
        parts.append(f"Result is UNSTABLE (std > 5 Nm — more seeds needed).")
    else:
        parts.append(f"Result is MODERATELY STABLE (MLP std={mlp_std:.2f}, GRU std={gru_std:.2f} Nm).")

    # Kinematic lift vs torDes
    med_mlp = median_run(mlp_results).rmse
    ratio   = med_mlp / tordes_rmse
    if ratio > 0.33:
        parts.append(
            f"Kinematic features provide WEAK lift over predict-torDes "
            f"(8-ft MLP={med_mlp:.2f} Nm, torDes baseline={tordes_rmse:.2f} Nm, ratio={ratio:.2f} — threshold 3×)."
        )
    else:
        parts.append(
            f"Kinematic features provide MEANINGFUL lift over predict-torDes "
            f"(8-ft MLP={med_mlp:.2f} Nm vs torDes={tordes_rmse:.2f} Nm, ratio={ratio:.2f})."
        )

    # Profile pattern
    plc_delta_mlp = pp_mlp.get("PLC", float("nan")) - V1_PROFILE["PLC"]["MLP"]
    tst_delta_mlp = pp_mlp.get("tStep", float("nan")) - V1_PROFILE["tStep"]["MLP"]
    if not np.isnan(tst_delta_mlp) and not np.isnan(plc_delta_mlp):
        if tst_delta_mlp > 5 * plc_delta_mlp and tst_delta_mlp > 2.0:
            parts.append("Per-profile: torEst/i were SPECIFICALLY responsible for easy-profile performance.")
        else:
            parts.append("Per-profile: RMSE increase is roughly uniform — torEst/i helped all profiles.")

    return " ".join(parts)


# ── CSV / text output ─────────────────────────────────────────────────────────

def save_seed_csv(mlp_results, gru_results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "seed", "overall_rmse", "overall_mae",
                    "best_val_mse", "best_epoch"])
        for arch, results in [("MLP", mlp_results), ("GRU", gru_results)]:
            for r in results:
                w.writerow([arch, r.seed, f"{r.rmse:.4f}", f"{r.mae:.4f}",
                             f"{r.best_val:.6f}", r.best_epoch])
    print(f"Wrote {path}")


def save_profile_csv(pp_mlp, pp_gru, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile",
                    "v1_mlp", "8ft_mlp", "delta_mlp",
                    "v1_gru", "8ft_gru", "delta_gru"])
        for p in PROFILES:
            v1m = V1_PROFILE[p]["MLP"]
            v1g = V1_PROFILE[p]["GRU"]
            fm  = pp_mlp.get(p, float("nan"))
            fg  = pp_gru.get(p, float("nan"))
            w.writerow([p, f"{v1m:.2f}", f"{fm:.2f}", f"{fm-v1m:+.2f}",
                        f"{v1g:.2f}", f"{fg:.2f}", f"{fg-v1g:+.2f}"])
    print(f"Wrote {path}")


# ── Console tables ────────────────────────────────────────────────────────────

def print_seed_table(mlp_results, gru_results):
    print(f"\n{'Arch':<6} {'Seed':>6} {'RMSE':>8} {'MAE':>8} "
          f"{'Best ep':>8} {'valMSE':>10} {'v/t ratio':>10}")
    print("─" * 62)
    for arch, results in [("MLP", mlp_results), ("GRU", gru_results)]:
        for r in results:
            _, tr, vl = r.history[r.best_epoch - 1]
            print(f"{arch:<6} {r.seed:>6} {r.rmse:>8.3f} {r.mae:>8.3f} "
                  f"{r.best_epoch:>8d} {r.best_val:>10.6f} {vl/tr:>10.3f}")
        rmses = [x.rmse for x in results]
        maes  = [x.mae  for x in results]
        print(f"{arch:<6} {'mean±std':>6} "
              f"{np.mean(rmses):>8.3f}±{np.std(rmses):.2f}  "
              f"{np.mean(maes):>7.3f}±{np.std(maes):.2f}")
        print()


def print_profile_table(pp_mlp, pp_gru):
    hdr = f"{'Profile':<8} {'v1 MLP':>8} {'8ft MLP':>8} {'Δ MLP':>7} | {'v1 GRU':>8} {'8ft GRU':>8} {'Δ GRU':>7}"
    print(f"\n{hdr}")
    print("─" * len(hdr))
    for p in PROFILES:
        v1m = V1_PROFILE[p]["MLP"]
        v1g = V1_PROFILE[p]["GRU"]
        fm  = pp_mlp.get(p, float("nan"))
        fg  = pp_gru.get(p, float("nan"))
        dm  = fm - v1m if not np.isnan(fm) else float("nan")
        dg  = fg - v1g if not np.isnan(fg) else float("nan")
        print(f"{p:<8} {v1m:>8.2f} {fm:>8.2f} {dm:>+7.2f} | "
              f"{v1g:>8.2f} {fg:>8.2f} {dg:>+7.2f}")


def print_regime_table(reg_gru):
    hdr = f"{'Regime':<14} {'v1 GRU':>8} {'8ft GRU':>9} {'Δ':>7}"
    print(f"\n{hdr}")
    print("─" * len(hdr))
    for r in REGIMES:
        v1 = V1_REGIME[r]["GRU"]
        ft = reg_gru.get(r, float("nan"))
        d  = ft - v1 if not np.isnan(ft) else float("nan")
        print(f"{r:<14} {v1:>8.2f} {ft:>9.2f} {d:>+7.2f}")


def print_baseline_table(zero_rmse, tordes_rmse, med_mlp, med_gru):
    print(f"\n{'Baseline':<25} {'RMSE (Nm)':>10}")
    print("─" * 37)
    print(f"{'predict-zero':<25} {zero_rmse:>10.3f}")
    print(f"{'predict-torDes':<25} {tordes_rmse:>10.3f}")
    print(f"{'8-ft MLP (median seed)':<25} {med_mlp:>10.3f}")
    print(f"{'8-ft GRU (median seed)':<25} {med_gru:>10.3f}")
    print(f"\n  MLP vs torDes ratio: {med_mlp/tordes_rmse:.2f}x  "
          f"(< 0.33x = meaningful lift)")


# ── Summary appendix ──────────────────────────────────────────────────────────

def append_summary_txt(mlp_results, gru_results, pp_mlp, pp_gru, reg_gru,
                       zero_rmse, tordes_rmse, verdict):
    mlp_rmses = [r.rmse for r in mlp_results]
    gru_rmses = [r.rmse for r in gru_results]
    lines = [
        "",
        "=" * 72,
        "[Prompt 16 — 8-Feature Evaluation (no torEst, no i)]",
        "=" * 72,
        "",
        f"MLP ({len(SEEDS)} seeds): {np.mean(mlp_rmses):.3f} ± {np.std(mlp_rmses):.3f} Nm RMSE",
        f"GRU ({len(SEEDS)} seeds): {np.mean(gru_rmses):.3f} ± {np.std(gru_rmses):.3f} Nm RMSE",
        f"(v1 MLP=13.53 Nm, v1 GRU=13.66 Nm)",
        "",
        "Per-profile (median seed):",
        f"  {'Profile':<8} {'v1MLP':>7} {'8ftMLP':>8} {'v1GRU':>7} {'8ftGRU':>8}",
    ]
    for p in PROFILES:
        lines.append(
            f"  {p:<8} {V1_PROFILE[p]['MLP']:>7.2f} {pp_mlp.get(p,float('nan')):>8.2f} "
            f"{V1_PROFILE[p]['GRU']:>7.2f} {pp_gru.get(p,float('nan')):>8.2f}"
        )
    lines += [
        "",
        f"Baselines — predict-zero: {zero_rmse:.3f} Nm  predict-torDes: {tordes_rmse:.3f} Nm",
        f"Verdict: {verdict}",
        "",
    ]
    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended to {SUMMARY_TXT}")


def append_summary_conc(mlp_results, gru_results, tordes_rmse, verdict):
    mlp_rmses = [r.rmse for r in mlp_results]
    gru_rmses = [r.rmse for r in gru_results]
    med_mlp   = median_run(mlp_results).rmse
    block = (
        "\n"
        "=" * 72 + "\n"
        "[Prompt 16 — 8-Feature Evaluation]\n"
        "=" * 72 + "\n"
        "\n"
        "Purpose: Replicate the collapse from 10-feature (13.53/13.66 Nm) to\n"
        "8-feature (no torEst, no i) across 3 seeds, and characterise whether\n"
        "kinematic-only features are sufficient to model actuator torque.\n"
        "\n"
        f"MLP: {np.mean(mlp_rmses):.2f} ± {np.std(mlp_rmses):.2f} Nm  "
        f"(v1 13.53 Nm, Δ={np.mean(mlp_rmses)-13.53:+.2f} Nm)\n"
        f"GRU: {np.mean(gru_rmses):.2f} ± {np.std(gru_rmses):.2f} Nm  "
        f"(v1 13.66 Nm, Δ={np.mean(gru_rmses)-13.66:+.2f} Nm)\n"
        f"predict-torDes baseline: {tordes_rmse:.2f} Nm\n"
        f"8-ft MLP / torDes ratio: {med_mlp/tordes_rmse:.2f}x\n"
        "\n"
        f"Verdict: {verdict}\n"
        "\n"
    )
    with open(SUMMARY_CONC_TXT, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"Appended to {SUMMARY_CONC_TXT}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.perf_counter()

    print("=" * 65)
    print("Prompt 16 — 8-Feature Evaluation")
    print("=" * 65)

    device = get_device()
    print(f"\nDevice: {device}")

    feature_cols = _get_feature_cols()
    print(f"\n[1] 8-feature set ({len(feature_cols)}): {feature_cols}")
    assert len(feature_cols) == 8, f"Expected 8 features, got {len(feature_cols)}"

    # ── Load data (scaler fitted once, shared across all seeds) ───────────────
    print("\n[2] Loading data and fitting scaler …")
    train_ds, val_ds, test_ds, scaler_X, scaler_y, feature_names = build_datasets(
        save_scalers=False   # don't overwrite main checkpoint scalers
    )
    print(f"    Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    # ── Build test set with profile / file labels ─────────────────────────────
    print("\n[3] Building test set with profile/regime labels …")
    X_test, y_nm, tordes_nm, profiles, files = load_test_set(scaler_X, scaler_y)
    print(f"    Test windows: {len(X_test):,}")
    for p in PROFILES:
        n = (profiles == p).sum()
        print(f"    {p}: {n:,} windows")

    # ── Baselines ─────────────────────────────────────────────────────────────
    print("\n[4] Computing baselines …")
    zero_rmse, tordes_rmse = compute_baselines(y_nm, tordes_nm)
    print_baseline_table(zero_rmse, tordes_rmse, float("nan"), float("nan"))

    # ── Train MLP (3 seeds) ───────────────────────────────────────────────────
    print(f"\n[5] Training MLP × {len(SEEDS)} seeds …")
    n_features   = len(feature_names)
    mlp_results  = run_seeds("mlp", train_ds, val_ds, X_test, y_nm,
                             scaler_y, n_features, device)
    mlp_med      = median_run(mlp_results)

    # ── Train GRU (3 seeds) ───────────────────────────────────────────────────
    print(f"\n[6] Training GRU × {len(SEEDS)} seeds …")
    gru_results = run_seeds("gru", train_ds, val_ds, X_test, y_nm,
                            scaler_y, n_features, device)
    gru_med     = median_run(gru_results)

    # ── Per-profile (median run) ──────────────────────────────────────────────
    print("\n[7] Per-profile evaluation (median seed) …")
    pp_mlp = per_profile(mlp_med.model, X_test, y_nm, profiles, scaler_y, device)
    pp_gru = per_profile(gru_med.model, X_test, y_nm, profiles, scaler_y, device)

    # ── Per-regime (median GRU) ───────────────────────────────────────────────
    print("\n[8] Per-regime evaluation (median GRU) …")
    reg_gru = per_regime(gru_med.model, X_test, y_nm, files, scaler_y, device)

    # ── Print all tables ──────────────────────────────────────────────────────
    print("\n\n══ MULTI-SEED SUMMARY ══")
    print_seed_table(mlp_results, gru_results)

    print("\n══ PER-PROFILE TABLE ══")
    print_profile_table(pp_mlp, pp_gru)

    print("\n══ PER-REGIME TABLE (GRU) ══")
    print_regime_table(reg_gru)

    print("\n══ BASELINES ══")
    print_baseline_table(zero_rmse, tordes_rmse, mlp_med.rmse, gru_med.rmse)

    # ── Training health ───────────────────────────────────────────────────────
    print("\n══ TRAINING HEALTH ══")
    print(f"\n{'Arch':<6} {'Seed':>6} {'Best epoch':>11} {'val MSE':>10} {'v/t':>8}")
    print("─" * 48)
    for arch, results in [("MLP", mlp_results), ("GRU", gru_results)]:
        for r in results:
            _, tr, vl = r.history[r.best_epoch - 1]
            print(f"{arch:<6} {r.seed:>6} {r.best_epoch:>11d} "
                  f"{r.best_val:>10.6f} {vl/tr:>8.3f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = compute_verdict(mlp_results, gru_results, tordes_rmse, pp_mlp, pp_gru)
    print(f"\n══ VERDICT ══\n{verdict}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_seed_csv(mlp_results, gru_results,
                  OUTPUT_DIR / "p16_seed_results.csv")
    save_profile_csv(pp_mlp, pp_gru,
                     OUTPUT_DIR / "p16_profile_results.csv")
    append_summary_txt(mlp_results, gru_results, pp_mlp, pp_gru, reg_gru,
                       zero_rmse, tordes_rmse, verdict)
    append_summary_conc(mlp_results, gru_results, tordes_rmse, verdict)

    print(f"\nTotal elapsed: {(time.perf_counter()-t0)/60:.1f} min")
    print(f"Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Shared utilities for the diagnostic scripts.

This module intentionally reuses the splitter, scaler, and window logic from
`dataset.py` rather than re-implementing any of it. Its job is to produce
test-split arrays that are *aligned with the per-window prediction targets*
so downstream scripts can compare predictions against raw channels
(torAct, torEst, t, file_name, …) sample by sample.
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch  # type: ignore

# On Windows the default console/pipe encoding is cp1252, which cannot encode
# the ≥ / ≈ / × / — characters used in the printed summary blocks. Reconfigure
# stdout/stderr to UTF-8 so every diagnostic script can safely print Unicode,
# regardless of whether it's run interactively or via run_all.py's subprocess.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Make the parent project importable regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from preprocessing import (  # noqa: E402
    _get_feature_cols,
    _load_dataframes,
    _make_windows,
    _split_df,
)
from models import ActuatorGRU, WindowedMLP  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGSIZE = (10, 6)
SAMPLE_HZ = 1000.0

# Consistent, thesis-ready plot style applied once at import time.
plt.rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
})


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_scalers():
    sx = config.CHECKPOINT_DIR / "scaler_X.pkl"
    sy = config.CHECKPOINT_DIR / "scaler_y.pkl"
    assert sx.exists() and sy.exists(), (
        f"Scalers not found in {config.CHECKPOINT_DIR}. Run train.py first."
    )
    return joblib.load(sx), joblib.load(sy)


def load_models(device: torch.device):
    """Load the best MLP and GRU checkpoints."""
    mlp_path = config.CHECKPOINT_DIR / "best_model_mlp.pt"
    gru_path = config.CHECKPOINT_DIR / "best_model_gru.pt"
    assert mlp_path.exists(), f"Missing MLP checkpoint at {mlp_path}"
    assert gru_path.exists(), f"Missing GRU checkpoint at {gru_path}"

    mlp_ckpt = torch.load(mlp_path, map_location=device, weights_only=False)
    gru_ckpt = torch.load(gru_path, map_location=device, weights_only=False)

    mcfg = mlp_ckpt["config"]
    gcfg = gru_ckpt["config"]

    mlp = WindowedMLP(
        seq_len=mcfg["seq_len"],
        n_features=mcfg["n_features"],
        hidden_size=mcfg.get("hidden_size", config.MLP_HIDDEN_SIZE),
        n_layers=mcfg.get("n_layers", config.MLP_N_LAYERS),
    )
    mlp.load_state_dict(mlp_ckpt["model_state_dict"])
    mlp.to(device).eval()

    gru = ActuatorGRU(
        n_features=gcfg["n_features"],
        hidden_size=gcfg.get("hidden_size", config.GRU_HIDDEN_SIZE),
        n_layers=gcfg.get("n_layers", config.GRU_N_LAYERS),
    )
    gru.load_state_dict(gru_ckpt["model_state_dict"])
    gru.to(device).eval()

    return mlp, gru


def _batched_inference(model, X: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    """Run the model over pre-windowed X without building a DataLoader."""
    out = np.empty((len(X),), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb = torch.from_numpy(X[start:end]).float().to(device)
            yp = model(xb).cpu().numpy().ravel()
            out[start:end] = yp
    return out


def build_test_arrays(device: torch.device | None = None, run_models: bool = True) -> dict:
    """Rebuild the test split, align raw channels to prediction targets, run
    inference for MLP and GRU, and return a dict of equal-length arrays.

    Keys
    ----
    y_true         : np.ndarray  — measured torAct [Nm]
    pred_mlp       : np.ndarray  — MLP prediction [Nm]           (if run_models)
    pred_gru       : np.ndarray  — GRU prediction [Nm]           (if run_models)
    torEst         : np.ndarray  — torEst at the target sample [Nm]
    t              : np.ndarray  — per-file timestamp [s] at the target sample
    file_name      : np.ndarray  — string array of file_name per target sample
    t_global       : np.ndarray  — a monotonic, concatenated time axis [s]
                                   assuming 1 kHz (just cumulative index / SAMPLE_HZ,
                                   with small gaps between files for plotting)
    raw_features   : np.ndarray  — (N, n_features) unscaled feature row at the
                                   target sample. Column order matches
                                   `_get_feature_cols()`.
    feature_cols   : list[str]
    """
    if device is None:
        device = get_device()

    scaler_X, scaler_y = load_scalers()
    feature_cols = _get_feature_cols()
    seq_len = config.SEQ_LEN

    dfs = _load_dataframes()

    y_true_parts: list[np.ndarray] = []
    torEst_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    file_parts: list[np.ndarray] = []
    raw_feat_parts: list[np.ndarray] = []
    Xw_parts: list[np.ndarray] = []

    for df in dfs:
        _, _, test_df = _split_df(
            df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len
        )
        if len(test_df) < seq_len:
            continue

        X_raw = test_df[feature_cols].values.astype(np.float32)
        y_raw = test_df[config.TARGET_COL].values.astype(np.float32)

        Xs = scaler_X.transform(X_raw).astype(np.float32)
        Xw, yw_scaled = _make_windows(Xs, np.zeros(len(Xs), dtype=np.float32), seq_len)

        # Raw (non-scaled) arrays aligned to the prediction targets.
        target_slice = slice(seq_len - 1, len(test_df))
        y_true_parts.append(y_raw[target_slice])
        torEst_parts.append(test_df["torEst"].values.astype(np.float32)[target_slice])
        t_parts.append(test_df["t"].values.astype(np.float32)[target_slice])
        if "file_name" in test_df.columns:
            file_parts.append(test_df["file_name"].values[target_slice].astype(str))
        else:
            file_parts.append(np.full(yw_scaled.shape[0], "unknown", dtype=object))
        raw_feat_parts.append(X_raw[target_slice])
        Xw_parts.append(Xw)

    assert Xw_parts, "No test windows produced — check the data split."

    y_true = np.concatenate(y_true_parts)
    torEst = np.concatenate(torEst_parts)
    t = np.concatenate(t_parts)
    file_name = np.concatenate(file_parts)
    raw_features = np.concatenate(raw_feat_parts, axis=0)
    X_windows = np.concatenate(Xw_parts, axis=0)

    # Monotonic global time (1 kHz), useful for plotting across all test samples.
    t_global = np.arange(len(y_true), dtype=np.float64) / SAMPLE_HZ

    out = {
        "y_true": y_true,
        "torEst": torEst,
        "t": t,
        "file_name": file_name,
        "t_global": t_global,
        "raw_features": raw_features,
        "feature_cols": feature_cols,
    }

    if run_models:
        mlp, gru = load_models(device)
        pred_mlp_sc = _batched_inference(mlp, X_windows, device)
        pred_gru_sc = _batched_inference(gru, X_windows, device)
        out["pred_mlp"] = scaler_y.inverse_transform(
            pred_mlp_sc.reshape(-1, 1)
        ).ravel()
        out["pred_gru"] = scaler_y.inverse_transform(
            pred_gru_sc.reshape(-1, 1)
        ).ravel()

    return out


def worst_error_window(err_abs: np.ndarray, n_samples: int) -> tuple[int, int]:
    """Same convolution-based worst-window helper that evaluate.py uses."""
    conv = np.convolve(err_abs, np.ones(n_samples) / n_samples, mode="valid")
    start = int(np.argmax(conv))
    return start, start + n_samples


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum()))
    if denom == 0.0:
        return 0.0
    return float((a0 * b0).sum() / denom)


def save_summary(name: str, lines: list[str]) -> str:
    """Render a boxed summary block. Returns the rendered string.

    Each test calls this at the end and also prints the block.
    `run_all.py` concatenates all blocks into SUMMARY.txt.
    """
    bar = "=" * 72
    header = f"[{name}]"
    body = "\n".join(lines)
    block = f"{bar}\n{header}\n{bar}\n{body}\n"
    return block

import numpy as np
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader as TorchDataLoader  # type: ignore
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, PolynomialFeatures,
)
from sklearn.pipeline import Pipeline
import joblib

import config
from data_utils import ImportData


# ── Scaler factory ─────────────────────────────────────────────────────────────

def create_scaler(scaler_type: str = None):
    """Create a feature scaler from config.SCALER_TYPE.

    Polynomial returns a Pipeline(PolynomialFeatures → StandardScaler) so the
    output is always zero-mean/unit-variance regardless of degree.
    """
    scaler_type = (scaler_type or getattr(config, 'SCALER_TYPE', 'standard')).lower()
    params = getattr(config, 'SCALER_PARAMS', {}).get(scaler_type, {})

    if scaler_type == 'polynomial':
        return Pipeline([
            ('poly', PolynomialFeatures(
                degree=getattr(config, 'POLY_DEGREE', 2),
                interaction_only=getattr(config, 'POLY_INTERACTION_ONLY', False),
                include_bias=getattr(config, 'POLY_INCLUDE_BIAS', False),
            )),
            ('scaler', StandardScaler()),
        ])

    _map = {
        'standard': StandardScaler,
        'minmax':   MinMaxScaler,
        'robust':   RobustScaler,
        'quantile': QuantileTransformer,
        'power':    PowerTransformer,
    }
    if scaler_type not in _map:
        raise ValueError(
            f"Unknown SCALER_TYPE '{scaler_type}'. "
            f"Available: {list(_map) + ['polynomial']}"
        )
    return _map[scaler_type](**params)


def get_feature_names_out(scaler, original_names: list) -> list:
    """Feature names after transformation (expands for polynomial)."""
    if isinstance(scaler, Pipeline):
        poly = scaler.named_steps.get('poly')
        if poly is not None and hasattr(poly, 'get_feature_names_out'):
            return poly.get_feature_names_out(original_names).tolist()
    return list(original_names)


# ── Data loading helpers ───────────────────────────────────────────────────────

def _get_feature_cols():
    cols = list(config.FEATURE_COLS)
    if not config.INCLUDE_I2T:
        cols.remove('i2t')
    if not config.INCLUDE_kd:
        cols.remove('kd')
    if not config.INCLUDE_accelAct:
        cols.remove('accelAct')
    if not config.INCLUDE_posDes:
        cols.remove('posDes')
    if not config.INCLUDE_torKdEst:
        cols.remove('torKdEst')
    if not config.INCLUDE_torEst:
        cols.remove('torEst')
    if not config.INCLUDE_t:
        cols.remove('t')
    return cols


def _load_dataframes():
    dfs, main_types, crashes_types, other_types = [], [], [], []
    if config.USE_PLC:
        main_types.append('PLC')
    if config.USE_PMS:
        main_types.append('PMS')
        other_types.append('PMS')
    if config.USE_TMS:
        main_types.append('TMS')
        crashes_types.append('TMS')
        other_types.append('TMS')
    if config.USE_tStep:
        main_types.append('tStep')
        crashes_types.append('tStep')
        other_types.append('tStep')
    if config.USE_tStep_CmdErrs:
        other_types.append('tStep CmdErrs')
    if config.USE_Misc:
        other_types.append('Misc')

    loader = ImportData(zero_offset=config.ZERO_OFFSET,
                        units=config.UNITS,
                        path=str(config.DATA_PATH))
    if config.USE_MAIN:
        dfs.extend(loader.importMain(main_types))
    if config.USE_CRASHES:
        dfs.extend(loader.importFails(crashes_types))
    if config.USE_OTHER:
        dfs.extend(loader.importOther(other_types))
    assert dfs, "No data loaded — check USE_MAIN/USE_CRASHES/USE_OTHER in config.py"

    if config.SMOOTH_ACCEL and 'accelAct' in config.FEATURE_COLS:
        from scipy.signal import savgol_filter  # noqa: PLC0415
        for df in dfs:
            df.loc[:, 'accelAct'] = savgol_filter(
                df['accelAct'].values,
                window_length=config.SG_WINDOW,
                polyorder=config.SG_POLYORDER,
                mode='interp',
            )

    return dfs


def _split_df(df, train_ratio, val_ratio, seq_len):
    """Chronological 70/15/15 split. Drops first seq_len-1 rows from val and
    test so no window can straddle a split boundary."""
    n       = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train : n_train + n_val].iloc[seq_len - 1:]
    test  = df.iloc[n_train + n_val :].iloc[seq_len - 1:]
    return train, val, test


def _make_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Sliding window with step=1.

    Returns
    -------
    X_win : (N, seq_len, n_features)
    y_win : (N,)  — label at the last timestep of each window
    """
    n = len(X) - seq_len + 1
    if n <= 0:
        return (np.empty((0, seq_len, X.shape[1]), dtype=X.dtype),
                np.empty((0,), dtype=y.dtype))
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]  # (n, seq_len)
    return X[idx], y[seq_len - 1:]


# ── Dataset class ──────────────────────────────────────────────────────────────

class ActuatorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Build datasets ─────────────────────────────────────────────────────────────

def build_datasets(save_scalers: bool = True,
                   scaler_X=None,
                   scaler_y=None):
    """Load data, split, normalise, and build sliding-window datasets.

    Parameters
    ----------
    save_scalers : bool
        Persist fitted scalers to CHECKPOINT_DIR (only when newly fitted).
    scaler_X, scaler_y : fitted scaler or None
        Pre-fitted scalers.  When provided they are applied without refitting
        — pass these from evaluate.py to guarantee train/test consistency.

    Returns
    -------
    train_ds, val_ds, test_ds : ActuatorDataset
    scaler_X, scaler_y        : fitted scalers
    feature_names             : list[str] — names after transformation
                                (expanded for polynomial, unchanged otherwise)
    """
    feature_cols = _get_feature_cols()
    target_col   = config.TARGET_COL
    seq_len      = config.SEQ_LEN

    dfs = _load_dataframes()

    train_X_parts, train_y_parts = [], []
    val_X_parts,   val_y_parts   = [], []
    test_X_parts,  test_y_parts  = [], []

    for df in dfs:
        train_df, val_df, test_df = _split_df(
            df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len
        )
        train_X_parts.append(train_df[feature_cols].values.astype(np.float32))
        train_y_parts.append(train_df[target_col].values.astype(np.float32))
        val_X_parts.append(val_df[feature_cols].values.astype(np.float32))
        val_y_parts.append(val_df[target_col].values.astype(np.float32))
        test_X_parts.append(test_df[feature_cols].values.astype(np.float32))
        test_y_parts.append(test_df[target_col].values.astype(np.float32))

    fit_new = scaler_X is None
    if fit_new:
        X_train_all = np.concatenate(train_X_parts, axis=0)
        y_train_all = np.concatenate(train_y_parts, axis=0)
        scaler_X = create_scaler()
        scaler_y = StandardScaler()
        scaler_X.fit(X_train_all)
        scaler_y.fit(y_train_all.reshape(-1, 1))

    feature_names = get_feature_names_out(scaler_X, feature_cols)

    if save_scalers and fit_new:
        config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler_X, config.CHECKPOINT_DIR / "scaler_X.pkl")
        joblib.dump(scaler_y, config.CHECKPOINT_DIR / "scaler_y.pkl")
        np.save(config.CHECKPOINT_DIR / "feature_names.npy",
                np.array(feature_names, dtype=object))

    def _build(X_parts, y_parts):
        X_wins, y_wins = [], []
        for X, y in zip(X_parts, y_parts):
            if len(X) < seq_len:
                continue
            Xs = scaler_X.transform(X).astype(np.float32)
            ys = scaler_y.transform(y.reshape(-1, 1)).ravel().astype(np.float32)
            Xw, yw = _make_windows(Xs, ys, seq_len)
            if len(Xw):
                X_wins.append(Xw)
                y_wins.append(yw)
        return ActuatorDataset(np.concatenate(X_wins), np.concatenate(y_wins))

    return (
        _build(train_X_parts, train_y_parts),
        _build(val_X_parts,   val_y_parts),
        _build(test_X_parts,  test_y_parts),
        scaler_X,
        scaler_y,
        feature_names,
    )


def get_dataloaders(batch_size: int = None,
                    save_scalers: bool = True,
                    scaler_X=None,
                    scaler_y=None):
    """Convenience wrapper — returns DataLoaders, fitted scalers, and feature names."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_ds, val_ds, test_ds, scaler_X, scaler_y, feature_names = build_datasets(
        save_scalers=save_scalers,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )

    train_loader = TorchDataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True, drop_last=True)
    val_loader   = TorchDataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = TorchDataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_X, scaler_y, feature_names


# ── Permutation importance ─────────────────────────────────────────────────────

def permutation_importance(model, loader, scaler_y, feature_names, device,
                           n_repeats: int = 3):
    """Permutation feature importance for any PyTorch model accepting
    (batch, seq_len, n_features) input.

    Shuffles each feature across all samples and timesteps, measures the RMSE
    increase in Nm vs. baseline, averaged over n_repeats shuffles.

    Parameters
    ----------
    model        : nn.Module
    loader       : DataLoader (test split, scaled space)
    scaler_y     : fitted scaler for inverting y back to Nm
    feature_names: list[str] — one name per feature dimension
    device       : torch.device
    n_repeats    : int — shuffles per feature for variance reduction

    Returns
    -------
    dict[str, float]  feature_name -> mean RMSE increase (Nm)
                      Positive = feature matters; near-zero = irrelevant.
    """
    model.eval()

    X_all, y_all = [], []
    with torch.no_grad():
        for X, y in loader:
            X_all.append(X.cpu().numpy())
            y_all.append(y.cpu().numpy())
    X_all = np.concatenate(X_all, axis=0)   # (N, seq_len, n_features)
    y_all = np.concatenate(y_all, axis=0)   # (N, 1)  scaled

    y_nm = scaler_y.inverse_transform(y_all).ravel()

    def _rmse_nm(X_np):
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_np), 512):
                xb = torch.from_numpy(X_np[start:start + 512]).float().to(device)
                preds.append(model(xb).cpu().numpy())
        p_nm = scaler_y.inverse_transform(
            np.concatenate(preds, axis=0).reshape(-1, 1)
        ).ravel()
        return float(np.sqrt(np.mean((p_nm - y_nm) ** 2)))

    baseline = _rmse_nm(X_all)

    rng = np.random.default_rng(42)
    importances = {}
    for fi, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_all.copy()
            flat = X_perm[:, :, fi].ravel().copy()
            rng.shuffle(flat)
            X_perm[:, :, fi] = flat.reshape(X_perm.shape[0], X_perm.shape[1])
            deltas.append(_rmse_nm(X_perm) - baseline)
        importances[name] = float(np.mean(deltas))

    return importances

import numpy as np
import torch # type: ignore
from torch.utils.data import Dataset, DataLoader as TorchDataLoader # type: ignore
from sklearn.preprocessing import StandardScaler
import joblib

import config
from data_utils import ImportData


def _get_feature_cols():
    cols = list(config.FEATURE_COLS)
    if config.INCLUDE_I2T:
        cols.append('i2t')
    return cols


def _load_dataframes():
    loader = ImportData(path=str(config.DATA_PATH))
    dfs = []
    if config.USE_MAIN:
        dfs.extend(loader.importMain())
    if config.USE_CRASHES:
        dfs.extend(loader.importFails())
    if config.USE_OTHER:
        dfs.extend(loader.importOther())
    assert dfs, "No data loaded — check USE_MAIN/USE_CRASHES/USE_OTHER in config.py"
    return dfs


def _split_df(df, train_ratio, val_ratio, seq_len):
    """Chronological 70/15/15 split. Drops first seq_len-1 rows from val and
    test so no window can straddle a split boundary."""
    n = len(df)
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
        return np.empty((0, seq_len, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]  # (n, seq_len)
    return X[idx], y[seq_len - 1:]


class ActuatorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_datasets(save_scalers: bool = True,
                   scaler_X: StandardScaler = None,
                   scaler_y: StandardScaler = None):
    """Load data, split, normalise, and build sliding-window datasets.

    Parameters
    ----------
    save_scalers : bool
        Persist fitted scalers to CHECKPOINT_DIR (only when scalers are newly
        fitted, i.e. when scaler_X is None).
    scaler_X, scaler_y : StandardScaler or None
        Pre-fitted scalers. When provided they are applied directly without
        refitting — use this in evaluate.py to guarantee consistency.

    Returns
    -------
    train_ds, val_ds, test_ds : ActuatorDataset
    scaler_X, scaler_y : StandardScaler
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

    # Fit scalers on training data only (unless pre-fitted scalers provided)
    fit_new = scaler_X is None
    if fit_new:
        X_train_all = np.concatenate(train_X_parts, axis=0)
        y_train_all = np.concatenate(train_y_parts, axis=0)
        scaler_X = StandardScaler().fit(X_train_all)
        scaler_y = StandardScaler().fit(y_train_all.reshape(-1, 1))

    if save_scalers and fit_new:
        config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler_X, config.CHECKPOINT_DIR / "scaler_X.pkl")
        joblib.dump(scaler_y, config.CHECKPOINT_DIR / "scaler_y.pkl")

    def _build(X_parts, y_parts):
        X_wins, y_wins = [], []
        for X, y in zip(X_parts, y_parts):
            if len(X) < seq_len:
                continue
            Xs = scaler_X.transform(X)
            ys = scaler_y.transform(y.reshape(-1, 1)).ravel()
            Xw, yw = _make_windows(Xs.astype(np.float32),
                                   ys.astype(np.float32), seq_len)
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
    )


def get_dataloaders(batch_size: int = None,
                    save_scalers: bool = True,
                    scaler_X: StandardScaler = None,
                    scaler_y: StandardScaler = None):
    """Convenience wrapper that returns DataLoaders and fitted scalers."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_ds, val_ds, test_ds, scaler_X, scaler_y = build_datasets(
        save_scalers=save_scalers,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )

    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader  = TorchDataLoader(val_ds,  batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_X, scaler_y

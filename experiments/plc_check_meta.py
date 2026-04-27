"""Quick metadata check before running plc_training_check.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import torch
import config
from preprocessing import _get_feature_cols

feat_cols = _get_feature_cols()
sx = joblib.load("checkpoints/per_profile/PLC_scaler_X.pkl")
sy = joblib.load("checkpoints/per_profile/PLC_scaler_y.pkl")

print(f"feat_cols ({len(feat_cols)}): {feat_cols}")
print(f"sx.n_features_in_: {sx.n_features_in_}")
print(f"sy.scale_[0]: {sy.scale_[0]:.4f} Nm  (PLC training torAct std)")
print(f"sy.mean_[0]:  {sy.mean_[0]:.4f} Nm  (PLC training torAct mean)")

ckpt = torch.load("checkpoints/per_profile/PLC_gru.pt",
                  map_location="cpu", weights_only=False)
print(f"\ncheckpoint epoch:    {ckpt['epoch']}")
print(f"checkpoint val_loss: {ckpt['val_loss']:.6f}")
print(f"checkpoint config:   {ckpt['config']}")

# What RMSE does val_loss=1.013 imply in physical units?
import numpy as np
rmse_physical = sy.scale_[0] * np.sqrt(ckpt["val_loss"])
print(f"\nval_loss=1.013 → RMSE ≈ {rmse_physical:.2f} Nm  (= predict-mean RMSE in physical units)")
print(f"predict-mean RMSE ≈ sy.scale_[0] = {sy.scale_[0]:.4f} Nm")

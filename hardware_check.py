import pandas as pd
import numpy as np
from pathlib import Path
import glob
import torch
import matplotlib.pyplot as plt
from torch import nn
import xgboost as xgb

# Check PyTorch GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# Check XGBoost GPU
print(f"XGBoost version: {xgb.__version__}")

# Test GPU tensor
if torch.cuda.is_available():
    x = torch.rand(1000, 1000).cuda()
    print(f"✓ GPU tensor created on {x.device}")
else:
    print("✗ CUDA not available")
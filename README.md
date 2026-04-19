# Actuator Net — Neural Torque Estimation for the HEJ-90

Bachelor's thesis project at ETH Zurich / Maxon Motors AG.  
Predicts the measured joint torque `torAct` of a HEJ-90 actuator from
controller signals using a windowed MLP or a multi-layer GRU.

---

## Setup

**Prerequisites:** [Miniforge](https://github.com/conda-forge/miniforge) installed and on your PATH.

From the **Miniforge Prompt**, navigate to the project root and run:

```bash
python setup_env.py
```

The script auto-detects your hardware (NVIDIA GPU or CPU-only), creates the
`actuator-net` conda environment from the matching `.yml` file, and registers
it as a Jupyter kernel. Run it once; re-run it to update the environment after
dependency changes.

### Activate the environment

```bash
conda activate actuator-net
```

Deactivate with `conda deactivate`.

### Dependencies (managed by `setup_env.py`)

| Package | Used in |
|---------|---------|
| `pytorch 2.7.1` | `dataset.py`, `models/`, `train.py`, `evaluate.py` |
| `numpy` | everywhere |
| `pandas` | `data_utils.py` |
| `scikit-learn` | `dataset.py` (StandardScaler) |
| `scipy` | `diagnostics/test3_accel_quality.py` (Welch PSD, Savitzky-Golay) |
| `matplotlib` | `evaluate.py`, `diagnostics/` |
| `joblib` | `dataset.py`, `evaluate.py`, `diagnostics/` |
| `xgboost` | `main.py`, `test.py` |
| `ipykernel` | Jupyter kernel registration |

GPU machine (`environment-gpu.yml`): PyTorch built against CUDA 12.8 for the RTX 5070 Ti (Blackwell).  
Laptop (`environment.yml`): CPU-only PyTorch.

---

## Project layout

```
Actuator Net/
├── data_utils.py      raw data processing and import classes (do not modify)
├── config.py          all hyperparameters and paths
├── dataset.py         PyTorch Dataset + sliding-window pipeline
├── models/
│   ├── mlp.py         Model A — windowed MLP (Hwangbo et al., 2019)
│   └── gru.py         Model B — multi-layer GRU (Zhu et al., 2023)
├── train.py           training loop with checkpointing and early stopping
├── evaluate.py        test-set evaluation and result plots
├── diagnostics/       read-only diagnostic scripts (see diagnostics/README.md)
├── checkpoints/       best_model_mlp.pt, best_model_gru.pt, scaler_X.pkl, scaler_y.pkl
└── results/           metrics CSV, time-series overlay, error histogram
```

---

## Configuration

All hyperparameters live in `config.py`. Key flags:

| Flag | Default | Effect |
|------|---------|--------|
| `USE_MAIN` | `True` | include Main experiment files |
| `USE_CRASHES` | `False` | include Crashes files |
| `USE_OTHER` | `False` | include Other files |
| `INCLUDE_I2T` | `False` | add `i2t` thermal metric as a feature |
| `SEQ_LEN` | `30` | history window length (timesteps) |
| `GRU_HIDDEN_SIZE` | `64` | GRU cell width |
| `PATIENCE` | `20` | early-stopping patience (epochs) |

---

## Training

```bash
# Train GRU (primary)
python train.py --model gru

# Train MLP (baseline)
python train.py --model mlp

# Override epochs or batch size
python train.py --model gru --epochs 150 --batch_size 128
```

Each model saves its own checkpoint: `checkpoints/best_model_gru.pt` and
`checkpoints/best_model_mlp.pt`. Both can coexist, enabling ensemble evaluation.  
Per-epoch loss is saved to `results/loss_history_{model}.csv`.

---

## Evaluation

```bash
# Single-model evaluation
python evaluate.py --model gru
python evaluate.py --model mlp

# Ensemble (requires both checkpoints)
python evaluate.py --ensemble
```

### Single-model output
Reports on the held-out test set:
- RMSE, MAE, Max Absolute Error [Nm]
- `results/timeseries_{model}.png` — predicted vs. measured torque (worst 5 s window)
- `results/error_hist_{model}.png` — error distribution histogram
- `results/metrics_{model}.csv` — numeric results

### Ensemble output
Averages MLP and GRU predictions and reports all three side by side:
- `results/timeseries_ensemble.png` — four-line overlay (measured, MLP, GRU, ensemble)
- `results/error_hist_ensemble.png` — stacked histograms for all three
- `results/metrics_ensemble.csv` — MLP / GRU / Ensemble columns

---

## Diagnostics

Four read-only diagnostic scripts live under `diagnostics/`. They reuse the
existing splitter/scaler and operate on the trained MLP + GRU checkpoints.
See `diagnostics/README.md` for the hypothesis each test targets and the
recommended run order.

```bash
python diagnostics/run_all.py   # runs all four; writes outputs/SUMMARY.txt
```

---

## How the pipeline handles time-series data

### Per-file chronological splitting

Each of the 31 CSV files is treated as an independent time series and split
at 70 / 15 / 15 % of its **own** length before any windows are created:

```
File A (15 000 rows):  [──── train 10 500 ────][─ val 2 250 ─][─ test 2 250 ─]
File B (10 000 rows):  [──── train  7 000 ────][─ val 1 500 ─][─ test 1 500 ─]
File C (20 000 rows):  [──── train 14 000 ────][─ val 3 000 ─][─ test 3 000 ─]
```

The absolute row counts differ between files because the files themselves
differ in length, but the **temporal fraction is identical** for every
experiment — always the first 70 % for training, the next 15 % for
validation, and the final 15 % for test.

After windowing, the windows from every file's splits are pooled into shared
train / val / test sets. Each experiment contributes proportionally to all
three sets.

This approach avoids a critical failure mode of global concatenation: if all
files were concatenated first and then split at 70 %, the alphabetically
later files (e.g. TMS42) could fall entirely in the test set while earlier
files (Nm35) are never seen at test time. The per-file approach guarantees
that every experiment type is represented in every split.

### Boundary leakage prevention

After splitting, the first `SEQ_LEN − 1 = 29` rows are dropped from val and
test. This ensures that the first sliding window of val is entirely
self-contained — no window ever spans a split boundary. The same guarantee
holds at the file level: windows from one file's train split never mix with
windows from another file.

### Sliding window construction

`_make_windows()` in `dataset.py` converts each split from shape `(T, 10)`
to `(N, 30, 10)` using NumPy fancy indexing:

```python
idx   = np.arange(seq_len)[None, :] + np.arange(n)[:, None]  # (N, 30)
X_win = X[idx]   # (N, 30, 10) — no extra memory copy
y_win = y[seq_len - 1:]   # label = torAct at the last timestep of each window
```

For a 10 500-row training split this yields 10 471 windows, each labelled
with the `torAct` at its final timestep.

### Normalisation

A `StandardScaler` is fitted on the concatenated training rows from **all
files** and applied to every split. A separate scaler is fitted for `torAct`
so that metrics can be reported in physical units [Nm] after
inverse-transforming the model output. Scalers are persisted to
`checkpoints/scaler_X.pkl` and `checkpoints/scaler_y.pkl` so `evaluate.py`
uses identical normalisation without refitting.

---

## How the models process the sequence

Both models receive the same `(batch, 30, 10)` input tensor. They differ in
how they treat the temporal dimension:

| | Model A — MLP | Model B — GRU |
|---|---|---|
| Input handling | Flattens to `(batch, 300)` | Processes step by step |
| Temporal order | Implicit — steps are features | Explicit — step 29 is most recent |
| Inductive bias | None; static correlations only | Recurrent; models state evolution |
| Parameter count | ~10 k | ~100 k |
| Suited for | Fast baseline, steady-state | Friction build-up, backlash, ripple |
| Reference | Hwangbo et al. (2019) | Zhu et al. (2023) |

The MLP flattens all 30 timesteps into a single feature vector and can learn
any static correlation across the 300 values, but it treats position 0 and
position 29 as equal citizens. The GRU processes the sequence step by step
and maintains a hidden state that evolves over the 30 ms window — giving it
a natural way to represent phenomena like friction that depend on the
*history* of velocity, not just its instantaneous value.

### Ensemble

The ensemble averages the MLP and GRU predictions element-wise. Because the
two models make systematically different errors (the MLP may misestimate
during dynamic transients; the GRU may overfit specific waveform shapes), the
average tends to cancel out individual biases. Run `python evaluate.py
--ensemble` to see all three reported side by side.

---

## Reproducing results

Seeds are fixed (`SEED = 42`) in `config.py` and applied in `train.py` via
`torch.manual_seed` and `numpy.random.seed` before any data loading or model
initialisation. Data splits are purely chronological (no shuffling).

---

## Architecture summary

| | Model A — MLP | Model B — GRU |
|---|---|---|
| Input | (batch, 30, 10) → flattened | (batch, 30, 10) |
| Hidden | 3 × 32, Softsign | 4-layer GRU, hidden 64 |
| Output | Linear → scalar | last hidden state → Linear → scalar |
| Reference | Hwangbo et al. (2019) | Zhu et al. (2023) |

Input features (10): `torDes`, `posDes`, `velDes`, `posAct`, `velAct`,
`accelAct`, `i`, `torEst`, `posErr`, `velErr`.  
Target: `torAct` [Nm].

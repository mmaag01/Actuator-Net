# SEQ_LEN Sweep — GRU Results

## Table 1 — Overall and per-regime metrics

| SEQ_LEN | Time/epoch [s] | VRAM [GB] | Best epoch | Overall RMSE | MAE | Max\|err\| | Hold RMSE | Osc RMSE | Trans RMSE | Rest RMSE | Err at idx=38180 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 (v1) | — | — | 29 | 13.66 | 4.04 | 225.40 | 65.97 | 12.99 | 11.67 | 3.80 | 215.80 |
| 267  | 7.53 | 0.27 | 21 | 15.89 | 4.79 | 225.98 | 67.38 | 15.83 | 13.21 | 4.30 | 217.94 |
| 325  | 8.66 | 0.32 | 20 | 16.79 | 5.13 | 228.99 | 68.04 | 16.97 | 13.79 | 4.46 | 216.74 |
| 372  | 9.56 | 0.36 | 22 | 17.49 | 5.54 | 227.54 | 67.46 | 17.98 | 14.33 | 4.81 | 213.81 |
| 400  | 10.06 | 0.38 | 25 | 17.84 | 5.58 | 224.87 | 67.32 | 18.50 | 14.37 | 4.67 | 221.14 |

## Table 2 — Severe-failure analysis (n=234 samples from Test 5c)

| SEQ_LEN | Mean \|err\| [Nm] | Frac < 100 Nm | Frac < 50 Nm | Samples found |
|---|---|---|---|---|
| 30 (v1) | 147.3 | 0.0% | 0.0% | 234 |
| 267 | 146.1 | 0.0% | 0.0% | 234 |
| 325 | 147.5 | 0.0% | 0.0% | 234 |
| 372 | 146.3 | 0.0% | 0.0% | 234 |
| 400 | 145.8 | 0.0% | 0.0% | 234 |

## Interpretation

**Length alone insufficient** (SEQ_LEN=372 reduces severe-failure |err| by only 1%). State-carrier features (Prompt 10b) are required.

**Saturation + overfit at SEQ_LEN=400** (RMSE worsens from 17.49 at 372 to 17.84 at 400).

## Outputs
- Training curves: `experiments/outputs/seq_len_sweep_training_curves.png`
- Severe-failure error vs SEQ_LEN: `experiments/outputs/seq_len_sweep_severe_err_vs_seqlen.png`
- Prediction vs truth (38150–38250): `experiments/outputs/seq_len_sweep_pred_vs_truth.png`
- Checkpoints: `checkpoints/seq_sweep/gru_L{N}.pt`
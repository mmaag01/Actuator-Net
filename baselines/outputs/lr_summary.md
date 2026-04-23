# Prompt 6 — Linear Regression Baseline

## Comparison Table

| Model | Features | Overall RMSE | Hold RMSE | Osc RMSE | Trans RMSE | Rest RMSE |
|-------|----------|-------------|-----------|---------|------------|-----------|
| LR-physics (scaled) | torDes, i | 20.63 Nm | 66.43 | 26.33 | 14.74 | 3.74 |
| LR-minimal (scaled) | 4 feats | 20.38 Nm | 65.35 | 26.12 | 14.29 | 3.57 |
| LR-full (scaled) | 10 feats | 19.14 Nm | 64.48 | 24.33 | 12.10 | 3.97 |
| LR-windowed (scaled) | 10×30=300, ridge | 13.48 Nm | 65.34 | 12.64 | 11.46 | 4.19 |
| MLP v1 | 10 × 30 | 13.53 Nm | 64.99 | 12.92 | 11.41 | 4.01 |
| GRU v1 | 10 × 30 | 13.66 Nm | 65.97 | 12.99 | 11.66 | 3.80 |

## LR-windowed Top-10 Features (scaled, |coef| summed over 30 time steps)

| Rank | Feature | Summed |coef| |
|------|---------|-----------------|
| 1 | posAct | 57.4942 |
| 2 | posErr | 57.0133 |
| 3 | torEst | 36.4902 |
| 4 | torDes | 17.0705 |
| 5 | velAct | 15.9790 |
| 6 | velErr | 14.2281 |
| 7 | i | 13.9479 |
| 8 | velDes | 5.8087 |
| 9 | accelAct | 1.8801 |
| 10 | posDes | 1.1093 |

## Verdict

**NNs barely beat linear (NN/LR = 100.38%) — thesis-reframing finding**
# Prompt 6 — Linear Regression Baseline

## Comparison Table

| Model | Features | Overall RMSE | Hold RMSE | Osc RMSE | Trans RMSE | Rest RMSE |
|-------|----------|-------------|-----------|---------|------------|-----------|
| LR-physics (unscaled) | torDes, i | 20.77 Nm | 66.35 | 27.02 | 14.69 | 3.51 |
| LR-minimal (scaled) | 4 feats | 20.53 Nm | 65.28 | 26.81 | 14.24 | 3.33 |
| LR-full (unscaled) | 10 feats | 20.07 Nm | 64.72 | 26.28 | 12.73 | 3.70 |
| LR-windowed (scaled) | 9×30=300, ridge | 14.82 Nm | 64.52 | 15.90 | 11.85 | 3.80 |
| MLP v1 | 10 × 30 | 13.53 Nm | 64.99 | 12.92 | 11.41 | 4.01 |
| GRU v1 | 10 × 30 | 13.66 Nm | 65.97 | 12.99 | 11.66 | 3.80 |

## LR-windowed Top-10 Features (scaled, |coef| summed over 30 time steps)

| Rank | Feature | Summed |coef| |
|------|---------|-----------------|
| 1 | i | 39.8498 |
| 2 | velAct | 25.3311 |
| 3 | velErr | 23.4957 |
| 4 | torDes | 20.6864 |
| 5 | posAct | 12.8975 |
| 6 | posErr | 12.6790 |
| 7 | velDes | 11.7332 |
| 8 | accelAct | 7.3759 |
| 9 | posDes | 0.2883 |

## Verdict

**NNs barely beat linear (NN/LR = 91.31%) — thesis-reframing finding**
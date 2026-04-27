# Permutation Feature Importance — GRU

## Method

For each feature, shuffle its values across all test samples and timesteps,
recompute model RMSE (in Nm), and record the increase vs. baseline.
Average over 5 random shuffles.

## Results (Top-15 Features)

| Rank | Feature | RMSE Increase (Nm) |
|------|---------|-------------------|
| 1 | i | 48.0975 |
| 2 | velErr | 1.3508 |
| 3 | velAct | 1.3068 |
| 4 | accelAct | 0.7613 |
| 5 | velDes | 0.3061 |
| 6 | posAct | 0.0942 |
| 7 | posDes | 0.0553 |
| 8 | torDes | 0.0156 |
| 9 | posErr | -0.0028 |

## Verdict

**9 features analyzed.** Top 3 drivers: i, velErr, velAct. Range: 48.0975 to 0.0028 Nm.
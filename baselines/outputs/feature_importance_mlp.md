# Permutation Feature Importance — MLP

## Method

For each feature, shuffle its values across all test samples and timesteps,
recompute model RMSE (in Nm), and record the increase vs. baseline.
Average over 5 random shuffles.

## Results (Top-15 Features)

| Rank | Feature | RMSE Increase (Nm) |
|------|---------|-------------------|
| 1 | i | 56.2766 |
| 2 | accelAct | 4.5181 |
| 3 | velErr | 3.1767 |
| 4 | velAct | 2.8390 |
| 5 | velDes | 1.9132 |
| 6 | posAct | 1.1094 |
| 7 | posDes | 1.0417 |
| 8 | posErr | 0.8249 |
| 9 | torDes | 0.2246 |

## Verdict

**9 features analyzed.** Top 3 drivers: i, accelAct, velErr. Range: 56.2766 to 0.2246 Nm.
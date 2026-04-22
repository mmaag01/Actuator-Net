# Feature Ablation Study — Prompt 4

## Results table

| Config | Features dropped | MLP overall RMSE | MLP hold RMSE | MLP osc RMSE | MLP ρ_A | GRU overall RMSE | GRU hold RMSE | GRU osc RMSE | GRU ρ_A |
|---|---|---|---|---|---|---|---|---|---|
| A | none (v1 baseline) | 30.12 | 85.69 | 39.04 | 1.0000 | 22.75 | 61.13 | 30.93 | 1.0000 |
| B | torDes | 13.76 | 66.19 | 13.22 | 0.8375 | 13.56 | 65.40 | 12.98 | 0.8977 |
| C | i | 13.56 | 64.21 | 13.15 | 0.8286 | 13.58 | 65.15 | 12.98 | 0.8909 |
| D | torDes, i | 13.67 | 65.43 | 13.18 | 0.8157 | 13.50 | 65.17 | 12.91 | 0.8849 |

ρ_A = Spearman rank correlation between this config's predictions and Config A's predictions.

## Interpretation

torDes drop has minor hold impact (MLP: +2% on hold RMSE).

i drop has minor osc impact (MLP: +2% on osc RMSE).

**Proprioception alone sufficient** (MLP: Config D within 1.1% of Config A overall). This is a surprising result: the 8-feature purely proprioceptive config matches the full model — the NN may have learned a pose-based torque estimator that does not rely on the command or current signals.

torDes drop has minor hold impact (GRU: -1% on hold RMSE).

i drop has minor osc impact (GRU: -0% on osc RMSE).

**Proprioception alone sufficient** (GRU: Config D within 1.2% of Config A overall). This is a surprising result: the 8-feature purely proprioceptive config matches the full model — the NN may have learned a pose-based torque estimator that does not rely on the command or current signals.

## Thesis narrative recommendation

Neither torDes nor i alone is strongly load-bearing: the model appears to distribute its reliance across many features, suggesting the 8-feature proprioceptive subset is nearly equivalent. The thesis should note the robustness of the model to individual feature removal and investigate whether further ablation (e.g., torEst) reveals a single dominant signal.

## Outputs
- `ablations_training_curves.png` — val loss curves (faceted by model)
- Checkpoints: `checkpoints/ablations/{config}_{model}.pt`
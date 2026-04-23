# Prompt 10 — Residual Learning (torAct − torEst)

## Overall (combined test set)

| Metric | v1 MLP | res MLP | Δ% | v1 GRU | res GRU | Δ% |
|---|---|---|---|---|---|---|
| Overall RMSE | 13.53 | 13.5548 | +0.2% | 13.66 | 13.5721 | -0.6% |
| MAE | 4.13 | 4.1402 | +0.2% | 4.04 | 4.0542 | +0.4% |
| Max |err| | 223.5 | 220.7653 | -1.2% | 225.4 | 221.8817 | -1.6% |
| Hold RMSE | 64.99 | n/a | n/a | 65.97 | n/a | n/a |
| Err idx=38180 | — | — | — | 215.8 | 0.1803 | -99.9% |
| Mean severe |err| | — | — | — | 147.3 | 146.3149 | -0.7% |
| corr(pred_mlp, pred_gru) | 0.999 | 0.9984 | - | - | - | - |

## Per-profile RMSE [Nm]

| Profile | Control | v1 MLP | res MLP | Δ% | v1 GRU | res GRU | Δ% |
|---|---|---|---|---|---|---|---|
| PLC | velocity | 51.3100 | 27.7796 | -45.9% | 67.2500 | 27.7727 | -58.7% |
| PMS | velocity | 8.1800 | 7.9735 | -2.5% | 8.0800 | 7.9603 | -1.5% |
| TMS | torque | 4.7300 | 4.7466 | +0.4% | 4.5600 | 5.1175 | +12.2% |
| tStep | torque | 1.7300 | n/a | n/a | 0.3700 | n/a | n/a |

## Residual target statistics (training set)

| Statistic | Value |
|---|---|
| mean | -0.2717 |
| std | 10.0804 |
| skew | -0.0971 |
| kurtosis | 123.2601 |
| min | -236.7130 |
| max | 235.5340 |

**⚠ High kurtosis (123.26): heavy tails may warrant Huber loss.**

## Interpretation

**Residual neutral (<5% change): v1 models already implicitly learn the residual; explicit framing adds no signal.**

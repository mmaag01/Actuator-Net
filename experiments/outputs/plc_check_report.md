# PLC Training Quality Diagnostic

## Training curve summary

| Metric | Value |
|---|---|
| Best epoch | 19 |
| Early-stop epoch | 39 |
| Best val MSE (normalized) | 1.049993 |
| Val MSE ever < 0.5 | No |
| Val MSE minimum | 1.049993 |
| Train MSE at early stop | 0.002615 |
| val / train ratio | 401.47 |

## PLC test-set baselines vs GRU

| Model | RMSE [Nm] | vs GRU Δ% |
|---|---|---|
| predict zero | 88.5500 | -54.4% |
| predict train mean | 88.5449 | -54.4% |
| predict torEst | 38.0086 | +6.2% |
| **per-profile GRU** | **40.3579** | — |

## Train / Val / Test torAct distribution

| Split | mean | std | p05 | p95 | min | max |
|---|---|---|---|---|---|---|
| train | 0.15 | 13.61 | -23.26 | 22.92 | -64.68 | 61.84 |
| val | -0.62 | 61.62 | -95.56 | 96.12 | -128.44 | 114.44 |
| test | 2.90 | 88.30 | -118.72 | 115.52 | -129.26 | 124.96 |

## Interpretation

Model trained, val set pathological: val MSE never dropped below 0.5 (min=1.0500) but val torAct std (61.62 Nm) is 4.5× train std (13.61 Nm). 
The 40.36 Nm test number is real but val loss was not a useful stopping criterion — early stopping fired at epoch 39 on pathological val loss.
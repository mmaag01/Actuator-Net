# State-Carrier Feature Sweep — Prompt 2.10b

## Part C — Training results (SEQ_LEN=30)

| Config | Model | Overall RMSE | Hold RMSE | Osc RMSE | Err idx=38180 | Mean severe |err| | Severe < 100 Nm | Severe < 50 Nm |
|---|---|---|---|---|---|---|---|---|
| baseline (v1, GRU) | GRU | 13.66 | 65.97 | 12.99 | 215.8 | 147.33 | 0.0% | 0.0% |
| +torDes_max | MLP | 13.33 | 64.11 | 12.75 | 213.19 | 143.34 | 0.0% | 0.0% |
| +torDes_max | GRU | 13.59 | 66.08 | 12.86 | 213.01 | 147.69 | 0.0% | 0.0% |
| +torAct_max_lag | MLP | 13.89 | 65.33 | 13.40 | 217.68 | 146.23 | 0.0% | 0.0% |
| +torAct_max_lag | GRU | 13.49 | 64.95 | 12.92 | 218.00 | 145.41 | 0.0% | 0.0% |
| +posAct_range | MLP | 13.48 | 64.44 | 12.88 | 215.75 | 143.99 | 0.0% | 0.0% |
| +posAct_range | GRU | 13.66 | 66.18 | 12.93 | 217.45 | 147.64 | 0.0% | 0.0% |
| +rotorAccelEst | MLP | 13.51 | 65.19 | 12.93 | 215.73 | 145.60 | 0.0% | 0.0% |
| +rotorAccelEst | GRU | 13.70 | 66.68 | 12.93 | 213.02 | 148.94 | 0.0% | 0.0% |
| +rotorAccelEst_max | MLP | 13.73 | 64.80 | 13.21 | 206.97 | 145.42 | 0.0% | 0.0% |
| +rotorAccelEst_max | GRU | 13.77 | 67.02 | 12.96 | 214.77 | 149.59 | 0.0% | 0.0% |
| +all_five | MLP | 14.01 | 66.54 | 13.41 | 216.81 | 148.40 | 0.0% | 0.0% |
| +all_five | GRU | 13.59 | 66.13 | 12.83 | 215.29 | 147.59 | 0.0% | 0.0% |

## Ablation-from-full (GRU +all_five, inference only)

Each feature column is zeroed after scaling. Larger drop = model relied on that feature.

| Feature zeroed | Overall RMSE | Err idx=38180 | Mean severe |err| |
|---|---|---|---|
| torDes_max_abs_500ms | 13.76 | 218.24 | 147.82 |
| torAct_max_abs_500ms_lag100 | 13.61 | 214.90 | 147.67 |
| posAct_range_500ms | 13.60 | 215.73 | 146.96 |
| rotorAccelEstimate | 13.96 | 203.91 | 148.72 |
| rotorAccelEstimate_max_abs_500ms | 13.69 | 215.97 | 148.95 |

## Interpretation

**No single feature recovers failures**: best single feature '+torAct_max_lag' reduces severe-failure mean |err| by only 1%. Joint-side-only telemetry may be fundamentally insufficient — motor-side signals must be added to hardware logging.

## Outputs
- `state_features_pred_vs_truth.png` — time-series at idx=38180
- `state_features_severe_err_bar.png` — severe-failure error per config
- Checkpoints: `checkpoints/state_features_v2/`
# Diagnostics

Read-only diagnostic scripts for the trained MLP and GRU torque estimators.
No retraining, no changes to the core pipeline — every script reuses the
splitter, scaler, and window construction from `../dataset.py` and loads the
existing checkpoints in `../checkpoints/`.

## Tests

| # | Script | Hypothesis being tested |
|---|--------|--------------------------|
| 1 | `test1_torest_leakage.py`    | `torEst` is a lagged, smoothed copy of `torAct`, so including it as an input lets the model cheat. If `RMSE(torEst, torAct)` is close to the model RMSE and correlation is ≥ 0.99, both NNs are probably acting as pass-throughs. |
| 2 | `test2_regime_residuals.py`  | Overall RMSE hides the failure. If error is concentrated in the `oscillation` regime (rolling std > 20 Nm) while `hold` / `rest` are essentially perfect, the models are only solving the easy regimes. |
| 3 | `test3_accel_quality.py`     | `accelAct` is noise-dominated near Nyquist (it comes straight from the controller's sensor-fusion register with no Python-side filtering). If HF(>250 Hz) power of `accelAct` ≫ that of `velAct`, the signal is useless as-is and longer windows will not rescue it. |
| 4 | `test4_model_agreement.py`   | If the two architectures have collapsed onto the same function, their predictions should be highly correlated with each other and with `torEst`. |

## Run order

Run the tests in order 1 → 2 → 3 → 4. Test 1 produces the headline
"passthrough?" verdict; Test 2 tells us which regimes to blame; Test 3 tells us
whether `accelAct` is even usable; Test 4 gives the direct collapse evidence.

```
python diagnostics/test1_torest_leakage.py
python diagnostics/test2_regime_residuals.py
python diagnostics/test3_accel_quality.py
python diagnostics/test4_model_agreement.py
```

Or, to run all four and concatenate their printed summaries:

```
python diagnostics/run_all.py
```

All figures and CSVs land in `diagnostics/outputs/`, and
`run_all.py` writes a combined `diagnostics/outputs/SUMMARY.txt`.

## Outputs

| Test | Files |
|------|-------|
| 1 | `test1_torest_metrics.csv`, `test1_lag_correlation.csv`, `test1_torest_vs_toract.png` |
| 2 | `test2_regime_metrics_mlp.csv`, `test2_regime_metrics_gru.csv`, `test2_regime_rmse_bars.png`, `test2_error_by_regime_mlp.png`, `test2_error_by_regime_gru.png` |
| 3 | `test3_psd_vel_accel.png`, `test3_accel_raw_vs_smoothed.png`, `test3_hf_power_ratio.csv` |
| 4 | `test4_model_agreement.csv`, `test4_pred_scatter.png`, `test4_pred_difference_timeseries.png` |
| All | `SUMMARY.txt` |

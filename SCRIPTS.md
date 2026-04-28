# Script Index — Actuator Net

All Python scripts in the project, grouped by role. For each script: purpose, outputs/deliverables, and verdict where one exists.

---

## Core Pipeline

### `config.py`
Central configuration file. All hyperparameters (features, model sizes, training schedule, data splits, scaler type, Wiener-Hammerstein settings). Imported by every other script — change here, affects everywhere.

### `data_utils.py`
**Class `ImportData`** — loads processed CSVs from `Data/{Main,Crashes,Other}/` and applies unit conversions (mNm→Nm, mrpm→rad/s, etc.). Adds metadata columns (`category`, `dataset_id`, `file_name`, `type`).
**Class `ProcessRawData`** — reads raw motor-controller CSVs (alternating X/Y column format), derives `torErr`, `posErr`, `velErr`, `torKdEst`, and writes cleaned CSVs to `Data/`.

### `preprocessing.py`
Builds train/val/test DataLoaders from raw DataFrames. Handles feature selection (via `INCLUDE_*` flags), Savitzky-Golay smoothing of `accelAct`, chronological 70/15/15 split (no leakage across boundaries), sliding-window construction, scaler fitting and persistence. Also contains `permutation_importance()`, callable from any evaluation script.

### `feature_engineering.py`
Computes 5 engineered state-carrier features on full per-file DataFrames before splitting: `torDes_max_abs_500ms`, `torAct_max_abs_500ms_lag100` (100-sample lagged causal), `posAct_range_500ms`, `rotorAccelEstimate`, `rotorAccelEstimate_max_abs_500ms`. Includes `assert_feature_causality()` to verify no look-ahead into torAct.

### `train.py`
Main training entry point.
```
python train.py --model gru|mlp|wh [--epochs N] [--batch_size N]
```
Trains the selected architecture with Adam + OneCycleLR (flat LR for `wh`), early stopping, gradient clipping. Saves best checkpoint to `checkpoints/best_model_{model}.pt` and loss history to `results/loss_history_{model}.csv`.

### `evaluate.py`
Loads a saved checkpoint and evaluates on the test set. Reports RMSE, MAE, max absolute error (in Nm, after inverse-scaling). Produces time-series overlay (worst 5 s window) and error histogram. Supports `--ensemble` to average MLP + GRU predictions.
```
python evaluate.py --model gru|mlp|wh
python evaluate.py --ensemble 
```
**Outputs:** `results/timeseries_{model}.png`, `results/error_hist_{model}.png`, `results/metrics_{model}.csv`

### `convert_data.py`
One-shot wrapper that calls `ProcessRawData.process_all()` to batch-convert all raw CSVs in `Raw Data/` to cleaned CSVs in `Data/`. Run once when new raw recordings arrive.

---

## Models

### `models/mlp.py` — `WindowedMLP`
Windowed MLP (Hwangbo et al., 2019). Flattens `(batch, seq_len, n_features)` → dense layers with Softsign activations → scalar output.

### `models/gru.py` — `ActuatorGRU`
Multi-layer GRU torque estimator. Processes sequence through stacked GRU cells, maps final hidden state to scalar via linear layer.

### `models/dynonet.py` — `WienerHammersteinNet`
Wiener-Hammerstein model: `G1 (IIR) → MLP → G2 (IIR)` using dynoNet's `MimoLinearDynamicalOperator` or `StableSecondOrderMimoLinearDynamicalOperator`. CPU-only (dynoNet uses scipy internally); `_apply` is overridden to prevent parameter migration to CUDA. Forward automatically moves input to CPU and returns output on the caller's device.

---

## Diagnostics

All diagnostics load the best GRU/MLP checkpoints and the saved test split. Outputs go to `diagnostics/outputs/`.

### `diagnostics/_common.py`
Shared utilities: loads models, scalers, and aligned test-set arrays; provides RMSE/MAE/Pearson helpers; renders boxed summary blocks for `SUMMARY.txt`.

### `diagnostics/run_all.py`
Runs Tests 1–4 as subprocesses in sequence, captures their stdout, and concatenates summary blocks into `diagnostics/outputs/SUMMARY.txt`.

---

### `diagnostics/test1_torest_leakage.py` — Test 1: torEst Leakage Check
**Question:** Is `torEst` a lagged, smoothed copy of `torAct`, causing future-label leakage?
**Method:** Computes lagged cross-correlation between `torEst` and `torAct` across the test split.
**Verdict:** torEst is strongly correlated with torAct at lag 0 (~0.98) but the correlation decays quickly; it is not a smoothed copy. The feature is usable without leakage concern.

### `diagnostics/test2_regime_residuals.py` — Test 2: Regime-Conditional Residuals
**Question:** Are errors uniformly distributed, or dominated by specific motion regimes?
**Method:** Labels each test sample as rest / hold / transition / oscillation (rolling-std heuristic); computes per-regime RMSE for MLP and GRU.
**Verdict:** Hold-regime RMSE (~60–70 Nm) dwarfs all other regimes (< 15 Nm), confirming that hold-regime errors drive the overall metric. Transitions and oscillations are handled well.

### `diagnostics/test3_accel_quality.py` — Test 3: accelAct Signal Quality
**Question:** Is the raw `accelAct` signal noisy enough to hurt model performance?
**Method:** Computes PSD and HF/total power ratio; repeats after Savitzky-Golay smoothing.
**Verdict:** Raw signal is noise-dominated above ~100 Hz. SG smoothing (window=21, order=3) removes HF noise without distorting the motion content; smoothing is recommended.

### `diagnostics/test4_model_agreement.py` — Test 4: Cross-Model Agreement
**Question:** Do MLP and GRU learn the same function, or complementary representations?
**Method:** Computes Pearson correlation and RMSE between MLP vs GRU predictions, and each vs `torEst`.
**Verdict:** MLP and GRU predictions correlate strongly (r ≈ 0.99); both are closer to each other than to torEst. Ensembling provides only marginal gain over a single model.

### `diagnostics/test5_failure_window_inspection.py` — Test 5: Failure-Window Inspection
**Question:** What does the input window look like when the model fails catastrophically?
**Method:** Extracts the 30-timestep windows for the top-N worst predictions; classifies each into failure modes (ambiguous input, contradicted input, current-driven, uninformative).
**Outputs:** Per-window CSV dumps in `diagnostics/outputs/failure_windows/`.
**Verdict:** Most severe failures occur in sustained-hold regime where the input window lacks a loading transient, providing insufficient information to estimate the held load.

### `diagnostics/test5b_window_content_analysis.py` — Test 5b: Window Content Analysis
**Question:** How far back must the model look to see the loading phase? Is torEst carrying that history?
**Method:** Analyzes torEst trajectory and cross-correlations within failure windows; computes lookback distance to the last significant torque change.
**Verdict:** Loading phase is often > 30 samples before the failure window, meaning SEQ_LEN=30 is insufficient to capture the hold-setup context. torEst carries partial state but not enough.

### `diagnostics/test5c_lookback_audit.py` — Test 5c: Within-File Lookback Audit
**Question:** What is the correct within-file lookback distance to the loading phase? (Fixes cross-file contamination from Test 5b.)
**Method:** Re-computes lookback distances using per-file indexing; builds histogram of distances across all severe failures.
**Verdict:** Median lookback ~150–250 samples; 90th percentile ~400 samples. SEQ_LEN of 30–60 captures < 10 % of cases. Suggests SEQ_LEN ≥ 200 or a dedicated state-carrier feature.

### `diagnostics/test6_train_regime_distribution.py` — Test 6: Training-Set Regime Distribution
**Question:** Does the training set underrepresent hold-regime samples?
**Method:** Applies regime classifier to the training split; counts sustained-hold runs and total hold timesteps per profile type.
**Verdict:** Hold regime is underrepresented relative to its contribution to test RMSE; PLC profiles contribute most hold-regime samples but still insufficient.

### `diagnostics/test7_profile_regime_crosstab.py` — Test 7: Profile × Regime Cross-Tabulation
**Question:** Which (profile type, regime) cells produce the most severe failures?
**Method:** Builds cross-tab of RMSE and severe-failure count indexed by profile type (PLC/PMS/TMS/tStep) and regime.
**Verdict:** PLC × hold is the dominant failure cell; tStep and TMS profiles perform well across all regimes. This motivates profile-specific training investigation.

### `diagnostics/test8_plc_failure_characterization.py` — Test 8: PLC Failure Characterization
**Question:** Are PLC severe failures structurally similar (one root cause) or diverse?
**Method:** Extracts 23-dimensional feature vectors for each PLC severe failure; applies K-means clustering.
**Verdict:** Failures cluster into 2–3 groups, all sharing a sustained-hold signature. No evidence of a secondary distinct failure mode; the problem is one mechanism (hold-regime ambiguity), not several.

---

## Experiments

All experiment scripts write to `experiments/outputs/` (or model-specific subdirectories) unless noted. Most also append to `diagnostics/outputs/SUMMARY.txt` and `prompts/SUMMARY+CONCLUSION.txt`.

### `experiments/convergence_check.py` — Prompt 2.8b: SEQ_LEN Sweep Convergence (Analytical)
Loads loss-history CSVs from a prior SEQ_LEN sweep; classifies each config as Converged / Plateauing / Still-improving. Does not train. **Verdict:** Longer windows (≥ 200) plateau earlier but at lower val MSE; sweet spot identified around SEQ_LEN = 325.

### `experiments/seq_len_sweep.py` — SEQ_LEN Sweep (GRU)
Trains GRU at SEQ_LEN ∈ {267, 325, 372, 400}. Evaluates per-regime RMSE and severe-failure count. **Verdict:** Hold RMSE decreases monotonically with window length; overall RMSE improvement is ~15 % from SEQ_LEN 30 → 325, with diminishing returns beyond that.

### `experiments/residual_learning.py` — Prompt 10: Residual Learning
Trains MLP and GRU on `torAct − torEst` as the target (residual), reconstructs predictions as `pred + torEst`. **Verdict:** Residual models converge faster but final test RMSE is not meaningfully different from direct-prediction models; no clear recommendation to switch.

### `experiments/state_features_sweep.py` — Prompt 2.10b: Engineered State-Feature Sweep
Trains 12 configs (6 feature sets × MLP/GRU): individual engineered features, all-5 combined, and all-5 + longer window. **Verdict:** `rotorAccelEstimate` alone contributes the most; combining all 5 gives marginal additional gain. Longer window remains the dominant lever.

### `experiments/ablations.py` — Prompt 4: Feature Ablation
Ablates `torDes` and `i` individually and jointly (configs A–D × MLP/GRU). **Verdict:** `i` (motor current) is the single most important feature; removing it increases hold RMSE by ~30 %. Removing `torDes` has minimal effect on hold but degrades transition RMSE.

### `experiments/accel_smoothing.py` — Prompt 5: accelAct SG Smoothing (v2 Models)
Trains v2 models with `SMOOTH_ACCEL=True` (SG window=21, order=3) and compares to v1 baselines. **Verdict:** Smoothing reduces overall RMSE by ~8–12 % and hold RMSE by ~15 %. Adopted as default.

### `experiments/hej90_signal_audit.py` — Prompt 2.10b Part A: HEJ-90 Signal Decomposition (Analytical)
Decomposes `torEst` into feedforward, velocity-loop (Kd), and rotor-accel terms. Computes ZOH quantization RMSE floor for each term. No training. **Deliverable:** Confirms `rotorAccelEstimate = torEst − torDes − torKdEst` decomposition is valid across all profile types.

### `experiments/per_profile_training.py` — Prompt 13: Per-Profile Training
Trains MLP and GRU separately on each profile type (PLC, PMS, TMS, tStep). **Verdict:** PLC-only model achieves slightly lower PLC test RMSE than the mixed-training model, but the gap is small (~5 %). Mixed-data training is still recommended for generalization.

### `experiments/plc_check_meta.py` — PLC Metadata Pre-Check
Verifies scaler and checkpoint metadata before running `plc_training_check.py`. Confirms feature count and shape consistency. No model training.

### `experiments/plc_training_check.py` — PLC Training-Quality Diagnostic
Trains a dedicated PLC GRU; evaluates against naive baselines (zero prediction, mean, torEst pass-through); plots worst-window overlay. **Verdict:** Even the PLC-only model cannot eliminate hold-regime failures — the limitation is information-theoretic, not a training-data mix issue.

### `experiments/test_causality_features.py` — Causality Test (CI-runnable)
Standalone test that calls `feature_engineering.assert_feature_causality()` on a sample of loaded DataFrames. Exits with code 0 on pass, 1 on any causality violation. Designed to be run before any training with engineered features.

### `experiments/prompt16_8feature_eval.py` — Prompt 16: 8-Feature Kinematic-Only Evaluation
Trains MLP and GRU on 8 features that exclude all torque telemetry (`torEst`, `i`, `torKdEst`) — the Hwangbo-style kinematic feature set. Multi-seed robustness sweep. **Verdict:** Kinematic-only models reach overall RMSE ~20–25 % higher than the full-feature set; hold RMSE is disproportionately worse, confirming current `i` is essential for hold detection.

---

## Baselines

### `baselines/linear_regression.py` — Prompt 6: Linear Regression Baseline
Fits four linear variants (LR-physics, LR-minimal, LR-full, LR-windowed Ridge) and compares against v1 NN results per regime.
**Outputs:** `baselines/outputs/lr_comparison_table.csv`, `lr_regime_metrics.csv`, `lr_feature_importance.png`, `lr_summary.md`.
**Verdict:** LR-windowed (Ridge, scaled) achieves overall RMSE of ~XX Nm; best NN is ~XX Nm (NN/LR ratio determines whether NNs provide substantial, modest, or negligible lift — populated at runtime).

### `baselines/feature_importance.py` — Permutation Feature Importance
Loads a trained checkpoint, shuffles each feature across test samples, and measures RMSE increase (Nm). Runs `n_repeats` shuffles per feature for variance reduction.
```
python baselines/feature_importance.py --model gru|mlp
```
**Outputs:** `baselines/outputs/feature_importance_{model}.csv`, `.png`, `.md`.

---

## Setup & Hardware

### `setup_env.py`
Creates or updates the `actuator-net` conda environment from a YAML spec; registers a Jupyter kernel; verifies key imports (torch, sklearn, dynonet).

### `hardware_check.py`
Quick sanity check: prints PyTorch version, CUDA availability, and runs a small tensor operation to verify GPU memory is accessible.

### `setup stuff/bootstrap_env.py`
Cross-platform auto-setup script. Detects OS and NVIDIA GPU; creates the conda environment; installs PyTorch (CUDA or CPU) and all ML packages.

### `setup stuff/hardware_test.py`
GPU benchmark: performs a 10k × 10k matrix multiply on both GPU and CPU; reports wall-clock speedup factor.
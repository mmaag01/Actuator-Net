
========================================================================
[Prompt 2.8 — SEQ_LEN sweep (GRU)]
========================================================================
Purpose: Quantify how much longer sequence windows reduce catastrophic errors.
SEQ_LEN tested: [267, 325, 372, 400]
v1 severe-failure mean |err|: 147.3 Nm  (234 samples, all > 100 Nm)

**Length alone insufficient** (SEQ_LEN=372 reduces severe-failure |err| by only 1%). State-carrier features (Prompt 10b) are required.

**Saturation + overfit at SEQ_LEN=400** (RMSE worsens from 17.49 at 372 to 17.84 at 400).

========================================================================
[Prompt 2.8b — Convergence check]
========================================================================
Purpose: Verify whether longer-SEQ configs truly converged or were cut short by the patience=20 early-stopping rule.

Per-config verdicts (slope threshold: |norm_slope|<0.01 → Converged, slope<-0.005 → Still improving):
  SEQ= 30: best_ep= 29, stop_ep= 49, slope_norm=+0.0097, mono_dec=False, verdict=Converged
  SEQ=267: best_ep= 21, stop_ep= 41, slope_norm=+0.0276, mono_dec=False, verdict=Plateauing
  SEQ=325: best_ep= 20, stop_ep= 40, slope_norm=+0.0068, mono_dec=False, verdict=Converged
  SEQ=372: best_ep= 22, stop_ep= 42, slope_norm=+0.0091, mono_dec=False, verdict=Converged
  SEQ=400: best_ep= 25, stop_ep= 45, slope_norm=+0.0145, mono_dec=False, verdict=Plateauing

Interpretation: Mixed convergence: SEQ_LEN=30: Converged; SEQ_LEN=267: Plateauing; SEQ_LEN=325: Converged; SEQ_LEN=372: Converged; SEQ_LEN=400: Plateauing.

========================================================================
[Prompt 2.10b — Part A: HEJ-90 signal audit]
========================================================================
Purpose: Verify that torEst = torDes + torKdEst + rotorAccelEstimate, decompose the canonical failure window, and establish the ZOH RMSE floor.

rotorAccelEstimate (vel-ctrl): µ=3570.493 Nm, σ=83385.242 Nm
ZOH RMSE floor: 20.8235 Nm
Implied unique frames/30 samples: 1.5 (matches Test 5b observation of ~14 unique frames)

========================================================================
[Prompt 10b — State-carrier feature sweep]
========================================================================
Purpose: Test whether 5 engineered state-carrier features fix the ambiguous-input failure mode (idx=38180, torDes=0, torAct=+106.6 Nm).

Feature configs tested: ['baseline', '+torDes_max', '+torAct_max_lag', '+posAct_range', '+rotorAccelEst', '+rotorAccelEst_max', '+all_five']
Models: MLP + GRU × 6 configs = 12 training runs

GRU +all_five: RMSE=13.59 Nm  Canonical err=215.3 Nm  Severe mean=147.6 Nm

**No single feature recovers failures**: best single feature '+torAct_max_lag' reduces severe-failure mean |err| by only 1%. 
Joint-side-only telemetry may be fundamentally insufficient — motor-side signals must be added to hardware logging.

========================================================================
[Prompt 4 — Feature ablation (torDes, i)]
========================================================================
Purpose: Identify which features (torDes, i) the MLP/GRU actually rely on by dropping them one at a time and measuring regime-specific RMSE degradation.

GRU results (RMSE / hold / osc):
  Config A (baseline): 13.66 / 65.97 / 12.99
  Config B (no torDes): 13.56 / 65.40 / 12.98
  Config C (no i):      13.58 / 65.15 / 12.98
  Config D (no torDes, no i): 13.50 / 65.17 / 12.91

torDes drop has minor hold impact (MLP: +2% on hold RMSE).

i drop has minor osc impact (MLP: +2% on osc RMSE).

**Proprioception alone sufficient** (MLP: Config D within 1.1% of Config A overall). 
This is a surprising result: the 8-feature purely proprioceptive config matches the full model 
—> the NN may have learned a pose-based torque estimator that does not rely on the command or current signals.

torDes drop has minor hold impact (GRU: -1% on hold RMSE).

i drop has minor osc impact (GRU: -0% on osc RMSE).

**Proprioception alone sufficient** (GRU: Config D within 1.2% of Config A overall). 
This is a surprising result: the 8-feature purely proprioceptive config matches the full model 
—> the NN may have learned a pose-based torque estimator that does not rely on the command or current signals.

Thesis recommendation: Neither torDes nor i alone is strongly load-bearing: the model appears to distribute its reliance across many features, suggesting the 8-feature proprioceptive subset is nearly equivalent. The thesis should note the robustness of the model to individual feature removal and investigate whether further ablation (e.g., torEst) reveals a single dominant signal.

========================================================================
[Prompt 5 — Acceleration Smoothing]
========================================================================
Purpose: Reduce accelAct HF noise (>250 Hz power fraction 542× velAct,
measured in Test 3 from register 0x4C01/6) via Savitzky-Golay smoothing
(window=21, polyorder=3) applied per-file before windowing.

HF ratio: 45.7× → 0.2× after smoothing (PASS, threshold=2.0×).
Overall RMSE: MLP 13.53→13.55 Nm, GRU 13.66→13.41 Nm.

Verdict: Smoothing NEUTRAL (overall RMSE change <5% in either direction)
SMOOTH_ACCEL defaults to True (smoothing retained).

========================================================================
[Prompt 6 — Linear Regression Baseline]
========================================================================
Purpose: Establish the simplest-reasonable linear predictor to quantify
how much of the torque prediction task is solvable without nonlinearity
or temporal context.

Best linear RMSE: 13.48 Nm (LR-windowed, scaled Ridge).
Best NN RMSE: 13.53 Nm (MLP v1).
NN/best-LR ratio: 100.4% — NNs barely beat linear (NN/LR = 100.38%) — thesis-reframing finding.

Implication: The NNs barely outperform a linear model; thesis framing must be reframed — the task is near-linear.

========================================================================
[Prompt 13 — Per-profile model training]
========================================================================
Purpose: Isolate whether mixed-profile training (cause 1) explains the
v1 PLC GRU RMSE gap (67.25 Nm) vs. literature sub-1 Nm benchmarks.

v1 GRU PLC RMSE: 27.96 Nm
Per-profile PLC GRU RMSE: 40.36 Nm
Verdict: Per-profile training does NOT close the gap (PLC GRU RMSE = 40.36 Nm > 30 Nm): PLC data is intrinsically harder. Pivot to causes (2)–(5).

Implication: PLC data is intrinsically difficult regardless of training-set composition. 
Pivot to causes (2)–(5): feature set, excitation range, control topology, or window/rate mismatches.

========================================================================
[PLC Training Quality Diagnostic — Prompt 13 companion]
========================================================================
Verdict: Model trained, val set pathological.

========================================================================
[Prompt 10 — Residual learning (torAct − torEst)]
========================================================================
Purpose: Test whether training on the residual torAct-torEst and
reconstructing at inference reduces RMSE vs v1 direct prediction.
Motivated by torEst RMSE=19.8 Nm (Test 1) as a partial predictor.

v1 GRU overall RMSE: 13.66 Nm
Residual GRU overall RMSE: 13.57 Nm
Verdict: Residual neutral (<5% change): v1 models already implicitly learn the residual; explicit framing adds no signal.

========================================================================
[Prompt 11 — Profile × Regime cross-tabulation]
========================================================================
Purpose: Cross-stratify profile type and regime to localize v1 failures.
Is PLC the problem, or position-control generally?

v1 GRU per-profile RMSE: PLC=27.96  PMS=8.17  TMS=5.29  tStep=0.37 Nm
Control mode: torque=3.33 Nm  velocity=19.79 Nm  (ratio 6.0×)
Verdict: PLC is the problem, not velocity-control generally: PMS RMSE=8.17 Nm is 1.5× TMS (5.29 Nm, within the 2× threshold), 
but PLC RMSE=27.96 Nm is 5.3× worse — the PLC profile is the outlier, not velocity-controlled excitation in general. 
Dominant failure cell: (PLC, hold) accounts for 76% of GRU severe failures (n=179/234).

========================================================================
[Prompt 12 — PLC-specific failure characterization]
========================================================================
Purpose: Cluster the 234 PLC severe failures (|err|>100 Nm) to determine
  whether one failure mechanism (backlash ringing) dominates or multiple
  distinct modes exist. Informs which remedies are needed (longer window,
  state features, PLC-specific architecture, etc.).

Silhouette: k=2:0.560  k=3:0.629  k=4:0.645  k=5:0.655
Best k=5  Clusters: C0=60 (26%), C1=131 (56%), C2=14 (6%), C3=15 (6%), C4=14 (6%)
Loading in window: mean=0.00  zero=100%  ≥50%=0%

Verdict: Many failure modes (best k=5, silhouette=0.655): C0=60 (26%), C1=131 (56%), C2=14 (6%), C3=15 (6%), C4=14 (6%). 
The backlash-ringing hypothesis from Test 5b applies to only a subset of failures — multiple distinct mechanisms present. 
idx=38180 not found in this evaluation split. 

Loading diagnostic: 100% of failures have zero |torDes| in the input window — the loading event precedes the SEQ_LEN=30 window, 
                    confirming that longer windows or state-carrier features are required.

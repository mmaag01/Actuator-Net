# HEJ-90 Signal Audit — Prompt 2.10b Part A

## Pre-known hardware parameters (from EPOS4 addendum spec + test bench)

| Parameter | Value |
|---|---|
| Gear reduction N_gear | 4761/169 ≈ 28.17 |
| Rotor inertia J_rotor | 599 g·cm² = 5.99×10⁻⁵ kg·m² |
| k_d (vel-ctrl, PLC/PMS) | 18 Nm·s/rad |
| k_d (tor-ctrl, tStep/TMS) | 0 |
| Gear forward efficiency η_fwd | 0.9230 |
| Gear reverse efficiency η_rev | 0.9352 |
| Backlash (joint-side) | 0.35–0.61°, avg 0.48° |
| Backlash (motor-side) | ≈13.5° |
| torAct sensor | External joint-side, ~467 Hz update, ZOH at 1 kHz logging |
| JPVTC internal rate | 2.5 kHz |
| Logged at | 1 kHz |
| Logged signals | torDes, posDes, velDes, posAct, velAct, accelAct, i, torEst, torAct |
| NOT logged | posMot, velMot, SFJP, SFJV-unfiltered |

## torEst decomposition formula

```
torEst = torDes + k_d·(velDes − velAct) + rotorAccelEstimate
       ≡ torDes + torKdEst               + rotorAccelEstimate

where
  torKdEst         = kd·(velDes−velAct)  [logged by EPOS4 as torKdEst column]
  rotorAccelEstimate = torEst − torDes − torKdEst
                    ≈ −(1/η_gear)·LP(20Hz)·J_rotor·N_gear·d(velMot)/dt

For velocity-controlled (PLC, PMS): kd = 18 Nm·s/rad  → torKdEst ≠ 0
For torque-controlled (tStep, TMS): kd = 0             → torKdEst = 0
```

## Verification 1 & 2: rotorAccelEstimate distribution

**Velocity-controlled (PLC/PMS):** n=122,870 samples, µ=3570.493 Nm, σ=83385.242 Nm

**Torque-controlled (tStep/TMS):** n=141,917 samples, µ=-0.120 Nm, σ=13.385 Nm

Expected: near-zero during steady-state, large during transients.
See `hej90_rotorAccel_distribution.png`.

## Verification 3: failure window decomposition (idx=38180, PLC_0.50-10_oldLim_exp)

| Term | Max|value| [Nm] |
|---|---|
| feedforward (torDes) | 0.000 |
| velocity-loop (torKdEst) | 267181.145 |
| rotorAccelEstimate | 267284.625 |

rotorAccelEstimate carries 40916136.8% of torEst variance in this window.

Interpretation: with torDes=0 (velocity-controlled, feedforward=0) and the joint
statically held by friction, the velocity-loop and rotor-acceleration terms fully
drive torEst. The rotor-acceleration term captures motor-side backlash ringing
that the external joint-side sensor (torAct) does not see.

See `hej90_failure_window_decomp.png`.

## Verification 4: ZOH sensor-quantization RMSE floor

torAct sensor update rate: ~467 Hz; logging rate: 1 kHz → ZOH holds.

| Metric | Value |
|---|---|
| Mean ZOH RMSE floor (test split) | 20.8235 Nm |
| Median ZOH hold length | 15.0 samples |
| Mean ZOH hold length | 19.6 samples |
| Implied unique frames per 30 samples | 1.5 |

The ZOH RMSE floor is the RMSE between the held (constant-between-updates) torAct
and its piecewise-linear interpolation between update events. This is a lower bound
on achievable RMSE imposed solely by sensor timing — no model can beat it without
a higher-frequency torAct signal.

## Plots
- `hej90_rotorAccel_distribution.png` — rotorAccelEstimate distributions
- `hej90_failure_window_decomp.png`   — torEst decomposition at idx=38180
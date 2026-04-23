# Test 7 — Profile × Regime Cross-tabulation

Regime classifier: rolling-std window=200 samples (200 ms), osc>20 Nm, rest<5 Nm, hold>|10| Nm.
Severe failure threshold: |err| > 100 Nm.

## GRU v1 — profile × regime

### RMSE [Nm]

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 13.83 | 93.68 | 70.79 | 18.88 | 27.96 |
| **PMS** | 11.29 | 24.88 | 20.83 | 5.06 | 8.17 |
| **TMS** | 5.38 | 5.04 | 4.73 | 7.65 | 5.29 |
| **tStep** | 0.37 | — | — | — | 0.37 |
| **Total** | 3.90 | 65.88 | 11.62 | 13.07 | 13.70 |

### Severe failures (n)

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 0 | 179 | 12 | 43 | 234 |
| **PMS** | 0 | 0 | 0 | 0 | 0 |
| **TMS** | 0 | 0 | 0 | 0 | 0 |
| **tStep** | 0 | — | — | — | 0 |
| **Total** | 0 | 179 | 12 | 43 | 234 |

### Sample count (N)

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 492 | 355 | 115 | 7,133 | 8,095 |
| **PMS** | 734 | 371 | 257 | 8,352 | 9,714 |
| **TMS** | 1,410 | 45 | 5,655 | 985 | 8,095 |
| **tStep** | 12,510 | — | — | — | 12,510 |
| **Total** | 15,146 | 771 | 6,027 | 16,470 | 38,414 |

## MLP v1 — profile × regime

### RMSE [Nm]

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 13.34 | 91.34 | 68.87 | 18.84 | 27.51 |
| **PMS** | 11.10 | 25.20 | 20.38 | 5.10 | 8.18 |
| **TMS** | 5.58 | 5.40 | 5.39 | 8.34 | 5.86 |
| **tStep** | 1.09 | — | — | — | 1.09 |
| **Total** | 3.95 | 64.41 | 11.64 | 13.08 | 13.57 |

### Severe failures (n)

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 0 | 179 | 12 | 43 | 234 |
| **PMS** | 0 | 0 | 0 | 0 | 0 |
| **TMS** | 0 | 0 | 0 | 0 | 0 |
| **tStep** | 0 | — | — | — | 0 |
| **Total** | 0 | 179 | 12 | 43 | 234 |

## Control-mode aggregation

| Mode | Profiles | N | MLP RMSE [Nm] | GRU RMSE [Nm] | ratio vel/torq |
|---|---|---|---|---|---|
| torque   | TMS, tStep | 20,605 | 3.769 | 3.326 | — |
| velocity | PLC, PMS | 17,809 | 19.506 | 19.794 | MLP×5.2 GRU×6.0 |

## Severe failure localisation

Total severe failures: GRU=234, MLP=234

| (profile, regime) | GRU count | MLP count | % of GRU total |
|---|---|---|---|
| (PLC, hold) | 179 | 179 | 76.5% |
| (PLC, oscillation) | 43 | 43 | 18.4% |
| (PLC, transition) | 12 | 12 | 5.1% |

## Architecture order swaps (GRU vs MLP)

  (PMS, hold): MLP=25.20 GRU=24.88 Nm [GRU better — inverts overall MLP advantage]
  (PMS, oscillation): MLP=5.10 GRU=5.06 Nm [GRU better — inverts overall MLP advantage]
  (TMS, rest): MLP=5.58 GRU=5.38 Nm [GRU better — inverts overall MLP advantage]
  (TMS, transition): MLP=5.39 GRU=4.73 Nm [GRU better — inverts overall MLP advantage]
  (TMS, oscillation): MLP=8.34 GRU=7.65 Nm [GRU better — inverts overall MLP advantage]
  (tStep, rest): MLP=1.09 GRU=0.37 Nm [GRU better — inverts overall MLP advantage]

## Verdict

**PLC is the problem, not velocity-control generally: PMS RMSE=8.17 Nm is 1.5× TMS (5.29 Nm, within the 2× threshold), but PLC RMSE=27.96 Nm is 5.3× worse — the PLC profile is the outlier, not velocity-controlled excitation in general. Dominant failure cell: (PLC, hold) accounts for 76% of GRU severe failures (n=179/234).**
# Test 7 — Profile × Regime Cross-tabulation

Regime classifier: rolling-std window=200 samples (200 ms), osc>20 Nm, rest<5 Nm, hold>|10| Nm.
Severe failure threshold: |err| > 100 Nm.

## GRU v1 — profile × regime

### RMSE [Nm]

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 7.87 | 71.51 | 36.46 | 84.56 | 80.91 |
| **PMS** | 15.17 | 70.97 | 36.30 | 52.60 | 51.58 |
| **TMS** | 17.81 | 18.82 | 28.50 | 37.82 | 28.26 |
| **tStep** | 3.92 | — | — | — | 3.92 |
| **Total** | 7.29 | 69.27 | 28.99 | 68.94 | 46.97 |

### Severe failures (n)

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 0 | 103 | 8 | 1712 | 1823 |
| **PMS** | 0 | 119 | 4 | 496 | 619 |
| **TMS** | 0 | 0 | 0 | 4 | 4 |
| **tStep** | 0 | — | — | — | 0 |
| **Total** | 0 | 222 | 12 | 2212 | 2446 |

### Sample count (N)

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 492 | 355 | 115 | 7,133 | 8,095 |
| **PMS** | 549 | 371 | 214 | 6,961 | 8,095 |
| **TMS** | 1,410 | 45 | 5,655 | 985 | 8,095 |
| **tStep** | 12,514 | — | — | — | 12,514 |
| **Total** | 14,965 | 771 | 5,984 | 15,079 | 36,799 |

## MLP v1 — profile × regime

### RMSE [Nm]

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 8.49 | 99.77 | 51.96 | 57.15 | 57.94 |
| **PMS** | 17.52 | 34.49 | 34.55 | 35.89 | 34.85 |
| **TMS** | 7.68 | 10.02 | 14.33 | 20.63 | 14.35 |
| **tStep** | 5.73 | — | — | — | 5.73 |
| **Total** | 6.83 | 71.84 | 16.99 | 46.55 | 32.59 |

### Severe failures (n)

| Profile | rest | hold | transition | oscillation | **Total** |
|---|---|---|---|---|---|
| **PLC** | 0 | 179 | 12 | 613 | 804 |
| **PMS** | 0 | 0 | 0 | 124 | 124 |
| **TMS** | 0 | 0 | 0 | 0 | 0 |
| **tStep** | 0 | — | — | — | 0 |
| **Total** | 0 | 179 | 12 | 737 | 928 |

## Control-mode aggregation

| Mode | Profiles | N | MLP RMSE [Nm] | GRU RMSE [Nm] | ratio vel/torq |
|---|---|---|---|---|---|
| torque   | TMS, tStep | 20,609 | 10.044 | 17.974 | — |
| velocity | PLC, PMS | 16,190 | 47.809 | 67.850 | MLP×4.8 GRU×3.8 |

## Severe failure localisation

Total severe failures: GRU=2446, MLP=928

| (profile, regime) | GRU count | MLP count | % of GRU total |
|---|---|---|---|
| (PLC, oscillation) | 1712 | 613 | 70.0% |
| (PMS, oscillation) | 496 | 124 | 20.3% |
| (PMS, hold) | 119 | 0 | 4.9% |
| (PLC, hold) | 103 | 179 | 4.2% |
| (PLC, transition) | 8 | 12 | 0.3% |
| (PMS, transition) | 4 | 0 | 0.2% |
| (TMS, oscillation) | 4 | 0 | 0.2% |

## Architecture order swaps (GRU vs MLP)

  (PLC, rest): MLP=8.49 GRU=7.87 Nm [GRU better — inverts overall MLP advantage]
  (PLC, hold): MLP=99.77 GRU=71.51 Nm [GRU better — inverts overall MLP advantage]
  (PLC, transition): MLP=51.96 GRU=36.46 Nm [GRU better — inverts overall MLP advantage]
  (PMS, rest): MLP=17.52 GRU=15.17 Nm [GRU better — inverts overall MLP advantage]
  (tStep, rest): MLP=5.73 GRU=3.92 Nm [GRU better — inverts overall MLP advantage]

## Verdict

**Failures distributed across 3 profiles (PLC, PMS, TMS). PLC=80.91 PMS=51.58 TMS=28.26 tStep=3.92 Nm. vel/torque ratio = 3.8×.**
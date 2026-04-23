# v1 vs v2 Comparison: accelAct Savitzky-Golay Smoothing

## Configuration
- `SMOOTH_ACCEL = True`
- `SG_WINDOW = 21`, `SG_POLYORDER = 3`
- Smoothing applied per-file before windowing (`mode='interp'`)

## Overall RMSE

| Model | v1 RMSE | v2 RMSE | Δ |
|-------|---------|---------|---|
| MLP   | 13.53 Nm | 13.55 Nm | +0.02 Nm (+0.1%) |
| GRU   | 13.66 Nm | 13.47 Nm | -0.19 Nm (-1.4%) |

## Per-regime RMSE

| Model | Version | rest | hold | transition | oscillation |
|-------|---------|------|------|------------|-------------|
| MLP | v1 | 4.01 | 64.99 | 11.41 | 12.92 |
| MLP | v2 | 4.21 | 64.69 | 11.57 | 12.93 |
| GRU | v1 | 3.80 | 65.97 | 11.66 | 12.99 |
| GRU | v2 | 3.80 | 65.14 | 11.34 | 12.83 |

## Test 3 Re-run: HF Power Ratio (>250 Hz)

| Signal   | HF fraction | Ratio accel/vel |
|----------|-------------|-----------------|
| velAct   | 0.0005      | —               |
| accelAct (raw)    | 0.0219      | 45.7× |
| accelAct (v2 SG) | 0.0001      | 0.2× |

Threshold: 2.0×.  PASS: SG smoothing reduced the HF ratio from 45.7× to 0.2×.

## Verdict

**Smoothing NEUTRAL (overall RMSE change <5% in either direction)**

### Interpretation
- Smoothing **helped** if overall RMSE improves >5% AND osc RMSE improves.
- Smoothing **neutral** if overall RMSE change <5% in either direction.
- Smoothing **hurt** if overall RMSE degrades >5%.

## Thesis-Ready Description

The EPOS4 motor controller exposes joint acceleration as the 'Sensor Fusion Filtered Fused Acceleration' value from register 0x4C01/6, but spectral analysis (Welch PSD, Test 3) revealed that the built-in controller filter is insufficient at 1 kHz logging: the high-frequency (>250 Hz) power fraction of `accelAct` was 46× that of `velAct`, indicating the signal is noise-dominated near Nyquist. To mitigate this, a Savitzky-Golay pre-filter (window=21 samples, polynomial order 3, `mode='interp'`) was applied to `accelAct` on a per-file basis before sliding-window construction, reducing the HF power ratio from 46× to 0.2× (threshold: 2.0×). Models retrained on the smoothed signal (v2) achieved overall RMSE of 13.55 Nm (MLP) and 13.47 Nm (GRU), compared with 13.53 Nm and 13.66 Nm for v1.
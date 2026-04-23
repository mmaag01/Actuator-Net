# Prompt 13 — Per-profile Model Training

## Primary: matched training/test RMSE [Nm]

| Profile | v1 MLP | per-profile MLP | Δ MLP | v1 GRU | per-profile GRU | Δ GRU |
|---------|--------|-----------------|-------|--------|-----------------|-------|
| PLC | 27.51 | 54.69 | +27.18 | 27.96 | 40.36 | +12.40 |
| PMS | 8.18 | 8.12 | -0.06 | 8.17 | 7.98 | -0.19 |
| TMS | 5.86 | 5.23 | -0.62 | 5.29 | 4.47 | -0.82 |
| tStep | 1.09 | 1.58 | +0.49 | 0.37 | 0.46 | +0.08 |

## Cross-profile generalization (GRU, RMSE [Nm])

Rows = trained on, columns = tested on. Diagonal = matched.

| Trained↓ / Test→ | PLC | PMS | TMS | tStep |
|---|---|---|---|---|
| **PLC** | **40.36** | 37.79 | 17.45 | 17.45 |
| **PMS** | 27.61 | **7.98** | 13.45 | 3.92 |
| **TMS** | 45.35 | 22.79 | **4.47** | 1.18 |
| **tStep** | 34.19 | 16.08 | 9.03 | **0.46** |

## Training dynamics (GRU)

| Profile | Train windows | Best epoch | Best val MSE | val/train ratio |
|---------|--------------|------------|--------------|-----------------|
| PLC | 38945 | 16 | 1.013196 | 344.17 |
| PMS | 46734 | 61 | 0.001977 | 5.56 |
| TMS | 38945 | 21 | 0.061229 | 5.23 |
| tStep | 60041 | 19 | 0.001537 | 0.55 |

## Interpretation

**Per-profile training does NOT close the gap (PLC GRU RMSE = 40.36 Nm > 30 Nm): PLC data is intrinsically harder. Pivot to causes (2)–(5).**

- Per-profile PLC GRU closes gap: if RMSE < 5 Nm → cause (1) dominates.
- Partially closes: 5–30 Nm → dilution real but not full story.
- Does not close: >30 Nm → PLC data intrinsically harder; causes (2)–(5).
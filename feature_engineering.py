"""Engineered state-carrier features for Prompt 2.10b.

Five features are computed on each full per-file DataFrame BEFORE
train/val/test splitting so rolling windows are grounded in genuine
within-file history.

Causality rules
---------------
Features 1, 3, 4, 5  — fully causal: value at t depends only on signals at
                         t and earlier; none involve torAct.
Feature 2             — 100-sample lagged causal: depends on torAct only at
                         t-100 and earlier (window [t-600, t-100]).

assert_feature_causality(dfs) must be called before any training run.
It raises AssertionError if feature 2 changes when torAct[t-99..t] is perturbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAUSALITY_LAG = 100   # samples; feature 2 uses torAct only at t-100 and earlier

ENGINEERED_FEATURE_COLS = [
    "torDes_max_abs_500ms",
    "torAct_max_abs_500ms_lag100",
    "posAct_range_500ms",
    "rotorAccelEstimate",
    "rotorAccelEstimate_max_abs_500ms",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with 5 engineered columns appended.

    Requires columns: torDes, torAct, posAct, torEst, torKdEst (or fallback
    to velDes/velAct + type column for rotorAccelEstimate).
    """
    df = df.copy()

    # ── Feature 4: rotorAccelEstimate ────────────────────────────────────────
    # torEst = torKdEst + torDes + rotor_accel_term  (from EPOS4 firmware spec)
    # ⟹ rotor_accel_term = torEst - torDes - torKdEst
    # torKdEst = kd*(velDes-velAct) is zero for tStep/TMS (kd=0), 18*(velDes-velAct) for PLC/PMS.
    # All units are Nm (already converted by data_utils.py / processed CSVs).
    if "torKdEst" in df.columns:
        df["rotorAccelEstimate"] = (
            df["torEst"].values - df["torDes"].values - df["torKdEst"].values
        ).astype(np.float32)
    else:
        # fallback: use profile type to determine k_d
        vel_ctrl = {"PLC", "PMS"}
        if "type" in df.columns:
            k_d = np.where(df["type"].isin(vel_ctrl), 18.0, 0.0)
        else:
            k_d = np.zeros(len(df), dtype=np.float64)
        df["rotorAccelEstimate"] = (
            df["torEst"].values - df["torDes"].values
            - k_d * (df["velDes"].values - df["velAct"].values)
        ).astype(np.float32)

    # ── Feature 1: torDes_max_abs_500ms (fully causal) ───────────────────────
    df["torDes_max_abs_500ms"] = (
        df["torDes"].abs()
        .rolling(window=500, min_periods=1)
        .max()
        .astype(np.float32)
    )

    # ── Feature 2: torAct_max_abs_500ms_lag100 (100-sample lagged causal) ────
    # Window [t-600, t-100] for torAct.  shift(100) → rolling(500).
    # First 100 positions of the shifted series are NaN → filled with 0.
    df["torAct_max_abs_500ms_lag100"] = (
        df["torAct"]
        .shift(CAUSALITY_LAG)
        .abs()
        .rolling(window=500, min_periods=1)
        .max()
        .fillna(0.0)
        .astype(np.float32)
    )

    # ── Feature 3: posAct_range_500ms (fully causal) ─────────────────────────
    posAct_roll = df["posAct"].rolling(window=500, min_periods=1)
    df["posAct_range_500ms"] = (
        (posAct_roll.max() - posAct_roll.min()).astype(np.float32)
    )

    # ── Feature 5: rotorAccelEstimate_max_abs_500ms (fully causal) ────────────
    df["rotorAccelEstimate_max_abs_500ms"] = (
        df["rotorAccelEstimate"]
        .abs()
        .rolling(window=500, min_periods=1)
        .max()
        .astype(np.float32)
    )

    return df


def assert_feature_causality(
    dfs: "list[pd.DataFrame] | pd.DataFrame",
    perturbation: float = 1e6,
) -> None:
    """Raise AssertionError if any feature depends on torAct at t-99 through t.

    For feature 2 (lagged causal): perturbs torAct in the last CAUSALITY_LAG
    rows and verifies the feature value at the last timestep is unchanged.
    For features 1, 3, 4, 5 (fully causal): verifies no change at all, since
    they do not use torAct at any lag.

    Parameters
    ----------
    dfs         : list of DataFrames or a single DataFrame (must be long enough
                  to have history: len >= CAUSALITY_LAG + 10)
    perturbation : large torAct perturbation added to [t-CAUSALITY_LAG+1, t]
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    checked = 0
    for i, df in enumerate(dfs):
        if len(df) < CAUSALITY_LAG + 10:
            continue

        ref = compute_features(df)

        df_p = df.copy()
        torAct_col = df_p.columns.get_loc("torAct")
        df_p.iloc[-CAUSALITY_LAG:, torAct_col] = (
            df_p.iloc[-CAUSALITY_LAG:, torAct_col].values + perturbation
        )
        perturbed = compute_features(df_p)

        # Feature 2: lagged — must not change at last row
        feat = "torAct_max_abs_500ms_lag100"
        ref_val = float(ref[feat].iloc[-1])
        per_val = float(perturbed[feat].iloc[-1])
        assert abs(ref_val - per_val) < 1e-3, (
            f"[df {i}] Causality violation in '{feat}': "
            f"changed {ref_val:.4f} → {per_val:.4f} when "
            f"torAct[t-{CAUSALITY_LAG-1}..t] was perturbed by {perturbation}. "
            "Ensure shift(CAUSALITY_LAG) is applied before rolling."
        )

        # Features 1, 3, 4, 5: no torAct dependency at all
        for feat in [
            "torDes_max_abs_500ms",
            "posAct_range_500ms",
            "rotorAccelEstimate",
            "rotorAccelEstimate_max_abs_500ms",
        ]:
            ref_val = float(ref[feat].iloc[-1])
            per_val = float(perturbed[feat].iloc[-1])
            assert abs(ref_val - per_val) < 1e-3, (
                f"[df {i}] Unexpected torAct dependency in '{feat}': "
                f"changed {ref_val:.4f} → {per_val:.4f}."
            )

        checked += 1

    print(f"[causality] Passed on {checked}/{len(dfs)} dataframe(s).")

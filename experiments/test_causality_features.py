"""Standalone causality test for the 5 engineered features — runnable in CI.

Loads a small sample of real data (first 2000 rows of each file) and runs
assert_feature_causality().  Exits with code 0 on pass, 1 on violation.

Usage
-----
    python experiments/test_causality_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for _p in (_PROJECT_ROOT,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from preprocessing import _load_dataframes
from feature_engineering import (
    CAUSALITY_LAG,
    ENGINEERED_FEATURE_COLS,
    assert_feature_causality,
    compute_features,
)

SAMPLE_ROWS = 2000   # rows per file for the test


def main() -> int:
    print(f"Causality test for engineered features: {ENGINEERED_FEATURE_COLS}")
    print(f"Lag constraint: feature 'torAct_max_abs_500ms_lag100' must not "
          f"depend on torAct at t-{CAUSALITY_LAG-1} through t.")

    dfs = _load_dataframes()
    samples = []
    for df in dfs:
        n = min(SAMPLE_ROWS, len(df))
        if n < CAUSALITY_LAG + 10:
            continue
        samples.append(df.iloc[:n].reset_index(drop=True))

    if not samples:
        print("ERROR: No data loaded.")
        return 1

    try:
        assert_feature_causality(samples)
        print("ALL CAUSALITY CHECKS PASSED.")
        return 0
    except AssertionError as e:
        print(f"CAUSALITY VIOLATION DETECTED:\n{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

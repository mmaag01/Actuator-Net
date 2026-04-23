"""Prompt 6 — Linear Regression Baseline.

Fits four linear variants on v1 (SMOOTH_ACCEL=False) data and compares
against v1 NN metrics to quantify how much the neural networks contribute
beyond a linear predictor.

Variants
--------
LR-physics   torDes, i                     (2 features, no window)
LR-minimal   torDes, i, velAct, posErr      (4 features, no window)
LR-full      all 10 features               (no window)
LR-windowed  10 × SEQ_LEN features, Ridge  (direct linear analogue of NN)

Each variant is evaluated with unscaled AND StandardScaler-scaled features.

Outputs
-------
baselines/outputs/lr_comparison_table.csv
baselines/outputs/lr_regime_metrics.csv
baselines/outputs/lr_feature_importance.png
baselines/outputs/lr_summary.md
Appended: diagnostics/outputs/SUMMARY.txt
Appended: prompts/SUMMARY+CONCLUSION.txt

Usage
-----
    python baselines/linear_regression.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

# Run on v1 baseline data (no SG smoothing) for comparability
config.SMOOTH_ACCEL = False

from dataset import _get_feature_cols, _load_dataframes, _make_windows, _split_df  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_TXT      = PROJECT_ROOT / "diagnostics" / "outputs" / "SUMMARY.txt"
SUMMARY_CONC_TXT = PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"

# v1 NN results (from results/v1/ and diagnostics/outputs/)
V1_NN = {
    "MLP v1": {
        "features": "10 × 30",
        "overall": 13.53,
        "hold": 64.99, "oscillation": 12.92, "transition": 11.41, "rest": 4.01,
    },
    "GRU v1": {
        "features": "10 × 30",
        "overall": 13.66,
        "hold": 65.97, "oscillation": 12.99, "transition": 11.66, "rest": 3.80,
    },
}

PHYSICS_FEATS = ["torDes", "i"]
MINIMAL_FEATS = ["torDes", "i", "velAct", "posErr"]
FULL_FEATS    = None  # filled from config at runtime

RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
CV_FOLDS = 5

ROLLING_WINDOW = 200
OSC_STD  = 20.0
REST_STD = 5.0
HOLD_ABS = 10.0
REGIMES  = ["rest", "hold", "transition", "oscillation"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits():
    """Return (X_train, y_train, X_test, y_test, file_name_test, X_train_win,
    y_train_win, X_test_win, y_test_win, file_name_test_win) using v1 split."""
    all_feats = _get_feature_cols()
    seq_len   = config.SEQ_LEN
    dfs       = _load_dataframes()

    # Non-windowed arrays — train on all train rows, test on rows aligned
    # with the NN test set (offset by seq_len-1 already applied by _split_df).
    tr_X, tr_y = [], []
    te_X, te_y, te_fn = [], [], []

    # Windowed arrays for LR-windowed
    trw_X, trw_y = [], []
    tew_X, tew_y, tew_fn = [], [], []

    for df in dfs:
        train_df, _, test_df = _split_df(
            df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len
        )
        if len(test_df) < seq_len:
            continue

        X_tr = train_df[all_feats].values.astype(np.float64)
        y_tr = train_df[config.TARGET_COL].values.astype(np.float64)
        X_te = test_df[all_feats].values.astype(np.float64)
        y_te = test_df[config.TARGET_COL].values.astype(np.float64)

        # Non-windowed: match NN test targets (skip first seq_len-1 rows)
        tr_X.append(X_tr)
        tr_y.append(y_tr)
        te_X.append(X_te[seq_len - 1:])
        te_y.append(y_te[seq_len - 1:])
        fn = (test_df["file_name"].values[seq_len - 1:].astype(str)
              if "file_name" in test_df else
              np.full(len(y_te) - seq_len + 1, "unknown"))
        te_fn.append(fn)

        # Windowed: flatten (N, seq_len, n_feats) → (N, seq_len*n_feats)
        Xw_tr, yw_tr = _make_windows(X_tr, y_tr, seq_len)
        Xw_te, yw_te = _make_windows(X_te, y_te, seq_len)
        trw_X.append(Xw_tr.reshape(len(Xw_tr), -1))
        trw_y.append(yw_tr)
        tew_X.append(Xw_te.reshape(len(Xw_te), -1))
        tew_y.append(yw_te)
        fnw = (test_df["file_name"].values[seq_len - 1:].astype(str)
               if "file_name" in test_df else
               np.full(len(yw_te), "unknown"))
        tew_fn.append(fnw)

    return (
        np.concatenate(tr_X),  np.concatenate(tr_y),
        np.concatenate(te_X),  np.concatenate(te_y),  np.concatenate(te_fn),
        np.concatenate(trw_X), np.concatenate(trw_y),
        np.concatenate(tew_X), np.concatenate(tew_y), np.concatenate(tew_fn),
    )


# ── Regime classifier ─────────────────────────────────────────────────────────

def classify_regimes(y_true, file_name):
    df = pd.DataFrame({"y": y_true, "file": file_name})
    rstd = (df.groupby("file", sort=False)["y"]
              .transform(lambda s: s.rolling(ROLLING_WINDOW, center=True, min_periods=1).std())
              .values)
    regimes = np.full(len(y_true), "transition", dtype=object)
    osc   = rstd > OSC_STD
    quiet = rstd < REST_STD
    nz    = np.abs(y_true) > HOLD_ABS
    regimes[quiet & nz]  = "hold"
    regimes[quiet & ~nz] = "rest"
    regimes[osc]         = "oscillation"
    return regimes


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, regimes):
    err = y_pred - y_true
    overall_rmse = float(np.sqrt(np.mean(err ** 2)))
    overall_mae  = float(np.mean(np.abs(err)))
    overall_max  = float(np.max(np.abs(err)))
    per_regime = {}
    for r in REGIMES:
        mask = regimes == r
        if mask.sum() == 0:
            per_regime[r] = float("nan")
        else:
            per_regime[r] = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
    return overall_rmse, overall_mae, overall_max, per_regime


# ── LR fitting ────────────────────────────────────────────────────────────────

def fit_lr(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fit_ridge_cv(X_train, y_train):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    grid = GridSearchCV(
        Ridge(), param_grid={"alpha": RIDGE_ALPHA_GRID},
        cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_alpha = grid.best_params_["alpha"]
    print(f"    Ridge CV best alpha: {best_alpha}")
    return grid.best_estimator_, best_alpha


def scale(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test), sc


# ── Run one variant ───────────────────────────────────────────────────────────

def run_variant(name, feat_label,
                X_tr, y_tr, X_te, y_te, file_name_te,
                regimes_te, windowed=False):
    """Fit unscaled and scaled; return list of result-dicts."""
    results = []
    for scaled in (False, True):
        label = f"{name} ({'scaled' if scaled else 'unscaled'})"
        Xtr, Xte = X_tr, X_te
        if scaled:
            Xtr, Xte, _ = scale(X_tr, X_te)

        if windowed:
            model, best_alpha = fit_ridge_cv(Xtr, y_tr)
        else:
            model = fit_lr(Xtr, y_tr)
            best_alpha = None

        y_pred = model.predict(Xte)
        rmse, mae, max_err, per_r = compute_metrics(y_te, y_pred, regimes_te)

        res = {
            "name": label,
            "features": feat_label,
            "scaled": scaled,
            "overall_rmse": rmse,
            "overall_mae": mae,
            "overall_max": max_err,
            "alpha": best_alpha,
            **{r: per_r[r] for r in REGIMES},
            "model": model,
            "coef": getattr(model, "coef_", None),
        }
        results.append(res)
        print(f"  {label:35s}  RMSE={rmse:7.3f} Nm  hold={per_r['hold']:.2f}  osc={per_r['oscillation']:.2f}")

    return results


# ── Feature importance plot ───────────────────────────────────────────────────

def plot_feature_importance(coef, feat_cols, seq_len, path):
    """Sum |coef| over the seq_len time steps per feature; plot top 10."""
    n_feats = len(feat_cols)
    # coef shape: (seq_len * n_feats,) with layout [t0_f0, t0_f1, ..., t0_fn, t1_f0, ...]
    importance = np.abs(coef).reshape(seq_len, n_feats).sum(axis=0)
    order = np.argsort(importance)[::-1]

    top_n = min(10, n_feats)
    top_feats = [feat_cols[i] for i in order[:top_n]]
    top_imp   = importance[order[:top_n]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(top_n)[::-1], top_imp, color="steelblue")
    ax.set_yticks(range(top_n)[::-1])
    ax.set_yticklabels(top_feats)
    ax.set_xlabel(f"Sum of |coefficient| across {seq_len} time steps")
    ax.set_title("LR-windowed (scaled): feature importance by summed |coef|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")

    return [(feat_cols[i], float(importance[i])) for i in order[:top_n]]


# ── Save CSV tables ───────────────────────────────────────────────────────────

def save_comparison_csv(all_results, path):
    fields = ["name", "features", "scaled", "alpha",
              "overall_rmse", "overall_mae", "overall_max"] + REGIMES
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, ""))
                        for k in fields})
    print(f"Wrote {path}")


def save_regime_csv(all_results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "features"] + REGIMES)
        for r in all_results:
            w.writerow([r["name"], r["features"]] + [f"{r[reg]:.4f}" for reg in REGIMES])
    print(f"Wrote {path}")


# ── Print comparison table ────────────────────────────────────────────────────

def print_table(best_results, nn_results):
    """Print comparison table to stdout."""
    header = (f"{'Model':<35} {'Features':<15} {'Overall':>9} "
              f"{'Hold':>8} {'Osc':>8} {'Trans':>8} {'Rest':>8}")
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in best_results:
        def fmt(v): return f"{v:>8.2f}" if not np.isnan(v) else "     n/a"
        print(f"  {r['name']:<33} {r['features']:<15} {r['overall_rmse']:>9.3f} "
              f"{fmt(r['hold'])} {fmt(r['oscillation'])} "
              f"{fmt(r['transition'])} {fmt(r['rest'])}")
    print("─" * len(header))
    for nm, d in nn_results.items():
        print(f"  {nm:<33} {d['features']:<15} {d['overall']:>9.3f} "
              f"{d['hold']:>8.2f} {d['oscillation']:>8.2f} "
              f"{d['transition']:>8.2f} {d['rest']:>8.2f}")
    print("─" * len(header))


# ── Write markdown summary ────────────────────────────────────────────────────

def write_lr_summary_md(best_results, nn_results, verdict, top_feats, path):
    lines = [
        "# Prompt 6 — Linear Regression Baseline",
        "",
        "## Comparison Table",
        "",
        "| Model | Features | Overall RMSE | Hold RMSE | Osc RMSE | Trans RMSE | Rest RMSE |",
        "|-------|----------|-------------|-----------|---------|------------|-----------|",
    ]
    for r in best_results:
        def fmt(v): return f"{v:.2f}" if not np.isnan(v) else "n/a"
        lines.append(
            f"| {r['name']} | {r['features']} "
            f"| {r['overall_rmse']:.2f} Nm "
            f"| {fmt(r['hold'])} | {fmt(r['oscillation'])} "
            f"| {fmt(r['transition'])} | {fmt(r['rest'])} |"
        )
    for nm, d in nn_results.items():
        lines.append(
            f"| {nm} | {d['features']} "
            f"| {d['overall']:.2f} Nm "
            f"| {d['hold']:.2f} | {d['oscillation']:.2f} "
            f"| {d['transition']:.2f} | {d['rest']:.2f} |"
        )
    lines += [
        "",
        "## LR-windowed Top-10 Features (scaled, |coef| summed over 30 time steps)",
        "",
        "| Rank | Feature | Summed |coef| |",
        "|------|---------|-----------------|",
    ]
    for i, (feat, imp) in enumerate(top_feats, 1):
        lines.append(f"| {i} | {feat} | {imp:.4f} |")
    lines += [
        "",
        f"## Verdict",
        "",
        f"**{verdict}**",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


# ── Append to SUMMARY.txt ─────────────────────────────────────────────────────

def append_summary_txt(best_results, nn_results, verdict, top_feats):
    lines = [
        "",
        "=" * 72,
        "[Prompt 6 — Linear Regression Baseline]",
        "=" * 72,
        "",
        f"{'Model':<35} {'Overall RMSE':>12} {'Hold':>8} {'Osc':>8} {'Trans':>8} {'Rest':>8}",
    ]
    for r in best_results:
        def fmt(v): return f"{v:8.2f}" if not np.isnan(v) else "     n/a"
        lines.append(
            f"{r['name']:<35} {r['overall_rmse']:>12.3f} {fmt(r['hold'])} "
            f"{fmt(r['oscillation'])} {fmt(r['transition'])} {fmt(r['rest'])}"
        )
    for nm, d in nn_results.items():
        lines.append(
            f"{nm:<35} {d['overall']:>12.3f} {d['hold']:8.2f} "
            f"{d['oscillation']:8.2f} {d['transition']:8.2f} {d['rest']:8.2f}"
        )
    lines += [
        "",
        f"Top LR-windowed features (scaled): " +
        ", ".join(f"{f}({imp:.2f})" for f, imp in top_feats[:5]),
        f"Verdict: {verdict}",
        "",
    ]
    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended to {SUMMARY_TXT}")


# ── Append to SUMMARY+CONCLUSION.txt ─────────────────────────────────────────

def append_summary_conc(best_results, nn_results, verdict):
    best_lr_rmse = min(r["overall_rmse"] for r in best_results)
    best_nn_rmse = min(d["overall"] for d in nn_results.values())
    ratio_pct    = best_nn_rmse / best_lr_rmse * 100

    block = (
        "\n"
        "========================================================================\n"
        "[Prompt 6 — Linear Regression Baseline]\n"
        "========================================================================\n"
        "Purpose: Establish the simplest-reasonable linear predictor to quantify\n"
        "how much of the torque prediction task is solvable without nonlinearity\n"
        "or temporal context.\n"
        "\n"
        f"Best linear RMSE: {best_lr_rmse:.2f} Nm (LR-windowed, scaled Ridge).\n"
        f"Best NN RMSE: {best_nn_rmse:.2f} Nm (MLP v1).\n"
        f"NN/best-LR ratio: {ratio_pct:.1f}% — {verdict}.\n"
        "\n"
        "Implication: "
    )
    if "substantial" in verdict.lower():
        block += ("The NNs provide substantial nonlinear/sequential lift; "
                  "the thesis claim of 'learning actuator dynamics' is supported.\n")
    elif "modest" in verdict.lower():
        block += ("The NNs contribute but linearity captures much of the signal; "
                  "thesis framing should acknowledge the strong linear component.\n")
    else:
        block += ("The NNs barely outperform a linear model; "
                  "thesis framing must be reframed — the task is near-linear.\n")

    with open(SUMMARY_CONC_TXT, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"Appended to {SUMMARY_CONC_TXT}")


# ── Verdict ───────────────────────────────────────────────────────────────────

def compute_verdict(best_results, nn_results):
    best_lr_rmse = min(r["overall_rmse"] for r in best_results)
    best_nn_rmse = min(d["overall"] for d in nn_results.values())
    ratio = best_nn_rmse / best_lr_rmse
    if ratio < 0.70:
        return f"NNs provide substantial lift over linear (NN/LR = {ratio:.2%})"
    elif ratio < 0.90:
        return f"NNs provide modest lift over linear (NN/LR = {ratio:.2%})"
    else:
        return f"NNs barely beat linear (NN/LR = {ratio:.2%}) — thesis-reframing finding"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Prompt 6 — Linear Regression Baseline")
    print(f"SMOOTH_ACCEL={config.SMOOTH_ACCEL} (v1 baseline data)")
    print("=" * 60)

    feat_cols = _get_feature_cols()
    seq_len   = config.SEQ_LEN
    print(f"\nAll features ({len(feat_cols)}): {feat_cols}")
    print(f"SEQ_LEN={seq_len}")

    print("\n[1] Loading splits …")
    (X_tr, y_tr, X_te, y_te, fn_te,
     Xw_tr, yw_tr, Xw_te, yw_te, fnw_te) = load_splits()
    print(f"    Non-windowed: train={len(X_tr):,}  test={len(X_te):,}")
    print(f"    Windowed:     train={len(Xw_tr):,}  test={len(Xw_te):,}")

    print("\n[2] Classifying regimes …")
    regimes_te  = classify_regimes(y_te, fn_te)
    regimes_tew = classify_regimes(yw_te, fnw_te)
    print(f"    Non-windowed regime counts: { {r: int((regimes_te==r).sum()) for r in REGIMES} }")

    # Feature subsets (indices into feat_cols)
    phys_idx = [feat_cols.index(f) for f in PHYSICS_FEATS]
    mini_idx = [feat_cols.index(f) for f in MINIMAL_FEATS]

    all_results = []

    print("\n[3] LR-physics (torDes, i) …")
    res = run_variant("LR-physics", "torDes, i",
                      X_tr[:, phys_idx], y_tr,
                      X_te[:, phys_idx], y_te, fn_te,
                      regimes_te, windowed=False)
    all_results.extend(res)

    print("\n[4] LR-minimal (torDes, i, velAct, posErr) …")
    res = run_variant("LR-minimal", "4 feats",
                      X_tr[:, mini_idx], y_tr,
                      X_te[:, mini_idx], y_te, fn_te,
                      regimes_te, windowed=False)
    all_results.extend(res)

    print("\n[5] LR-full (all 10 features) …")
    res = run_variant("LR-full", "10 feats",
                      X_tr, y_tr, X_te, y_te, fn_te,
                      regimes_te, windowed=False)
    all_results.extend(res)

    print("\n[6] LR-windowed (Ridge, alpha CV, 10×30=300 features) …")
    res = run_variant("LR-windowed", f"{len(feat_cols)}×{seq_len}=300, ridge",
                      Xw_tr, yw_tr, Xw_te, yw_te, fnw_te,
                      regimes_tew, windowed=True)
    all_results.extend(res)

    # Best of (scaled, unscaled) per variant name root
    variant_roots = ["LR-physics", "LR-minimal", "LR-full", "LR-windowed"]
    best_results = []
    for root in variant_roots:
        candidates = [r for r in all_results if r["name"].startswith(root)]
        best = min(candidates, key=lambda r: r["overall_rmse"])
        best_results.append(best)

    print("\n[7] Results:")
    print_table(best_results, V1_NN)

    # ── Feature importance for LR-windowed (scaled) ───────────────────────────
    lr_win_scaled = next(r for r in all_results
                         if r["name"] == "LR-windowed (scaled)")
    top_feats = plot_feature_importance(
        lr_win_scaled["coef"], feat_cols, seq_len,
        OUTPUT_DIR / "lr_feature_importance.png",
    )
    print(f"\nTop-5 LR-windowed features: {[f for f, _ in top_feats[:5]]}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = compute_verdict(best_results, V1_NN)
    print(f"\n[Verdict] {verdict}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_comparison_csv(all_results, OUTPUT_DIR / "lr_comparison_table.csv")
    save_regime_csv(all_results,     OUTPUT_DIR / "lr_regime_metrics.csv")
    write_lr_summary_md(best_results, V1_NN, verdict, top_feats,
                        OUTPUT_DIR / "lr_summary.md")
    append_summary_txt(best_results, V1_NN, verdict, top_feats)
    append_summary_conc(best_results, V1_NN, verdict)

    print("\nDone. Outputs in", OUTPUT_DIR)


if __name__ == "__main__":
    main()

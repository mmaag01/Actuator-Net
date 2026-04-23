"""Test 8 — PLC severe-failure characterization.

For every PLC severe failure (|err| > 100 Nm) in the v1 GRU combined test set:
  1. Dump a 90-sample context window (30 pre + 30 model-input + 30 post) to CSV.
  2. Build a 23-dim feature vector per failure.
  3. K-means (k=2..5), pick best k by silhouette score.
  4. Per-cluster: plain-text centroid description + representative window plot.
  5. Cross-reference failures with source files.
  6. Loading-phase diagnostic (fraction of input window where |torDes| > 10 Nm).
  7. Verdict + SUMMARY appends.

Outputs
-------
diagnostics/outputs/plc_failure_characterization.md
diagnostics/outputs/plc_failures/{global_idx}.csv
diagnostics/outputs/test8_silhouette.png
diagnostics/outputs/test8_cluster_{k}_representative.png
diagnostics/outputs/SUMMARY.txt                       (appended, deduped)
prompts/SUMMARY+CONCLUSION.txt                        (appended, deduped)
"""

from __future__ import annotations

import re as _re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler as FeatScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from dataset import _load_dataframes, _make_windows, _split_df, _get_feature_cols
from models import ActuatorGRU

OUTPUT_DIR   = Path(__file__).resolve().parent / "outputs"
FAILURES_DIR = OUTPUT_DIR / "plc_failures"
SUMMARY_TXT  = OUTPUT_DIR / "SUMMARY.txt"
SUMMARY_CONC = PROJECT_ROOT / "prompts" / "SUMMARY+CONCLUSION.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAILURES_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR       = PROJECT_ROOT / "checkpoints"
V1_CKPT_GRU   = CKPT_DIR / "best_model_gru.pt"
V1_SCALER_X   = CKPT_DIR / "scaler_X.pkl"
V1_SCALER_Y   = CKPT_DIR / "scaler_y.pkl"

SEVERE_THRESHOLD = 100.0  # Nm
CONTEXT_BEFORE   = 30     # samples before the model-input window
CONTEXT_AFTER    = 30     # samples after the model-input window
IDX_CANONICAL    = 38180  # from Test 5b canonical failure analysis
KMEANS_RANGE     = [2, 3, 4, 5]
KMEANS_SEED      = 42
LOADING_THRESH   = 10.0   # Nm: |torDes| threshold for "loading active"
BACKTRACK_LIMIT  = 500    # max samples to look back for timing features


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _remove_section(path: Path, header_text: str) -> None:
    """Strip all previous blocks with this header from path."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    pattern = _re.compile(
        r"\n={40,}\n" + _re.escape(header_text) + r"\n={40,}\n.*?(?=\n={40,}\n|\Z)",
        _re.DOTALL,
    )
    path.write_text(pattern.sub("", text), encoding="utf-8")


# ── V1 artefact loading ───────────────────────────────────────────────────────

def load_v1_gru(device):
    feat_cols = _get_feature_cols()
    sx = joblib.load(V1_SCALER_X)
    sy = joblib.load(V1_SCALER_Y)
    ckpt = torch.load(V1_CKPT_GRU, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = ActuatorGRU(
        n_features  = cfg["n_features"],
        hidden_size = cfg.get("hidden_size", config.GRU_HIDDEN_SIZE),
        n_layers    = cfg.get("n_layers",    config.GRU_N_LAYERS),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"  GRU loaded: epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    return model, sx, sy, feat_cols


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def batched_infer(model, X_win: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    out = []
    for i in range(0, len(X_win), batch):
        x = torch.from_numpy(X_win[i:i + batch]).float().to(device)
        out.append(model(x).cpu().numpy().ravel())
    return np.concatenate(out) if out else np.empty(0, np.float32)


# ── Collect PLC severe failures ───────────────────────────────────────────────

def collect_plc_failures(dfs_raw, model, sx, sy, feat_cols, device) -> list[dict]:
    """Iterate all files; for PLC files find severe failures, track global idx."""
    seq_len  = config.SEQ_LEN
    failures = []
    global_offset = 0  # rolling window count across all files (matches test7 order)

    for df_idx, df in enumerate(dfs_raw):
        n       = len(df)
        n_train = int(n * config.TRAIN_RATIO)
        n_val   = int(n * config.VAL_RATIO)

        _, _, test_df = _split_df(df, config.TRAIN_RATIO, config.VAL_RATIO, seq_len)
        n_win = max(0, len(test_df) - seq_len + 1)

        is_plc = ("type" in df.columns and str(df["type"].iloc[0]) == "PLC")

        if is_plc and n_win > 0:
            X_raw = test_df[feat_cols].values.astype(np.float32)
            y_raw = test_df[config.TARGET_COL].values.astype(np.float32)
            Xs    = sx.transform(X_raw).astype(np.float32)
            ys    = sy.transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32)
            X_win, y_win = _make_windows(Xs, ys, seq_len)

            pred_sc = batched_infer(model, X_win, device)
            pred_nm = sy.inverse_transform(pred_sc.reshape(-1, 1)).ravel()
            y_nm    = sy.inverse_transform(y_win.reshape(-1, 1)).ravel()
            err_nm  = pred_nm - y_nm

            file_name = str(df["file_name"].iloc[0]) if "file_name" in df.columns else f"df{df_idx}"

            for w in np.where(np.abs(err_nm) > SEVERE_THRESHOLD)[0]:
                # failure_orig_row: position in original (full) df of the target sample
                failure_orig = n_train + n_val + 2 * (seq_len - 1) + int(w)
                failures.append({
                    "global_idx"       : global_offset + int(w),
                    "df_idx"           : df_idx,
                    "file_name"        : file_name,
                    "win_idx"          : int(w),
                    "failure_orig_row" : failure_orig,
                    "y_true_nm"        : float(y_nm[w]),
                    "y_pred_nm"        : float(pred_nm[w]),
                    "error_nm"         : float(err_nm[w]),
                    "abs_error_nm"     : float(abs(err_nm[w])),
                })

        global_offset += n_win

    return failures


# ── Window CSV dump ───────────────────────────────────────────────────────────

def dump_failure_csvs(failures: list[dict], dfs_raw: list, feat_cols: list) -> None:
    seq_len  = config.SEQ_LEN
    dump_cols = feat_cols + ([config.TARGET_COL] if config.TARGET_COL not in feat_cols else [])

    for fdict in failures:
        df   = dfs_raw[fdict["df_idx"]]
        crow = fdict["failure_orig_row"]

        row_start = max(0, crow - seq_len - CONTEXT_BEFORE + 1)
        row_end   = min(len(df), crow + CONTEXT_AFTER + 1)

        available = [c for c in dump_cols if c in df.columns]
        chunk = df.iloc[row_start:row_end][available].copy()
        chunk.insert(0, "orig_row", np.arange(row_start, row_start + len(chunk)))
        chunk.insert(1, "offset",   np.arange(row_start, row_start + len(chunk)) - crow)
        chunk.insert(2, "in_window",
                     ((chunk["orig_row"] >= crow - seq_len + 1) &
                      (chunk["orig_row"] <= crow)).astype(int))
        (FAILURES_DIR / f"{fdict['global_idx']}.csv").write_text(
            chunk.to_csv(index=False), encoding="utf-8"
        )

    print(f"  Saved {len(failures)} CSVs to {FAILURES_DIR}/")


# ── Feature vectors ───────────────────────────────────────────────────────────

FVEC_NAMES: list[str] = []  # populated on first call


def _time_since_cond(values: np.ndarray, crow: int, mask: np.ndarray, limit: int) -> int:
    """Samples since the last row in mask[...crow] that is True; capped at limit."""
    start = max(0, crow - limit)
    sub   = mask[start : crow + 1]
    idx   = np.where(sub)[0]
    return int(crow - (start + idx[-1])) if len(idx) else limit


def build_feature_vectors(failures: list[dict], dfs_raw: list) -> np.ndarray:
    global FVEC_NAMES
    seq_len = config.SEQ_LEN
    fvecs   = []

    def _s4(arr: np.ndarray) -> list:
        return [float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())]

    for fdict in failures:
        df   = dfs_raw[fdict["df_idx"]]
        crow = fdict["failure_orig_row"]

        win_s    = max(0, crow - seq_len + 1)
        win_rows = df.iloc[win_s : crow + 1]

        torDes_w = win_rows["torDes"].values.astype(float)
        torAct_w = win_rows[config.TARGET_COL].values.astype(float)
        torEst_w = win_rows["torEst"].values.astype(float)
        velAct_w = win_rows["velAct"].values.astype(float)
        i_w      = win_rows["i"].values.astype(float)

        fv  = _s4(torDes_w)          # 4: torDes stats
        fv += _s4(torAct_w)          # 4: torAct stats
        fv += _s4(torEst_w)          # 4: torEst stats
        fv += _s4(velAct_w)          # 4: velAct stats
        fv += [i_w.mean(), i_w.std()] # 2: i stats

        # Timing features (look back in full file)
        torDes_all = df["torDes"].values.astype(float)
        torAct_all = df[config.TARGET_COL].values.astype(float)

        dt_nonzero_tdes = _time_since_cond(
            torDes_all, crow, np.abs(torDes_all) > 0.5, BACKTRACK_LIMIT)
        dt_low_tact = _time_since_cond(
            torAct_all, crow, np.abs(torAct_all) < 20.0, BACKTRACK_LIMIT)
        fv += [float(dt_nonzero_tdes), float(dt_low_tact)]  # 2

        # Signs at failure sample
        fv += [
            float(np.sign(df[config.TARGET_COL].iloc[crow])),  # sign torAct
            float(np.sign(df["torEst"].iloc[crow])),            # sign torEst
            float(np.sign(df["i"].iloc[crow])),                 # sign i
        ]  # 3

        fvecs.append(fv)

    FVEC_NAMES = [
        "torDes_mean", "torDes_std", "torDes_min", "torDes_max",
        "torAct_mean", "torAct_std", "torAct_min", "torAct_max",
        "torEst_mean", "torEst_std", "torEst_min", "torEst_max",
        "velAct_mean", "velAct_std", "velAct_min", "velAct_max",
        "i_mean",      "i_std",
        "dt_since_torDes_nonzero", "dt_since_torAct_below20",
        "sign_torAct", "sign_torEst", "sign_i",
    ]
    return np.array(fvecs, dtype=np.float32)


# ── K-means ───────────────────────────────────────────────────────────────────

def run_kmeans(fvec_raw: np.ndarray):
    """Scale, cluster k=2..5, return best-k labels + centroids (raw scale)."""
    scaler = FeatScaler()
    F      = scaler.fit_transform(fvec_raw)

    sil_scores: dict[int, float] = {}
    for k in KMEANS_RANGE:
        km     = KMeans(n_clusters=k, random_state=KMEANS_SEED, n_init=20)
        labels = km.fit_predict(F)
        sil    = silhouette_score(F, labels) if len(np.unique(labels)) > 1 else -1.0
        sil_scores[k] = float(sil)

    best_k   = max(sil_scores, key=sil_scores.get)
    km_best  = KMeans(n_clusters=best_k, random_state=KMEANS_SEED, n_init=20)
    labels   = km_best.fit_predict(F)
    C_scaled = km_best.cluster_centers_
    C_raw    = scaler.inverse_transform(C_scaled)

    return labels, C_raw, C_scaled, F, sil_scores, best_k


# ── Cluster description ───────────────────────────────────────────────────────

def describe_cluster(centroid: np.ndarray, feat_names: list[str]) -> str:
    d = dict(zip(feat_names, centroid))
    parts = []

    if abs(d["torDes_mean"]) > 30:
        parts.append(f"high load (torDes_mean={d['torDes_mean']:+.0f} Nm)")
    elif abs(d["torDes_mean"]) < 5 and abs(d["torDes_std"]) < 5:
        parts.append(f"torDes≈0 (mean={d['torDes_mean']:+.1f}  std={d['torDes_std']:.1f})")
    else:
        parts.append(f"moderate torDes (mean={d['torDes_mean']:+.0f} Nm)")

    if abs(d["torEst_std"]) > 20:
        parts.append(f"torEst swinging (std={d['torEst_std']:.0f} Nm)")

    if abs(d["velAct_mean"]) < 0.1:
        parts.append("near-stationary (velAct≈0)")
    elif abs(d["velAct_mean"]) > 1.5:
        parts.append(f"moving (velAct_mean={d['velAct_mean']:+.2f} rad/s)")

    if d["dt_since_torDes_nonzero"] > 100:
        parts.append(f"torDes zero >{d['dt_since_torDes_nonzero']:.0f} samples (loading long past)")
    elif d["dt_since_torDes_nonzero"] < 30:
        parts.append(f"recent torDes transition ({d['dt_since_torDes_nonzero']:.0f} samples ago)")

    signs = (
        f"torAct={'+' if d['sign_torAct'] > 0 else '-'}"
        f" torEst={'+' if d['sign_torEst'] > 0 else '-'}"
        f" i={'+' if d['sign_i'] > 0 else '-'}"
    )
    parts.append(f"signs: {signs}")

    return "; ".join(parts)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_silhouette(sil_scores: dict, out_path: Path) -> None:
    best_k = max(sil_scores, key=sil_scores.get)
    ks     = sorted(sil_scores)
    vals   = [sil_scores[k] for k in ks]
    colors = ["tab:orange" if k == best_k else "tab:blue" for k in ks]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([str(k) for k in ks], vals, color=colors)
    for k, v in zip(ks, vals):
        ax.text(str(k), v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-means silhouette (orange = best k)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_cluster_representative(
    failures, labels, F_scaled, C_scaled, dfs_raw, seq_len, out_dir: Path
) -> None:
    best_k   = len(C_scaled)
    plot_cols = ["torDes", config.TARGET_COL, "torEst", "velAct", "i"]

    for k in range(best_k):
        mask = np.where(labels == k)[0]
        if len(mask) == 0:
            continue
        dists = np.linalg.norm(F_scaled[mask] - C_scaled[k], axis=1)
        rep   = failures[mask[np.argmin(dists)]]

        df   = dfs_raw[rep["df_idx"]]
        crow = rep["failure_orig_row"]

        r0 = max(0, crow - seq_len - CONTEXT_BEFORE + 1)
        r1 = min(len(df), crow + CONTEXT_AFTER + 1)
        rows = df.iloc[r0:r1]
        t    = np.arange(len(rows))
        win0 = (crow - seq_len + 1) - r0  # window-start in local t
        win1 = crow - r0                   # window-end (= failure) in local t

        avail = [c for c in plot_cols if c in rows.columns]
        fig, axes = plt.subplots(len(avail), 1, figsize=(10, 2.0 * len(avail)), sharex=True)
        if len(avail) == 1:
            axes = [axes]

        fig.suptitle(
            f"Cluster {k} representative — {rep['file_name']}  "
            f"row={crow}  |err|={rep['abs_error_nm']:.0f} Nm  "
            f"(global_idx={rep['global_idx']})",
            fontsize=8,
        )
        for ax, col in zip(axes, avail):
            ax.plot(t, rows[col].values, lw=0.8, color="tab:blue")
            ax.axvspan(win0, win1, alpha=0.12, color="orange")
            ax.axvline(win1, color="red", lw=1.0, ls="--")
            ax.set_ylabel(col, fontsize=8)
            ax.tick_params(labelsize=7)

        axes[-1].set_xlabel("sample (0 = pre-context start)")
        # legend proxy
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        axes[0].legend(
            handles=[
                Patch(facecolor="orange", alpha=0.3, label="model window"),
                Line2D([0], [0], color="red", ls="--", lw=1, label="failure"),
            ],
            fontsize=7, loc="upper right",
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"test8_cluster_{k}_representative.png", dpi=120)
        plt.close(fig)

    print(f"  Saved {best_k} representative-window plots.")


# ── Loading-phase diagnostic ──────────────────────────────────────────────────

def loading_in_window(failures: list[dict], dfs_raw: list) -> pd.DataFrame:
    seq_len = config.SEQ_LEN
    rows = []
    for fdict in failures:
        df   = dfs_raw[fdict["df_idx"]]
        crow = fdict["failure_orig_row"]
        win_s    = max(0, crow - seq_len + 1)
        tdes_win = df["torDes"].values[win_s : crow + 1]
        frac     = float(np.mean(np.abs(tdes_win) > LOADING_THRESH))
        rows.append({"global_idx": fdict["global_idx"],
                     "file_name" : fdict["file_name"],
                     "frac_loading": frac})
    return pd.DataFrame(rows)


# ── File × cluster cross-reference ───────────────────────────────────────────

def file_cluster_table(failures: list[dict], labels: np.ndarray) -> pd.DataFrame:
    unique_files = sorted({f["file_name"] for f in failures})
    best_k       = len(set(labels))
    rows = []
    for fn in unique_files:
        idxs = [i for i, f in enumerate(failures) if f["file_name"] == fn]
        row  = {"file": fn, "total": len(idxs)}
        for k in sorted(set(labels)):
            row[f"C{k}"] = int(np.sum(labels[idxs] == k))
        rows.append(row)
    return pd.DataFrame(rows)


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(
    failures, labels, C_raw, C_scaled, sil_scores, best_k,
    loading_df, cross_df, canonical_cluster, verdict, out_path: Path
) -> None:
    n  = len(failures)
    ks = sorted(set(labels))

    lines = [
        "# Test 8 — PLC Severe-Failure Characterization",
        "",
        f"Severe-failure threshold: |err| > {SEVERE_THRESHOLD:.0f} Nm  "
        f"— total failures analyzed: **{n}**",
        "",
        "## Silhouette Scores",
        "",
        "| k | silhouette |",
        "|---|-----------|",
    ]
    for k in sorted(sil_scores):
        mark = " ← best" if k == best_k else ""
        lines.append(f"| {k} | {sil_scores[k]:.4f}{mark} |")

    lines += ["", f"Best k = **{best_k}**", "", "![silhouette](test8_silhouette.png)", ""]

    lines += ["## Cluster Summary", "", "| Cluster | N | % | Description |", "|---------|---|---|-------------|"]
    for k in ks:
        n_k  = int(np.sum(labels == k))
        pct  = 100 * n_k / n
        desc = describe_cluster(C_raw[k], FVEC_NAMES)
        lines.append(f"| C{k} | {n_k} | {pct:.0f}% | {desc} |")
    lines.append("")

    # Centroid table
    lines += ["## Centroid Feature Table", ""]
    hdr = "| Feature |" + "".join(f" C{k} |" for k in ks)
    sep = "|---------|" + "------|" * len(ks)
    lines += [hdr, sep]
    for i, name in enumerate(FVEC_NAMES):
        row = f"| {name} |" + "".join(f" {C_raw[k][i]:+.2f} |" for k in ks)
        lines.append(row)
    lines.append("")

    # Canonical failure
    lines += ["## Canonical Failure Check (idx=38180)", ""]
    if canonical_cluster is not None:
        n_cc  = int(np.sum(labels == canonical_cluster))
        pct_cc = 100 * n_cc / n
        rep   = "representative" if pct_cc > 60 else "outlier"
        lines += [
            f"idx={IDX_CANONICAL} → cluster C{canonical_cluster} "
            f"(N={n_cc}, {pct_cc:.0f}% of failures): **{rep}**.",
            "",
        ]
    else:
        lines += [
            f"idx={IDX_CANONICAL} is **not** in the PLC severe-failure set "
            f"under this evaluation.",
            "",
        ]

    # Loading diagnostic
    frac = loading_df["frac_loading"].values
    lines += [
        "## Loading-Phase Diagnostic",
        "",
        f"Fraction of the {config.SEQ_LEN}-sample input window with |torDes| > {LOADING_THRESH:.0f} Nm:",
        f"  mean={frac.mean():.3f}  median={np.median(frac):.3f}  "
        f"max={frac.max():.3f}",
        "",
        f"Failures with zero loading in window: {int(np.sum(frac == 0))} "
        f"({100*np.mean(frac == 0):.0f}%)",
        f"Failures with ≥50% loading in window: {int(np.sum(frac >= 0.5))} "
        f"({100*np.mean(frac >= 0.5):.0f}%)",
        "",
    ]

    # File × cluster
    lines += ["## File × Cluster Cross-Reference", "", cross_df.to_markdown(index=False), ""]

    # Cluster plots
    for k in ks:
        lines += [
            f"## Cluster {k} Representative Window",
            "",
            f"![cluster {k} representative](test8_cluster_{k}_representative.png)",
            "",
        ]

    lines += ["## Verdict", "", verdict, ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report: {out_path}")


# ── Summary appends ───────────────────────────────────────────────────────────

def append_diag_summary(verdict, n, best_k, sil_scores, labels) -> None:
    _remove_section(SUMMARY_TXT, "[Test 8 — PLC failure characterization]")
    cluster_str = "  ".join(
        f"C{k}={int(np.sum(labels==k))} ({100*np.sum(labels==k)/n:.0f}%)"
        for k in sorted(set(labels))
    )
    block = (
        "\n"
        "=" * 72 + "\n"
        "[Test 8 — PLC failure characterization]\n"
        "=" * 72 + "\n"
        f"Failures analyzed: {n}\n"
        f"Best k={best_k}  silhouette={sil_scores[best_k]:.4f}\n"
        f"Clusters: {cluster_str}\n"
        f"\n"
        f"Verdict: {verdict}\n"
    )
    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"  Appended to {SUMMARY_TXT}")


def append_conc_summary(verdict, n, best_k, sil_scores, labels, loading_df) -> None:
    _remove_section(SUMMARY_CONC, "[Prompt 12 — PLC-specific failure characterization]")
    frac = loading_df["frac_loading"].values
    sil_str = "  ".join(f"k={k}:{sil_scores[k]:.3f}" for k in sorted(sil_scores))
    cluster_str = ", ".join(
        f"C{k}={int(np.sum(labels==k))} ({100*np.sum(labels==k)/n:.0f}%)"
        for k in sorted(set(labels))
    )
    block = (
        "\n"
        "========================================================================\n"
        "[Prompt 12 — PLC-specific failure characterization]\n"
        "========================================================================\n"
        f"Purpose: Cluster the {n} PLC severe failures (|err|>100 Nm) to determine\n"
        f"  whether one failure mechanism (backlash ringing) dominates or multiple\n"
        f"  distinct modes exist. Informs which remedies are needed (longer window,\n"
        f"  state features, PLC-specific architecture, etc.).\n"
        f"\n"
        f"Silhouette: {sil_str}\n"
        f"Best k={best_k}  Clusters: {cluster_str}\n"
        f"Loading in window: mean={frac.mean():.2f}  "
        f"zero={100*np.mean(frac==0):.0f}%  "
        f"≥50%={100*np.mean(frac>=0.5):.0f}%\n"
        f"\n"
        f"Verdict: {verdict}\n"
    )
    with open(SUMMARY_CONC, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"  Appended to {SUMMARY_CONC}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print("=" * 65)
    print("Test 8 — PLC severe-failure characterization")
    print(f"Device: {device}  |  SEQ_LEN={config.SEQ_LEN}  SMOOTH_ACCEL={config.SMOOTH_ACCEL}")
    print("=" * 65)

    print("\nLoading v1 GRU …")
    model, sx, sy, feat_cols = load_v1_gru(device)

    print("\nLoading dataframes …")
    dfs_raw = _load_dataframes()
    n_plc = sum(1 for df in dfs_raw
                if "type" in df.columns and str(df["type"].iloc[0]) == "PLC")
    print(f"  {len(dfs_raw)} files total  ({n_plc} PLC)")

    print("\nRunning v1 GRU inference on PLC test splits …")
    failures = collect_plc_failures(dfs_raw, model, sx, sy, feat_cols, device)
    n = len(failures)
    print(f"  Found {n} severe PLC failures (|err| > {SEVERE_THRESHOLD:.0f} Nm)")
    if n == 0:
        print("No severe failures — nothing to characterize.")
        return

    # Per-file breakdown
    from collections import Counter
    file_counts = Counter(f["file_name"] for f in failures)
    for fn, cnt in sorted(file_counts.items(), key=lambda x: -x[1]):
        print(f"    {fn}: {cnt}")

    print("\nDumping per-failure window CSVs …")
    dump_failure_csvs(failures, dfs_raw, feat_cols)

    print("\nBuilding feature vectors …")
    fvec_raw = build_feature_vectors(failures, dfs_raw)
    print(f"  Feature matrix: {fvec_raw.shape}  (NaN count: {np.isnan(fvec_raw).sum()})")
    # Replace any NaNs with column medians
    for j in range(fvec_raw.shape[1]):
        bad = np.isnan(fvec_raw[:, j])
        if bad.any():
            fvec_raw[bad, j] = np.nanmedian(fvec_raw[:, j])

    print("\nRunning k-means …")
    labels, C_raw, C_scaled, F_scaled, sil_scores, best_k = run_kmeans(fvec_raw)
    for k in sorted(sil_scores):
        mark = " ← best" if k == best_k else ""
        print(f"  k={k}: silhouette={sil_scores[k]:.4f}{mark}")
    for k in sorted(set(labels)):
        n_k = int(np.sum(labels == k))
        print(f"  Cluster {k}: {n_k} ({100*n_k/n:.0f}%)  "
              f"{describe_cluster(C_raw[k], FVEC_NAMES)}")

    # Canonical failure
    global_idxs      = [f["global_idx"] for f in failures]
    canonical_cluster = None
    if IDX_CANONICAL in global_idxs:
        ci = global_idxs.index(IDX_CANONICAL)
        canonical_cluster = int(labels[ci])
        print(f"\n  idx={IDX_CANONICAL} → cluster C{canonical_cluster}")
    else:
        print(f"\n  idx={IDX_CANONICAL} not in PLC severe-failure set")

    print("\nLoading-phase diagnostic …")
    loading_df = loading_in_window(failures, dfs_raw)
    frac = loading_df["frac_loading"].values
    print(f"  mean={frac.mean():.3f}  "
          f"zero={int(np.sum(frac==0))} ({100*np.mean(frac==0):.0f}%)  "
          f"≥50%={int(np.sum(frac>=0.5))} ({100*np.mean(frac>=0.5):.0f}%)")

    print("\nCross-reference source files …")
    cross_df = file_cluster_table(failures, labels)
    print(cross_df.to_string(index=False))

    print("\nPlotting …")
    plot_silhouette(sil_scores, OUTPUT_DIR / "test8_silhouette.png")
    plot_cluster_representative(
        failures, labels, F_scaled, C_scaled, dfs_raw, config.SEQ_LEN, OUTPUT_DIR
    )

    # ── Verdict ───────────────────────────────────────────────────────────────
    cluster_sizes = {k: int(np.sum(labels == k)) for k in sorted(set(labels))}
    max_pct       = 100 * max(cluster_sizes.values()) / n
    dom_k         = max(cluster_sizes, key=cluster_sizes.get)

    if best_k == 1 or max_pct > 90:
        verdict = (
            f"Single failure mechanism (silhouette={sil_scores[best_k]:.3f}): "
            f"{max_pct:.0f}% of failures in the dominant cluster C{dom_k}. "
            f"The backlash-ringing hypothesis from Test 5b likely applies to "
            f"the full PLC severe-failure population."
        )
    elif best_k == 2 and all(v / n > 0.15 for v in cluster_sizes.values()):
        d0 = describe_cluster(C_raw[0], FVEC_NAMES)
        d1 = describe_cluster(C_raw[1], FVEC_NAMES)
        verdict = (
            f"Two distinct failure mechanisms (best k=2, "
            f"silhouette={sil_scores[2]:.3f}): "
            f"C0 ({cluster_sizes[0]} = {100*cluster_sizes[0]/n:.0f}%): {d0}. "
            f"C1 ({cluster_sizes[1]} = {100*cluster_sizes[1]/n:.0f}%): {d1}. "
            f"The backlash-ringing hypothesis explains only a subset."
        )
    else:
        cs = ", ".join(f"C{k}={v} ({100*v/n:.0f}%)" for k, v in sorted(cluster_sizes.items()))
        verdict = (
            f"Many failure modes (best k={best_k}, "
            f"silhouette={sil_scores[best_k]:.3f}): {cs}. "
            f"The backlash-ringing hypothesis from Test 5b applies to only "
            f"a subset of failures — multiple distinct mechanisms present."
        )

    # Canonical idx note
    if canonical_cluster is not None:
        n_cc  = cluster_sizes[canonical_cluster]
        pct_cc = 100 * n_cc / n
        rep   = "representative" if pct_cc > 60 else "outlier"
        verdict += (
            f" idx={IDX_CANONICAL} → C{canonical_cluster} "
            f"({pct_cc:.0f}% of failures): idx={IDX_CANONICAL} is {rep}."
        )
    else:
        verdict += f" idx={IDX_CANONICAL} not found in this evaluation split."

    # Loading note
    frac_zero = float(np.mean(frac == 0))
    frac_high = float(np.mean(frac >= 0.5))
    if frac_zero > 0.8:
        verdict += (
            f" Loading diagnostic: {100*frac_zero:.0f}% of failures have zero "
            f"|torDes| in the input window — the loading event precedes the "
            f"SEQ_LEN={config.SEQ_LEN} window, confirming that longer windows "
            f"or state-carrier features are required."
        )
    elif frac_high > 0.3:
        verdict += (
            f" Loading diagnostic: {100*frac_high:.0f}% of failures have active "
            f"loading (|torDes|>{LOADING_THRESH:.0f} Nm) in ≥50% of the window — "
            f"longer windows would partially capture the cause for these."
        )
    else:
        verdict += (
            f" Loading diagnostic: mixed (mean loading fraction={frac.mean():.2f}); "
            f"some failures are within the window, others are not."
        )

    print(f"\n[Verdict] {verdict}")

    print("\nWriting report …")
    write_report(
        failures, labels, C_raw, C_scaled, sil_scores, best_k,
        loading_df, cross_df, canonical_cluster, verdict,
        OUTPUT_DIR / "plc_failure_characterization.md",
    )
    append_diag_summary(verdict, n, best_k, sil_scores, labels)
    append_conc_summary(verdict, n, best_k, sil_scores, labels, loading_df)

    print("\nDone.")


if __name__ == "__main__":
    main()

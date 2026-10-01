"""Step 17: Exact SHAP values of the refitted final models (primary 7-feature track).

For each model, the final model is refitted on the full training set exactly as in step 15
(same configuration, tuned parameters, standardization and random seeds), and its test-set
predictions are checked against the saved step 15 predictions.

Exact interventional Shapley values are then computed on the log-odds scale. For every one of
the 2^7 = 128 predictor coalitions S,

    v(S) = mean over background rows b of logit f(x_S, b_notS),

with 100 training participants (seed 42) as the background and the 500 test participants as
the explained rows. The Shapley values follow from the exact Shapley weights. Only the test
participants' predictor values are read; their outcome labels are not.

Checks per model:
- the refit reproduces the step 15 test predictions;
- additivity: base value + sum of SHAP values = logit of the prediction;
- for LR, the SHAP values equal beta_j * (x_j - mean of background x_j);
- for XGBoost, LightGBM and CatBoost, they equal interventional TreeSHAP on the raw margin.

Checkpointing: coalition values are saved per model and chunk of explained participants in
step_17_shap_checkpoints/ (git-ignored, per participant). A rerun reuses completed chunks
whose fingerprint matches, so models can be added later (for example TabFM) without
recomputing the others.

Outputs:
- step_17_shap_values_7feat.npz (git-ignored): SHAP values, base values and predictions
- step_17_shap_importance_7feat.csv: mean |SHAP|, share and rank per model and predictor
- step_17_shap_checks_7feat.csv: reproduction, additivity and cross-checks per model
- step_17_shap_agreement_7feat.csv: Spearman correlation of predictor importance between models
- step_17_shap_{beeswarm_LR_CatBoost,beeswarm_all,importance_heatmap,dependence}_7feat.pdf

Usage:
    python step_17_shap.py                          # the 10 models except TabFM
    python step_17_shap.py --models TabFM           # add TabFM later (long)
    python step_17_shap.py --plots-only             # redraw figures from saved results
    python step_17_shap.py --smoke --out-dir <dir>  # tiny run to test the code
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from benchmark_evaluation import MODEL_COLORS, fit_benchmark_model
from benchmark_models import TUNED_MODELS, create_benchmark_model, file_sha256
from step_15_benchmark import load_benchmark_config, load_tuned_params
from step_15_tuning import BEST_PARAMS_FILE, build_model_config

TRACK = "7feat"
ALL_MODELS = ["LR", "SVM", "RF", "XGBoost", "LightGBM", "CatBoost", "RealMLP", "TabM",
              "TabICLv2", "TabPFN", "TabFM"]
DEFAULT_MODELS = [m for m in ALL_MODELS if m != "TabFM"]
DISPLAY_MODEL = {"TabICLv2": "TabICL v2.0", "TabPFN": "TabPFN v3.5", "TabFM": "TabFM v1.0"}  # name used in figures, as in the rest of the manuscript
N_BACKGROUND = 100
BACKGROUND_SEED = 42
CHUNK_SIZE = 50
LOGIT_CLIP = 1e-7
# Rows per predict call: large for fast models, small where memory or in-context cost grows.
# TabFM was faster with 5,000 rows per call (305 rows/s) than with 20,000 (213 rows/s).
PREDICT_BATCH_ROWS = {"TabM": 20_000, "RealMLP": 50_000, "TabICLv2": 20_000, "TabPFN": 20_000,
                      "TabFM": 5_000}
DEFAULT_BATCH_ROWS = 262_144
# Inference-only settings applied after fitting; they must not change predictions (checked
# by the reproduction test against step 15). None are used: TabFM's batch_size (ensemble
# members per forward pass) gave at most 9% speed-up at 2 or 4, changed predictions by
# about 1e-6, and exhausted the 16 GB GPU at 8 (tested 2026-09-30).
INFERENCE_SETTINGS: dict = {}
# Approximate prediction throughput (rows/s) measured on 2026-09-30, used only for the ETA
EXPECTED_ROWS_PER_S = {"LR": 2e6, "SVM": 7e4, "RF": 1.4e5, "XGBoost": 2e5, "LightGBM": 4e5,
                       "CatBoost": 1e6, "RealMLP": 5e5, "TabM": 2e5, "TabICLv2": 2e4,
                       "TabPFN": 6.9e3, "TabFM": 305}
DISPLAY = {
    "SEX": "Sex (male)",
    "EDUCYRS": "Education (years)",
    "fampd_bin": "Family history of PD",
    "sym_other": "Other initial symptom",
    "upsit_pctl": "UPSIT percentile",
    "gds": "GDS score",
    "MIA_CAUDATE_mean": "Caudate SBR (DAT)",
}
CONTINUOUS = ["EDUCYRS", "gds", "upsit_pctl", "MIA_CAUDATE_mean"]
CHECKPOINT_DIR = "step_17_shap_checkpoints"


# =====================================================================
# Exact Shapley values
# =====================================================================

def coalition_design(m: int) -> np.ndarray:
    """present[s, j] is True when predictor j is taken from x in coalition s (bit j of s)."""
    masks = np.arange(2 ** m)
    return ((masks[:, None] >> np.arange(m)) & 1).astype(bool)


def shapley_weights(present: np.ndarray) -> np.ndarray:
    """W such that phi = V @ W, where V[i, s] = v(S) for coalition s."""
    n_coal, m = present.shape
    sizes = present.sum(axis=1)
    W = np.zeros((n_coal, m))
    for s in range(n_coal):
        for j in range(m):
            if not present[s, j]:
                w = math.factorial(sizes[s]) * math.factorial(m - sizes[s] - 1) / math.factorial(m)
                W[s | (1 << j), j] += w
                W[s, j] -= w
    return W


def coalition_values(predict, X_explain: np.ndarray, background: np.ndarray,
                     present: np.ndarray, batch_rows: int) -> np.ndarray:
    """V[i, s] = mean over background rows of logit f(x_i on S, background elsewhere)."""
    n_e, m = X_explain.shape
    n_b, n_c = background.shape[0], present.shape[0]
    rows = np.where(present[None, :, None, :], X_explain[:, None, None, :],
                    background[None, None, :, :]).reshape(-1, m)
    probs = np.concatenate([np.asarray(predict(rows[i:i + batch_rows]), dtype=float)
                            for i in range(0, len(rows), batch_rows)])
    p = np.clip(probs, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p)).reshape(n_e, n_c, n_b).mean(axis=2)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


SHAP_CODE = "".join(inspect.getsource(f) for f in (coalition_design, shapley_weights, coalition_values))


# =====================================================================
# Data, models and checks
# =====================================================================

def load_data(config: dict):
    """Training rows with outcome; test predictor values only (outcome labels are not read)."""
    data_cfg = config["data_files"]
    track_cfg = data_cfg[f"{TRACK}_track"]
    train = pd.read_csv(track_cfg["train"])
    features = [c for c in train.columns if c not in data_cfg["forbidden_columns"]]
    test = pd.read_csv(track_cfg["test"], usecols=[data_cfg["subject_column"]] + features)
    return (train[features].to_numpy(float), train[data_cfg["target_column"]].to_numpy(int),
            test[features].to_numpy(float), test[data_cfg["subject_column"]].to_numpy(), features)


def use_final_model_shrinkage(model, Xtr: np.ndarray, y_train: np.ndarray) -> None:
    """Replace the LR's shrinkage factor by that of the final model (step 14).

    The step 15 LR of the 7-predictor track estimates the shrinkage factor by a bootstrap
    over the seven selected predictors, which ignores variable selection. The final model
    uses the factor from the bootstrap over all 28 candidates (step 14).
    """
    from step_14_shrinkage import shrink_model

    summary = pd.read_csv("step_14_shrinkage_summary.csv").set_index("Metric")["Value"]
    s_final = float(summary["Shrinkage_factor_S"])
    unshrunk = model.coef_ / model.shrinkage_
    model.shrinkage_ = s_final
    model.intercept_, model.coef_ = shrink_model(Xtr[:, model.mask_], y_train, unshrunk, s_final)


def model_fingerprint(key: str, cfg: dict, config: dict, settings: dict) -> str:
    track_cfg = config["data_files"][f"{TRACK}_track"]
    parts = {
        "model": key,
        "cfg": cfg,
        "settings": settings,
        "inference": INFERENCE_SETTINGS.get(key, {}),
        "train_sha256": file_sha256(track_cfg["train"]),
        "test_sha256": file_sha256(track_cfg["test"]),
        "code_sha256": [file_sha256(f) for f in ["benchmark_models.py", "benchmark_evaluation.py"]],
        "shap_code": SHAP_CODE,
    }
    if key == "LR":  # added only for LR so that the other models' checkpoints stay valid
        parts["final_model_shrinkage"] = file_sha256("step_14_shrinkage_summary.csv")
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def treeshap_check(key: str, model, background: np.ndarray, X_explain: np.ndarray,
                   phi: np.ndarray) -> tuple[float, str]:
    """Max |interventional TreeSHAP (raw margin) - exact SHAP| for the GBDTs."""
    import shap

    est = model.model
    try:
        if key == "XGBoost":
            tree_model = est.get_booster()
            best = getattr(est, "best_iteration", None)
            if best is not None:
                tree_model = tree_model[: best + 1]
        elif key == "LightGBM":
            import lightgbm as lgb

            best = est.best_iteration_ or est.booster_.current_iteration()
            tree_model = lgb.Booster(model_str=est.booster_.model_to_string(num_iteration=best))
        else:
            tree_model = est
        explainer = shap.TreeExplainer(tree_model, data=background,
                                       feature_perturbation="interventional", model_output="raw")
        sv = explainer.shap_values(X_explain, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]
        return float(np.abs(sv - phi).max()), ""
    except Exception as exc:  # reported, not fatal: the exact values do not depend on it
        return float("nan"), f"{type(exc).__name__}: {exc}"[:300]


def format_seconds(seconds: float) -> str:
    seconds = int(round(max(seconds, 0)))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def run_model(key, config, tuned_path, data, background_idx, settings, out_dir, progress):
    X_train, y_train, X_test_raw, patno, features = data
    n_explain = settings["n_explain"]
    tuned = None
    if key in TUNED_MODELS:
        tuned = load_tuned_params(config, TRACK, key, tuned_path, allow_incomplete=False)["params"]
    cfg = build_model_config(config, key, tuned_params=tuned)
    fingerprint = model_fingerprint(key, cfg, config, settings)
    model_dir = os.path.join(out_dir, CHECKPOINT_DIR, TRACK, key)
    os.makedirs(model_dir, exist_ok=True)
    result_path = os.path.join(model_dir, "result.npz")

    if os.path.isfile(result_path):
        saved = np.load(result_path, allow_pickle=True)
        if str(saved["fingerprint"]) == fingerprint:
            print(f"[{key}] loaded from checkpoint", flush=True)
            progress["done_rows"] += progress["rows_per_model"]
            return {k: saved[k] for k in saved.files}

    started = time.time()
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test_raw)[:n_explain]
    model = create_benchmark_model(key, cfg, random_state=42)
    fit_benchmark_model(model, Xtr, y_train, random_state=42)
    if key == "LR":
        use_final_model_shrinkage(model, Xtr, y_train)
    for attr, value in INFERENCE_SETTINGS.get(key, {}).items():
        setattr(model.model, attr, value)
    fit_s = time.time() - started

    p_test = np.asarray(model.predict_proba(Xte), dtype=float)
    # The final LR is the one fitted on the 28 candidate predictors (step 14), whose predictions
    # are stored in the step 15 28-predictor track; see step_21_lr_shrinkage_consistency.py
    ref_track = "full" if key == "LR" else TRACK
    saved_pred = pd.read_csv(f"step_15_benchmark_predictions_{ref_track}.csv",
                             usecols=["PATNO", f"prob_{key}"])
    if not (saved_pred["PATNO"].to_numpy()[:n_explain] == patno[:n_explain]).all():
        raise RuntimeError("Test participants are not in the step 15 order")
    repro = float(np.abs(p_test - saved_pred[f"prob_{key}"].to_numpy()[:n_explain]).max())
    if repro > 1e-4:
        raise RuntimeError(f"[{key}] refit does not reproduce step 15 predictions (max diff {repro:.2e})")
    print(f"[{key}] refit in {fit_s:.1f}s; max |p - p(step 15)| = {repro:.1e}", flush=True)

    background = Xtr[background_idx]
    present = coalition_design(Xtr.shape[1])
    batch_rows = PREDICT_BATCH_ROWS.get(key, DEFAULT_BATCH_ROWS)
    chunks = [(s, min(s + settings["chunk_size"], n_explain))
              for s in range(0, n_explain, settings["chunk_size"])]
    V_parts = []
    for k, (a, b) in enumerate(chunks):
        path = os.path.join(model_dir, f"chunk_{k:03d}.npz")
        if os.path.isfile(path):
            ck = np.load(path)
            if str(ck["fingerprint"]) == fingerprint:
                V_parts.append(ck["V"])
                progress["done_rows"] += (b - a) * len(present) * len(background)
                continue
        t0 = time.time()
        V = coalition_values(model.predict_proba, Xte[a:b], background, present, batch_rows)
        np.savez(path, V=V, fingerprint=fingerprint)
        V_parts.append(V)
        rows = (b - a) * len(present) * len(background)
        progress["done_rows"] += rows
        progress["done_expected_s"] += rows / EXPECTED_ROWS_PER_S[key]
        progress["done_actual_s"] += time.time() - t0
        scale = progress["done_actual_s"] / max(progress["done_expected_s"], 1e-9)
        remaining = sum(r / EXPECTED_ROWS_PER_S[m] for m, r in progress["remaining_rows"].items())
        remaining -= progress["done_expected_s"]
        print(f"[{key}] chunk {k + 1}/{len(chunks)} | model {progress['model_index']}/"
              f"{progress['n_models']} | total {100 * progress['done_rows'] / progress['total_rows']:.0f}% "
              f"| elapsed {format_seconds(time.time() - progress['started'])}, "
              f"ETA {format_seconds(remaining * scale)}", flush=True)

    V = np.concatenate(V_parts)
    phi = V @ shapley_weights(present)
    base = V[:, 0]
    logit_pred = logit(p_test)
    checks = {
        "model": key,
        "fit_seconds": fit_s,
        "repro_max_abs_diff_prob": repro,
        "additivity_max_err": float(np.abs(base + phi.sum(axis=1) - logit_pred).max()),
        "base_value": float(base.mean()),
        "base_value_spread": float(base.max() - base.min()),
        "lr_linear_max_diff": float("nan"),
        "treeshap_max_diff": float("nan"),
        "treeshap_note": "",
    }
    lr_std_coef = np.full(Xtr.shape[1], np.nan)
    if key == "LR":
        expected = np.zeros_like(phi)
        mask = model.mask_
        expected[:, mask] = (Xte[:, mask] - background[:, mask].mean(axis=0)) * model.coef_
        checks["lr_linear_max_diff"] = float(np.abs(phi - expected).max())
        lr_std_coef = np.zeros(Xtr.shape[1])
        lr_std_coef[mask] = model.coef_
    elif key in ("XGBoost", "LightGBM", "CatBoost"):
        checks["treeshap_max_diff"], checks["treeshap_note"] = treeshap_check(key, model, background, Xte, phi)
    elif key == "RF":
        checks["treeshap_note"] = "not applicable: RF outputs averaged probabilities, not a log-odds margin"
    result = {"phi": phi, "base": base, "logit_pred": logit_pred, "V": V,
              "checks": json.dumps(checks), "lr_std_coef": lr_std_coef, "fingerprint": fingerprint}
    np.savez(result_path, **result)
    print(f"[{key}] done in {format_seconds(time.time() - started)}; additivity max err "
          f"{checks['additivity_max_err']:.1e}; LR check {checks['lr_linear_max_diff']:.1e}; "
          f"TreeSHAP check {checks['treeshap_max_diff']:.1e} {checks['treeshap_note']}", flush=True)
    return result


# =====================================================================
# Summaries and figures
# =====================================================================

def swarm_offsets(vals: np.ndarray, nbins: int = 100, width: float = 0.38) -> np.ndarray:
    """Vertical offsets that pile up points with similar values (as in shap's beeswarm)."""
    n = len(vals)
    span = vals.max() - vals.min()
    quant = np.round(nbins * (vals - vals.min()) / (span if span > 0 else 1.0))
    inds = np.argsort(quant + np.random.default_rng(0).standard_normal(n) * 1e-6)
    ys = np.zeros(n)
    layer, last = 0, None
    for ind in inds:
        if quant[ind] != last:
            layer = 0
        ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last = quant[ind]
    top = np.abs(ys).max()
    return ys * (width / top) if top > 0 else ys


def beeswarm(ax, phi, X, features, order, title, show_labels=True, own_colorbar=False, fontsize=None, scale=1.0):
    """shap's own beeswarm plot of the exact SHAP values, drawn on the given axes.

    The SHAP values come from our exact enumeration (not from a shap explainer), so they are
    wrapped in a shap.Explanation; base values are not used by the plot.
    """
    import shap

    expl = shap.Explanation(values=phi, base_values=np.zeros(len(phi)), data=X,
                            feature_names=[DISPLAY.get(f, f) for f in features])
    kwargs = {} if order is None else {"order": np.asarray(order)}  # None: shap's own order
    shap.plots.beeswarm(expl, max_display=len(features), ax=ax, show=False,
                        color_bar=False, s=8, plot_size=None, **kwargs)
    if own_colorbar:
        sm = plt.cm.ScalarMappable(cmap=shap.plots.colors.red_blue, norm=plt.Normalize(0, 1))
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, aspect=40)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Predictor value", fontsize=fontsize or 10)
        cb.ax.tick_params(labelsize=fontsize or 10)
        cb.outline.set_visible(False)
    ax.set_title(DISPLAY_MODEL.get(title, title))
    ax.set_xlabel("SHAP value (log-odds)")
    # shap sets its own, larger sizes: use the sizes of the other figures (tick 7.5, label 8, title 9)
    # (scale enlarges all text, used for Figure 6 whose frames are larger than those of Figure S3)
    ax.tick_params(axis="both", labelsize=7.5 * scale)
    ax.xaxis.label.set_size(8 * scale)
    ax.title.set_size(9 * scale)
    if not show_labels:
        ax.tick_params(labelleft=False)


def colorbar(fig, axes):
    import shap

    sm = plt.cm.ScalarMappable(cmap=shap.plots.colors.red_blue, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Predictor value")


def summarize_and_plot(results: dict, data, out_dir: str, n_explain: int) -> None:
    _, _, X_test_raw, patno, features = data
    X = X_test_raw[:n_explain]
    models = [m for m in ALL_MODELS if m in results]

    rows, checks = [], []
    for m in models:
        phi = results[m]["phi"]
        mean_abs = np.abs(phi).mean(axis=0)
        ranks = (-mean_abs).argsort().argsort() + 1
        lr_coef = results[m]["lr_std_coef"]
        for j, f in enumerate(features):
            rows.append({"Model": m, "Predictor": f, "Display": DISPLAY.get(f, f),
                         "Mean_abs_SHAP": mean_abs[j], "Share": mean_abs[j] / mean_abs.sum(),
                         "Rank": int(ranks[j]),
                         "LR_std_coef": lr_coef[j] if m == "LR" else np.nan})
        checks.append(json.loads(str(results[m]["checks"])))
    imp = pd.DataFrame(rows)
    imp.to_csv(os.path.join(out_dir, f"step_17_shap_importance_{TRACK}.csv"), index=False)
    pd.DataFrame(checks).to_csv(os.path.join(out_dir, f"step_17_shap_checks_{TRACK}.csv"), index=False)

    share = imp.pivot(index="Model", columns="Predictor", values="Mean_abs_SHAP").loc[models, features]
    agree = []
    for a in models:
        for b in models:
            agree.append({"Model_A": a, "Model_B": b,
                          "Spearman_importance": spearmanr(share.loc[a], share.loc[b]).statistic})
    pd.DataFrame(agree).to_csv(os.path.join(out_dir, f"step_17_shap_agreement_{TRACK}.csv"), index=False)

    np.savez(os.path.join(out_dir, f"step_17_shap_values_{TRACK}.npz"),
             models=np.array(models), features=np.array(features), patno=patno[:n_explain], X=X,
             **{f"phi_{m}": results[m]["phi"] for m in models},
             **{f"base_{m}": results[m]["base"] for m in models},
             **{f"logit_pred_{m}": results[m]["logit_pred"] for m in models})

    ref = "LR" if "LR" in results else models[0]
    order = list(np.argsort(-np.abs(results[ref]["phi"]).mean(axis=0)))

    pair = [m for m in ("LR", "CatBoost") if m in results]
    # Figure 6: both panels use the predictor order of LR (labels on the left panel only), the common
    # frame size of the 2-D figures, and share one colour bar
    import figstyle as fs
    import shap

    fs.apply()

    def colour_bar(cax, orientation, size=fs.ANNOTATION_SIZE):
        sm = plt.cm.ScalarMappable(cmap=shap.plots.colors.red_blue, norm=plt.Normalize(0, 1))
        cb = cax.figure.colorbar(sm, cax=cax, orientation=orientation)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Predictor value", fontsize=size)
        cb.ax.tick_params(labelsize=size)
        cb.outline.set_visible(False)

    zoom = 1.4  # Figure 6 has frames 1.4 times larger than Figure S3, so its text is 1.4 times larger
    left_mm, cb_gap, cb_w = 46.0, 3.0, 2.2
    fig, axes = fs.axes_grid(1, len(pair), left=left_mm, wgap=4.0, right=cb_gap + cb_w + 21.0, top=10.0,
                             bottom=17.0)
    width_mm = fig.get_figwidth() / fs.MM
    height_mm = fig.get_figheight() / fs.MM
    for i, m in enumerate(pair):
        beeswarm(axes[0, i], results[m]["phi"], X, features, order, m, show_labels=(i == 0), scale=zoom)
    x_right = left_mm + len(pair) * fs.AX_W + (len(pair) - 1) * 4.0
    cb_h = 0.85 * fs.AX_H  # tall and slim, centred on the frames
    cax = fig.add_axes([(x_right + cb_gap) / width_mm, (17.0 + 0.5 * (fs.AX_H - cb_h)) / height_mm,
                        cb_w / width_mm, cb_h / height_mm])
    colour_bar(cax, "vertical", size=fs.ANNOTATION_SIZE * zoom)
    fig.savefig(os.path.join(out_dir, f"step_17_shap_beeswarm_LR_CatBoost_{TRACK}.pdf"),
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    # Supplementary figure S3: three columns of frames with the same aspect ratio (smaller size), all in
    # the LR predictor order, so only the left column is labelled; one colour bar in the free slot
    ncol = 3
    nrow = math.ceil(len(models) / ncol)
    ax_w, ax_h = 50.0, 50.0 / (fs.AX_W / fs.AX_H)
    left_mm, wgap, hgap = 34.0, 6.0, 14.0
    fig, axes = fs.axes_grid(nrow, ncol, ax_w=ax_w, ax_h=ax_h, left=left_mm, wgap=wgap, hgap=hgap)
    width_mm = fig.get_figwidth() / fs.MM
    height_mm = fig.get_figheight() / fs.MM
    for i, ax in enumerate(axes.ravel()):
        if i < len(models):
            beeswarm(ax, results[models[i]]["phi"], X, features, order, models[i],
                     show_labels=(i % ncol == 0))
            if i + ncol < len(models):  # the horizontal axis title is only needed at the bottom of each column
                ax.set_xlabel("")
        else:
            ax.axis("off")
    # Vertical colour bar to the right of the last panel, with the relative size of the one in Figure 6
    row, col = divmod(len(models) - 1, ncol)
    x0 = left_mm + col * (ax_w + wgap) + ax_w + 4.0
    cb_h, cb_w2 = 0.85 * ax_h, 2.2 * ax_w / fs.AX_W
    y0 = 11.0 + (nrow - 1 - row) * (ax_h + hgap) + 0.5 * (ax_h - cb_h)
    cax = fig.add_axes([x0 / width_mm, y0 / height_mm, cb_w2 / width_mm, cb_h / height_mm])
    colour_bar(cax, "vertical")
    fig.savefig(os.path.join(out_dir, f"step_17_shap_beeswarm_all_{TRACK}.pdf"), bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)

    # Predictors in rows (ordered by rank in the reference model), models in columns
    # Rows ordered by the mean rank across models (ties keep the LR order)
    mean_rank = imp.groupby("Predictor")["Rank"].mean()
    pred_order = sorted((features[j] for j in order), key=lambda f: mean_rank[f])
    share_mat = imp.pivot(index="Predictor", columns="Model", values="Share").loc[pred_order, models]
    rank_mat = imp.pivot(index="Predictor", columns="Model", values="Rank").loc[pred_order, models]
    fig, ax = plt.subplots(figsize=(0.62 * len(models) + 2.0, 0.62 * len(pred_order) + 1.3))
    ax.imshow(share_mat.to_numpy(), cmap="GnBu", vmin=0, aspect="equal")
    for r in range(share_mat.shape[0]):
        for c in range(share_mat.shape[1]):
            val = share_mat.to_numpy()[r, c]
            ax.text(c, r, str(int(rank_mat.to_numpy()[r, c])), ha="center", va="center", fontsize=12.6,
                    color="white" if val > 0.6 * share_mat.to_numpy().max() else "black")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([DISPLAY_MODEL.get(m, m) for m in models], rotation=35, ha="right", fontsize=14)
    ax.set_yticks(range(len(pred_order)))
    ax.set_yticklabels([DISPLAY.get(f, f) for f in pred_order], fontsize=14)
    fig.savefig(os.path.join(out_dir, f"step_17_shap_importance_heatmap_{TRACK}.pdf"), bbox_inches="tight")
    plt.close(fig)

    cont = [f for f in CONTINUOUS if f in features]
    ncol_dep = 2
    nrow_dep = math.ceil(len(cont) / ncol_dep)
    import figstyle as fs

    fs.apply()  # common font sizes and frame size of the 2-D figures
    fig, axes = fs.axes_grid(nrow_dep, ncol_dep)
    for ax in axes.ravel()[len(cont):]:
        ax.axis("off")
    for i, f in enumerate(cont):
        j = features.index(f)
        ax = axes[i // ncol_dep, i % ncol_dep]
        for m in pair:
            ax.scatter(X[:, j], results[m]["phi"][:, j], s=8, alpha=0.55, linewidths=0,
                       color=MODEL_COLORS.get(m, "#333333"), label=m, rasterized=True)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel(DISPLAY.get(f, f))
        if f == "EDUCYRS":
            ax.set_xticks([0, 5, 10, 15, 20])
        if i % ncol_dep == 0:
            ax.set_ylabel("SHAP value (log-odds)")
        ax.grid(alpha=0.25)
    # One framed legend, inside the first panel (upper right); opaque markers larger than the data points
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=MODEL_COLORS.get(m, "#333333"),
                      label=m) for m in pair]
    # Same frame style as the legends of the other figures; legend text smaller than the axis labels
    axes[0, 0].legend(handles=handles, loc="upper right", frameon=True)
    fig.savefig(os.path.join(out_dir, f"step_17_shap_dependence_{TRACK}.pdf"), bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    print(f"Saved SHAP summaries and figures for {len(models)} models to {os.path.abspath(out_dir)}", flush=True)


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Exact SHAP values of the refitted final models.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--plots-only", action="store_true",
                        help="Redraw summaries and figures from the saved per-model results")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run (6 explained participants, 5 background rows) to test the code")
    args = parser.parse_args()

    config = load_benchmark_config("benchmark_config.json")
    data = load_data(config)
    settings = {"n_background": 5 if args.smoke else N_BACKGROUND, "background_seed": BACKGROUND_SEED,
                "chunk_size": 3 if args.smoke else CHUNK_SIZE, "logit_clip": LOGIT_CLIP,
                "n_explain": 6 if args.smoke else len(data[2])}
    background_idx = np.random.default_rng(BACKGROUND_SEED).choice(
        len(data[0]), settings["n_background"], replace=False)
    os.makedirs(args.out_dir, exist_ok=True)

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    # Figures always use every model whose result exists, so TabFM can be added later
    models = [m for m in ALL_MODELS if m in requested or
              os.path.isfile(os.path.join(args.out_dir, CHECKPOINT_DIR, TRACK, m, "result.npz"))]
    if args.plots_only:
        requested = []

    rows_per_model = settings["n_explain"] * 2 ** data[2].shape[1] * settings["n_background"]
    progress = {"started": time.time(), "done_rows": 0, "total_rows": rows_per_model * max(len(requested), 1),
                "rows_per_model": rows_per_model, "n_models": len(requested), "model_index": 0,
                "done_expected_s": 0.0, "done_actual_s": 0.0,
                "remaining_rows": {m: rows_per_model for m in requested}}
    results = {}
    for m in models:
        if m in requested:
            progress["model_index"] += 1
            results[m] = run_model(m, config, BEST_PARAMS_FILE, data, background_idx, settings,
                                   args.out_dir, progress)
        else:
            saved = np.load(os.path.join(args.out_dir, CHECKPOINT_DIR, TRACK, m, "result.npz"), allow_pickle=True)
            results[m] = {k: saved[k] for k in saved.files}
    summarize_and_plot(results, data, args.out_dir, settings["n_explain"])
    print(f"Step 17 finished in {format_seconds(time.time() - progress['started'])}.", flush=True)


if __name__ == "__main__":
    main()

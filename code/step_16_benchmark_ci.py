"""Step 16: Per-model test-set bootstrap CIs and recalibration sensitivity analysis.

Reads the saved step 15 test-set predictions and Youden thresholds; no model is refit
and no threshold is re-estimated. For each track (7-feature primary, 28-feature
sensitivity):

1. Per-model participant-level bootstrap 95% CIs (2,000 resamples shared by all models)
   for discrimination, calibration and threshold metrics at the model's pre-specified
   training out-of-fold Youden threshold.
2. A forest plot of test-set AUPRC (primary metric, with the prevalence baseline) and
   AUROC per model.
3. Recalibration sensitivity analysis (pre-specified from v1.4 on; post hoc in v1.2):
   every model, including LR, is logistically recalibrated with
   logit(p') = a + b * logit(p), where a and b are fitted on that model's training
   out-of-fold predictions only. Only LR has a calibration step (bootstrap shrinkage) in
   step 15, so this puts all models on an equal footing for calibration metrics.
   Recalibration is monotone, so AUROC/AUPRC are unchanged; the Youden threshold is
   re-derived on the recalibrated out-of-fold predictions.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import statsmodels.api as sm

from benchmark_evaluation import (
    MODEL_COLORS,
    bootstrap_indices,
    bootstrap_model_ci,
    find_optimal_youden_threshold,
    paired_bootstrap_test_comparison,
)

CONFIG_FILE = "benchmark_config.json"
TRACKS = ["7feat", "full"]
# Input predictions and output files are named with these prefixes; step 21 reruns this
# analysis on the corrected LR predictions with its own prefix.
IN_PREFIX = "step_15_benchmark"
OUT_PREFIX = "step_16_benchmark"
TRACK_TITLES = {
    "7feat": "Primary analysis (7 LASSO-selected features)",
    "full": "Sensitivity analysis (28 candidate features)",
}


def load_track(track: str, config: dict):
    preds = pd.read_csv(f"{IN_PREFIX}_predictions_{track}.csv")
    metrics = pd.read_csv(f"{IN_PREFIX}_test_metrics_{track}.csv")
    test = pd.read_csv(config["data_files"][f"{track}_track"]["test"])
    if not (preds["PATNO"].to_numpy() == test["PATNO"].to_numpy()).all():
        raise RuntimeError(f"Prediction rows do not match the {track} test set")
    if not (preds["MCI"].to_numpy() == test["MCI"].to_numpy()).all():
        raise RuntimeError(
            f"Outcome mismatch between predictions and the {track} test set"
        )
    thresholds = dict(zip(metrics["Model"], metrics["Threshold"]))
    return preds, thresholds


def overall_ci(track, preds, thresholds, n_bootstrap, seed):
    y = preds["MCI"].to_numpy().astype(int)
    indices = bootstrap_indices(len(y), n_bootstrap, seed)
    models = list(thresholds)
    results = Parallel(n_jobs=-1)(
        delayed(bootstrap_model_ci)(
            y, preds[f"prob_{m}"].to_numpy(), thresholds[m], indices
        )
        for m in models
    )
    rows = []
    for m, res in zip(models, results):
        row = {"Model": m, "Threshold": thresholds[m]}
        for metric, s in res.items():
            row[metric] = s["point"]
            row[f"{metric}_CI_lower"] = s["ci_lower"]
            row[f"{metric}_CI_upper"] = s["ci_upper"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT_PREFIX}_model_ci_{track}.csv", index=False)
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recalibrated_sensitivity(track, preds, thresholds, n_bootstrap, seed):
    """Logistic recalibration of every model on its own training out-of-fold predictions."""
    oof = pd.read_csv(f"{IN_PREFIX}_oof_predictions_{track}.csv")
    y_train = oof["MCI"].to_numpy().astype(int)
    y_test = preds["MCI"].to_numpy().astype(int)
    indices = bootstrap_indices(len(y_test), n_bootstrap, seed)

    recal_test, rows = {}, []
    for m in thresholds:
        res = sm.GLM(
            y_train,
            sm.add_constant(_logit(oof[f"prob_{m}"].to_numpy())),
            family=sm.families.Binomial(),
        ).fit()
        a, b = float(res.params[0]), float(res.params[1])

        def apply(p):
            return 1.0 / (1.0 + np.exp(-(a + b * _logit(p))))

        threshold = find_optimal_youden_threshold(
            y_train, apply(oof[f"prob_{m}"].to_numpy())
        )
        recal_test[m] = apply(preds[f"prob_{m}"].to_numpy())
        ci = bootstrap_model_ci(y_test, recal_test[m], threshold, indices)

        row = {"Model": m, "Recal_a": a, "Recal_b": b, "Threshold": threshold}
        for metric in [
            "AUROC",
            "AUPRC",
            "Brier",
            "LogLoss",
            "Calib_Slope",
            "Calib_Intercept",
            "Sensitivity",
            "Specificity",
        ]:
            row[metric] = ci[metric]["point"]
            row[f"{metric}_CI_lower"] = ci[metric]["ci_lower"]
            row[f"{metric}_CI_upper"] = ci[metric]["ci_upper"]
        rows.append(row)

    out = pd.DataFrame(rows)
    # Paired Brier difference against recalibrated LR on the same resamples as step 15
    deltas = []
    for m in out["Model"]:
        if m == "LR":
            deltas.append((np.nan, np.nan, np.nan))
            continue
        d = paired_bootstrap_test_comparison(
            y_test,
            recal_test["LR"],
            recal_test[m],
            n_bootstrap=n_bootstrap,
            random_state=seed,
        )["Delta_Brier"]
        deltas.append((d["diff_point"], d["ci_lower"], d["ci_upper"]))
    (
        out["Delta_Brier_vs_LR"],
        out["Delta_Brier_CI_lower"],
        out["Delta_Brier_CI_upper"],
    ) = zip(*deltas)
    out.to_csv(f"{OUT_PREFIX}_recalibrated_{track}.csv", index=False)
    return out


def forest_plot(track, overall, prevalence):
    """AUPRC (primary) and AUROC with 95% CIs; dashed line = LR, dotted line = chance."""
    models = overall["Model"].tolist()
    df = overall.set_index("Model").loc[models]
    ypos = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(models) + 1.6), sharey=True)
    for ax, metric, chance in [(axes[0], "AUPRC", prevalence), (axes[1], "AUROC", 0.5)]:
        for i, m in enumerate(models):
            ax.errorbar(
                df.loc[m, metric],
                ypos[i],
                xerr=[
                    [df.loc[m, metric] - df.loc[m, f"{metric}_CI_lower"]],
                    [df.loc[m, f"{metric}_CI_upper"] - df.loc[m, metric]],
                ],
                fmt="o",
                color=MODEL_COLORS.get(m, "#333333"),
                capsize=2,
                markersize=5,
                linewidth=1,
            )
        ax.axvline(chance, color="grey", linestyle=":", linewidth=1)
        ax.axvline(
            df.loc["LR", metric], color=MODEL_COLORS["LR"], linestyle="--", linewidth=1
        )
        ax.set_xlabel(f"Test-set {metric} (95% bootstrap CI)")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    axes[0].set_title(f"AUPRC (primary); dotted line = prevalence {prevalence:.3f}")
    axes[1].set_title("AUROC; dotted line = 0.5")
    fig.suptitle(TRACK_TITLES[track])
    fig.tight_layout()
    fig.savefig(f"{OUT_PREFIX}_forest_{track}.pdf", bbox_inches="tight")
    plt.close(fig)


def main(tracks=None):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
    n_bootstrap = config["protocol"]["bootstrap_samples"]
    seed = config["protocol"]["bootstrap_seed"]

    for track in tracks or TRACKS:
        print(f"\n=== {track} track ===")
        preds, thresholds = load_track(track, config)
        overall = overall_ci(track, preds, thresholds, n_bootstrap, seed)
        cols = [
            "Model",
            "AUPRC",
            "AUPRC_CI_lower",
            "AUPRC_CI_upper",
            "AUROC",
            "AUROC_CI_lower",
            "AUROC_CI_upper",
            "Calib_Slope",
            "Calib_Slope_CI_lower",
            "Calib_Slope_CI_upper",
        ]
        print(overall[cols].round(3).to_string(index=False))

        forest_plot(track, overall, prevalence=float(preds["MCI"].mean()))

        recal = recalibrated_sensitivity(track, preds, thresholds, n_bootstrap, seed)
        print(
            recal[
                [
                    "Model",
                    "Recal_a",
                    "Recal_b",
                    "AUPRC",
                    "Brier",
                    "Calib_Slope",
                    "Calib_Slope_CI_lower",
                    "Calib_Slope_CI_upper",
                    "Calib_Intercept",
                    "Delta_Brier_vs_LR",
                    "Delta_Brier_CI_lower",
                    "Delta_Brier_CI_upper",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
        print(f"Saved step_16 outputs for the {track} track")


if __name__ == "__main__":
    main()

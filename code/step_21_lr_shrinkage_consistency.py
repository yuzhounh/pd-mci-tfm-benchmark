"""Step 21: Use one final LR model throughout the 7-predictor comparison.

The final model is the LASSO + bootstrap-shrunk LR of step 14, whose shrinkage bootstrap
repeats LASSO selection from the 28 candidate predictors (shrinkage factor 0.814). Its
predictions are identical to those of the LR in the step 15 28-predictor track (checked
below), which fits the same pipeline on the candidate predictors. The step 15 LR of the
7-predictor track instead ran the bootstrap on the seven selected predictors only, so its
shrinkage ignored the variable selection (factor 0.900) and its probabilities differ from
the final model by a monotone transformation: AUROC and AUPRC are the same, calibration,
Brier score and the threshold are not.

This script writes corrected copies of the step 15 7-predictor results in which only the
LR columns are replaced: test-set predictions of the final model, and out-of-fold
predictions (5 folds, as for all models) from the same seven predictors with the same
shrinkage factor in every fold, i.e. without reselecting variables inside the folds. The LR
Youden threshold, cross-validation metrics and recalibration come from these out-of-fold
predictions. The other ten models are untouched. It then repeats the paired bootstrap of
step 15 and the step 16 analyses (per-model CIs, recalibration, forest plot) on them.
The step 15 and step 16 files are left as released in v1.4.

Outputs: step_21_benchmark_{predictions,oof_predictions,test_metrics,bootstrap_diffs,
model_ci,recalibrated}_7feat.csv and step_21_benchmark_forest_7feat.pdf
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import step_16_benchmark_ci as s16
from benchmark_evaluation import (compute_binary_metrics, find_optimal_youden_threshold,
                                  paired_bootstrap_test_comparison, run_cross_validation)
from step_14_shrinkage import fit_unpenalized, predict_prob, shrink_model

PREFIX = "step_21_benchmark"


class FixedVariableShrunkLR:
    """Unpenalized LR on the given (already selected) predictors, shrunk by a fixed factor.

    The slopes are multiplied by the factor from step 14 and the intercept is re-estimated with
    the shrunk linear predictor as offset (step 14's shrink_model). No variable selection.
    """

    uses_early_stopping = False

    def __init__(self, shrinkage: float):
        self.shrinkage = shrinkage

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        model = fit_unpenalized(X_train, y_train)
        self.intercept_, self.coef_ = shrink_model(X_train, y_train, model.coef_[0], self.shrinkage)
        return self

    def predict_proba(self, X):
        return predict_prob(X, self.intercept_, self.coef_)


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("benchmark_config.json", encoding="utf-8") as f:
        config = json.load(f)
    n_boot = config["protocol"]["bootstrap_samples"]
    seed = config["protocol"]["bootstrap_seed"]

    test7 = pd.read_csv("step_15_benchmark_predictions_7feat.csv")
    testf = pd.read_csv("step_15_benchmark_predictions_full.csv")
    oof7 = pd.read_csv("step_15_benchmark_oof_predictions_7feat.csv")
    ooff = pd.read_csv("step_15_benchmark_oof_predictions_full.csv")
    met7 = pd.read_csv("step_15_benchmark_test_metrics_7feat.csv")
    metf = pd.read_csv("step_15_benchmark_test_metrics_full.csv")
    step14 = pd.read_csv("step_14_shrunk_predictions.csv")

    for a, b in ((test7, testf), (oof7, ooff)):
        if not ((a["PATNO"] == b["PATNO"]).all() and (a["MCI"] == b["MCI"]).all()):
            raise RuntimeError("Participants or outcomes differ between the two tracks")
    diff14 = float(np.abs(step14["y_prob"].to_numpy() - testf["prob_LR"].to_numpy()).max())
    if diff14 > 1e-8:
        raise RuntimeError(f"Full-track LR differs from the step 14 final model ({diff14:.1e})")
    print(f"Full-track LR test predictions equal the step 14 final model (max diff {diff14:.1e})")

    # Test-set predictions: the final model. Out-of-fold predictions: the same seven predictors and
    # the same shrinkage factor (0.814) in every fold, as for all other models (variables are fixed
    # after LASSO; nothing is reselected inside the folds).
    track = config["data_files"]["7feat_track"]
    train = pd.read_csv(track["train"])
    feats = [c for c in train.columns if c not in config["data_files"]["forbidden_columns"]]
    X_tr, y_tr = train[feats].to_numpy(float), train[config["data_files"]["target_column"]].to_numpy(int)
    subj = train[config["data_files"]["subject_column"]].to_numpy()
    if not (subj == oof7["PATNO"].to_numpy()).all():
        raise RuntimeError("Training rows are not in the step 15 order")
    s_final = float(pd.read_csv("step_14_shrinkage_summary.csv").set_index("Metric").loc["Shrinkage_factor_S", "Value"])
    proto = config["protocol"]
    oof_lr, cv_mean, cv_std = run_cross_validation(
        lambda: FixedVariableShrunkLR(s_final), X_tr, y_tr, subj,
        n_folds=proto["cv_folds"], random_state=proto["cv_seed"])
    thr = find_optimal_youden_threshold(y_tr, oof_lr)
    print(f"LR (fixed seven predictors, shrinkage {s_final:.4f}): out-of-fold Youden threshold {thr:.4f}")

    test7["prob_LR"] = testf["prob_LR"]
    oof7["prob_LR"] = oof_lr
    lr_test = compute_binary_metrics(test7["MCI"].to_numpy().astype(int), test7["prob_LR"].to_numpy(),
                                     threshold=thr)
    for col in met7.columns:
        if col in lr_test:
            met7.loc[met7["Model"] == "LR", col] = lr_test[col]
    met7.loc[met7["Model"] == "LR", "Threshold"] = thr
    met7.loc[met7["Model"] == "LR", "Runtime_s"] = np.nan
    cv7 = pd.read_csv("step_15_benchmark_cv_metrics_7feat.csv")
    cv_row = {"Model": "LR", "Optimal_Threshold": thr}
    for k in ["AUROC", "AUPRC", "Brier", "LogLoss", "Calib_Slope", "Calib_Intercept"]:
        cv_row[f"{k}_mean"], cv_row[f"{k}_std"] = cv_mean[k], cv_std[k]
    cv7.loc[cv7["Model"] == "LR", list(cv_row)] = list(cv_row.values())
    test7.to_csv(f"{PREFIX}_predictions_7feat.csv", index=False)
    oof7.to_csv(f"{PREFIX}_oof_predictions_7feat.csv", index=False)
    met7.to_csv(f"{PREFIX}_test_metrics_7feat.csv", index=False)
    cv7.to_csv(f"{PREFIX}_cv_metrics_7feat.csv", index=False)

    y = test7["MCI"].to_numpy().astype(int)
    rows = []
    for m in met7["Model"]:
        if m == "LR":
            continue
        d = paired_bootstrap_test_comparison(
            y_test=y, baseline_probs=test7["prob_LR"].to_numpy(),
            model_probs=test7[f"prob_{m}"].to_numpy(), n_bootstrap=n_boot, random_state=seed)
        row = {"Model": m, "Comparison": f"{m} vs LR"}
        for k in ("AUPRC", "AUROC", "Brier"):
            row[f"Delta_{k}"] = d[f"Delta_{k}"]["diff_point"]
            row[f"Delta_{k}_95CI"] = f"{d[f'Delta_{k}']['ci_lower']:.4f}–{d[f'Delta_{k}']['ci_upper']:.4f}"
            row[f"Delta_{k}_pval"] = d[f"Delta_{k}"]["p_value"]
        rows.append(row)
    diffs = pd.DataFrame(rows)
    diffs.to_csv(f"{PREFIX}_bootstrap_diffs_7feat.csv", index=False)

    # Rank-based differences must not change: compare with the released step 15 file
    old = pd.read_csv("step_15_benchmark_bootstrap_diffs_7feat.csv").set_index("Model")
    new = diffs.set_index("Model")
    for k in ("Delta_AUPRC", "Delta_AUROC"):
        err = float((old[k] - new[k]).abs().max())
        print(f"Max change in {k} versus step 15: {err:.1e}")
        if err > 1e-9:
            raise RuntimeError(f"{k} changed although LR ranks are identical")
    print(diffs.assign(ci=diffs["Delta_Brier_95CI"])[["Model", "Delta_Brier", "ci", "Delta_Brier_pval"]]
          .round(4).to_string(index=False))

    s16.IN_PREFIX = PREFIX
    s16.OUT_PREFIX = PREFIX
    s16.main(tracks=["7feat"])


if __name__ == "__main__":
    main()

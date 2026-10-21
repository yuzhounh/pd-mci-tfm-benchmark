"""Sensitivity of the final LR to dropping the unstable predictors.

Exploratory, post hoc: the test set has already been used. Reduced models drop "Other symptom"
(sym_other), or "Other symptom" and FH-PD, from the seven final predictors. Each is refitted with the
step 13 estimator on the training set and shrunk with the locked bootstrap shrinkage factor of the
final model (intercept re-estimated with the shrunk linear predictor as offset), exactly as in
step_14_shrinkage.shrink_model. The seven-predictor refit must reproduce final_model.json.
Test-set AUPRC/AUROC/Brier and calibration slope with paired participant-level bootstrap (2,000
resamples, seed 42; benchmark_evaluation). Output: step_23_lr_reduced_sensitivity.csv.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from benchmark_evaluation import bootstrap_indices, paired_bootstrap_test_comparison
from step_14_shrinkage import (
    calibration_slope,
    fit_unpenalized,
    predict_prob,
    shrink_model,
)

final = json.loads(open("final_model.json", encoding="utf-8").read())
S = float(final["shrinkage_factor"])
train, test = pd.read_csv("PPMI_9_train.csv"), pd.read_csv("PPMI_9_test.csv")
y_tr, y_te = train.MCI.to_numpy(), test.MCI.to_numpy()
FEATURES = final["feature_order"]
SETS = {
    "7 predictors (final)": [],
    "without Other symptom": ["sym_other"],
    "without Other symptom and FH-PD": ["sym_other", "fampd_bin"],
}

probs = {}
for name, drop in SETS.items():
    cols = [f for f in FEATURES if f not in drop]
    model = fit_unpenalized(train[cols].to_numpy(float), y_tr)
    intercept, coef = shrink_model(train[cols].to_numpy(float), y_tr, model.coef_[0], S)
    probs[name] = predict_prob(test[cols].to_numpy(float), intercept, coef)

ref = final["intercept"] + test[FEATURES].to_numpy(float) @ np.array(
    [final["coefficients"][f] for f in FEATURES]
)
ref = 1 / (1 + np.exp(-ref))
diff = float(np.max(np.abs(ref - probs["7 predictors (final)"])))
if diff > 1e-8:
    raise SystemExit(
        f"Seven-predictor refit does not reproduce the final model: {diff}"
    )

draws = bootstrap_indices(len(y_te), 2000, 42)
base = probs["7 predictors (final)"]
rows = []
for name, p in probs.items():
    point = {
        "AUPRC": average_precision_score(y_te, p),
        "AUROC": roc_auc_score(y_te, p),
        "Brier": brier_score_loss(y_te, p),
        "Calibration_slope": calibration_slope(y_te, p),
    }
    boot = {k: [] for k in point}
    for i in draws:
        boot["AUPRC"].append(average_precision_score(y_te[i], p[i]))
        boot["AUROC"].append(roc_auc_score(y_te[i], p[i]))
        boot["Brier"].append(brier_score_loss(y_te[i], p[i]))
    paired = paired_bootstrap_test_comparison(
        y_te, base, p, n_bootstrap=2000, random_state=42
    )
    row = {"model": name}
    for k, v in point.items():
        row[k] = v
        if boot[k]:
            row[k + "_ci_low"], row[k + "_ci_high"] = np.percentile(
                boot[k], [2.5, 97.5]
            )
    for k in ["AUPRC", "AUROC", "Brier"]:
        d = paired["Delta_" + k]
        row.update(
            {
                f"Delta_{k}": d["diff_point"],
                f"Delta_{k}_ci_low": d["ci_lower"],
                f"Delta_{k}_ci_high": d["ci_upper"],
                f"Delta_{k}_p": d["p_value"],
            }
        )
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv("step_23_lr_reduced_sensitivity.csv", index=False)
print("reproduction max |diff|:", diff, "| shrinkage factor", round(S, 4))
pd.set_option("display.width", 220)
print(out.round(4).T.to_string())

"""Export and independently validate the final shrunk LR from saved results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

CODE = Path(__file__).resolve().parent
FEATURES = [
    "SEX",
    "EDUCYRS",
    "fampd_bin",
    "sym_other",
    "upsit_pctl",
    "gds",
    "MIA_CAUDATE_mean",
]
BINARY = {"SEX", "fampd_bin", "sym_other"}
UNITS = {
    "SEX": "0=female, 1=male",
    "EDUCYRS": "years, capped at 20",
    "fampd_bin": "0=no, 1=yes",
    "sym_other": "0=no, 1=yes",
    "upsit_pctl": "age- and sex-adjusted percentile",
    "gds": "points (15-item GDS)",
    "MIA_CAUDATE_mean": "bilateral caudate specific binding ratio (MIAKAT, white-matter reference)",
}


def predict_final_model(data: pd.DataFrame, model: dict) -> np.ndarray:
    """Apply the documented input coding and training-derived imputation; return risk."""
    missing = set(model["feature_order"]) - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing predictor columns: {sorted(missing)}; supply NaN for unavailable values"
        )
    x = data[model["feature_order"]].apply(pd.to_numeric, errors="raise").copy()
    for feature in model["feature_order"]:
        if feature in BINARY and not x[feature].dropna().isin([0, 1]).all():
            raise ValueError(f"{feature} must be coded 0/1 or missing")
        x[feature] = x[feature].fillna(
            model["predictors"][feature]["imputation"]["value"]
        )
    if not np.isfinite(x.to_numpy(float)).all():
        raise ValueError("Predictors must be finite numeric values or missing")
    x["EDUCYRS"] = x["EDUCYRS"].clip(upper=model["predictors"]["EDUCYRS"]["upper_cap"])
    return expit(
        model["intercept"]
        + x.to_numpy(float)
        @ np.array([model["coefficients"][f] for f in model["feature_order"]])
    )


def export_final_model() -> dict:
    read = lambda name: pd.read_csv(CODE / name)
    parameters_path = CODE / "model_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    coefficients = read("step_14_shrunk_coefficients.csv").set_index("Variable")[
        "beta_shrunk"
    ]
    summary = read("step_14_shrinkage_summary.csv").set_index("Metric")["Value"]
    metric = (
        read("step_21_benchmark_test_metrics_7feat.csv").set_index("Model").loc["LR"]
    )
    ci = read("step_21_benchmark_model_ci_7feat.csv").set_index("Model").loc["LR"]
    imputation = json.loads(
        (CODE / "step_08_imputation_parameters.json").read_text(encoding="utf-8")
    )
    train = read("PPMI_9_train.csv")
    final = {
        "schema_version": "1.0",
        "model_version": "v1.9",
        "status": "final_manuscript_model",
        "model_type": "Logistic regression with bootstrap uniform shrinkage",
        "target": "Concurrent PPMI cognitive-test-based PD-MCI indicator at baseline",
        "intended_population": "Baseline participants in the PPMI PD cohort; no disease-duration restriction",
        "coefficient_scale": "original predictor units; do not z-standardize inputs",
        "intercept": float(coefficients["Intercept"]),
        "coefficients": {f: float(coefficients[f]) for f in FEATURES},
        "feature_order": FEATURES,
        "shrinkage_factor": float(summary["Shrinkage_factor_S"]),
        "thresholds": {"youden": float(metric.Threshold)},
        "threshold_method": "Youden index of pooled five-fold training OOF predictions; seven predictors and shrinkage factor fixed in each fold",
        "predictors": {
            f: {
                "type": "binary" if f in BINARY else "numeric_score_or_measure",
                "unit": UNITS[f],
                "imputation": imputation[f],
                "training_observed_range": [
                    float(train[f].min()),
                    float(train[f].max()),
                ],
            }
            for f in FEATURES
        },
        "test_performance": {
            k: float(metric[k]) for k in metric.index if k != "Runtime_s"
        },
        "test_performance_ci": {k: float(ci[k]) for k in ci.index},
        "sources": {
            "coefficients": "step_14_shrunk_coefficients.csv",
            "performance_and_threshold": "step_21_benchmark_test_metrics_7feat.csv",
            "confidence_intervals": "step_21_benchmark_model_ci_7feat.csv",
        },
    }
    final["predictors"]["EDUCYRS"]["upper_cap"] = 20
    final["predictors"]["gds"]["instrument_range"] = [0, 15]
    final["predictors"]["upsit_pctl"]["instrument_range"] = [0, 100]
    test = read("PPMI_9_test.csv")
    saved = read("step_21_benchmark_predictions_7feat.csv")
    if not np.array_equal(test.PATNO, saved.PATNO) or not np.array_equal(
        test.MCI, saved.MCI
    ):
        raise RuntimeError(
            "Final export validation: participant/outcome alignment failed"
        )
    probabilities = predict_final_model(test, final)
    difference = float(np.max(np.abs(probabilities - saved.prob_LR.to_numpy())))
    if difference > 1e-10:
        raise RuntimeError(
            f"Final export does not reproduce saved probabilities: {difference}"
        )
    positive = probabilities >= final["thresholds"]["youden"]
    y = test.MCI.to_numpy(int)
    confusion = {
        "TN": int(np.sum(~positive & (y == 0))),
        "FP": int(np.sum(positive & (y == 0))),
        "FN": int(np.sum(~positive & (y == 1))),
        "TP": int(np.sum(positive & (y == 1))),
    }
    if confusion != {"TN": 324, "FP": 98, "FN": 46, "TP": 32}:
        raise RuntimeError(
            f"Final export classification differs from validated results: {confusion}"
        )
    final["test_confusion_matrix"] = confusion
    parameters["legacy_development_status"] = (
        "Top-level coefficients/performance are the unshrunk development fit; shrinkage is the step 14 development export with its older ten-fold threshold. Use final_model for the manuscript model."
    )
    parameters["scale_ranges"]["Education"]["max"] = 20
    parameters["final_model"] = final
    parameters_path.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Standalone copy of the final model only; development fields stay in model_parameters.json.
    (CODE / "final_model.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = {
        "model_version": final["model_version"],
        "n_test": len(test),
        "maximum_probability_difference": difference,
        "threshold": float(metric.Threshold),
        "confusion_matrix": confusion,
        "passed": True,
        "sources_sha256": {
            name: hashlib.sha256((CODE / name).read_bytes()).hexdigest()
            for name in final["sources"].values()
        },
    }
    (CODE / "final_model_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Final model exported and validated: max probability difference {difference:.2e}"
    )
    return final


if __name__ == "__main__":
    export_final_model()

"""Targeted SHAP diagnostics from saved coalitions; no full SHAP recomputation."""

from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap
from benchmark_models import create_benchmark_model
from benchmark_evaluation import fit_benchmark_model
from step_15_tuning import build_model_config
from step_17_shap import coalition_design, coalition_values, shapley_weights

CODE = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(CODE)
    config = json.loads(Path("benchmark_config.json").read_text())
    tuned = json.loads(Path("step_15_tuning_best_params.json").read_text())
    train = pd.read_csv("PPMI_9_train.csv")
    test = pd.read_csv("PPMI_9_test.csv")
    features = [c for c in train if c not in {"PATNO", "MCI"}]
    scaler = StandardScaler().fit(train[features])
    x, t = scaler.transform(train[features]), scaler.transform(test[features])
    y = train.MCI.to_numpy(int)
    background = x[np.random.default_rng(42).choice(len(x), 100, replace=False)]
    model = create_benchmark_model(
        "LightGBM",
        build_model_config(config, "LightGBM", tuned["7feat"]["LightGBM"]["params"]),
        random_state=42,
    )
    fit_benchmark_model(model, x, y, random_state=42)
    result_path = Path("step_17_shap_checkpoints/7feat/LightGBM/result.npz")
    saved = dict(np.load(result_path, allow_pickle=True))
    phi = saved["phi"]
    best = model.model.best_iteration_ or model.model.booster_.current_iteration()
    booster = lgb.Booster(
        model_str=model.model.booster_.model_to_string(num_iteration=best)
    )
    raw = booster.predict(t, raw_score=True)
    report = {
        "LightGBM": {
            "n_explained": len(t),
            "n_background": len(background),
            "best_iteration": best,
            "raw_margin_logit_difference": float(
                abs(raw - logit(model.predict_proba(t))).max()
            ),
        }
    }
    for dtype in (np.float64, np.float32):
        explainer = shap.TreeExplainer(
            booster,
            data=background.astype(dtype),
            feature_perturbation="interventional",
            model_output="raw",
        )
        sv = np.asarray(explainer.shap_values(t.astype(dtype), check_additivity=False))
        error = abs(sv - phi)
        worst = np.unravel_index(np.argmax(error), error.shape)
        report["LightGBM"][np.dtype(dtype).name] = {
            "maximum_shap_difference": float(error.max()),
            "maximum_tree_shap_additivity_error": float(
                abs(explainer.expected_value + sv.sum(axis=1) - raw).max()
            ),
            "base_value_difference": float(
                abs(explainer.expected_value - saved["base"][0])
            ),
            "importance_rank_correlation": float(
                spearmanr(abs(sv).mean(axis=0), abs(phi).mean(axis=0)).statistic
            ),
            "worst_feature": features[worst[1]],
        }
        present = coalition_design(len(features))
        v = coalition_values(
            model.predict_proba, t[[worst[0]]], background, present, 262144
        )
        report["LightGBM"][np.dtype(dtype).name]["direct_enumeration_saved_error"] = (
            float(abs(v @ shapley_weights(present) - phi[[worst[0]]]).max())
        )
    probe = np.vstack([t, background])
    probability_dtype_error = float(
        abs(
            model.predict_proba(probe) - model.predict_proba(probe.astype(np.float32))
        ).max()
    )
    report["LightGBM"]["float32_conversion_probability_error"] = probability_dtype_error
    report["LightGBM"]["foreground_prediction_cast_difference"] = float(
        abs(model.predict_proba(t) - model.predict_proba(t.astype(np.float32))).max()
    )
    report["LightGBM"]["background_prediction_cast_difference"] = float(
        abs(
            model.predict_proba(background)
            - model.predict_proba(background.astype(np.float32))
        ).max()
    )
    if (
        report["LightGBM"]["float32"]["maximum_shap_difference"] > 1e-6
        or report["LightGBM"]["float64"]["direct_enumeration_saved_error"] > 1e-9
    ):
        raise RuntimeError("LightGBM dtype diagnostic failed")
    report["LightGBM"][
        "interpretation"
    ] = "Float64 TreeSHAP comparison path produces a dtype-dependent discrepancy; float32 foreground and background in the reference explainer restore agreement with original saved enumeration and global ranks. Do not cast actual prediction or coalition inputs: dtype conversion can change predictions near tree splits. Saved model probabilities and enumerated SHAP values are unchanged."
    # Refresh diagnostic metadata only, preserving all coalition and SHAP arrays.
    checks = json.loads(str(saved["checks"]))
    checks["treeshap_float64_max_diff"] = report["LightGBM"]["float64"][
        "maximum_shap_difference"
    ]
    checks["treeshap_max_diff"] = report["LightGBM"]["float32"][
        "maximum_shap_difference"
    ]
    checks["treeshap_note"] = (
        "float32 foreground/background; previous float64 comparison had a dtype-dependent discrepancy"
    )
    saved["checks"] = json.dumps(checks)
    np.savez(result_path, **saved)
    frame = pd.read_csv("step_17_shap_checks_7feat.csv")
    index = frame.model.eq("LightGBM")
    for key in ("treeshap_max_diff", "treeshap_note", "treeshap_float64_max_diff"):
        frame.loc[index, key] = checks[key]
    frame.to_csv("step_17_shap_checks_7feat.csv", index=False)

    tabicl = create_benchmark_model(
        "TabICLv2", build_model_config(config, "TabICLv2"), random_state=42
    )
    fit_benchmark_model(tabicl, x, y, random_state=42)
    # Match the large coalition-inference batch and the scored-test batch sizes.
    probe = np.repeat(t[:100], 200, axis=0)
    report["TabICLv2"] = {
        "n_probe_rows": len(probe),
        "n_unique_participants": 100,
        "batch_rows": [20000, 500],
    }
    for setting in ("auto", False):
        # AMP is resolved during fit; changing the attribute afterwards does not
        # reconfigure the prepared inference manager.
        from tabicl import TabICLClassifier

        parameters = tabicl.model.get_params()
        parameters["use_amp"] = setting
        classifier = TabICLClassifier(**parameters).fit(x, y)
        whole = classifier.predict_proba(probe)[:, 1].astype(float)
        split = np.concatenate(
            [
                classifier.predict_proba(probe[i : i + 500])[:, 1]
                for i in range(0, len(probe), 500)
            ]
        ).astype(float)
        report["TabICLv2"][str(setting)] = {
            "maximum_probability_batch_difference": float(abs(whole - split).max()),
            "maximum_logit_batch_difference": float(
                abs(logit(whole) - logit(split)).max()
            ),
        }
    report["TabICLv2"][
        "interpretation"
    ] = "Batch sensitivity is measured separately from saved exact-coalition sums; the saved model explanations retain the original benchmark inference settings."
    Path("step_17_shap_diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if report["TabICLv2"]["False"]["maximum_logit_batch_difference"] >= 1e-5:
        raise RuntimeError(
            "TabICL full-precision batch check exceeded tolerance; measured diagnostics were saved"
        )


if __name__ == "__main__":
    main()

"""Verify saved final artifacts and write current provenance; never refit or train."""

from __future__ import annotations
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.special import expit
import benchmark_evaluation as be
from export_final_model import predict_final_model
from step_15_tuning import tuning_fingerprint

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
MODELS = [
    "LR",
    "SVM",
    "RF",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RealMLP",
    "TabM",
    "TabICLv2",
    "TabPFN",
    "TabFM",
]
BASELINE_INPUTS = {
    "PPMI_8_train.csv": "7f62746f137a99733c4f2e81e0f583af5e149bbee85fe1c7fdbe0fbd96b10c5b",
    "PPMI_8_test.csv": "a71305a098dc81faf17d3a2c138a59c7c0fed0aa8856ccde8c9e57bcfb8f4190",
    "PPMI_9_train.csv": "a150c78214d75e6098b94ffcb2330aa448f575e6634e735e5973e1a717f30b78",
    "PPMI_9_test.csv": "d47e6c8eacfc5828f2392faa6f7ac740c2076088fcb3fea5256c0b2f6bfe7e94",
    "PPMI_Curated_Data_Cut_Public_20260810.xlsx": "a844495c6c015e470e73254daae8caa6b8fc6754720a993bca77334a8cc764ef",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return pd.read_csv(CODE / name)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_tables():
    output = ROOT / "tmp" / "final_table_validation"
    checked = []
    for filename, variable, target, real in [
        (
            "step_20_manuscript_tables.py",
            "OUT_DIR",
            output / "main",
            ROOT / "manuscript/tables",
        ),
        (
            "step_22_supplementary_tables.py",
            "OUT",
            output / "supplement",
            ROOT / "manuscript/tables",
        ),
    ]:
        source = (CODE / filename).read_text(encoding="utf-8")
        if variable == "OUT":
            source = source.split(
                "# ---------------------------------------------------------------- figures",
                1,
            )[0]
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == variable for t in node.targets
            ):
                node.value = ast.Constant(str(target))
        ast.fix_missing_locations(tree)
        exec(
            compile(tree, filename, "exec"),
            {"__name__": "validate_saved_tables", "__file__": str(CODE / filename)},
        )
        folder = target if variable == "OUT_DIR" else target / "tables"
        expected = (
            [f"table{i}.tex" for i in range(1, 4)]
            if variable == "OUT_DIR"
            else [f"S{i}.tex" for i in range(2, 13)]
        )
        for name in expected:
            require(
                (real / name).read_text(encoding="utf-8").splitlines()
                == (folder / name).read_text(encoding="utf-8").splitlines(),
                f"Stale table: {real / name}",
            )
            checked.append((real / name).relative_to(ROOT).as_posix())
    return checked


def main():
    os.chdir(CODE)
    checks = {}
    pyfiles = list(CODE.glob("*.py")) + list((ROOT / "release").glob("*.py"))
    for p in pyfiles:
        ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
    checks["python_syntax"] = {"files": len(pyfiles), "passed": True}
    for name, value in BASELINE_INPUTS.items():
        require(
            sha(CODE / name) == value, f"Input changed from reviewed baseline: {name}"
        )
    train, test = read("PPMI_8_train.csv"), read("PPMI_8_test.csv")
    require(
        (len(train), len(test), int(train.MCI.sum()), int(test.MCI.sum()))
        == (1165, 500, 183, 78),
        "Cohort/event counts changed",
    )
    require(
        not train.PATNO.duplicated().any()
        and not test.PATNO.duplicated().any()
        and not set(train.PATNO) & set(test.PATNO),
        "Duplicate or overlapping participants",
    )
    raw_labels = (
        read("PPMI_0_data.csv")
        .query("EVENT_ID == 'BL'")
        .set_index("PATNO")["MCI_testscores"]
    )
    cohort = pd.concat([train, test]).set_index("PATNO")
    require(
        np.array_equal(cohort.MCI, raw_labels.reindex(cohort.index)),
        "Outcome differs from supplied PPMI indicator",
    )
    checks["cohort"] = {
        "train": 1165,
        "test": 500,
        "events": [183, 78],
        "overlap": 0,
        "input_hashes_unchanged": True,
        "supplied_outcome_matches": True,
    }
    config = json.loads((CODE / "benchmark_config.json").read_text(encoding="utf-8"))
    best = json.loads(
        (CODE / "step_15_tuning_best_params.json").read_text(encoding="utf-8")
    )
    for track, entries in best.items():
        for model, entry in entries.items():
            require(
                entry["n_trials_completed"] == 100
                and entry["fingerprint"] == tuning_fingerprint(config, track, model),
                f"Stale tuning: {track}/{model}",
            )
    checks["tuning"] = {
        "fingerprints_matched": sum(len(v) for v in best.values()),
        "trials_each": 100,
    }
    errors = {}
    for track, step in [("7feat", "21"), ("full", "15")]:
        pred = read(f"step_{step}_benchmark_predictions_{track}.csv")
        require(
            np.array_equal(test.PATNO, pred.PATNO)
            and np.array_equal(test.MCI, pred.MCI),
            "Prediction alignment failed",
        )
        metrics = read(f"step_{step}_benchmark_test_metrics_{track}.csv").set_index(
            "Model"
        )
        require(set(metrics.index) == set(MODELS), f"Incomplete {track} metrics")
        differences = []
        for model in MODELS:
            observed = be.compute_binary_metrics(
                pred.MCI.to_numpy(),
                pred[f"prob_{model}"].to_numpy(),
                metrics.loc[model, "Threshold"],
            )
            differences.extend(
                abs(value - metrics.loc[model, key]) for key, value in observed.items()
            )
        errors[track] = max(differences)
        require(errors[track] < 2e-7, f"Saved metrics do not reproduce: {track}")
        diff = read(f"step_{step}_benchmark_bootstrap_diffs_{track}.csv")
        for metric in ("AUPRC", "AUROC", "Brier"):
            require(
                diff[[f"Delta_{metric}_CI_lower", f"Delta_{metric}_CI_upper"]]
                .notna()
                .all()
                .all(),
                "Missing full-precision intervals",
            )
    checks["metrics"] = {"models_per_track": 11, "maximum_absolute_errors": errors}
    final = json.loads((CODE / "model_parameters.json").read_text(encoding="utf-8"))[
        "final_model"
    ]
    data = read("PPMI_9_test.csv")
    coef = read("step_14_shrunk_coefficients.csv").set_index("Variable").beta_shrunk
    independent = expit(
        float(coef.Intercept)
        + data[final["feature_order"]].to_numpy()
        @ coef[final["feature_order"]].to_numpy()
    )
    saved = read("step_21_benchmark_predictions_7feat.csv")
    delta = max(
        float(np.max(np.abs(independent - saved.prob_LR))),
        float(np.max(np.abs(predict_final_model(data, final) - independent))),
    )
    require(delta < 1e-10, "Final equation/export differs from saved LR")
    require(
        final["model_version"] == "v1.9"
        and final["predictors"]["EDUCYRS"]["upper_cap"] == 20,
        "Stale model version/cap",
    )
    require(
        final["thresholds"]["youden"] == 0.2003183883873819
        and final["test_confusion_matrix"] == {"TN": 324, "FP": 98, "FN": 46, "TP": 32},
        "Stale threshold/classification",
    )
    # Exercise export-only input handling; this is not a clinical missingness validation.
    unavailable = pd.DataFrame([{f: np.nan for f in final["feature_order"]}])
    expected = expit(
        final["intercept"]
        + sum(
            final["coefficients"][f] * final["predictors"][f]["imputation"]["value"]
            for f in final["feature_order"]
        )
    )
    require(
        abs(float(predict_final_model(unavailable, final)[0]) - expected) < 1e-12,
        "Imputation/export mismatch",
    )
    for feature, value in [("SEX", 2), ("gds", np.inf)]:
        bad = data.iloc[:1].copy()
        bad[feature] = value
        try:
            predict_final_model(bad, final)
        except ValueError:
            pass
        else:
            raise RuntimeError(f"Invalid {feature} accepted")
    checks["final_export"] = {
        "maximum_probability_error": delta,
        "threshold": final["thresholds"]["youden"],
        "confusion": final["test_confusion_matrix"],
        "coding_missingness_finite_input_checks": "passed",
    }
    shap = np.load(CODE / "step_17_shap_values_7feat.npz", allow_pickle=False)
    importance = read("step_17_shap_importance_7feat.csv").set_index(
        ["Model", "Predictor"]
    )
    require(
        set(shap["models"]) == set(MODELS)
        and np.array_equal(shap["patno"], test.PATNO),
        "Incomplete/misaligned SHAP",
    )
    for model in MODELS:
        require(shap[f"phi_{model}"].shape == (500, 7), "Invalid SHAP dimensions")
        means = np.abs(shap[f"phi_{model}"]).mean(axis=0)
        for feature, value in zip(shap["features"], means):
            require(
                abs(value - importance.loc[(model, feature), "Mean_abs_SHAP"]) < 1e-10,
                "SHAP aggregate mismatch",
            )
    diag = json.loads((CODE / "step_17_shap_diagnostics.json").read_text())
    require(
        diag["LightGBM"]["float32"]["maximum_shap_difference"] < 1e-6
        and diag["LightGBM"]["float64"]["direct_enumeration_saved_error"] < 1e-9
        and diag["TabICLv2"]["False"]["maximum_logit_batch_difference"] < 1e-5,
        "SHAP diagnostic tolerance failed",
    )
    checks["shap"] = {
        "models": 11,
        "participants": 500,
        "features": 7,
        "aggregates_match_arrays": True,
        "targeted_diagnostics": diag,
        "full_shap_rerun": False,
    }
    freq = read("step_14_bootstrap_selection_frequency.csv")
    require(
        len(freq) == 28
        and freq.In_final_model.sum() == 7
        and np.allclose(
            freq.Selection_frequency * 500, np.round(freq.Selection_frequency * 500)
        ),
        "Invalid selection frequencies",
    )
    checks["tables"] = {
        "regenerated_in_tmp_and_matched": validate_tables(),
        "selection_frequency_candidates": 28,
    }
    builds = json.loads((CODE / "manuscript_build_manifest.json").read_text())
    checks["pdfs"] = {}
    for name, record in builds.items():
        require(
            sha(ROOT / record["pdf"]) == record["pdf_sha256"], "Stale final PDF hash"
        )
        for path, value in record["inputs_sha256"].items():
            require(sha(ROOT / path) == value, f"PDF build input changed: {path}")
        text = subprocess.check_output(
            ["pdftotext", "-layout", str(ROOT / record["pdf"]), "-"]
        ).decode("utf-8")
        for token in (
            ["0.00004", "liblinear", "July 2010", "46 of the 78"]
            if name == "main"
            else [
                "Table S11",
                "0.00434",
                "0.0251",
                "TRIPOD+AI checklist is provided as a separate file",
            ]
        ):
            require(token in " ".join(text.split()), f"Missing compiled text: {token}")
        checks["pdfs"][name] = {
            "pages": text.count("\f"),
            "text_anchors": "passed",
            "build_inputs_and_pdf_hashes": "matched",
            "layout_warnings": record["layout_warnings"],
        }
    require(set(checks["pdfs"]) == {"main", "supplement"}, "Missing final document")
    artifacts = {}
    for folder in [CODE, ROOT / "manuscript"]:
        for p in sorted(folder.rglob("*")):
            if (
                not p.is_file()
                or p.name == "final_artifact_manifest.json"
                or p.suffix
                not in {
                    ".py",
                    ".R",
                    ".json",
                    ".csv",
                    ".pdf",
                    ".tex",
                    ".bib",
                    ".md",
                    ".txt",
                }
            ):
                continue
            rel = p.relative_to(ROOT).as_posix()
            if (
                any(
                    part
                    in {
                        "__pycache__",
                        "step_15_checkpoints",
                        "step_17_shap_checkpoints",
                    }
                    for part in p.parts
                )
                or re.match(r"PPMI_\d", p.name)
                or "predictions" in p.name
                or "_quicktest" in p.name
            ):
                continue
            artifacts[rel] = sha(p)
    for relative in [
        "README.md",
        "CITATION.cff",
        ".gitignore",
        "review/implementation_plan.md",
        "review/version.md",
    ]:
        artifacts[relative] = sha(ROOT / relative)
    manifest = {
        "target_version": "v1.9",
        "observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "basis": "Current file bytes and saved results, not a claim that historical fits were regenerated with edited reporting code",
        "baseline": "v1.8 / 5406599; analysis inputs unchanged",
        "validation": checks,
        "restricted_input_sha256_only": {
            **BASELINE_INPUTS,
            **{
                p.name: sha(p)
                for p in CODE.glob("*predictions*.csv")
                if "quicktest" not in p.name
            },
            "step_17_shap_values_7feat.npz": sha(
                CODE / "step_17_shap_values_7feat.npz"
            ),
        },
        "artifact_sha256": artifacts,
        "authority": {
            "final_equation": "step_14_shrunk_coefficients.csv",
            "primary_results": "step_21_benchmark_*_7feat",
            "sensitivity_results": "step_15/16_benchmark_*_full",
            "model_export": "model_parameters.json/final_model",
            "shap": "step_17_shap_*_7feat",
            "tables": "step_20 and step_22",
            "pdf_build": "manuscript_build_manifest.json",
        },
    }
    (CODE / "final_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "target_version": "v1.9",
                "checks_passed": list(checks),
                "artifact_hashes": len(artifacts),
                "metric_errors": errors,
                "export_probability_error": delta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

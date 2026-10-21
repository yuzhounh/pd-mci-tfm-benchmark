"""Step 15b: PD-MCI Tabular Deep Learning & Foundation Model Benchmark (config 1.4.0).

Compares, against the v1.0 LR:
1. LR: the complete v1.0 algorithm (LASSO 1SE selection -> unpenalized LR -> bootstrap shrinkage)
2. SVM and Random Forest (classical machine learning)
3. XGBoost, LightGBM and CatBoost (GBDT)
4. RealMLP and TabM (tabular deep learning trained from scratch)
5. TabICLv2, TabPFN v3.5 and TabFM (pretrained tabular foundation models, zero-shot)

Models 2-4 use the hyperparameters found by step_15_tuning.py; the run refuses a tuned
model whose tuning result is missing, incomplete or has a stale fingerprint.

Key methodological safeguards:
- Primary analysis: 7 LASSO-selected features; the 28 candidate features are a
  sensitivity analysis. Primary metric: test-set Delta AUPRC versus LR.
- In-fold standardization on training folds to strictly prevent data leakage.
- Early stopping uses an inner split of the training rows only; the rows being scored
  are never used for early stopping.
- Optimal decision thresholds derived exclusively from training set out-of-fold predictions.
- Evaluation on the locked 500-participant test set. `--quick-test` never touches the test
  set: it evaluates on a stratified 30% split of the training set and writes `_quicktest`
  outputs.
- Participant-level paired bootstrap (2,000 resamples) computing delta AUPRC, delta AUROC,
  and delta Brier score with 95% CI relative to LR.
- Each finished model is checkpointed (step_15_checkpoints/, git-ignored because it holds
  predictions); a rerun skips models whose checkpoint fingerprint still matches.
- A model that fails stops the run instead of silently disappearing from the results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from benchmark_models import (
    TUNED_MODELS,
    create_benchmark_model,
    file_sha256,
    tabpfn_checkpoint_path,
)
from step_15_tuning import BEST_PARAMS_FILE, build_model_config, tuning_fingerprint
from benchmark_evaluation import (
    compute_binary_metrics,
    find_optimal_youden_threshold,
    run_cross_validation,
    run_test_evaluation,
    paired_bootstrap_test_comparison,
    plot_benchmark_roc_pr,
    plot_benchmark_calibration,
    plot_benchmark_dca,
)

ALL_MODELS = [
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
PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "catboost",
    "xgboost",
    "lightgbm",
    "optuna",
    "pytabkit",
    "tabm",
    "tabicl",
    "tabpfn",
    "tabfm",
]
TRACK_ORDER = ["7feat", "full"]
TRACK_TITLES = {
    "7feat": "Primary Analysis (7 LASSO-Selected Features)",
    "full": "Sensitivity Analysis (28 Candidate Features)",
}
CHECKPOINT_DIR = "step_15_checkpoints"


def load_benchmark_config(config_path: str = "benchmark_config.json") -> dict:
    """Load benchmark configuration file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_benchmark_data(
    train_file: str,
    test_file: str,
    forbidden_cols: list[str],
    target_col: str = "MCI",
    subject_col: str = "PATNO",
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """
    Load and sanitize train and test sets, strictly excluding forbidden/leakage columns.
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Invariant assertions
    if len(np.unique(train_df[subject_col])) != len(train_df):
        raise RuntimeError(
            f"Participant invariant violated in {train_file}: duplicate {subject_col}"
        )
    if len(np.unique(test_df[subject_col])) != len(test_df):
        raise RuntimeError(
            f"Participant invariant violated in {test_file}: duplicate {subject_col}"
        )

    feature_cols = [c for c in train_df.columns if c not in forbidden_cols]

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[target_col].values.astype(int)
    subjects_train = train_df[subject_col].values

    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df[target_col].values.astype(int)
    subjects_test = test_df[subject_col].values

    return X_train, y_train, subjects_train, X_test, y_test, subjects_test, feature_cols


def quick_test_config(model_key: str, model_cfg: dict) -> dict:
    """Shrink compute budgets for a fast smoke test."""
    fixed = model_cfg.get("fixed_params", {})
    if model_key == "LR":
        model_cfg["n_shrinkage_bootstrap"] = 20
    elif model_key == "TabM":
        fixed.update(seeds=[42], max_epochs=5)
    elif model_key == "CatBoost":
        fixed["iterations"] = 30
    elif model_key in ["XGBoost", "LightGBM"]:
        fixed["n_estimators"] = 30
    elif model_key == "RealMLP":
        fixed["n_epochs"] = 8
    elif model_key in ["TabICLv2", "TabFM", "TabPFN"]:
        model_cfg["n_estimators"] = 1
    return model_cfg


def load_tuned_params(
    config: dict, track: str, model_key: str, path: str, allow_incomplete: bool
) -> dict:
    """Best tuned parameters, refused if missing, incomplete or from a stale configuration."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; run step_15_tuning.py first")
    with open(path, encoding="utf-8") as f:
        entry = json.load(f).get(track, {}).get(model_key)
    if entry is None:
        raise RuntimeError(
            f"No tuning result for {model_key} on the {track} track in {path}"
        )
    if entry["fingerprint"] != tuning_fingerprint(config, track, model_key):
        raise RuntimeError(
            f"Tuning result for {model_key} ({track}) is stale: its fingerprint "
            "does not match the current configuration, data or code"
        )
    n_required = config["protocol"]["tuning"]["n_trials"]
    if entry["n_trials_completed"] < n_required and not allow_incomplete:
        raise RuntimeError(
            f"Tuning of {model_key} ({track}) has {entry['n_trials_completed']} "
            f"of {n_required} trials"
        )
    return entry


def checkpoint_fingerprint(
    config: dict, track: str, model_key: str, model_cfg: dict, quick_test: bool
) -> str:
    """Hash of everything that determines one model's CV and test predictions."""
    data_cfg = config["data_files"][f"{track}_track"]
    parts = {
        "model": model_key,
        "model_cfg": model_cfg,
        "track": track,
        "quick_test": quick_test,
        "protocol": {
            k: config["protocol"][k] for k in ["cv_folds", "cv_seed", "device"]
        },
        "train_sha256": file_sha256(data_cfg["train"]),
        "test_sha256": file_sha256(data_cfg["test"]),
        "code_sha256": [
            file_sha256(f) for f in ["benchmark_models.py", "benchmark_evaluation.py"]
        ],
    }
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def collect_run_info(
    config: dict, selected_models: list[str], quick_test: bool
) -> dict:
    """Software, hardware and checkpoint provenance for the run."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    info = {
        "quick_test": quick_test,
        "models": selected_models,
        "device": config["protocol"].get("device", "cpu"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": versions,
        "config": config,
    }
    info["tabpfn_checkpoints"] = {}
    for key in selected_models:
        if key.startswith("TabPFN"):
            version = config["models"][key]["model_version"]
            path = tabpfn_checkpoint_path(version)
            info["tabpfn_checkpoints"][key] = {
                "version": version,
                "file": os.path.basename(path),
                "sha256": file_sha256(path) if os.path.isfile(path) else None,
            }
    return info


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def report_progress(progress: dict, label: str) -> None:
    """One plain progress line per finished model: overall done/total, percent, elapsed, ETA."""
    progress["done"] += 1
    elapsed = time.time() - progress["started"]
    eta = elapsed / progress["done"] * (progress["total"] - progress["done"])
    print(
        f"[progress] {progress['done']}/{progress['total']} "
        f"({100 * progress['done'] / progress['total']:.0f}%) {label} done | "
        f"elapsed {format_seconds(elapsed)}, ETA {format_seconds(eta)}",
        flush=True,
    )


def run_track_benchmark(
    track_name: str,
    config: dict,
    selected_models: list[str],
    quick_test: bool = False,
    tuned_params_path: str = BEST_PARAMS_FILE,
    allow_incomplete_tuning: bool = False,
    progress: dict | None = None,
) -> dict:
    """Run the benchmark for one feature track ('7feat' or 'full'); returns tuning provenance."""
    print("\n" + "=" * 80)
    print(f"RUNNING BENCHMARK TRACK: {track_name.upper()}")
    print("=" * 80)

    data_cfg = config["data_files"]
    track_key = f"{track_name}_track"
    if track_key not in data_cfg:
        raise ValueError(f"Unknown track: {track_name}")

    train_file = data_cfg[track_key]["train"]
    test_file = data_cfg[track_key]["test"]
    forbidden_cols = data_cfg["forbidden_columns"]

    X_train, y_train, subjects_train, X_test, y_test, subjects_test, feature_cols = (
        prepare_benchmark_data(train_file, test_file, forbidden_cols)
    )

    suffix = track_name
    eval_label = "locked test set"
    if quick_test:
        # Smoke test: never read predictions or metrics from the locked test set
        X_train, X_test, y_train, y_test, subjects_train, subjects_test = (
            train_test_split(
                X_train,
                y_train,
                subjects_train,
                test_size=0.3,
                stratify=y_train,
                random_state=config["protocol"]["cv_seed"],
            )
        )
        suffix = f"{track_name}_quicktest"
        eval_label = "quick-test holdout (30% of training set)"

    print(
        f"Dataset: Train N={len(X_train)} (PD-MCI={np.sum(y_train)}, {np.mean(y_train):.1%}) | "
        f"Evaluation N={len(X_test)} [{eval_label}] (PD-MCI={np.sum(y_test)}, {np.mean(y_test):.1%})"
    )
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    n_folds = 2 if quick_test else config["protocol"]["cv_folds"]
    n_bootstrap = 100 if quick_test else config["protocol"]["bootstrap_samples"]

    oof_predictions = {"PATNO": subjects_train, "MCI": y_train}
    test_predictions = {"PATNO": subjects_test, "MCI": y_test}

    cv_results = []
    test_results = []
    optimal_thresholds = {}
    tuning_info = {}
    ckpt_dir = os.path.join(CHECKPOINT_DIR, suffix)
    os.makedirs(ckpt_dir, exist_ok=True)

    for model_key in selected_models:
        tuned = None
        if model_key in TUNED_MODELS:
            entry = load_tuned_params(
                config,
                track_name,
                model_key,
                tuned_params_path,
                allow_incomplete_tuning,
            )
            tuned = entry["params"]
            tuning_info[model_key] = entry
        model_cfg = build_model_config(config, model_key, tuned_params=tuned)
        if quick_test:
            model_cfg = quick_test_config(model_key, model_cfg)

        fingerprint = checkpoint_fingerprint(
            config, track_name, model_key, model_cfg, quick_test
        )
        ckpt_path = os.path.join(ckpt_dir, f"{model_key}.joblib")
        if os.path.isfile(ckpt_path):
            ckpt = joblib.load(ckpt_path)
            if ckpt["fingerprint"] == fingerprint:
                print(
                    f"\n[{model_key}] Loaded from checkpoint {ckpt_path} "
                    f"(test AUPRC {ckpt['test_row']['AUPRC']:.4f})"
                )
                oof_predictions[f"prob_{model_key}"] = ckpt["oof_probs"]
                test_predictions[f"prob_{model_key}"] = ckpt["test_probs"]
                optimal_thresholds[model_key] = ckpt["cv_row"]["Optimal_Threshold"]
                cv_results.append(ckpt["cv_row"])
                test_results.append(ckpt["test_row"])
                if progress is not None:
                    report_progress(progress, f"{track_name}/{model_key} (checkpoint)")
                continue
            print(f"\n[{model_key}] Checkpoint {ckpt_path} is stale; rerunning")

        print(
            f"\n[{model_key}] Starting {n_folds}-fold CV evaluation on training set..."
        )
        model_factory = lambda: create_benchmark_model(
            model_key, model_cfg, random_state=42
        )
        started = time.time()

        # 1. Cross-validation on training set (Leakage-free in-fold scaling)
        oof_probs, cv_mean, cv_std = run_cross_validation(
            model_factory=model_factory,
            X=X_train,
            y=y_train,
            subjects=subjects_train,
            n_folds=n_folds,
            random_state=config["protocol"]["cv_seed"],
        )

        oof_predictions[f"prob_{model_key}"] = oof_probs
        opt_thresh = find_optimal_youden_threshold(y_train, oof_probs)
        optimal_thresholds[model_key] = opt_thresh

        cv_row = {"Model": model_key, "Optimal_Threshold": opt_thresh}
        for m_name in [
            "AUROC",
            "AUPRC",
            "Brier",
            "LogLoss",
            "Calib_Slope",
            "Calib_Intercept",
        ]:
            cv_row[f"{m_name}_mean"] = cv_mean.get(m_name, float("nan"))
            cv_row[f"{m_name}_std"] = cv_std.get(m_name, float("nan"))
        cv_results.append(cv_row)

        print(
            f"[{model_key}] CV AUROC: {cv_mean['AUROC']:.4f} ± {cv_std['AUROC']:.4f} | "
            f"AUPRC: {cv_mean['AUPRC']:.4f} ± {cv_std['AUPRC']:.4f} | "
            f"Optimal Cutoff (Youden): {opt_thresh:.4f}"
        )

        # 2. Final fit on training set and evaluation on the evaluation set
        print(f"[{model_key}] Evaluating on {eval_label}...")
        test_probs, test_metrics = run_test_evaluation(
            model_factory=model_factory,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            threshold=opt_thresh,
            random_state=42,
        )

        test_predictions[f"prob_{model_key}"] = test_probs
        test_row = {"Model": model_key}
        test_row.update(test_metrics)
        test_row["Runtime_s"] = time.time() - started
        test_results.append(test_row)

        joblib.dump(
            {
                "fingerprint": fingerprint,
                "oof_probs": oof_probs,
                "test_probs": test_probs,
                "cv_row": cv_row,
                "test_row": test_row,
            },
            ckpt_path,
        )

        print(
            f"[{model_key}] Test AUPRC: {test_metrics['AUPRC']:.4f} | "
            f"AUROC: {test_metrics['AUROC']:.4f} | "
            f"Brier: {test_metrics['Brier']:.4f} | "
            f"Calib Slope: {test_metrics['Calib_Slope']:.4f} | "
            f"{test_row['Runtime_s']:.0f}s",
            flush=True,
        )
        if progress is not None:
            report_progress(progress, f"{track_name}/{model_key}")

    # Save DataFrames
    oof_df = pd.DataFrame(oof_predictions)
    test_df = pd.DataFrame(test_predictions)
    cv_df = pd.DataFrame(cv_results)
    test_summary_df = pd.DataFrame(test_results)

    oof_pred_file = f"step_15_benchmark_oof_predictions_{suffix}.csv"
    test_pred_file = f"step_15_benchmark_predictions_{suffix}.csv"
    cv_file = f"step_15_benchmark_cv_metrics_{suffix}.csv"
    test_metrics_file = f"step_15_benchmark_test_metrics_{suffix}.csv"
    boot_diff_file = f"step_15_benchmark_bootstrap_diffs_{suffix}.csv"

    oof_df.to_csv(oof_pred_file, index=False)
    test_df.to_csv(test_pred_file, index=False)
    cv_df.to_csv(cv_file, index=False)
    test_summary_df.to_csv(test_metrics_file, index=False)

    print(f"\nSaved out-of-fold predictions to: {oof_pred_file}")
    print(f"Saved test predictions to: {test_pred_file}")
    print(f"Saved CV metrics to: {cv_file}")
    print(f"Saved test metrics to: {test_metrics_file}")

    # 3. Paired Participant-Level Bootstrap against LR
    boot_diff_rows = []
    if "prob_LR" in test_df.columns:
        print(
            f"\nRunning {n_bootstrap} paired bootstrap resamples relative to LR baseline..."
        )
        base_probs = test_df["prob_LR"].values

        for m_key in test_summary_df["Model"].tolist():
            if m_key == "LR":
                continue
            m_probs = test_df[f"prob_{m_key}"].values
            diff_res = paired_bootstrap_test_comparison(
                y_test=y_test,
                baseline_probs=base_probs,
                model_probs=m_probs,
                n_bootstrap=n_bootstrap,
                random_state=config["protocol"]["bootstrap_seed"],
            )

            row = {
                "Model": m_key,
                "Comparison": f"{m_key} vs LR",
                "Delta_AUPRC": diff_res["Delta_AUPRC"]["diff_point"],
                "Delta_AUPRC_95CI": f"{diff_res['Delta_AUPRC']['ci_lower']:.4f}–{diff_res['Delta_AUPRC']['ci_upper']:.4f}",
                "Delta_AUPRC_pval": diff_res["Delta_AUPRC"]["p_value"],
                "Delta_AUROC": diff_res["Delta_AUROC"]["diff_point"],
                "Delta_AUROC_95CI": f"{diff_res['Delta_AUROC']['ci_lower']:.4f}–{diff_res['Delta_AUROC']['ci_upper']:.4f}",
                "Delta_AUROC_pval": diff_res["Delta_AUROC"]["p_value"],
                "Delta_Brier": diff_res["Delta_Brier"]["diff_point"],
                "Delta_Brier_95CI": f"{diff_res['Delta_Brier']['ci_lower']:.4f}–{diff_res['Delta_Brier']['ci_upper']:.4f}",
                "Delta_Brier_pval": diff_res["Delta_Brier"]["p_value"],
            }
            for metric in ("AUPRC", "AUROC", "Brier"):
                row[f"Delta_{metric}_CI_lower"] = diff_res[f"Delta_{metric}"][
                    "ci_lower"
                ]
                row[f"Delta_{metric}_CI_upper"] = diff_res[f"Delta_{metric}"][
                    "ci_upper"
                ]
            boot_diff_rows.append(row)

        boot_diff_df = pd.DataFrame(boot_diff_rows)
        boot_diff_df.to_csv(boot_diff_file, index=False)
        print(f"Saved paired bootstrap differences to: {boot_diff_file}")

        print("\n" + "=" * 80)
        print(
            f"PAIRED BOOTSTRAP TEST PERFORMANCE DIFFERENCES vs LR ({suffix.upper()} TRACK):"
        )
        print("=" * 80)
        print(boot_diff_df.to_string(index=False))

    # 4. Comparative Visualizations (PDF format)
    plot_dict = {
        m_key: test_df[f"prob_{m_key}"].values
        for m_key in test_summary_df["Model"].tolist()
        if f"prob_{m_key}" in test_df.columns
    }

    track_title = TRACK_TITLES[track_name]
    if quick_test:
        track_title += " - QUICK TEST, NOT LOCKED TEST SET"
    roc_pr_pdf = f"step_15_benchmark_roc_pr_{suffix}.pdf"
    calib_pdf = f"step_15_benchmark_calibration_{suffix}.pdf"
    dca_pdf = f"step_15_benchmark_dca_{suffix}.pdf"

    print("\nGenerating publication figures...")
    plot_benchmark_roc_pr(
        plot_dict, y_test, save_path=roc_pr_pdf, track_title=track_title
    )
    print(f"Saved ROC/PR curves to: {roc_pr_pdf}")

    plot_benchmark_calibration(
        plot_dict, y_test, save_path=calib_pdf, track_title=track_title
    )
    print(f"Saved Calibration curves to: {calib_pdf}")

    plot_benchmark_dca(plot_dict, y_test, save_path=dca_pdf, track_title=track_title)
    print(f"Saved Decision Curve Analysis to: {dca_pdf}")

    print("\n" + "=" * 80)
    print(
        f"FINAL PERFORMANCE SUMMARY ON {eval_label.upper()} ({suffix.upper()} TRACK):"
    )
    print("=" * 80)
    summary_cols = [
        "Model",
        "AUPRC",
        "AUROC",
        "Brier",
        "Calib_Slope",
        "Sensitivity",
        "Specificity",
        "F1",
        "Threshold",
    ]
    print(test_summary_df[summary_cols].round(4).to_string(index=False))
    return tuning_info


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(
        description="Run PD-MCI Tabular Deep Learning & Foundation Model Benchmark."
    )
    parser.add_argument(
        "--track",
        choices=["full", "7feat", "both"],
        default="both",
        help="Feature track to execute: 'full' (28 features), '7feat' (7 features), or 'both'",
    )
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"Comma-separated model keys or 'all' (options: {','.join(ALL_MODELS)},LR_plain)",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Smoke test with fewer folds/epochs/bootstrap samples, evaluated on a split of "
        "the training set (the locked test set is not used)",
    )
    parser.add_argument(
        "--tuned-params",
        default=BEST_PARAMS_FILE,
        help="Tuning result file from step_15_tuning.py",
    )
    args = parser.parse_args()

    config = load_benchmark_config("benchmark_config.json")
    if (
        config["protocol"].get("device", "cpu").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "benchmark_config.json requests CUDA but torch.cuda is not available"
        )

    # Resolve models
    if args.models.lower() == "all":
        selected_models = list(ALL_MODELS)
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 80)
    print("PD-MCI TABULAR DEEP LEARNING & FOUNDATION MODEL BENCHMARK")
    print(f"Models to evaluate: {selected_models}")
    print(f"Tracks to run: {args.track}")
    print(f"Quick test mode: {args.quick_test}")
    print("=" * 80)

    run_info = collect_run_info(config, selected_models, args.quick_test)
    run_info_file = (
        "step_15_benchmark_run_info_quicktest.json"
        if args.quick_test
        else "step_15_benchmark_run_info.json"
    )

    started = time.time()
    tracks = TRACK_ORDER if args.track == "both" else [args.track]

    run_info["tuning"] = {}
    progress = {
        "done": 0,
        "total": len(tracks) * len(selected_models),
        "started": started,
    }
    for tr in tracks:
        run_info["tuning"][tr] = run_track_benchmark(
            track_name=tr,
            config=config,
            selected_models=selected_models,
            quick_test=args.quick_test,
            tuned_params_path=args.tuned_params,
            # Smoke tests may use a partial tuning run; the formal run needs all trials
            allow_incomplete_tuning=args.quick_test,
            progress=progress,
        )

    run_info["tracks"] = tracks
    run_info["total_runtime_s"] = time.time() - started
    with open(run_info_file, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print(f"\nRun provenance saved to: {run_info_file}")

    print(
        f"\nAll benchmark tasks completed successfully in {time.time() - started:.1f}s."
    )


if __name__ == "__main__":
    main()

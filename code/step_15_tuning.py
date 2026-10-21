"""Step 15a: Bayesian hyperparameter tuning of the benchmark models trained from scratch.

For each track and each tuned model (SVM, RF, XGBoost, LightGBM, CatBoost, RealMLP,
TabM), Optuna's TPE sampler (seed 42) searches the pre-specified space below for 100
trials. The objective is the mean AUPRC over the same 5 training-set CV folds that the
benchmark uses (in-fold standardization, inner 10% early-stopping split), so tuning
reads the training set only; the locked test set is never loaded.

Checkpointing: every finished trial is stored in an SQLite study
(step_15_tuning_studies.db). A rerun resumes each study from its completed trials. The
study name carries a fingerprint of everything that defines the tuning problem (search
space source, fixed parameters, training data, CV settings, model and CV code, device),
so a changed configuration starts a new study instead of reusing stale trials.
A resumed TPE sampler is re-seeded, so an interrupted run need not reproduce the exact
trial sequence of an uninterrupted one.

Outputs (aggregate only, no row-level data):
- step_15_tuning_best_params.json: best parameters per track and model, with CV AUPRC,
  trial count and fingerprint; step_15_benchmark.py refuses tuned models whose entry is
  missing, incomplete or has a different fingerprint.
- step_15_tuning_trials_{track}.csv: every trial with its parameters and CV metrics.

Usage:
    python step_15_tuning.py                      # all tracks and tuned models
    python step_15_tuning.py --track 7feat --models TabM
    python step_15_tuning.py --n-trials 2 --out-dir <scratch>   # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import optuna
import pandas as pd
import torch

import benchmark_evaluation
import benchmark_models
from benchmark_evaluation import run_cross_validation
from benchmark_models import TUNED_MODELS, create_benchmark_model, file_sha256

CONFIG_FILE = "benchmark_config.json"
STORAGE_FILE = "step_15_tuning_studies.db"
BEST_PARAMS_FILE = "step_15_tuning_best_params.json"

MODEL_CLASSES = {
    "SVM": "SVMBenchmarkModel",
    "RF": "RFBenchmarkModel",
    "XGBoost": "XGBoostBenchmarkModel",
    "LightGBM": "LightGBMBenchmarkModel",
    "CatBoost": "CatBoostBenchmarkModel",
    "RealMLP": "RealMLPBenchmarkModel",
    "TabM": "TabMBenchmarkModel",
}


# =====================================================================
# Pre-specified search spaces
# =====================================================================


def search_space(model_key: str, trial: optuna.Trial) -> dict:
    """
    Hyperparameters sampled for one trial. GBDT tree counts are chosen by early stopping
    (fixed maximum in benchmark_config.json). Ranges follow the parent project where it
    had the same model (without its class weighting); the RealMLP space is the authors'
    default HPO space (pytabkit RealMLPParamSampler); the TabM space follows the TabM paper.

    Revision 2 (2026-09-29): after the first tuning round (training set only; results kept
    in step_15_tuning_round1/), the top trials clustered at artificial range limits, so
    these ranges were widened once: RF min_samples_leaf 1-30 -> 1-100 (log) and
    max_features 0.1 -> 0.03; XGBoost learning_rate 0.005 -> 0.001, max_depth 2 -> 1,
    subsample 0.5 -> 0.3; LightGBM num_leaves 4 -> 2, min_child_samples 5 -> 1; CatBoost
    depth 2 -> 1; TabM learning_rate 5e-3 -> 3e-2, weight_decay 0.1 -> 1.0. Natural limits
    (TabM n_blocks 1, min_samples_leaf 1, sampling fractions 1.0) were left as they are.
    Pre-specified rule: the space is widened only this once; a parameter still at a range
    limit afterwards is accepted.
    """
    if model_key == "SVM":
        params = {
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        }
        if params["kernel"] == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        return params
    if model_key == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 3, 5, 8, 12, 16]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100, log=True),
            "max_features": trial.suggest_float("max_features", 0.03, 1.0),
        }
    if model_key == "XGBoost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if model_key == "LightGBM":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 64, log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 1, 100, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_key == "CatBoost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "depth": trial.suggest_int("depth", 1, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float(
                "random_strength", 1e-3, 10.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        }
    if model_key == "RealMLP":
        return {
            "num_emb_type": trial.suggest_categorical(
                "num_emb_type", ["none", "pbld", "pl", "plr"]
            ),
            "add_front_scale": trial.suggest_categorical(
                "add_front_scale", [True, False]
            ),
            "lr": trial.suggest_float("lr", 2e-2, 3e-1, log=True),
            "p_drop": trial.suggest_categorical("p_drop", [0.0, 0.15, 0.3]),
            "wd": trial.suggest_categorical("wd", [0.0, 2e-2]),
            "plr_sigma": trial.suggest_float("plr_sigma", 0.05, 0.5, log=True),
            "act": trial.suggest_categorical("act", ["relu", "selu", "mish"]),
            "hidden_sizes": trial.suggest_categorical(
                "hidden_sizes", ["256x3", "64x5", "512x1"]
            ),
            "ls_eps": trial.suggest_categorical("ls_eps", [0.0, 0.1]),
        }
    if model_key == "TabM":
        return {
            "d_block": trial.suggest_int("d_block", 64, 512, step=64),
            "n_blocks": trial.suggest_int("n_blocks", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1.0, log=True),
        }
    raise ValueError(f"No search space for {model_key}")


# =====================================================================
# Configuration helpers shared with step_15_benchmark.py
# =====================================================================


def build_model_config(
    config: dict,
    model_key: str,
    tuned_params: dict | None = None,
    overrides: dict | None = None,
) -> dict:
    """Model configuration with the protocol device, tuned parameters and fixed overrides."""
    cfg = json.loads(json.dumps(config["models"][model_key]))
    cfg.setdefault("device", config["protocol"].get("device", "cpu"))
    if tuned_params is not None:
        cfg["tuned_params"] = dict(tuned_params)
    if overrides:
        cfg["fixed_params"] = {**cfg["fixed_params"], **overrides}
    return cfg


def tuning_fingerprint(config: dict, track: str, model_key: str) -> str:
    """Hash of everything that defines one tuning problem."""
    tuning = config["protocol"]["tuning"]
    parts = {
        "model": model_key,
        "track": track,
        "search_space": inspect.getsource(search_space),
        "model_config": config["models"][model_key],
        "tuning_overrides": tuning["tuning_overrides"].get(model_key, {}),
        "objective": tuning["objective"],
        "sampler": [tuning["sampler"], tuning["sampler_seed"]],
        "cv": [config["protocol"]["cv_folds"], config["protocol"]["cv_seed"]],
        "device": config["protocol"].get("device", "cpu"),
        "train_sha256": file_sha256(config["data_files"][f"{track}_track"]["train"]),
        "model_code": inspect.getsource(
            getattr(benchmark_models, MODEL_CLASSES[model_key])
        ),
        "cv_code": inspect.getsource(benchmark_evaluation.run_cross_validation)
        + inspect.getsource(benchmark_evaluation.fit_benchmark_model),
    }
    if model_key == "TabM":
        parts["model_code"] += inspect.getsource(benchmark_models._SingleTabMNetwork)
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_training_data(config: dict, track: str):
    """Training rows only; the locked test set is not read."""
    data_cfg = config["data_files"]
    train = pd.read_csv(data_cfg[f"{track}_track"]["train"])
    if train[data_cfg["subject_column"]].duplicated().any():
        raise RuntimeError(
            f"Participant invariant violated in the {track} training set"
        )
    features = [c for c in train.columns if c not in data_cfg["forbidden_columns"]]
    X = train[features].values.astype(float)
    y = train[data_cfg["target_column"]].values.astype(int)
    subjects = train[data_cfg["subject_column"]].values
    return X, y, subjects, features


# =====================================================================
# Tuning
# =====================================================================


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


@contextmanager
def output_lock(out_dir: str, timeout_s: float = 600.0):
    """
    Exclusive lock for the read-modify-write of the shared output files, so tuning
    processes running in parallel (e.g. one per track) cannot overwrite each other's
    results.
    """
    lock_path = os.path.join(out_dir, "step_15_tuning_outputs.lock")
    started = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - started > timeout_s:
                raise TimeoutError(
                    f"{lock_path} held for over {timeout_s:.0f}s; remove it if "
                    "no other tuning process is running"
                )
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        os.remove(lock_path)


def save_json_atomic(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def export_trials(study: optuna.Study, model_key: str, path: str) -> None:
    rows = []
    for t in study.trials:
        rows.append(
            {
                "Model": model_key,
                "Trial": t.number,
                "State": t.state.name,
                "CV_AUPRC": t.value,
                "CV_AUPRC_std": t.user_attrs.get("cv_auprc_std"),
                "CV_AUROC": t.user_attrs.get("cv_auroc"),
                "CV_Brier": t.user_attrs.get("cv_brier"),
                "Duration_s": t.duration.total_seconds() if t.duration else None,
                "Params": json.dumps(t.params, sort_keys=True),
            }
        )
    new = pd.DataFrame(rows)
    if os.path.isfile(path):
        old = pd.read_csv(path)
        new = pd.concat([old[old["Model"] != model_key], new], ignore_index=True)
    new.to_csv(path, index=False)


def tune_model(
    config: dict, track: str, model_key: str, data, n_trials: int, out_dir: str
) -> None:
    X, y, subjects, _ = data
    tuning = config["protocol"]["tuning"]
    fingerprint = tuning_fingerprint(config, track, model_key)
    overrides = tuning["tuning_overrides"].get(model_key, {})

    storage = f"sqlite:///{os.path.join(out_dir, STORAGE_FILE)}"
    study = optuna.create_study(
        study_name=f"{track}__{model_key}__{fingerprint[:16]}",
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=tuning["sampler_seed"]),
        load_if_exists=True,
    )
    done = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    remaining = max(0, n_trials - done)
    print(
        f"\n[{track}/{model_key}] fingerprint {fingerprint[:16]}: {done} trials done, "
        f"{remaining} to run",
        flush=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = search_space(model_key, trial)
        cfg = build_model_config(
            config, model_key, tuned_params=params, overrides=overrides
        )
        _, cv_mean, cv_std = run_cross_validation(
            model_factory=lambda: create_benchmark_model(
                model_key, cfg, random_state=42
            ),
            X=X,
            y=y,
            subjects=subjects,
            n_folds=config["protocol"]["cv_folds"],
            random_state=config["protocol"]["cv_seed"],
        )
        trial.set_user_attr("cv_auprc_std", cv_std["AUPRC"])
        trial.set_user_attr("cv_auroc", cv_mean["AUROC"])
        trial.set_user_attr("cv_brier", cv_mean["Brier"])
        return cv_mean["AUPRC"]

    started = time.time()

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n_done = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
        n_new = n_done - done
        elapsed = time.time() - started
        eta = elapsed / max(n_new, 1) * max(n_trials - n_done, 0)
        print(
            f"[{track}/{model_key}] trial {n_done}/{n_trials} AUPRC {trial.value:.4f} | "
            f"best {study.best_value:.4f} (trial {study.best_trial.number}) | "
            f"elapsed {format_seconds(elapsed)}, ETA {format_seconds(eta)}",
            flush=True,
        )

    if remaining:
        # A failing trial raises and stops the run; completed trials stay in the study
        study.optimize(objective, n_trials=remaining, callbacks=[progress])

    completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    best = study.best_trial
    best_path = os.path.join(out_dir, BEST_PARAMS_FILE)
    with output_lock(out_dir):
        export_trials(
            study,
            model_key,
            os.path.join(out_dir, f"step_15_tuning_trials_{track}.csv"),
        )
        results = {}
        if os.path.isfile(best_path):
            with open(best_path, encoding="utf-8") as f:
                results = json.load(f)
        results.setdefault(track, {})[model_key] = {
            "params": best.params,
            "cv_auprc": best.value,
            "cv_auprc_std": best.user_attrs.get("cv_auprc_std"),
            "cv_auroc": best.user_attrs.get("cv_auroc"),
            "cv_brier": best.user_attrs.get("cv_brier"),
            "best_trial": best.number,
            "n_trials_completed": len(completed),
            "fingerprint": fingerprint,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json_atomic(results, best_path)
    print(
        f"[{track}/{model_key}] best CV AUPRC {best.value:.4f} (trial {best.number}): "
        f"{json.dumps(best.params)}",
        flush=True,
    )


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
    tuning = config["protocol"]["tuning"]

    parser = argparse.ArgumentParser(
        description="Tune the benchmark models trained from scratch."
    )
    parser.add_argument("--track", choices=["7feat", "full", "both"], default="both")
    parser.add_argument("--models", default=",".join(tuning["models"]))
    parser.add_argument(
        "--n-trials",
        type=int,
        default=tuning["n_trials"],
        help="Override only for smoke tests; the benchmark requires the configured count",
    )
    parser.add_argument(
        "--out-dir", default=".", help="Directory for the study database and outputs"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in TUNED_MODELS]
    if unknown:
        raise ValueError(f"Not tunable: {unknown}")
    if (
        config["protocol"].get("device", "cpu").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "benchmark_config.json requests CUDA but torch.cuda is not available"
        )
    os.makedirs(args.out_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tracks = tuning["tracks"] if args.track == "both" else [args.track]
    started = time.time()
    for track in tracks:
        data = load_training_data(config, track)
        print(
            f"\n=== Track {track}: {len(data[1])} training participants, "
            f"{int(data[1].sum())} PD-MCI, {len(data[3])} features",
            flush=True,
        )
        for model_key in models:
            tune_model(config, track, model_key, data, args.n_trials, args.out_dir)
    print(f"\nTuning finished in {format_seconds(time.time() - started)}.", flush=True)


if __name__ == "__main__":
    main()

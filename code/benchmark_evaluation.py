"""Benchmark Evaluation Utilities for PD-MCI Tabular Deep Learning & Foundation Models.

Provides:
- In-fold leakage-free cross validation
- Comprehensive discrimination, calibration, and classification metrics
- Participant-level paired bootstrap relative to LR
- Comparative publication-quality multi-model plots (ROC, PR, Calibration, DCA)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Callable

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import statsmodels.api as sm

# Set font to Arial
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


# =====================================================================
# Metric Computation
# =====================================================================

def compute_calibration_slope_intercept(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate calibration slope and intercept using standard logistic recalibration:
    logit(y) = intercept + slope * logit(p)
    Ideal values: slope = 1.0, intercept = 0.0
    """
    eps = 1e-6
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(y_prob_clipped / (1.0 - y_prob_clipped))

    X_cal = sm.add_constant(logit_p)
    try:
        glm = sm.GLM(y_true, X_cal, family=sm.families.Binomial())
        res = glm.fit()
        intercept = float(res.params[0])
        slope = float(res.params[1])
    except Exception:
        intercept, slope = float("nan"), float("nan")

    return slope, intercept


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate discrimination, probability calibration, and threshold-based metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # Discrimination
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # Probability accuracy & Calibration
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1.0 - 1e-15))
    slope, intercept = compute_calibration_slope_intercept(y_true, y_prob)

    # Threshold-based metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity & NPV
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Brier": float(brier),
        "LogLoss": float(ll),
        "Calib_Slope": float(slope),
        "Calib_Intercept": float(intercept),
        "Accuracy": float(acc),
        "Balanced_Acc": float(bacc),
        "Sensitivity": float(rec),
        "Specificity": float(spec),
        "PPV": float(prec),
        "NPV": float(npv),
        "F1": float(f1),
        "Threshold": float(threshold),
    }


def find_optimal_youden_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> float:
    """Determine the optimal probability threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = float(thresholds[best_idx])
    # Bound within [0.01, 0.99]
    return float(np.clip(best_threshold, 0.01, 0.99))


# =====================================================================
# Cross-Validation & Test Evaluation Runners
# =====================================================================

def fit_benchmark_model(model: Any, X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """
    Fit a benchmark model on (X, y). Models with early stopping get a stratified 10%
    early-stopping split carved out of (X, y) itself; all other models use every row.
    Rows that will later be scored are therefore never used for early stopping.
    """
    if getattr(model, "uses_early_stopping", False):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
        fit_idx, es_idx = next(skf.split(X, y))
        return model.fit(X_train=X[fit_idx], y_train=y[fit_idx], X_val=X[es_idx], y_val=y[es_idx])
    return model.fit(X_train=X, y_train=y)


def run_cross_validation(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """
    Run stratified K-fold cross-validation with strict in-fold standardization.
    Returns:
    - y_oof_prob: out-of-fold predicted probabilities for training participants
    - cv_mean_metrics: average metrics across folds
    - cv_std_metrics: standard deviation of metrics across folds
    """
    if len(np.unique(subjects)) != len(subjects):
        raise RuntimeError("Single-visit participant invariant violated: duplicate PATNO found")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y), dtype=float)
    fold_metrics_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_f, y_train_f = X[train_idx], y[train_idx]
        X_val_f, y_val_f = X[val_idx], y[val_idx]

        # Strict in-fold standardization
        scaler = StandardScaler()
        X_train_f_scaled = scaler.fit_transform(X_train_f)
        X_val_f_scaled = scaler.transform(X_val_f)

        # The validation fold is only scored, never used for early stopping
        model = model_factory()
        fit_benchmark_model(model, X_train_f_scaled, y_train_f, random_state=random_state)

        val_probs = model.predict_proba(X_val_f_scaled)
        oof_probs[val_idx] = val_probs

        metrics_f = compute_binary_metrics(y_val_f, val_probs, threshold=0.5)
        fold_metrics_list.append(metrics_f)

    # Compute fold mean and std
    fold_df = pd.DataFrame(fold_metrics_list)
    cv_mean = fold_df.mean().to_dict()
    cv_std = fold_df.std().to_dict()

    return oof_probs, cv_mean, cv_std


def run_test_evaluation(
    model_factory: Callable[[], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit model on full training set (standardized once) and evaluate on locked test set.
    Only early-stopping models hold out a 10% training split for early stopping.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = model_factory()
    fit_benchmark_model(model, X_train_scaled, y_train, random_state=random_state)

    test_probs = model.predict_proba(X_test_scaled)
    test_metrics = compute_binary_metrics(y_test, test_probs, threshold=threshold)

    return test_probs, test_metrics


# =====================================================================
# Participant-Level Paired Bootstrap
# =====================================================================

def paired_bootstrap_test_comparison(
    y_test: np.ndarray,
    baseline_probs: np.ndarray,
    model_probs: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute paired bootstrap difference on the same test set participants:
    Delta Metric = Metric(Model) - Metric(Baseline LR)
    Returns:
    - point_diff: point estimate difference
    - ci_low: 2.5th percentile
    - ci_high: 97.5th percentile
    - p_value: two-sided bootstrap p-value
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_test)

    # Point estimates
    base_auroc = roc_auc_score(y_test, baseline_probs)
    base_auprc = average_precision_score(y_test, baseline_probs)
    base_brier = brier_score_loss(y_test, baseline_probs)

    model_auroc = roc_auc_score(y_test, model_probs)
    model_auprc = average_precision_score(y_test, model_probs)
    model_brier = brier_score_loss(y_test, model_probs)

    diff_auroc_point = model_auroc - base_auroc
    diff_auprc_point = model_auprc - base_auprc
    diff_brier_point = model_brier - base_brier

    diffs_auroc = []
    diffs_auprc = []
    diffs_brier = []

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b = y_test[boot_idx]

        # Check if both classes exist in resample
        if len(np.unique(y_b)) < 2:
            continue

        p_base_b = baseline_probs[boot_idx]
        p_model_b = model_probs[boot_idx]

        d_auc = roc_auc_score(y_b, p_model_b) - roc_auc_score(y_b, p_base_b)
        d_pr = average_precision_score(y_b, p_model_b) - average_precision_score(y_b, p_base_b)
        d_br = brier_score_loss(y_b, p_model_b) - brier_score_loss(y_b, p_base_b)

        diffs_auroc.append(d_auc)
        diffs_auprc.append(d_pr)
        diffs_brier.append(d_br)

    diffs_auroc = np.array(diffs_auroc)
    diffs_auprc = np.array(diffs_auprc)
    diffs_brier = np.array(diffs_brier)

    def _summarize_diff(arr: np.ndarray, point: float, higher_is_better: bool = True):
        low, high = np.percentile(arr, [2.5, 97.5])
        # Two-sided empirical p-value for H0: diff == 0
        if higher_is_better:
            p_val = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        else:
            p_val = 2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0))
        p_val = min(1.0, max(0.0, float(p_val)))
        return {
            "diff_point": float(point),
            "ci_lower": float(low),
            "ci_upper": float(high),
            "p_value": float(p_val),
        }

    return {
        "Delta_AUROC": _summarize_diff(diffs_auroc, diff_auroc_point, higher_is_better=True),
        "Delta_AUPRC": _summarize_diff(diffs_auprc, diff_auprc_point, higher_is_better=True),
        "Delta_Brier": _summarize_diff(diffs_brier, diff_brier_point, higher_is_better=False),
    }


# =====================================================================
# Per-Model Bootstrap CIs and Subgroup Metrics
# =====================================================================

CI_METRICS = [
    "AUROC", "AUPRC", "Brier", "LogLoss", "Calib_Slope", "Calib_Intercept",
    "Sensitivity", "Specificity", "PPV", "NPV", "Balanced_Acc", "F1",
]


def bootstrap_indices(n: int, n_bootstrap: int, random_state: int) -> List[np.ndarray]:
    """Participant-level resamples shared by all models, so model CIs are paired."""
    rng = np.random.RandomState(random_state)
    return [rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap)]


def bootstrap_model_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    indices: List[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Point estimate and percentile 95% CI for each metric in CI_METRICS."""
    point = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    draws = {m: [] for m in CI_METRICS}
    for idx in indices:
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        metrics_b = compute_binary_metrics(y_b, y_prob[idx], threshold=threshold)
        for m in CI_METRICS:
            draws[m].append(metrics_b[m])

    summary = {}
    for m in CI_METRICS:
        values = np.asarray(draws[m], dtype=float)
        values = values[~np.isnan(values)]
        low, high = np.percentile(values, [2.5, 97.5]) if len(values) else (np.nan, np.nan)
        summary[m] = {
            "point": float(point[m]),
            "ci_lower": float(low),
            "ci_upper": float(high),
            "n_bootstrap": int(len(values)),
        }
    return summary


# =====================================================================
# Publication Plots: ROC, PR, Calibration, DCA
# =====================================================================

MODEL_COLORS = {
    "LR": "#1f77b4",          # Steel Blue
    "LR_plain": "#7f7f7f",    # Grey
    "SVM": "#bcbd22",         # Olive
    "RF": "#393b79",          # Indigo
    "XGBoost": "#d62728",     # Crimson Red
    "LightGBM": "#ad494a",    # Brick
    "CatBoost": "#ff7f0e",    # Orange
    "RealMLP": "#98df8a",     # Light Green
    "TabM": "#2ca02c",        # Green
    "TabICLv2": "#9467bd",    # Purple
    "TabPFN": "#8c564b",      # Brown
    "TabFM": "#17becf",       # Teal
}

MODEL_LINESTYLES = {
    "LR": "-",
    "LR_plain": ":",
    "SVM": ":",
    "RF": "-.",
    "XGBoost": ":",
    "LightGBM": "-.",
    "CatBoost": "--",
    "RealMLP": "--",
    "TabM": "-",
    "TabICLv2": "-.",
    "TabPFN": "--",
    "TabFM": "-",
}



def plot_benchmark_roc_pr(
    predictions_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    save_path: str = "benchmark_roc_pr.pdf",
    track_title: str = "Full-Feature Analysis (28 Features)",
):
    """Plot multi-model ROC and Precision-Recall comparison curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # 1. ROC Curves
    ax_roc = axes[0]
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUROC = 0.5000)")

    for name, probs in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = roc_auc_score(y_test, probs)
        color = MODEL_COLORS.get(name, "#333333")
        ls = MODEL_LINESTYLES.get(name, "-")
        ax_roc.plot(
            fpr,
            tpr,
            label=f"{name:<9} AUROC = {score:.4f}",
            color=color,
            linestyle=ls,
            linewidth=2.0,
        )

    ax_roc.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=12)
    ax_roc.set_ylabel("Sensitivity (True Positive Rate)", fontsize=12)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.grid(True, alpha=0.25)
    ax_roc.legend(loc="lower right", fontsize=10, frameon=True)
    ax_roc.text(-0.08, 1.05, "A", transform=ax_roc.transAxes, fontsize=15, fontweight="bold")

    # 2. PR Curves
    ax_pr = axes[1]
    prevalence = np.mean(y_test)
    ax_pr.axhline(
        y=prevalence,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline (Prevalence = {prevalence:.4f})",
    )

    for name, probs in predictions_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        score = average_precision_score(y_test, probs)
        color = MODEL_COLORS.get(name, "#333333")
        ls = MODEL_LINESTYLES.get(name, "-")
        ax_pr.plot(
            rec,
            prec,
            label=f"{name:<9} AUPRC = {score:.4f}",
            color=color,
            linestyle=ls,
            linewidth=2.0,
        )

    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=12)
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.grid(True, alpha=0.25)
    ax_pr.legend(loc="upper right", fontsize=10, frameon=True)
    ax_pr.text(-0.08, 1.05, "B", transform=ax_pr.transAxes, fontsize=15, fontweight="bold")

    fig.suptitle(f"Discriminative Performance Benchmark ({track_title})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_calibration(
    predictions_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    save_path: str = "benchmark_calibration.pdf",
    track_title: str = "Full-Feature Analysis (28 Features)",
    n_bins: int = 6,
):
    """Plot multi-model Calibration curves compared to ideal line."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Ideal Perfect Calibration")

    for name, probs in predictions_dict.items():
        # Quantile or uniform bins
        prob_true, prob_pred = sm.tools.eval_measures.calibration_curve(
            y_test, probs, n_bins=n_bins
        ) if hasattr(sm.tools.eval_measures, "calibration_curve") else (
            # Fallback to sklearn calibration_curve
            __import__("sklearn.calibration", fromlist=["calibration_curve"]).calibration_curve(
                y_test, probs, n_bins=n_bins, strategy="uniform"
            )
        )
        slope, intercept = compute_calibration_slope_intercept(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        color = MODEL_COLORS.get(name, "#333333")
        ls = MODEL_LINESTYLES.get(name, "-")
        label = f"{name:<9} Slope={slope:.2f}, Brier={brier:.4f}"
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            markersize=5,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.8,
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Event Frequency", fontsize=12)
    ax.set_title(f"Model Probability Calibration ({track_title})", fontsize=13, fontweight="bold")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_dca(
    predictions_dict: Dict[str, np.ndarray],
    y_test: np.ndarray,
    save_path: str = "benchmark_dca.pdf",
    track_title: str = "Full-Feature Analysis (28 Features)",
):
    """Plot multi-model Decision Curve Analysis (DCA)."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    thresholds = np.linspace(0.02, 0.70, 70)
    n_total = len(y_test)
    prevalence = np.mean(y_test)

    # Treat None
    ax.plot(thresholds, np.zeros_like(thresholds), "k--", label="Treat None (Net Benefit = 0)")

    # Treat All
    net_benefit_all = prevalence - (1 - prevalence) * (thresholds / (1.0 - thresholds))
    ax.plot(thresholds, net_benefit_all, "grey", linestyle=":", label="Treat All")

    for name, probs in predictions_dict.items():
        net_benefits = []
        for pt in thresholds:
            y_pred = (probs >= pt).astype(int)
            tp = np.sum((y_pred == 1) & (y_test == 1))
            fp = np.sum((y_pred == 1) & (y_test == 0))
            nb = (tp / n_total) - (fp / n_total) * (pt / (1.0 - pt))
            net_benefits.append(nb)

        color = MODEL_COLORS.get(name, "#333333")
        ls = MODEL_LINESTYLES.get(name, "-")
        ax.plot(thresholds, net_benefits, label=name, color=color, linestyle=ls, linewidth=2.0)

    ax.set_xlabel("Threshold Probability ($p_t$)", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(f"Decision Curve Analysis ({track_title})", fontsize=13, fontweight="bold")
    ax.set_xlim([0.02, 0.70])
    ax.set_ylim([-0.05, prevalence + 0.05])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=10, frameon=True)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

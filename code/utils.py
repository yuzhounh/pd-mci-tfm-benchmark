"""Shared helpers for model evaluation and figures.

Includes data loading, cross-validation, threshold selection, performance
metrics, and the ROC/PR, SHAP, calibration, decision-curve and clinical-impact
plots.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import shap
from matplotlib.font_manager import FontProperties

# Use Arial for all figures
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"

import figstyle as fs


def load_data(train_file, test_file, feature_mapping_file):
    """Load training set, test set and feature mapping file"""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    feature_mapping = pd.read_csv(feature_mapping_file)

    # Create feature name mapping dictionary
    feature_name_mapping = dict(
        zip(feature_mapping["feature"], feature_mapping["abbreviation"])
    )

    return train_data, test_data, feature_name_mapping


def prepare_data(data, subject_col="PATNO", label_col="MCI"):
    """Prepare data, separate features and labels"""
    # Remove subject ID column
    X = data.drop([subject_col, label_col], axis=1)
    y = data[label_col]
    subjects = data[subject_col]

    return X, y, subjects


def subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=42):
    """Stratified CV for a dataset with one baseline row per participant."""
    subjects = np.asarray(subjects)
    if len(np.unique(subjects)) != len(subjects):
        raise RuntimeError("Cross-validation expected one row per participant")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))


def f1_score_threshold(y_true, y_prob):
    """
    Calculate optimal threshold for maximum F1-score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    f1_scores = f1_scores[:-1]
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]


def youden_threshold(y_true, y_prob):
    """Calculate optimal threshold based on Youden index"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_scores = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(youden_scores)
    return thresholds[optimal_idx]


def calculate_all_metrics(y_true, y_pred, y_prob, threshold=None):
    """Calculate all evaluation metrics"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Return results in specified order
    result = {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "Brier Score": brier_score_loss(y_true, y_prob),
        "Optimal Threshold": threshold if threshold is not None else 0.5,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "NPV": npv,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "Cohen's Kappa": cohen_kappa,
    }

    return result


def save_figure(fig, save_path):
    """Save figure (PDF only)"""
    base, ext = os.path.splitext(save_path)
    if ext.lower() != ".pdf":
        save_path = base + ".pdf"
    fig.savefig(save_path, bbox_inches="tight")


def plot_roc_pr_curves(
    y_true,
    y_prob,
    save_path,
    model_name="Model",
    groups=None,
    n_bootstrap=2000,
    random_state=42,
):
    """Plot ROC and PR curves with row- or group-bootstrap 95% CIs."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    roc_grid = np.linspace(0, 1, 201)
    pr_grid = np.linspace(0, 1, 201)
    roc_ci = pr_ci = None

    if n_bootstrap:
        rng = np.random.RandomState(random_state)
        roc_samples = []
        pr_samples = []

        if groups is None:
            sampling_units = np.arange(len(y_true))
            unit_indices = {unit: np.array([unit]) for unit in sampling_units}
        else:
            groups = np.asarray(groups)
            if len(groups) != len(y_true):
                raise ValueError("groups must have the same length as y_true")
            sampling_units = np.unique(groups)
            unit_indices = {
                group: np.flatnonzero(groups == group) for group in sampling_units
            }

        for _ in range(n_bootstrap):
            sampled_units = rng.choice(
                sampling_units, size=len(sampling_units), replace=True
            )
            indices = np.concatenate([unit_indices[unit] for unit in sampled_units])
            y_boot = y_true[indices]
            p_boot = y_prob[indices]
            if len(np.unique(y_boot)) < 2:
                continue

            fpr_boot, tpr_boot, _ = roc_curve(y_boot, p_boot)
            roc_samples.append(np.interp(roc_grid, fpr_boot, tpr_boot))

            precision_boot, recall_boot, _ = precision_recall_curve(y_boot, p_boot)
            pr_samples.append(
                np.interp(pr_grid, recall_boot[::-1], precision_boot[::-1])
            )

        if roc_samples and pr_samples:
            roc_ci = np.percentile(np.asarray(roc_samples), [2.5, 97.5], axis=0)
            pr_ci = np.percentile(np.asarray(pr_samples), [2.5, 97.5], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── ROC curve ──────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    curve_color = "#1f77b4"
    h1 = [
        ax1.plot(fpr, tpr, color=curve_color, linewidth=2)[0],
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)[0],
    ]
    n1, v1 = [model_name, "Random"], [f"{auc_roc:.4f}", f"{0.5:.4f}"]
    if roc_ci is not None:
        ax1.fill_between(
            roc_grid,
            roc_ci[0],
            roc_ci[1],
            color=curve_color,
            alpha=0.18,
            linewidth=0,
        )
        h1.append(ax1.collections[-1])
        n1.append("95% CI")
        v1.append("")

    ax1.set_xlabel("1 - Specificity", fontsize=12)
    ax1.set_ylabel("Sensitivity", fontsize=12)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, alpha=0.3)

    fs.table_legend(
        ax1, h1, n1, [("AUROC", v1)], "Model", fontsize=9, loc="lower right"
    )

    ax1.text(
        -0.09,
        1.05,
        "A",
        transform=ax1.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    # ── PR curve ───────────────────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)

    h2 = [
        ax2.plot(recall, precision, color=curve_color, linewidth=2)[0],
        ax2.axhline(y=baseline, color="k", linestyle="--", alpha=0.5),
    ]
    n2, v2 = [model_name, "Random"], [f"{auc_pr:.4f}", f"{baseline:.4f}"]
    if pr_ci is not None:
        ax2.fill_between(
            pr_grid,
            pr_ci[0],
            pr_ci[1],
            color=curve_color,
            alpha=0.18,
            linewidth=0,
        )
        h2.append(ax2.collections[-1])
        n2.append("95% CI")
        v2.append("")

    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.grid(True, alpha=0.3)

    fs.table_legend(
        ax2, h2, n2, [("AUPRC", v2)], "Model", fontsize=9, loc="upper right"
    )

    ax2.text(
        -0.09,
        1.05,
        "B",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_shap_summary(shap_values, X, feature_names, model_name, save_path):
    """Plot SHAP summary plots"""
    import os

    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]

    # # Bar plot
    # fig = plt.figure(figsize=(10, 8))
    # shap.summary_plot(
    #     shap_values, X, feature_names=feature_names, plot_type="bar", show=False
    # )
    # plt.title(f"{model_name} - SHAP Feature Importance (Bar)")
    # plt.tight_layout()
    # save_figure(fig, os.path.join(base_dir, f"{base_name}_bar.pdf"))
    # plt.close()

    # Summary plot
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f"{model_name}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, os.path.join(base_dir, f"{base_name}_summary.pdf"))
    plt.close()

    # # Save SHAP values data
    # shap_df = pd.DataFrame(shap_values, columns=feature_names)
    # shap_df.to_csv(os.path.join(base_dir, f"{base_name}_values.csv"), index=False)


def get_permutation_importance(model, X, y, feature_names, random_state=42):
    """Calculate permutation importance"""
    perm_importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=random_state, scoring="roc_auc"
    )
    return perm_importance.importances_mean


def print_results(model_name, metrics, optimal_threshold):
    """Print results"""
    print(f"\n{'='*50}")
    print(f"{model_name} Model Results")
    print(f"{'='*50}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    print(f"\nConfusion Matrix:")
    print(metrics["Confusion Matrix"])

    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        if metric != "Confusion Matrix":
            print(f"{metric}: {value}")


def plot_nomogram(coefficients, intercept, feature_names, save_path="nomogram.pdf"):
    """
    Plot a nomogram.

    Parameters:
        coefficients: model coefficients
        intercept: intercept
        feature_names: list of feature names
        save_path: output path
    """
    fig, ax = plt.subplots(figsize=(12, len(feature_names) * 0.8 + 4))

    # Point range of each feature
    n_features = len(feature_names)

    # Point range (features assumed standardised, roughly -3 to 3)
    feature_range = np.linspace(-3, 3, 100)

    # One row per feature
    y_pos = np.arange(n_features + 2, 0, -1)

    # 1. Total-points scale
    ax.plot([0, 100], [y_pos[0], y_pos[0]], "k-", linewidth=2)
    for i in range(0, 101, 10):
        ax.plot([i, i], [y_pos[0] - 0.1, y_pos[0] + 0.1], "k-", linewidth=1)
        ax.text(i, y_pos[0] + 0.3, str(i), ha="center", fontsize=10)
    ax.text(50, y_pos[0] + 0.8, "Total Points", ha="center", fontsize=12)

    # 2. A scale for each feature
    max_abs_coef = np.max(np.abs(coefficients))

    for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
        y = y_pos[i + 1]

        # Point contribution of the feature (normalised to 0-100)
        points = (feature_range * coef / max_abs_coef) * 50 + 50

        # Feature scale line
        ax.plot([0, 100], [y, y], "k-", linewidth=1.5)

        # Tick marks
        for j in range(0, 101, 20):
            # Corresponding feature value
            feature_val = (j - 50) / 50 * max_abs_coef / coef if coef != 0 else 0
            ax.plot([j, j], [y - 0.1, y + 0.1], "k-", linewidth=1)
            ax.text(j, y - 0.4, f"{feature_val:.2f}", ha="center", fontsize=8)

        # Feature name
        ax.text(-5, y, name, ha="right", va="center", fontsize=10)

    # 3. Predicted-probability scale
    y = y_pos[-1]
    ax.plot([0, 100], [y, y], "k-", linewidth=2)

    # Points corresponding to each probability
    prob_range = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    logit_range = np.log(prob_range / (1 - prob_range))

    # Normalise to 0-100
    logit_min = np.log(0.01 / 0.99)
    logit_max = np.log(0.99 / 0.01)
    points_range = ((logit_range - logit_min) / (logit_max - logit_min)) * 100

    for prob, point in zip(prob_range, points_range):
        if 0 <= point <= 100:
            ax.plot([point, point], [y - 0.1, y + 0.1], "k-", linewidth=1)
            ax.text(point, y - 0.4, f"{prob:.1f}", ha="center", fontsize=8)

    ax.text(50, y + 0.8, "Predicted Probability", ha="center", fontsize=12)

    # Axes settings
    ax.set_xlim(-10, 105)
    ax.set_ylim(0, n_features + 3)
    ax.axis("off")
    ax.set_title("Nomogram for Logistic Regression", fontsize=14, pad=20)

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_decision_curve_analysis(y_true, y_prob, save_path="decision_curve.pdf"):
    """
    Plot a decision curve analysis (DCA).

    Parameters:
        y_true: observed labels
        y_prob: predicted probabilities
        save_path: output path
    """
    # Threshold probabilities
    thresholds = np.linspace(0.01, 0.99, 100)

    net_benefits = []
    net_benefit_all = []

    for threshold in thresholds:
        # Classify at the threshold
        y_pred = (y_prob >= threshold).astype(int)

        # True and false positives
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)

        # Net benefit
        net_benefit = (TP / n) - (FP / n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)

        # Net benefit of treating all
        prevalence = np.sum(y_true) / n
        net_benefit_treat_all = prevalence - (1 - prevalence) * (
            threshold / (1 - threshold)
        )
        net_benefit_all.append(net_benefit_treat_all)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))

    # Model decision curve
    ax.plot(thresholds, net_benefits, label="Model", linewidth=2, color="#2E86AB")

    # Treat-all curve
    ax.plot(
        thresholds,
        net_benefit_all,
        label="Treat All",
        linewidth=2,
        linestyle="--",
        color="#A23B72",
    )

    # Treat-none curve (net benefit 0)
    ax.axhline(y=0, color="#F18F01", linestyle="--", linewidth=2, label="Treat None")

    # Axes settings
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Decision Curve Analysis", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_decision_curve_analysis_v2(y_true, y_prob, save_path="LR_decision_curve.pdf"):
    """
    Plot a decision curve analysis (DCA).

    Three curves:
        1. Model:      nb = TP/n - (FP/n) * (pt / (1-pt))
        2. Treat All:  nb = prevalence - (1-prevalence) * (pt / (1-pt))
        3. Treat None: nb = 0
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    prevalence = np.mean(y_true)

    thresholds = np.arange(0, 1.00, 0.01)
    nb_model = np.zeros(len(thresholds))
    nb_treat_all = np.zeros(len(thresholds))

    for idx, pt in enumerate(thresholds):
        if pt >= 1.0:
            nb_model[idx] = 0.0
            nb_treat_all[idx] = -np.inf
            continue

        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        nb_model[idx] = tp / n - (fp / n) * (pt / (1 - pt))
        nb_treat_all[idx] = prevalence - (1 - prevalence) * (pt / (1 - pt))

    nb_treat_none = np.zeros(len(thresholds))

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(
        thresholds,
        nb_model,
        color="#E64B35",
        linewidth=2.0,
        linestyle="-",
        label="Model",
    )
    ax.plot(
        thresholds,
        nb_treat_all,
        color="#4DBBD5",
        linewidth=2.0,
        linestyle="--",
        label="Treat All",
    )
    ax.plot(
        thresholds,
        nb_treat_none,
        color="black",
        linewidth=1.5,
        linestyle="-",
        label="Treat None",
    )

    # y-axis range
    valid = np.isfinite(nb_treat_all)
    y_max = max(np.nanmax(nb_model), np.nanmax(nb_treat_all[valid]), 0) + 0.05
    ax.set_ylim(-0.1, y_max)
    ax.set_xlim(0, 1)

    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis (DCA)")

    ax.legend(loc="upper right")
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

    plt.tight_layout()
    save_figure(fig, save_path)
    print(f"Decision curve saved to: {save_path}")


def plot_calibration_curve_analysis(
    y_true, y_prob, save_path="calibration_curve.pdf", n_bins=10
):
    """
    Plot the calibration curve.

    Parameters:
        y_true: observed labels
        y_prob: predicted probabilities
        save_path: output path
        n_bins: number of bins
    """
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    # Brier score
    brier_score = brier_score_loss(y_true, y_prob)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: calibration curve
    ax1.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        color="gray",
        label="Perfectly Calibrated",
        alpha=0.7,
    )
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#E64B35",
        label="Model",
    )

    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Annotate the Brier score
    ax1.text(
        0.95,
        0.05,
        f"Brier Score: {brier_score:.4f}",
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    # Right: histogram of predicted probabilities
    ax2.hist(
        y_prob[y_true == 0],
        bins=20,
        alpha=0.6,
        color="#4DBBD5",
        label="Negative Class",
        edgecolor="black",
    )
    ax2.hist(
        y_prob[y_true == 1],
        bins=20,
        alpha=0.6,
        color="#E64B35",
        label="Positive Class",
        edgecolor="black",
    )

    ax2.set_xlabel("Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_clinical_impact_curve(y_true, y_prob, save_path="clinical_impact_curve.pdf"):
    """
    Plot the clinical impact curve.

    Parameters:
        y_true: observed labels
        y_prob: predicted probabilities
        save_path: output path
    """
    # Threshold probabilities
    thresholds = np.linspace(0.01, 0.99, 100)

    n = len(y_true)
    n_high_risk = []  # number classified as high risk
    n_high_risk_with_outcome = []  # number of high-risk cases with the outcome

    for threshold in thresholds:
        # Classify at the threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Number classified as high risk (per 1000)
        n_classified_high_risk = np.sum(y_pred) / n * 1000
        n_high_risk.append(n_classified_high_risk)

        # Number of true positives among them (per 1000)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        n_tp = TP / n * 1000
        n_high_risk_with_outcome.append(n_tp)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot the number classified as high risk
    ax.plot(
        thresholds,
        n_high_risk,
        label="Number High Risk",
        linewidth=2.5,
        color="#E63946",
        linestyle="-",
    )

    # Plot the number of true positives
    ax.plot(
        thresholds,
        n_high_risk_with_outcome,
        label="Number High Risk with Outcome",
        linewidth=2.5,
        color="#2A9D8F",
        linestyle="--",
    )

    # Axes settings
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Number per 1000 Visits", fontsize=12)
    ax.set_title("Clinical Impact Curve", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(n_high_risk) * 1.1])

    # Explanatory text
    prevalence = np.sum(y_true) / n
    ax.text(
        0.95,
        0.07,
        f"Prevalence: {prevalence:.2%}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def plot_combined_analysis(
    y_true, y_prob, save_path="combined_analysis.pdf", n_bins=10
):
    """
    Combine the calibration curve, probability distribution, DCA and CIC in a 2x2 grid.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    prevalence = np.mean(y_true)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_cal, ax_hist, ax_dca, ax_cic = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Top left: calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    brier_score = brier_score_loss(y_true, y_prob)

    ax_cal.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="gray",
        label="Perfectly Calibrated",
        alpha=0.7,
    )
    ax_cal.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        linewidth=1.5,
        markersize=4,
        color="#E64B35",
        label="Model",
    )
    ax_cal.set_xlabel("Mean Predicted Probability")
    ax_cal.set_ylabel("Fraction of Positives")
    ax_cal.set_title("Calibration Curve")
    ax_cal.legend(loc="upper left", fontsize=9)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.set_xlim([0, 1])
    ax_cal.set_ylim([0, 1])
    ax_cal.text(
        0.95,
        0.05,
        f"Brier Score: {brier_score:.4f}",
        transform=ax_cal.transAxes,
        fontsize=10,
        va="bottom",
        ha="right",
    )

    # Top right: distribution of predicted probabilities
    ax_hist.hist(
        y_prob[y_true == 0],
        bins=20,
        alpha=0.6,
        color="#4DBBD5",
        label="Negative Class",
        edgecolor="black",
    )
    ax_hist.hist(
        y_prob[y_true == 1],
        bins=20,
        alpha=0.6,
        color="#E64B35",
        label="Positive Class",
        edgecolor="black",
    )
    ax_hist.set_xlabel("Predicted Probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Predicted Probability Distribution")
    ax_hist.legend(loc="upper right", fontsize=9)
    ax_hist.grid(True, alpha=0.3, axis="y")

    # Bottom left: decision curve
    thresholds = np.arange(0, 1.00, 0.01)
    nb_model = np.zeros(len(thresholds))
    nb_treat_all = np.zeros(len(thresholds))

    for idx, pt in enumerate(thresholds):
        if pt >= 1.0:
            nb_model[idx] = 0.0
            nb_treat_all[idx] = -np.inf
            continue
        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        nb_model[idx] = tp / n - (fp / n) * (pt / (1 - pt))
        nb_treat_all[idx] = prevalence - (1 - prevalence) * (pt / (1 - pt))

    valid = np.isfinite(nb_treat_all)
    y_max = max(np.nanmax(nb_model), np.nanmax(nb_treat_all[valid]), 0) + 0.05

    ax_dca.plot(thresholds, nb_model, color="#E64B35", linewidth=1.5, label="Model")
    ax_dca.plot(
        thresholds,
        nb_treat_all,
        color="#0088AA",
        linewidth=1.5,
        linestyle="--",
        label="Treat All",
    )
    ax_dca.plot(
        thresholds,
        np.zeros_like(thresholds),
        color="black",
        linewidth=1.0,
        label="Treat None",
    )
    ax_dca.set_ylim(-0.1, y_max)
    ax_dca.set_xlim(0, 1)
    ax_dca.set_xlabel("Threshold Probability")
    ax_dca.set_ylabel("Net Benefit")
    ax_dca.set_title("Decision Curve Analysis")
    ax_dca.legend(loc="upper right", fontsize=9)
    ax_dca.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)

    # Bottom right: clinical impact curve
    thresholds_cic = np.linspace(0.01, 0.99, 100)
    n_high_risk, n_high_risk_with_outcome = [], []

    for pt in thresholds_cic:
        y_pred = (y_prob >= pt).astype(int)
        n_high_risk.append(np.sum(y_pred) / n * 1000)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        n_high_risk_with_outcome.append(tp / n * 1000)

    ax_cic.plot(
        thresholds_cic,
        n_high_risk,
        label="Number High Risk",
        linewidth=1.5,
        color="#E63946",
    )
    ax_cic.plot(
        thresholds_cic,
        n_high_risk_with_outcome,
        label="Number High Risk with Outcome",
        linewidth=1.5,
        color="#0088AA",
        linestyle="--",
    )
    ax_cic.set_xlabel("Threshold Probability")
    ax_cic.set_ylabel("Number per 1000 Visits")
    ax_cic.set_title("Clinical Impact Curve")
    ax_cic.legend(loc="upper right", fontsize=9)
    ax_cic.grid(True, alpha=0.3)
    ax_cic.set_xlim([0, 1])
    ax_cic.set_ylim([0, max(n_high_risk) * 1.1])
    ax_cic.text(
        0.95,
        0.05,
        f"Prevalence: {prevalence:.2%}",
        transform=ax_cic.transAxes,
        fontsize=10,
        va="bottom",
        ha="right",
    )

    # Panel letters
    for ax, label in zip(axes.flat, ["A", "B", "C", "D"]):
        ax.text(
            -0.08,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()
    print(f"Combined analysis saved to: {save_path}")

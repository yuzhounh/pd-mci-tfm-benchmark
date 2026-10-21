"""Step 13: unpenalised logistic regression on the LASSO-selected predictors.

Fits the model on the training set, evaluates it on the test set with bootstrap
confidence intervals, exports the odds-ratio table, performance metrics and
figures, and writes model_parameters.json. Uniform shrinkage is applied in
step 14.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.calibration import calibration_curve
import statsmodels.api as sm
import shap
from utils import *
import warnings

warnings.filterwarnings("ignore")

# Use Arial
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


class LRModel:
    """Fit, evaluate and export the logistic regression model."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.optimal_threshold = None

    @staticmethod
    def _format_p_value(p: float) -> str:
        if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
            return ""
        if p < 0.001:
            return "<0.001"
        return f"{p:.4f}"

    @staticmethod
    def _format_or_ci(or_value: float, ci_low: float, ci_high: float) -> str:
        if any(
            v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
            for v in [or_value, ci_low, ci_high]
        ):
            return ""
        # Join the CI limits with an en dash and keep 3 decimals
        return f"{or_value:.4f} ({ci_low:.4f}–{ci_high:.4f})"

    def export_or_ci_p_table(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        out_csv_path: str = "step_13_lr_or_table.csv",
        include_intercept: bool = False,
    ) -> pd.DataFrame:
        """
        Fit a statsmodels logistic regression with model-based SE and export a
        clinically-prioritized medical coefficient table: Variable, OR (95% CI), p, β, SE, z.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y, name="Outcome")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        X_sm = sm.add_constant(X, has_constant="add")

        glm = sm.GLM(y.astype(float), X_sm.astype(float), family=sm.families.Binomial())
        res = glm.fit()

        self.fitted_glm = res
        self.statsmodels_params = res.params.to_dict()
        self.statsmodels_bse = res.bse.to_dict()
        self.statsmodels_pvalues = res.pvalues.to_dict()

        params = res.params
        ses = res.bse
        pvals = res.pvalues
        zvals = res.tvalues  # for a binomial GLM, tvalues are Wald z statistics

        # OR and 95% CI
        z_score = 1.96
        or_vals = np.exp(params)
        ci_low = np.exp(params - z_score * ses)
        ci_high = np.exp(params + z_score * ses)

        rows = []
        for name in params.index.tolist():
            if (name == "const") and (not include_intercept):
                continue

            display_name = "Intercept" if name == "const" else name

            # Clinical column order, with controlled precision for all exported values
            rows.append(
                {
                    "Variable": display_name,
                    "OR (95% CI)": self._format_or_ci(
                        float(or_vals[name]), float(ci_low[name]), float(ci_high[name])
                    ),
                    "p": self._format_p_value(float(pvals[name])),
                    "β": f"{float(params[name]):.4f}",
                    "SE": f"{float(ses[name]):.4f}",
                    "z": f"{float(zvals[name]):.4f}",
                }
            )

        table = pd.DataFrame(rows)

        # Keep the same feature order used in the main pipeline
        if not include_intercept:
            order = {fn: i for i, fn in enumerate(feature_names)}
            table["_order"] = table["Variable"].map(order).fillna(1e9)
            table = table.sort_values(["_order", "Variable"]).drop(columns=["_order"])
        else:
            order = {"Intercept": -1, **{fn: i for i, fn in enumerate(feature_names)}}
            table["_order"] = table["Variable"].map(order).fillna(1e9)
            table = table.sort_values(["_order", "Variable"]).drop(columns=["_order"])

        # Save as CSV
        table.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

        # Print to the terminal
        print("\n" + "=" * 90)
        print(
            "Logistic Regression Coefficients (Single-Visit Analysis) - Medical Journal Style"
        )
        print("=" * 90)
        print(
            table[["Variable", "OR (95% CI)", "p", "β", "SE", "z"]].to_string(
                index=False
            )
        )
        print(f"\nCoefficient table saved to: {out_csv_path}")

        return table

    def export_model_parameters_json(
        self,
        out_json_path: str = "model_parameters.json",
        feature_names: list[str] = None,
        feature_name_mapping: dict = None,
        bootstrap_ci_path: str = "step_13_lr_bootstrap_ci.csv",
    ):
        """
        Export the full-precision, machine-readable model parameter file model_parameters.json,
        the single source of the model for Python, the R nomogram and the web calculator.
        """
        if not hasattr(self, "statsmodels_params"):
            print(
                "Warning: statsmodels parameters not available, skipping JSON export."
            )
            return

        intercept = float(self.statsmodels_params.get("const", 0.0))
        coefs = {}
        for k, v in self.statsmodels_params.items():
            if k != "const":
                coefs[k] = float(v)

        selected_feature_mapping = {
            name: (feature_name_mapping or {}).get(name, name) for name in coefs
        }

        # Scale limits based on clinical instrument definitions
        scale_metadata = {
            "Sex": {"min": 0, "max": 1, "description": "0=Female, 1=Male"},
            "Education": {"min": 0, "max": 20, "unit": "years", "upper_cap": 20},
            "GDS": {"min": 0, "max": 15, "description": "15-item short form, 0-15"},
            "PIGD": {
                "min": 0.0,
                "max": 4.0,
                "description": "MDS-UPDRS PIGD score (OFF state)",
            },
            "H&Y": {
                "min": 0,
                "max": 5,
                "description": "Modified Hoehn and Yahr stage, 0-5",
            },
            "UPDRS-III": {
                "min": 0,
                "max": 132,
                "description": "MDS-UPDRS Part III motor score (OFF state)",
            },
        }

        # Read bootstrap performance metrics if available
        metrics_summary = {}
        if os.path.exists(bootstrap_ci_path):
            try:
                b_df = pd.read_csv(bootstrap_ci_path)
                for _, row in b_df.iterrows():
                    m = row["Metric"]
                    metrics_summary[m] = {
                        "point_estimate": float(row["Point_Estimate"]),
                        "ci_lower": float(row["CI_Lower"]),
                        "ci_upper": float(row["CI_Upper"]),
                    }
            except Exception as e:
                print(f"Could not parse bootstrap CI for json: {e}")

        model_dict = {
            "model_type": "Unpenalized Multivariable Logistic Regression (Baseline Single-Visit)",
            "target": "PD-MCI",
            "outcome_definition": "PPMI-derived operational PD-MCI indicator",
            "intercept": intercept,
            "coefficients": coefs,
            "feature_order": [f for f in (feature_names or list(coefs.keys()))],
            "feature_abbreviations": selected_feature_mapping,
            "thresholds": {
                "default": 0.5,
                "youden": round(float(getattr(self, "youden_median", 0.5)), 4),
                "f1": round(float(getattr(self, "f1_median", 0.5)), 4),
            },
            "clinical_notes": {
                "motor_assessment_state": "H&Y, MDS-UPDRS Part III, and PIGD use OFF-medication assessments in treated participants or assessments in untreated participants.",
                "intended_use": "Baseline participant-level risk stratification aid for exploratory screening, not a standalone diagnostic confirmation.",
            },
            "scale_ranges": scale_metadata,
            "test_performance_bootstrap": metrics_summary,
        }

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(model_dict, f, indent=2, ensure_ascii=False)

        print(f"\nUnified model parameters exported to: {out_json_path}")

    def create_model(self):
        """Create an unpenalised logistic regression model."""
        model_params = {
            "penalty": None,
            # Converges to the same full-precision maximum-likelihood solution as the statsmodels GLM, so
            # the saved predictions, OR table, nomogram and model_parameters.json share one set of estimates.
            "solver": "newton-cholesky",
            "tol": 1e-12,
            "max_iter": 10000,
            # 'class_weight': 'balanced',
            "random_state": self.random_state,
        }

        return LogisticRegression(**model_params)

    def train_final_model(self, X, y, subjects):
        """Fit the final model on the full training set."""
        print("Training LR model without penalty...")

        # Create the model
        self.model = self.create_model()

        # Fit the model
        self.model.fit(X, y)

        # Optimal thresholds
        folds = subject_level_cross_validation(
            X, y, subjects, n_folds=10, random_state=self.random_state
        )
        youden_thresholds = []
        f1_thresholds = []

        for train_idx, val_idx in folds:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Fit the model
            fold_model = self.create_model()
            fold_model.fit(X_train_fold, y_train_fold)

            # Predicted probabilities
            y_val_prob = fold_model.predict_proba(X_val_fold)[:, 1]

            # Optimal thresholds
            youden_thresh = youden_threshold(y_val_fold, y_val_prob)
            f1_thresh = f1_score_threshold(y_val_fold, y_val_prob)

            youden_thresholds.append(youden_thresh)
            f1_thresholds.append(f1_thresh)

        # Median thresholds
        self.default = 0.5
        self.f1_median = np.median(f1_thresholds)
        self.youden_median = np.median(youden_thresholds)

        # Print the thresholds
        print("\n" + "=" * 60)
        print("Optimal threshold calculation results:")
        print("=" * 60)
        print(f"1. Default threshold:   0.5000")
        print(f"2. F1-score:   {self.f1_median:.4f}")
        print(f"3. Youden index: {self.youden_median:.4f}")
        print("=" * 60)

        # Show the thresholds of each fold
        print("\nThreshold details by fold:")
        print("Fold\tYouden\tF1-score")
        print("-" * 35)
        for i, (y_thresh, f_thresh) in enumerate(
            zip(youden_thresholds, f1_thresholds), 1
        ):
            print(f"{i:2d}\t{y_thresh:.4f}\t{f_thresh:.4f}")
        print("-" * 35)
        print(f"Median\t{self.youden_median:.4f}\t{self.f1_median:.4f}")
        print()

    def get_cross_validation_performance(self, X, y, subjects):
        """Cross-validated performance on the training set."""
        print("Calculating training set cross-validation performance...")

        folds = subject_level_cross_validation(
            X, y, subjects, n_folds=10, random_state=self.random_state
        )

        fold_results = {"default": [], "f1": [], "youden": []}

        youden_thresholds = []
        f1_thresholds = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Create and fit the model
            fold_model = self.create_model()
            fold_model.fit(X_train_fold, y_train_fold)

            # Predicted probabilities
            y_val_prob = fold_model.predict_proba(X_val_fold)[:, 1]

            # Optimal thresholds
            f1_thresh = f1_score_threshold(y_val_fold, y_val_prob)
            youden_thresh = youden_threshold(y_val_fold, y_val_prob)

            f1_thresholds.append(f1_thresh)
            youden_thresholds.append(youden_thresh)

            # Metrics at the three thresholds
            thresholds = {"default": 0.5, "f1": f1_thresh, "youden": youden_thresh}

            for thresh_type, threshold in thresholds.items():
                y_pred = (y_val_prob >= threshold).astype(int)
                metrics = calculate_all_metrics(
                    y_val_fold, y_pred, y_val_prob, threshold
                )
                fold_results[thresh_type].append(metrics)

        # Mean and SD for each threshold
        summary_results = {}

        for thresh_type in ["default", "f1", "youden"]:
            metrics_df = pd.DataFrame(fold_results[thresh_type])
            means = metrics_df.mean()
            stds = metrics_df.std()

            summary_results[f"train_{thresh_type}_mean"] = means.to_dict()
            summary_results[f"train_{thresh_type}_std"] = stds.to_dict()

        return summary_results

    def evaluate_feature_importance(self, X, y, feature_names, save_path):
        """Evaluate feature importance."""
        print("Calculating LR feature importance...")

        sample_size = min(500, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]

        explainer = shap.LinearExplainer(self.model, X_sample)
        shap_values = explainer.shap_values(X_sample)

        plot_shap_summary(shap_values, X_sample, feature_names, "LR", save_path)

        if hasattr(self.model, "coef_"):
            coef_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": self.model.coef_[0],
                    "abs_coefficient": np.abs(self.model.coef_[0]),
                }
            )
            coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
            intercept_row = pd.DataFrame(
                {
                    "feature": ["Intercept"],
                    "coefficient": [self.model.intercept_[0]],
                    "abs_coefficient": [np.abs(self.model.intercept_[0])],
                }
            )
            coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
            coef_save_path = "step_13_lr_feature_importance.csv"
            coef_df.to_csv(coef_save_path, index=False)
            print(f"LR coefficients saved to: {coef_save_path}")

        return shap_values

    def evaluate_and_save_results(self, X_test, y_test, save_path="lr_predictions.csv"):
        """Evaluate performance on the test set."""
        print("Calculating test set performance...")
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Test-set performance at the three thresholds
        test_performance = {}

        # Default threshold
        y_pred_default = (y_prob >= self.default).astype(int)
        test_performance["test_default"] = calculate_all_metrics(
            y_test, y_pred_default, y_prob, self.default
        )

        # F1 threshold
        y_pred_f1 = (y_prob >= self.f1_median).astype(int)
        test_performance["test_f1"] = calculate_all_metrics(
            y_test, y_pred_f1, y_prob, self.f1_median
        )

        # Youden threshold
        y_pred_youden = (y_prob >= self.youden_median).astype(int)
        test_performance["test_youden"] = calculate_all_metrics(
            y_test, y_pred_youden, y_prob, self.youden_median
        )

        # Save the predictions
        predictions_data = pd.DataFrame({"y_true": y_test, "y_prob": y_prob})
        predictions_data.to_csv(save_path, index=False)
        print(f"Predictions saved to: {save_path}")

        return test_performance, y_prob

    def bootstrap_test_performance(
        self,
        X_test,
        y_test,
        y_prob,
        n_bootstrap=2000,
        ci_alpha=0.05,
        save_path="step_13_lr_bootstrap_ci.csv",
    ):
        """
        Bootstrap confidence intervals on the test set.

        Each patient has one baseline record, so independent participant rows are resampled with replacement.
        Percentile 95% CIs are computed over n_bootstrap resamples.
        Threshold-dependent metrics use the pre-specified default, F1-optimal and
        Youden-optimal thresholds; thresholds are not re-selected in the resamples.

        Parameters
        ----------
        X_test : ndarray, shape (n_participants, n_features)
        y_test : ndarray, shape (n_participants,)
        y_prob : ndarray, shape (n_participants,)
            Predicted probabilities.
        n_bootstrap : int
            Number of bootstrap resamples.
        ci_alpha : float
            Significance level (default 0.05, i.e. 95% CI).
        save_path : str
            Output CSV path.
        """
        from sklearn.linear_model import LogisticRegression as LR_calibration
        from sklearn.metrics import brier_score_loss

        print(f"\n{'=' * 80}")
        print(f"PARTICIPANT-LEVEL BOOTSTRAP ({n_bootstrap} iterations)")
        print(f"{'=' * 80}")

        n_participants = len(y_test)

        rng = np.random.RandomState(42)

        thresholds = {
            "Default": self.default,
            "F1": self.f1_median,
            "Youden": self.youden_median,
        }
        classification_metric_names = [
            "Accuracy",
            "Balanced_Accuracy",
            "Precision",
            "NPV",
            "Sensitivity",
            "Specificity",
            "F1_Score",
            "Cohen_Kappa",
        ]

        def _classification_metrics(y_true_values, y_prob_values, threshold):
            y_pred_values = (y_prob_values >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_true_values, y_pred_values, labels=[0, 1]
            ).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
            f1_denominator = 2 * tp + fp + fn
            return {
                "Accuracy": (tp + tn) / (tp + tn + fp + fn),
                "Balanced_Accuracy": np.nanmean([sensitivity, specificity]),
                "Precision": precision,
                "NPV": npv,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "F1_Score": (2 * tp / f1_denominator if f1_denominator > 0 else np.nan),
                "Cohen_Kappa": cohen_kappa_score(y_true_values, y_pred_values),
            }

        metric_names = [
            "AUROC",
            "AUPRC",
            "Brier",
            "Calibration_Slope",
            "Joint_Recalibration_Intercept",
        ]
        metric_names.extend(
            f"{metric}_{threshold_name}"
            for threshold_name in thresholds
            for metric in classification_metric_names
        )
        bootstrap_results = {m: [] for m in metric_names}

        for b in range(n_bootstrap):
            # Resample participant rows with replacement
            indices = rng.choice(n_participants, size=n_participants, replace=True)

            y_true_b = y_test[indices]
            y_prob_b = y_prob[indices]

            # Skip degenerate resamples (all positive or all negative)
            if len(np.unique(y_true_b)) < 2:
                continue

            # 3. Metrics
            bootstrap_results["AUROC"].append(roc_auc_score(y_true_b, y_prob_b))
            bootstrap_results["AUPRC"].append(
                average_precision_score(y_true_b, y_prob_b)
            )
            bootstrap_results["Brier"].append(brier_score_loss(y_true_b, y_prob_b))

            # Joint logistic recalibration: logit[P(Y=1)] = a + b * logit(p), estimating a and b together.
            # Here a is the joint recalibration intercept, not calibration-in-the-large with b fixed at 1.
            try:
                logit_prob = np.log(
                    np.clip(y_prob_b, 1e-10, 1 - 1e-10)
                    / (1 - np.clip(y_prob_b, 1e-10, 1 - 1e-10))
                )
                cal_model = LR_calibration(penalty=None, solver="lbfgs", max_iter=5000)
                cal_model.fit(logit_prob.reshape(-1, 1), y_true_b)
                bootstrap_results["Calibration_Slope"].append(cal_model.coef_[0][0])
                bootstrap_results["Joint_Recalibration_Intercept"].append(
                    cal_model.intercept_[0]
                )
            except Exception:
                pass  # extreme resamples can make the calibration fit fail

            # Classification metrics at the three pre-specified thresholds
            for threshold_name, threshold in thresholds.items():
                metrics_b = _classification_metrics(y_true_b, y_prob_b, threshold)
                for metric_name, metric_value in metrics_b.items():
                    bootstrap_results[f"{metric_name}_{threshold_name}"].append(
                        metric_value
                    )

            if (b + 1) % 500 == 0:
                print(f"  Completed {b + 1}/{n_bootstrap} iterations...")

        # Point estimates and percentile CIs
        lo = ci_alpha / 2 * 100
        hi = (1 - ci_alpha / 2) * 100

        rows = []
        print(
            f"\n{'Metric':<25} {'Point Est':>10} {'95% CI Lower':>14} {'95% CI Upper':>14}"
        )
        print(f"{'-'*25} {'-'*10} {'-'*14} {'-'*14}")

        for metric in metric_names:
            values = np.array(bootstrap_results[metric])
            values = values[~np.isnan(values)]

            if len(values) == 0:
                continue

            # Point estimates use the full original test set
            if metric == "AUROC":
                point = roc_auc_score(y_test, y_prob)
            elif metric == "AUPRC":
                point = average_precision_score(y_test, y_prob)
            elif metric == "Brier":
                point = brier_score_loss(y_test, y_prob)
            elif metric == "Calibration_Slope":
                logit_p = np.log(
                    np.clip(y_prob, 1e-10, 1 - 1e-10)
                    / (1 - np.clip(y_prob, 1e-10, 1 - 1e-10))
                )
                cal_m = LR_calibration(penalty=None, solver="lbfgs", max_iter=5000)
                cal_m.fit(logit_p.reshape(-1, 1), y_test)
                point = cal_m.coef_[0][0]
            elif metric == "Joint_Recalibration_Intercept":
                logit_p = np.log(
                    np.clip(y_prob, 1e-10, 1 - 1e-10)
                    / (1 - np.clip(y_prob, 1e-10, 1 - 1e-10))
                )
                cal_m = LR_calibration(penalty=None, solver="lbfgs", max_iter=5000)
                cal_m.fit(logit_p.reshape(-1, 1), y_test)
                point = cal_m.intercept_[0]
            else:
                metric_name, threshold_name = metric.rsplit("_", 1)
                point = _classification_metrics(
                    y_test, y_prob, thresholds[threshold_name]
                )[metric_name]

            ci_lower = np.percentile(values, lo)
            ci_upper = np.percentile(values, hi)

            rows.append(
                {
                    "Metric": metric,
                    "Point_Estimate": round(point, 4),
                    "CI_Lower": round(ci_lower, 4),
                    "CI_Upper": round(ci_upper, 4),
                    "N_Bootstrap": len(values),
                }
            )

            print(f"  {metric:<25} {point:>10.4f} {ci_lower:>14.4f} {ci_upper:>14.4f}")

        ci_df = pd.DataFrame(rows)
        ci_df.to_csv(save_path, index=False)
        print(f"\nBootstrap CI saved to: {save_path}")

        return ci_df

    def plot_decision_curve(self, y_true, y_prob, save_path="lr_decision_curve.pdf"):
        """Plot the decision curve."""
        print("Plotting decision curve...")
        plot_decision_curve_analysis_v2(y_true, y_prob, save_path)
        print(f"Decision curve saved to: {save_path}")

    def plot_calibration_curve(
        self, y_true, y_prob, save_path="lr_calibration_curve.pdf"
    ):
        """Plot the calibration curve."""
        print("Plotting calibration curve...")
        plot_calibration_curve_analysis(y_true, y_prob, save_path)
        print(f"Calibration curve saved to: {save_path}")

    def plot_clinical_impact_curve(
        self, y_true, y_prob, save_path="lr_clinical_impact_curve.pdf"
    ):
        """Plot the clinical impact curve."""
        print("Plotting clinical impact curve...")
        plot_clinical_impact_curve(y_true, y_prob, save_path)
        print(f"Clinical impact curve saved to: {save_path}")

    def plot_roc_pr_curves(self, y_true, y_prob, save_path="lr_roc_pr_curves.pdf"):
        """Plot ROC and PR curves with participant-level bootstrap 95% pointwise bands."""
        print("Plotting ROC and PR curves...")
        plot_roc_pr_curves(
            y_true,
            y_prob,
            save_path,
            model_name="LR",
            groups=None,
            n_bootstrap=2000,
            random_state=self.random_state,
        )
        print(f"ROC and PR curves saved to: {save_path}")

    def plot_combined_analysis(
        self, y_true, y_prob, save_path="lr_combined_analysis.pdf"
    ):
        """Plot the combined figure (calibration, probability distributions, DCA and CIC)."""
        print("Plotting combined analysis...")
        plot_combined_analysis(y_true, y_prob, save_path)
        print(f"Combined analysis saved to: {save_path}")


def main():
    """Fit the model on the training set and evaluate it on the test set."""
    # Random seed
    np.random.seed(42)

    # Load data
    train_data, test_data, feature_name_mapping = load_data(
        "PPMI_9_train.csv", "PPMI_9_test.csv", "PPMI_feature_mapping.csv"
    )

    # Prepare data
    X_train, y_train, subjects_train = prepare_data(train_data)
    X_test, y_test, subjects_test = prepare_data(test_data)

    # Keep DataFrame/Series versions for the OR / 95% CI / P value table (statsmodels)
    X_train_df = X_train.copy()
    y_train_s = y_train.copy()
    # Convert to numpy arrays
    X_train = X_train.values
    y_train = y_train.values
    subjects_train = subjects_train.values
    X_test = X_test.values
    y_test = y_test.values

    # Feature names
    feature_names = [
        feature_name_mapping.get(col, col)
        for col in train_data.columns
        if col not in ["PATNO", "MCI"]
    ]

    # Create the LR model
    lr_model = LRModel(random_state=42)

    # Fit the model
    lr_model.train_final_model(X_train, y_train, subjects_train)

    if (
        len(np.unique(subjects_train)) != len(subjects_train)
        or subjects_test.duplicated().any()
    ):
        raise RuntimeError("Single-visit invariant failed before model fitting")

    # Standard reporting table: OR, 95% CI, P, beta, SE, z
    lr_model.export_or_ci_p_table(
        X=X_train_df,
        y=y_train_s,
        feature_names=feature_names,
        out_csv_path="step_13_lr_or_table.csv",
        include_intercept=True,
    )

    # Cross-validated performance
    train_performance = lr_model.get_cross_validation_performance(
        X_train, y_train, subjects_train
    )

    # Feature importance
    lr_model.evaluate_feature_importance(
        X_train, y_train, feature_names, "step_13_lr_shap"
    )

    # Test-set performance
    test_performance, y_prob = lr_model.evaluate_and_save_results(
        X_test, y_test, "step_13_lr_predictions.csv"
    )

    # Participant-level bootstrap 95% CIs
    lr_model.bootstrap_test_performance(
        X_test,
        y_test,
        y_prob,
        n_bootstrap=2000,
        save_path="step_13_lr_bootstrap_ci.csv",
    )

    # Export the full-precision model parameter file (used by the nomogram and web calculator)
    lr_model.export_model_parameters_json(
        out_json_path="model_parameters.json",
        feature_names=feature_names,
        feature_name_mapping=feature_name_mapping,
        bootstrap_ci_path="step_13_lr_bootstrap_ci.csv",
    )

    # Plot ROC and PR curves
    lr_model.plot_roc_pr_curves(y_test, y_prob, "step_13_lr_roc_pr_curves.pdf")

    # Plot the combined figure
    lr_model.plot_combined_analysis(y_test, y_prob, "step_13_lr_combined_analysis.pdf")

    # Save the performance results
    train_df = pd.DataFrame(train_performance)
    train_df.to_csv("step_13_lr_train_performance.csv")
    print(f"\nTraining set performance saved to: step_13_lr_train_performance.csv")

    test_df = pd.DataFrame(test_performance)
    test_df.to_csv("step_13_lr_test_performance.csv")
    print(f"Test set performance saved to: step_13_lr_test_performance.csv")

    # Print the results
    print("\n" + "=" * 80)
    print("LR model training set performance (mean±std):")
    print("=" * 80)
    print(train_df.round(4))

    print("\n" + "=" * 80)
    print("LR model test set performance:")
    print("=" * 80)
    print(test_df.round(4))


if __name__ == "__main__":
    main()

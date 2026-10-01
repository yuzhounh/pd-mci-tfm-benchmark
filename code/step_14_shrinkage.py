"""
Bootstrap uniform shrinkage of the step 13 logistic-regression model.

方法学说明（Harrell / Steyerberg 的 bootstrap 均一收缩）：
1. 在训练集上有放回抽样得到 bootstrap 样本；
2. 在 bootstrap 样本上完整重复建模流程：LASSO（10 折分层 CV、折内标准化、
   AUPRC、1SE 规则，与 step 12 一致）选择变量 → 无惩罚 LR 拟合（与 step 13 一致）；
3. 将该模型应用到原始训练集，估计校准斜率 b_i 与 AUC；
4. 收缩因子 S = mean(b_i)；optimism = mean(AUC_boot - AUC_orig)；
5. 最终模型所有非截距系数乘以 S，再以 offset 方式重新估计截距，
   使训练集平均预测概率等于实际患病率。

收缩因子只由训练集估计，测试集仅用于最终评估。
"""

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from step_13_lr import LRModel
from utils import (
    calculate_all_metrics,
    f1_score_threshold,
    plot_combined_analysis,
    subject_level_cross_validation,
    youden_threshold,
)

CANDIDATE_TRAIN_FILE = "PPMI_8_train.csv"
SELECTED_TRAIN_FILE = "PPMI_9_train.csv"
SELECTED_TEST_FILE = "PPMI_9_test.csv"
FEATURE_MAPPING_FILE = "PPMI_feature_mapping.csv"
MODEL_JSON = "model_parameters.json"

NON_FEATURE_COLS = ["SITE", "subgroup", "EVENT_ID", "visit_date", "PATNO", "MCI"]

N_BOOTSTRAP = 500
RANDOM_STATE = 42
# Same LASSO grid and CV settings as step_12_lasso.py
LAMBDA_SEQ = 10 ** np.arange(0, 4, 0.05)
N_FOLDS = 10


# ====================== Refit pipeline inside each bootstrap ======================

def lasso_select(X, y):
    """Replicate step 12: 10-fold stratified CV, in-fold scaling, AUPRC, 1SE rule."""
    folds = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    )
    scores = np.zeros((N_FOLDS, len(LAMBDA_SEQ)))
    for j, lam in enumerate(LAMBDA_SEQ):
        for k, (tr, va) in enumerate(folds):
            scaler = StandardScaler().fit(X[tr])
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=1.0 / lam,
                random_state=RANDOM_STATE, max_iter=1000,
            )
            model.fit(scaler.transform(X[tr]), y[tr])
            scores[k, j] = average_precision_score(
                y[va], model.predict_proba(scaler.transform(X[va]))[:, 1]
            )

    mean_scores = scores.mean(axis=0)
    se_scores = scores.std(axis=0) / np.sqrt(N_FOLDS)
    best_idx = int(np.argmax(mean_scores))
    idx_1se = int(np.max(np.where(mean_scores >= mean_scores[best_idx] - se_scores[best_idx])[0]))

    X_scaled = StandardScaler().fit_transform(X)

    def nonzero_at(idx):
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0 / LAMBDA_SEQ[idx],
            random_state=RANDOM_STATE, max_iter=1000,
        )
        model.fit(X_scaled, y)
        return np.abs(model.coef_[0]) > 1e-6

    mask = nonzero_at(idx_1se)
    fallback = False
    if not mask.any():
        mask = nonzero_at(best_idx)
        fallback = True
    return mask, fallback


def fit_unpenalized(X, y):
    """Same estimator as step 13 (LRModel.create_model)."""
    return LRModel(random_state=RANDOM_STATE).create_model().fit(X, y)


def calibration_slope(y, prob):
    logit_p = np.log(np.clip(prob, 1e-10, 1 - 1e-10) / (1 - np.clip(prob, 1e-10, 1 - 1e-10)))
    res = sm.GLM(y, sm.add_constant(logit_p), family=sm.families.Binomial()).fit()
    return float(res.params[1])


def one_bootstrap(b, seed, X_all, y):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(y), size=len(y), replace=True)
    X_b, y_b = X_all[idx], y[idx]
    if len(np.unique(y_b)) < 2:
        return None

    mask, fallback = lasso_select(X_b, y_b)
    model = fit_unpenalized(X_b[:, mask], y_b)
    prob_boot = model.predict_proba(X_b[:, mask])[:, 1]
    prob_orig = model.predict_proba(X_all[:, mask])[:, 1]
    return {
        "bootstrap": b,
        "n_selected": int(mask.sum()),
        "fallback_best_lambda": fallback,
        "auc_boot": roc_auc_score(y_b, prob_boot),
        "auc_orig": roc_auc_score(y, prob_orig),
        "slope_orig": calibration_slope(y, prob_orig),
        "mask": mask,
    }


# ====================== Shrunk model ======================

def shrink_model(X, y, coef, shrinkage):
    """Multiply slopes by S, then re-estimate the intercept with the shrunk linear predictor as offset."""
    coef_s = coef * shrinkage
    offset = X @ coef_s
    res = sm.GLM(
        y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=offset
    ).fit()
    return float(res.params[0]), coef_s


def predict_prob(X, intercept, coef):
    return 1.0 / (1.0 + np.exp(-(intercept + X @ coef)))


def cv_thresholds(X, y, subjects, shrinkage):
    """Median Youden/F1 thresholds from 10-fold CV, applying the same shrinkage inside each fold."""
    youden, f1 = [], []
    for tr, va in subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=RANDOM_STATE):
        model = fit_unpenalized(X[tr], y[tr])
        intercept_s, coef_s = shrink_model(X[tr], y[tr], model.coef_[0], shrinkage)
        prob = predict_prob(X[va], intercept_s, coef_s)
        youden.append(youden_threshold(y[va], prob))
        f1.append(f1_score_threshold(y[va], prob))
    return float(np.median(youden)), float(np.median(f1))


def main():
    np.random.seed(RANDOM_STATE)

    feature_mapping = pd.read_csv(FEATURE_MAPPING_FILE)
    abbr = dict(zip(feature_mapping["feature"], feature_mapping["abbreviation"]))

    # Candidate pool (input to step 12)
    candidates = pd.read_csv(CANDIDATE_TRAIN_FILE)
    candidate_cols = [c for c in candidates.columns if c not in NON_FEATURE_COLS]
    X_all = candidates[candidate_cols].to_numpy(dtype=float)
    y_all = candidates["MCI"].to_numpy()

    # Final selected model (step 13)
    train = pd.read_csv(SELECTED_TRAIN_FILE)
    test = pd.read_csv(SELECTED_TEST_FILE)
    selected_cols = [c for c in train.columns if c not in ["PATNO", "MCI"]]
    if not train["PATNO"].equals(candidates["PATNO"]):
        raise RuntimeError("PPMI_8_train and PPMI_9_train row order differs")

    X_train = train[selected_cols].to_numpy(dtype=float)
    y_train = train["MCI"].to_numpy()
    subjects_train = train["PATNO"].to_numpy()
    X_test = test[selected_cols].to_numpy(dtype=float)
    y_test = test["MCI"].to_numpy()

    apparent = fit_unpenalized(X_train, y_train)
    apparent_auc = roc_auc_score(y_train, apparent.predict_proba(X_train)[:, 1])

    # ---------------- Bootstrap ----------------
    print(f"=== Bootstrap shrinkage ({N_BOOTSTRAP} resamples, full LASSO + LR refit) ===")
    seeds = np.random.SeedSequence(RANDOM_STATE).generate_state(N_BOOTSTRAP)
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(one_bootstrap)(b, int(seed), X_all, y_all) for b, seed in enumerate(seeds)
    )
    results = [r for r in results if r is not None]

    masks = np.array([r.pop("mask") for r in results])
    draws = pd.DataFrame(results)
    draws["optimism_auc"] = draws["auc_boot"] - draws["auc_orig"]
    draws.to_csv("step_14_shrinkage_bootstrap_draws.csv", index=False)

    shrinkage = float(draws["slope_orig"].mean())
    optimism = float(draws["optimism_auc"].mean())

    summary = pd.DataFrame([
        ("N_bootstrap_valid", len(draws)),
        ("Shrinkage_factor_S", shrinkage),
        ("S_SD", draws["slope_orig"].std()),
        ("S_P2.5", draws["slope_orig"].quantile(0.025)),
        ("S_P97.5", draws["slope_orig"].quantile(0.975)),
        ("Apparent_AUC_train", apparent_auc),
        ("Optimism_AUC", optimism),
        ("Optimism_corrected_AUC_train", apparent_auc - optimism),
        ("Mean_features_selected", draws["n_selected"].mean()),
        ("Fallback_best_lambda_count", int(draws["fallback_best_lambda"].sum())),
    ], columns=["Metric", "Value"])
    summary.to_csv("step_14_shrinkage_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    freq = pd.DataFrame({
        "Feature": candidate_cols,
        "Abbreviation": [abbr.get(c, c) for c in candidate_cols],
        "Selection_frequency": masks.mean(axis=0),
        "In_final_model": [c in selected_cols for c in candidate_cols],
    }).sort_values("Selection_frequency", ascending=False)
    freq.to_csv("step_14_bootstrap_selection_frequency.csv", index=False)
    print("\nBootstrap selection frequency:")
    print(freq.to_string(index=False))

    # ---------------- Shrunk final model ----------------
    intercept_s, coef_s = shrink_model(X_train, y_train, apparent.coef_[0], shrinkage)
    coef_table = pd.DataFrame({
        "Variable": ["Intercept"] + selected_cols,
        "Abbreviation": ["Intercept"] + [abbr.get(c, c) for c in selected_cols],
        "beta_unshrunk": np.r_[apparent.intercept_[0], apparent.coef_[0]],
        "beta_shrunk": np.r_[intercept_s, coef_s],
    })
    coef_table["OR_unshrunk"] = np.exp(coef_table["beta_unshrunk"])
    coef_table["OR_shrunk"] = np.exp(coef_table["beta_shrunk"])
    coef_table.loc[0, ["OR_unshrunk", "OR_shrunk"]] = np.nan
    coef_table.to_csv("step_14_shrunk_coefficients.csv", index=False)
    print("\nShrunk coefficients:")
    print(coef_table.round(4).to_string(index=False))

    prob_train_s = predict_prob(X_train, intercept_s, coef_s)
    print(f"\nTraining mean predicted = {prob_train_s.mean():.4f}, observed prevalence = {y_train.mean():.4f}")

    youden_t, f1_t = cv_thresholds(X_train, y_train, subjects_train, shrinkage)
    print(f"CV thresholds (shrunk): Youden = {youden_t:.4f}, F1 = {f1_t:.4f}")

    # ---------------- Test-set evaluation ----------------
    y_prob = predict_prob(X_test, intercept_s, coef_s)
    pd.DataFrame({"y_true": y_test, "y_prob": y_prob}).to_csv(
        "step_14_shrunk_predictions.csv", index=False
    )

    evaluator = LRModel(random_state=RANDOM_STATE)
    evaluator.default, evaluator.f1_median, evaluator.youden_median = 0.5, f1_t, youden_t
    test_perf = {
        f"test_{name}": calculate_all_metrics(y_test, (y_prob >= t).astype(int), y_prob, t)
        for name, t in [("default", 0.5), ("f1", f1_t), ("youden", youden_t)]
    }
    test_df = pd.DataFrame(test_perf)
    test_df.to_csv("step_14_shrunk_test_performance.csv")
    print("\nShrunk model test performance:")
    print(test_df.round(4))

    ci_df = evaluator.bootstrap_test_performance(
        X_test, y_test, y_prob, n_bootstrap=2000,
        save_path="step_14_shrunk_bootstrap_ci.csv",
    )
    plot_combined_analysis(y_test, y_prob, "step_14_shrunk_combined_analysis.pdf")

    # ---------------- Append to the unified parameter file ----------------
    with open(MODEL_JSON, encoding="utf-8") as f:
        model_dict = json.load(f)
    model_dict["shrinkage"] = {
        "method": "Bootstrap uniform shrinkage (full LASSO 1SE + unpenalized LR refit per resample)",
        "n_bootstrap": int(len(draws)),
        "shrinkage_factor": shrinkage,
        "optimism_corrected_auc_train": apparent_auc - optimism,
        "intercept": intercept_s,
        "coefficients": dict(zip(selected_cols, map(float, coef_s))),
        "thresholds": {"default": 0.5, "youden": round(youden_t, 4), "f1": round(f1_t, 4)},
        "test_performance_bootstrap": {
            row["Metric"]: {
                "point_estimate": float(row["Point_Estimate"]),
                "ci_lower": float(row["CI_Lower"]),
                "ci_upper": float(row["CI_Upper"]),
            }
            for _, row in ci_df.iterrows()
        },
    }
    with open(MODEL_JSON, "w", encoding="utf-8") as f:
        json.dump(model_dict, f, indent=2, ensure_ascii=False)
    print(f"\nShrunk model appended to {MODEL_JSON} under 'shrinkage'")


if __name__ == "__main__":
    main()

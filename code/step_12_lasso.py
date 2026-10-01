import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# File name variables
INPUT_TRAIN_FILE = 'PPMI_8_train.csv'
INPUT_TEST_FILE = 'PPMI_8_test.csv'
OUTPUT_TRAIN_FILE = 'PPMI_9_train.csv'
OUTPUT_TEST_FILE = 'PPMI_9_test.csv'
FEATURE_MAPPING_FILE = 'PPMI_feature_mapping.csv'

# Updated output filenames
COEF_PATH_FILE = 'step_12_lasso_coefficients_path.csv'
LASSO_PLOT_FILE = 'step_12_lasso_plot.pdf'
FEATURE_WEIGHT_CSV = 'step_12_feature_weight.csv'
FEATURE_WEIGHT_PDF = 'step_12_feature_weight.pdf'

# Set font to Arial
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# ====================== Core Functions ======================

def subject_level_cross_validation(X, y, subjects, n_folds=10, random_state=42):
    """Stratified row-level CV for one baseline row per participant."""
    if len(np.unique(subjects)) != len(subjects):
        raise RuntimeError("LASSO expected one row per participant")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return list(skf.split(X, y))


def evaluate_model_cv(X, y, subjects, model, cv_folds, scoring_func=average_precision_score):
    """Model evaluation using cross-validation with in-fold standardization to prevent data leakage"""
    scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # 折内独立标准化：训练折 fit_transform，验证折 transform，严格杜绝分布泄漏
        fold_scaler = StandardScaler()
        X_train_fold_scaled = fold_scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = fold_scaler.transform(X_val_fold)
        
        model.fit(X_train_fold_scaled, y_train_fold)
        
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_val_fold_scaled)[:, 1]
        else:
            y_pred = model.predict(X_val_fold_scaled)
            
        score = scoring_func(y_val_fold, y_pred)
        scores.append(score)
    
    return np.array(scores)


def compute_coefficient_path(X, y, lambda_seq, random_state=42):
    """
    Compute LASSO coefficient path on the full standardized training set.
    
    方法学说明：10折交叉验证（K-fold CV）采用折内独立标准化，以无偏方式确定满足 1SE 规则
    的最优稀疏超参数 lambda_1SE。确定最优超参数后，遵循统计学习标准两阶段范式（Hastie et al.），
    基于整个训练集的标准化特征矩阵计算系数路径并提取最终入选特征与权重。
    """
    coefficients_path = np.zeros((len(lambda_seq), X.shape[1]))
    
    for i, lam in enumerate(lambda_seq):
        model = LogisticRegression(
            penalty='l1', solver='liblinear', C=1.0/lam, 
            # class_weight='balanced', # OFF
            random_state=random_state, max_iter=1000
        )
        model.fit(X, y)
        coefficients_path[i] = model.coef_[0]
    
    coefficients_df = pd.DataFrame(coefficients_path)
    coefficients_df.index = lambda_seq
    coefficients_df.to_csv(COEF_PATH_FILE, index_label='lambda')
    
    return coefficients_path


def get_feature_count(coeffs):
    return np.sum(np.abs(coeffs) > 1e-6)

def lasso_kfold_cv_with_coefficients(X, y, subjects, lambda_seq, n_folds=10, random_state=42):
    """LASSO hyperparameter tuning using K-fold CV with 1SE Rule and in-fold standardization"""
    cv_folds = subject_level_cross_validation(X, y, subjects, n_folds, random_state)
    
    print(f"\nStarting {n_folds}-fold stratified CV (one baseline row per participant)...")
    
    auc_pr_scores = np.zeros((n_folds, len(lambda_seq)))
    
    for i, lam in enumerate(lambda_seq):
        model = LogisticRegression(
            penalty='l1', solver='liblinear', C=1.0/lam, 
            # class_weight='balanced', # OFF
            random_state=random_state, max_iter=1000
        )
        scores = evaluate_model_cv(X, y, subjects, model, cv_folds, average_precision_score)
        auc_pr_scores[:, i] = scores
        
        if (i + 1) % 10 == 0:
            print(f"Lambda progress: {(i + 1) / len(lambda_seq) * 100:.0f}%")
    
    mean_auc_pr = np.mean(auc_pr_scores, axis=0)
    # 1SE rule uses the standard error of the mean across folds (SD / sqrt(K),
    # consistent with cv.glmnet's cvsd.
    se_auc_pr = np.std(auc_pr_scores, axis=0) / np.sqrt(n_folds)

    # 1. Find the Best (Max Score)
    best_idx = np.argmax(mean_auc_pr)
    best_score = mean_auc_pr[best_idx]
    best_se = se_auc_pr[best_idx]
    
    print("\nComputing coefficient path on full standardized training set...")
    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X)
    coefficients_path = compute_coefficient_path(X_scaled, y, lambda_seq, random_state)
    
    # Check features at best
    feat_count_best = get_feature_count(coefficients_path[best_idx])
    print(f"Best AUPRC: {best_score:.4f} (λ={lambda_seq[best_idx]:.4f}), Features: {feat_count_best}")

    # 2. Find 1SE Target
    # We want the simplest model (largest lambda -> higher index in our seq usually, check lambda_seq order)
    # lambda_seq was: lambda_seq = 10**np.arange(0, 3, 0.05) -> 1, ..., 1000. 
    # C = 1/lambda. So larger lambda = smaller C = Stronger Regularization = Fewer features.
    # So we want the Largest Lambda (Highest Index) that satisfies the condition.
    
    cutoff_score = best_score - best_se
    
    # Find all indices where score >= cutoff
    eligible_indices = np.where(mean_auc_pr >= cutoff_score)[0]
    
    # We want the one with strongest regularization. 
    # lambda_seq is increasing (1 -> 1000). 
    # Stronger regularization = Larger Lambda = Larger Index.
    # So we take the max index from eligible ones.
    best_1se_idx = np.max(eligible_indices)
    
    feat_count_1se = get_feature_count(coefficients_path[best_1se_idx])
    print(f"1SE Cutoff: {cutoff_score:.4f} (Best {best_score:.4f} - SE {best_se:.4f})")
    print(f"1SE AUPRC: {mean_auc_pr[best_1se_idx]:.4f} (λ={lambda_seq[best_1se_idx]:.4f}), Features: {feat_count_1se}")

    return mean_auc_pr, se_auc_pr, coefficients_path, best_idx, best_1se_idx


def standardize_data(train_data, test_data):
    """Data standardization"""
    subject_col = 'PATNO'
    label_col = 'MCI'
    
    feature_cols = [col for col in train_data.columns if col not in [subject_col, label_col]]
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    train_scaled = pd.concat([train_data[[subject_col]], X_train_scaled, train_data[[label_col]]], axis=1)
    test_scaled = pd.concat([test_data[[subject_col]], X_test_scaled, test_data[[label_col]]], axis=1)
    
    return train_scaled, test_scaled, scaler


def plot_results(lambda_seq, mean_auc_pr_scores, se_auc_pr_scores, coefficients_path, feature_names, feature_dict, best_lambda_idx, best_1se_idx, out_file=None, lambda_max=None):
    """Plot AUPRC and coefficient path

    out_file: output path (default LASSO_PLOT_FILE). lambda_max: show only lambda <= lambda_max;
    the analysis itself is unaffected (used for the supplementary figure, where every
    coefficient is zero beyond about 10^2).
    """
    if lambda_max is not None:
        keep = lambda_seq <= lambda_max * (1 + 1e-9)
        lambda_seq = lambda_seq[keep]
        mean_auc_pr_scores = mean_auc_pr_scores[keep]
        se_auc_pr_scores = se_auc_pr_scores[keep]
        coefficients_path = coefficients_path[keep]
    abbreviated_names = [feature_dict.get(name, name) for name in feature_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    # AUPRC vs Lambda
    ax1.plot(lambda_seq, mean_auc_pr_scores, '-', linewidth=1.5, marker='o', markersize=3)
    ax1.fill_between(lambda_seq,
                     mean_auc_pr_scores - se_auc_pr_scores,
                     mean_auc_pr_scores + se_auc_pr_scores,
                     alpha=0.2)
    ax1.set_xlabel('$\\lambda$', fontsize=12)
    ax1.set_ylabel('AUPRC', fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Mark feature counts on top axis (optional, but finding points is enough)
    
    # Best Point
    ax1.axvline(lambda_seq[best_lambda_idx], color='r', linestyle='--', alpha=0.6, label=f'Best: {mean_auc_pr_scores[best_lambda_idx]:.4f}')
    ax1.plot(lambda_seq[best_lambda_idx], mean_auc_pr_scores[best_lambda_idx], 'ro', markersize=6)
    
    # 1SE Point
    ax1.axvline(lambda_seq[best_1se_idx], color='g', linestyle='--', alpha=0.6, label=f'1SE: {mean_auc_pr_scores[best_1se_idx]:.4f}')
    ax1.plot(lambda_seq[best_1se_idx], mean_auc_pr_scores[best_1se_idx], 'go', markersize=6)
    
    ax1.legend(loc='upper right')
    ax1.text(-0.08, 1.1, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # Coefficient path
    feature_indices_to_plot = [i for i in range(len(feature_names)) if np.any(np.abs(coefficients_path[:, i]) > 1e-6)]
    
    # If too many features, filter by magnitude at 1SE lambda (since that's our effective target)
    if len(feature_indices_to_plot) > 20:
        coeffs_at_1se = np.abs(coefficients_path[best_1se_idx])
        # If 1SE is very sparse, looking at best might be better for "potential" features
        coeffs_at_best = np.abs(coefficients_path[best_lambda_idx])
        max_coeffs = np.maximum(coeffs_at_1se, coeffs_at_best)
        
        top_features = np.argsort(max_coeffs)[-20:]
        feature_indices_to_plot = [i for i in feature_indices_to_plot if i in top_features]
    
    for i in feature_indices_to_plot:
        ax2.plot(lambda_seq, coefficients_path[:, i], label=abbreviated_names[i], linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('$\\lambda$', fontsize=12)
    ax2.set_ylabel('Coefficient', fontsize=12)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    ax2.axvline(lambda_seq[best_lambda_idx], color='r', linestyle='--', alpha=0.4)
    ax2.axvline(lambda_seq[best_1se_idx], color='g', linestyle='--', alpha=0.4)
    
    ax2.text(-0.08, 1.1, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
    
    if len(feature_indices_to_plot) <= 15:
        ax2.legend(bbox_to_anchor=(1.02, 1.03), loc='upper left', fontsize=10)
    else:
        ax2.legend(bbox_to_anchor=(1.02, 1.03), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save as pdf format
    fig.savefig(out_file or LASSO_PLOT_FILE, bbox_inches='tight')
    plt.close(fig)

    return best_1se_idx # Return 1SE as the chosen one for plotting weights later


def extract_selected_features(selected_features, train_data_original, test_data_original):
    """Extract selected features and save (using original unscaled data)"""
    print(f"\n=== Extract Selected Features ===")
    
    subject_col = 'PATNO'
    label_col = 'MCI'
    
    columns_to_keep = [subject_col] + selected_features + [label_col]
    
    train_selected = train_data_original[columns_to_keep]
    test_selected = test_data_original[columns_to_keep]
    
    train_selected.to_csv(OUTPUT_TRAIN_FILE, index=False)
    test_selected.to_csv(OUTPUT_TEST_FILE, index=False)
    
    print(f"Saved: {OUTPUT_TRAIN_FILE}, {OUTPUT_TEST_FILE}")
    
    return train_selected, test_selected


def main():
    """Main function"""
    # Load data
    print("=== Load Data ===")
    train_data = pd.read_csv(INPUT_TRAIN_FILE)
    test_data = pd.read_csv(INPUT_TEST_FILE)
    
    # Load feature mapping
    feature_mapping = pd.read_csv(FEATURE_MAPPING_FILE)
    feature_dict = dict(zip(feature_mapping['feature'], feature_mapping['abbreviation']))
    
    # Remove unnecessary columns
    columns_to_drop = ['SITE', 'subgroup', 'EVENT_ID', 'visit_date']
    train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
    test_data = test_data.drop(columns=columns_to_drop, errors='ignore')
    
    # Keep original unscaled data for later saving
    train_data_original = train_data.copy()
    test_data_original = test_data.copy()
    
    # Standardization
    print("\n=== Standardization ===")
    train_scaled, test_scaled, scaler = standardize_data(train_data, test_data)
    
    # K-fold CV with in-fold standardization
    print("\n=== K-fold CV (1SE Rule, in-fold standardization) ===")
    # Extended range to ensure we find sparse solutions
    lambda_seq = 10**np.arange(0, 4, 0.05) 
    
    subject_col = 'PATNO'
    label_col = 'MCI'
    feature_cols = [col for col in train_data.columns if col not in [subject_col, label_col]]
    
    X_train_raw = train_data[feature_cols].values
    y_train = train_data[label_col].values
    subjects_train = train_data[subject_col].values

    mean_auc_pr, se_auc_pr, coefficients_path, best_lambda_idx, best_1se_idx = lasso_kfold_cv_with_coefficients(
        X_train_raw, y_train, subjects_train, lambda_seq, n_folds=10
    )
    
    # Plot results
    print("\n=== Visualization ===")
    feature_names = feature_cols
    plot_results(lambda_seq, mean_auc_pr, se_auc_pr, coefficients_path, feature_names, feature_dict, best_lambda_idx, best_1se_idx)
    
    # Output results - WE USE 1SE INDEX NOW
    chosen_idx = best_1se_idx
    best_lambda = lambda_seq[chosen_idx]
    best_auc_pr = mean_auc_pr[chosen_idx]
    best_coeffs = coefficients_path[chosen_idx]
    
    print(f"\nFinal Selection (1SE Rule):")
    print(f"Lambda: {best_lambda:.6f}")
    print(f"AUPRC: {best_auc_pr:.4f}")
    
    # Select non-zero features
    non_zero_mask = np.abs(best_coeffs) > 1e-6
    selected_features = [feature_names[i] for i in range(len(feature_names)) if non_zero_mask[i]]
    selected_weights = [best_coeffs[i] for i in range(len(best_coeffs)) if non_zero_mask[i]]
    
    print(f"Selected features: {len(selected_features)}/{len(feature_names)}")
    
    if len(selected_features) == 0:
        print("WARNING: No features selected! Falling back to 'Best' lambda.")
        chosen_idx = best_lambda_idx
        best_coeffs = coefficients_path[chosen_idx]
        non_zero_mask = np.abs(best_coeffs) > 1e-6
        selected_features = [feature_names[i] for i in range(len(feature_names)) if non_zero_mask[i]]
        selected_weights = [best_coeffs[i] for i in range(len(best_coeffs)) if non_zero_mask[i]]
        print(f"Fallback features: {len(selected_features)}")

    # Save feature weights
    results_df = pd.DataFrame({
        'Feature': selected_features,
        'Abbreviation': [feature_dict.get(f, f) for f in selected_features],
        'Weight': selected_weights
    })
    results_df = results_df.sort_values('Weight', key=abs, ascending=False)
    results_df.to_csv(FEATURE_WEIGHT_CSV, index=False)
    print("\nSelected features:")
    print(results_df.to_string(index=False))
    # Plot feature weight bar chart (Nature style)
    plt.figure(figsize=(5.5, max(4, len(selected_features)*0.4)))
    results_sorted = results_df.copy().sort_values('Weight', key=abs, ascending=True)
    alphas = [0.95 if w > 0 else 0.75 for w in results_sorted['Weight']]
    colors = ['#3CADC4' if w > 0 else '#E64B35' for w in results_sorted['Weight']]
    
    # Plot bars with individual alpha values
    for i, (weight, color, alpha) in enumerate(zip(results_sorted['Weight'], colors, alphas)):
        plt.barh(i, abs(weight), color=color, alpha=alpha)
    
    plt.yticks(range(len(results_sorted)), results_sorted['Abbreviation'])
    plt.xlabel('Weight', fontsize=12)
    
    # Adjust xlim
    max_w = results_sorted['Weight'].abs().max() if not results_sorted.empty else 0.1
    plt.xlim(0, max_w * 1.3)

    for i, weight in enumerate(results_sorted['Weight']):
        plt.text(abs(weight) + 0.005, i, f'{weight:.4f}', ha='left', va='center')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3CADC4', alpha=0.95, label='Positive'),
                      Patch(facecolor='#E64B35', alpha=0.75, label='Negative')]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(FEATURE_WEIGHT_PDF, bbox_inches='tight')
    
    # Extract selected features (using original unscaled data)
    extract_selected_features(selected_features, train_data_original, test_data_original)
    
    print("\n=== Complete ===")
    print(f"Output files:")
    print(f"- {FEATURE_WEIGHT_CSV}")
    print(f"- {FEATURE_WEIGHT_PDF}")
    print(f"- {LASSO_PLOT_FILE}")


if __name__ == "__main__":
    main()

"""Step 10: Spearman correlation heatmap of the candidate predictors (training set)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use Arial
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"

# Read the data
df = pd.read_csv("PPMI_7_train.csv")

# All candidate features (used to identify redundant features; this is not the final model feature set)
features = [
    "age_at_visit",
    "SEX",
    "EDUCYRS",
    "fampd_bin",
    "BMI",
    "agediag",
    "ageonset",
    "sym_tremor",
    "sym_rigid",
    "sym_brady",
    "sym_posins",
    "sym_other",
    "PDTRTMNT",
    "LEDD",
    "quip",
    "ess",
    "rem",
    "gds",
    "stai_state",
    "stai_trait",
    "scopa",
    "orthostasis",
    "NHY",
    "pigd",
    "td_pigd",
    "updrs2_score",
    "updrs3_score",
    "APOE_e4",
    "has_GBA",
    "has_LRRK2",
    "duration",
]

# Read the feature-abbreviation mapping
_csv_df = pd.read_csv("PPMI_feature_mapping.csv")
feature_name_mapping = dict(zip(_csv_df["feature"], _csv_df["abbreviation"]))

# Raw values of these features
df_features = df[features]

# Spearman rank correlation matrix
# The pool contains binary (SEX, fampd_bin, sym_*), ordinal (NHY) and non-normal continuous variables,
# for which Spearman rank correlation is more robust than Pearson correlation.
correlation_matrix = df_features.corr(method="spearman")

# Correlation threshold for the heatmap
CORRELATION_THRESHOLD = 0.6


# Features whose absolute correlation with any other feature reaches the threshold;
# used for the heatmap only, not as the retained feature set.
def get_features_above_correlation_threshold(corr_matrix, threshold=0.2):
    """
    Select features whose absolute correlation with any other feature exceeds the threshold (for the heatmap).
    """
    features_selected = []
    features_unselected = []
    for i, feature in enumerate(corr_matrix.columns):
        # Correlations with all other features (excluding itself)
        correlations = corr_matrix.iloc[i, :].drop(feature)
        if (abs(correlations) >= threshold).any():
            features_selected.append(feature)
        else:
            features_unselected.append(feature)
    return features_selected, features_unselected


# Features shown in the heatmap
features_for_heatmap, features_with_consistently_low_corr = (
    get_features_above_correlation_threshold(
        correlation_matrix, threshold=CORRELATION_THRESHOLD
    )
)

print("=" * 80)
print(
    f"Features for the heatmap (absolute correlation >= {CORRELATION_THRESHOLD} with any other feature):"
)
print("=" * 80)
print(f"\nFeatures shown: {len(features_for_heatmap)}")
print(
    f"Features with consistently low correlation (not shown): {len(features_with_consistently_low_corr)}"
)

if features_with_consistently_low_corr:
    print(
        f"\nFeatures with absolute correlation < {CORRELATION_THRESHOLD} with all others (not in the heatmap):"
    )
    for feature in features_with_consistently_low_corr:
        max_corr = correlation_matrix[feature].drop(feature).abs().max()
        print(f"  - {feature:20s} (maximum correlation: {max_corr:.4f})")
else:
    print(f"\nEvery feature reaches the threshold with at least one other feature")

print("\nFeatures shown in the heatmap (not the final model features):")
for feature in features_for_heatmap:
    max_corr = correlation_matrix[feature].drop(feature).abs().max()
    print(f"  - {feature:20s} (maximum correlation: {max_corr:.4f})")

# Heatmap of the features above the threshold (to visualise redundancy; highly correlated features are removed later)
correlation_matrix_filtered = correlation_matrix.loc[
    features_for_heatmap, features_for_heatmap
]

# Use abbreviations as row and column labels
correlation_matrix_filtered_renamed = correlation_matrix_filtered.rename(
    index=feature_name_mapping, columns=feature_name_mapping
)

# Mask the upper triangle
mask = np.triu(np.ones_like(correlation_matrix_filtered_renamed, dtype=bool))

plt.figure(figsize=(6, 5))
sns.heatmap(
    correlation_matrix_filtered_renamed,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("step_10_correlation_heatmap.pdf", bbox_inches="tight")

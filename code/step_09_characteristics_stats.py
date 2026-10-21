"""
Step 09: baseline characteristics and univariable association analysis.

Outputs:
  - Overall characteristics of the training and held-out test sets (main Table 1).
  - Participant-level characteristics stratified by PD-NC / PD-MCI (Supplementary Table S10).
  - Univariable logistic GLM odds ratios with model-based 95% CIs,
      and P values adjusted by Benjamini-Hochberg separately within each dataset.

The univariable associations are descriptive and do not guide LASSO feature selection.
"""

import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import (
    shapiro,
    mannwhitneyu,
    ttest_ind,
    fisher_exact,
    f_oneway,
    kruskal,
    chi2_contingency,
)
from statsmodels.stats.diagnostic import lilliefors as ks_lilliefors
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations

warnings.filterwarnings("ignore")


# ================================================================
# (1) Configuration
# ================================================================

# Group column and display labels
GROUP_COL = "MCI"
GROUP_LABELS = {0: "PD-NC", 1: "PD-MCI"}

# Manual variable types (take precedence over automatic detection)
# Allowed values: 'Continuous', 'Nominal' (unordered categories), 'Ordinal' (ordered categories)
VARIABLE_TYPE_OVERRIDE: dict[str, str] = {
    "SEX": "Nominal",  # sex: unordered binary
    "NHY": "Ordinal",  # H&Y stage: ordered
    "APOE_e4": "Ordinal",  # number of APOE e4 alleles: ordered
    "quip": "Continuous",  # QUIP impulse-control disorder total score
}

# Display names of all 31 candidate features
FEATURE_NAME_MAPPING: dict[str, str] = {
    "age_at_visit": "Age",
    "SEX": "Sex",
    "EDUCYRS": "Education",
    "fampd_bin": "FH-PD",
    "BMI": "BMI",
    "agediag": "AAD",
    "ageonset": "AAO",
    "sym_tremor": "Tremor",
    "sym_rigid": "Rigidity",
    "sym_brady": "Bradykinesia",
    "sym_posins": "Postural instability",
    "sym_other": "Other symptoms",
    "PDTRTMNT": "On medication",
    "LEDD": "LEDD",
    "quip": "QUIP",
    "ess": "ESS",
    "rem": "RBDSQ",
    "gds": "GDS",
    "stai_state": "STAI-S",
    "stai_trait": "STAI-T",
    "scopa": "SCOPA-AUT",
    "orthostasis": "OH",
    "NHY": "H&Y Stage",
    "pigd": "PIGD",
    "td_pigd": "Motor Phenotype",
    "updrs2_score": "UPDRS-II",
    "updrs3_score": "UPDRS-III",
    "APOE_e4": "APOE-ε4",
    "has_GBA": "GBA",
    "has_LRRK2": "LRRK2",
    "duration": "Duration",
}

# Decimals for continuous variables (0 for integer scales and doses, 1 for clinical scores, 2 for other measures)
FEATURE_PRECISION: dict[str, int] = {
    "EDUCYRS": 0,
    "LEDD": 0,
    "quip": 0,
    "ess": 0,
    "rem": 0,
    "gds": 0,
    "stai_state": 0,
    "stai_trait": 0,
    "scopa": 0,
    "updrs2_score": 0,
    "pigd": 1,
    "updrs3_score": 1,
    "age_at_visit": 2,
    "BMI": 2,
    "agediag": 2,
    "ageonset": 2,
    "duration": 2,
}

# Univariable OR settings and level labels
FEATURE_OR_CONFIG: dict[str, dict] = {
    "age_at_visit": {"scale": 1.0, "unit": "per 1-year increase"},
    "SEX": {
        "scale": 1.0,
        "ref": 0,
        "ref_label": "Female",
        "label": "Male vs Female",
        "level_labels": {0: "Female", 1: "Male"},
    },
    "EDUCYRS": {"scale": 1.0, "unit": "per 1-year increase"},
    "fampd_bin": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "No",
        "label": "Yes vs No",
        "level_labels": {0.0: "No", 1.0: "Yes"},
    },
    "BMI": {"scale": 1.0, "unit": "per 1-unit increase"},
    "agediag": {"scale": 1.0, "unit": "per 1-year increase"},
    "ageonset": {"scale": 1.0, "unit": "per 1-year increase"},
    "sym_tremor": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "No",
        "label": "Yes vs No",
        "level_labels": {0.0: "No", 1.0: "Yes"},
    },
    "sym_rigid": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "No",
        "label": "Yes vs No",
        "level_labels": {0.0: "No", 1.0: "Yes"},
    },
    "sym_brady": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "No",
        "label": "Yes vs No",
        "level_labels": {0.0: "No", 1.0: "Yes"},
    },
    "sym_posins": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "No",
        "label": "Yes vs No",
        "level_labels": {0.0: "No", 1.0: "Yes"},
    },
    "sym_other": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "No",
        "label": "Yes vs No",
        "level_labels": {0.0: "No", 1.0: "Yes"},
    },
    "PDTRTMNT": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "No",
        "label": "Yes vs No",
        "level_labels": {0.0: "No", 1.0: "Yes"},
    },
    "LEDD": {"scale": 100.0, "unit": "per 100 mg/day increase"},
    "quip": {"scale": 1.0, "unit": "per 1-point increase"},
    "ess": {"scale": 1.0, "unit": "per 1-point increase"},
    "rem": {"scale": 1.0, "unit": "per 1-point increase"},
    "gds": {"scale": 1.0, "unit": "per 1-point increase"},
    "stai_state": {"scale": 1.0, "unit": "per 1-point increase"},
    "stai_trait": {"scale": 1.0, "unit": "per 1-point increase"},
    "scopa": {"scale": 1.0, "unit": "per 1-point increase"},
    "orthostasis": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "No",
        "label": "Yes vs No",
        "level_labels": {0.0: "No", 1.0: "Yes"},
    },
    "NHY": {
        "scale": 1.0,
        "unit": "per 1-stage increase",
        "level_labels": {
            0.0: "Stage 0",
            1.0: "Stage 1",
            2.0: "Stage 2",
            3.0: "Stage 3",
            4.0: "Stage 4",
            5.0: "Stage 5",
        },
    },
    "pigd": {"scale": 1.0, "unit": "per 1-unit increase"},
    "td_pigd": {
        "scale": 1.0,
        "ref": 0.0,
        "ref_label": "PIGD/Indeterminate",
        "label": "TD vs PIGD/Indeterminate",
        "level_labels": {0.0: "PIGD/Indeterminate", 1.0: "TD"},
    },
    "updrs2_score": {"scale": 1.0, "unit": "per 1-point increase"},
    "updrs3_score": {"scale": 1.0, "unit": "per 1-point increase"},
    "APOE_e4": {
        "scale": 1.0,
        "unit": "per 1-allele increase",
        "level_labels": {0.0: "0", 1.0: "1", 2.0: "2"},
    },
    "has_GBA": {
        "scale": 1.0,
        "ref": 0,
        "ref_label": "Non-carrier",
        "label": "Carrier vs Non-carrier",
        "level_labels": {0: "No", 1: "Yes"},
    },
    "has_LRRK2": {
        "scale": 1.0,
        "ref": 0,
        "ref_label": "Non-carrier",
        "label": "Carrier vs Non-carrier",
        "level_labels": {0: "No", 1: "Yes"},
    },
    "duration": {"scale": 1.0, "unit": "per 1-year increase"},
}

# Normality threshold
NORMALITY_ALPHA = 0.05

# When to switch from chi-squared to Fisher's exact test
FISHER_2X2_MIN_EXPECTED = 5  # 2x2: any expected count < 5
FISHER_RXC_PROPORTION = 0.20  # rxc: more than 20% of cells with expected count < 5
FISHER_RXC_MIN_ANY = 1  # rxc: any expected count < 1

# Number of Monte Carlo permutations (rxc Fisher test)
MONTE_CARLO_N = 10_000
MONTE_CARLO_SEED = 42

# Skip variables with a higher missing rate
MISSING_SKIP_THRESHOLD = 0.50

# FDR threshold
FDR_ALPHA = 0.05


# ================================================================
# (2) Helpers
# ================================================================


def format_p(p: float) -> str:
    """Format a P value: <0.001 for very small values, otherwise 3 decimals."""
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "NE"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_or_ci(or_val: float, ci_low: float, ci_high: float) -> str:
    """Format OR (95% CI) with 2 decimals and an en dash."""
    if any(
        v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
        for v in [or_val, ci_low, ci_high]
    ):
        return "NE"
    return f"{or_val:.2f} ({ci_low:.2f}-{ci_high:.2f})"


def markdown_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table (without tabulate)."""
    cols = df.columns.tolist()
    # Maximum width of each column
    widths = [max(len(str(v)) for v in [c] + df[c].tolist()) for c in cols]
    widths = [max(w, 3) for w in widths]

    def row_str(values):
        cells = [str(v).ljust(widths[i]) for i, v in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    header = row_str(cols)
    separator = "| " + " | ".join(["-" * w for w in widths]) + " |"
    rows = [row_str(df.iloc[i].tolist()) for i in range(len(df))]
    return "\n".join([header, separator] + rows)


# ================================================================
# (3) Variable types
# ================================================================


def auto_detect_type(series: pd.Series) -> tuple[str, str]:
    """
    Detect the variable type automatically.
    Returns (type, reason).
    """
    data = series.dropna()
    n_unique = data.nunique()

    # Are all values integers?
    all_int = all(
        isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer())
        for x in data
    )

    if n_unique <= 2:
        return "Nominal", f"{n_unique} unique values (<= 2): binary nominal variable"
    elif n_unique <= 8 and all_int:
        return "Nominal", (
            f"{n_unique} unique integer values (<= 8): nominal variable; "
            'set "Ordinal" in VARIABLE_TYPE_OVERRIDE if it is ordered'
        )
    else:
        return (
            "Continuous",
            f"{n_unique} unique values (> 8) or non-integer values: continuous variable",
        )


def get_variable_type(series: pd.Series, col_name: str) -> tuple[str, str]:
    """Return the variable type (manual setting first)."""
    if col_name in VARIABLE_TYPE_OVERRIDE:
        vtype = VARIABLE_TYPE_OVERRIDE[col_name]
        return vtype, f'set to "{vtype}" in VARIABLE_TYPE_OVERRIDE'
    return auto_detect_type(series)


# ================================================================
# (4) Normality tests
# ================================================================


def normality_test_single(arr: np.ndarray) -> tuple[float, str]:
    """
    Test one group for normality:
    n <= 50: Shapiro-Wilk; n > 50: Kolmogorov-Smirnov (Lilliefors).
    Returns (P value, test name).
    """
    n = len(arr)
    if n < 3:
        return 1.0, "N/A (sample too small)"
    if n <= 50:
        _, p = shapiro(arr)
        return float(p), "Shapiro-Wilk"
    else:
        _, p = ks_lilliefors(arr, dist="norm")
        return float(p), "KS-Lilliefors"


def check_normality(
    data: pd.DataFrame, feature: str, group_col: str
) -> tuple[bool, str]:
    """
    Test each group; the variable counts as normal only if every group has P >= 0.05.
    Returns (is_normal, details).
    """
    groups = sorted(data[group_col].unique())
    details = []

    for g in groups:
        arr = data[data[group_col] == g][feature].dropna().values
        p, met = normality_test_single(arr)
        details.append(f"Group {g} [{met}, p={p:.3f}, n={len(arr)}]")

    # Collect the P values for the decision
    all_normal = True
    for g in groups:
        arr = data[data[group_col] == g][feature].dropna().values
        p, _ = normality_test_single(arr)
        if p < NORMALITY_ALPHA:
            all_normal = False
            break

    return all_normal, "; ".join(details)


# ================================================================
# (5) Descriptive statistics
# ================================================================


def describe_continuous(arr: np.ndarray, is_normal: bool, feat: str = None) -> str:
    """Describe a continuous variable with feature-specific precision (0, 1 or 2 decimals)."""
    if len(arr) == 0:
        return "N/A"
    prec = FEATURE_PRECISION.get(feat, 2)
    if is_normal:
        if prec == 0:
            return f"{arr.mean():.0f} ± {arr.std(ddof=1):.0f}"
        elif prec == 1:
            return f"{arr.mean():.1f} ± {arr.std(ddof=1):.1f}"
        else:
            return f"{arr.mean():.2f} ± {arr.std(ddof=1):.2f}"
    else:
        q1, q3 = np.percentile(arr, [25, 75])
        m = np.median(arr)
        if prec == 0:
            return f"{m:.0f} ({q1:.0f}, {q3:.0f})"
        elif prec == 1:
            return f"{m:.1f} ({q1:.1f}, {q3:.1f})"
        else:
            return f"{m:.2f} ({q1:.2f}, {q3:.2f})"


def describe_categorical_levels(
    data: pd.DataFrame, feature: str, group_col: str
) -> list[dict]:
    """
    Describe a categorical or ordinal variable level by level.
    Each level gives a dict with 'n (%)' strings for the overall sample and each group.
    """
    groups = sorted(data[group_col].unique())
    overall_data = data[feature].dropna()
    group_data = {g: data[data[group_col] == g][feature].dropna() for g in groups}
    all_levels = sorted(overall_data.unique())

    rows = []
    for level in all_levels:
        n_ov = int((overall_data == level).sum())
        pct_ov = n_ov / len(overall_data) * 100
        row = {"level": level, "overall": f"{n_ov} ({pct_ov:.1f}%)"}
        for g in groups:
            gd = group_data[g]
            n_g = int((gd == level).sum())
            pct = n_g / len(gd) * 100 if len(gd) > 0 else 0.0
            row[f"group_{g}"] = f"{n_g} ({pct:.1f}%)"
        rows.append(row)
    return rows


# ================================================================
# (6) Group comparisons
# ================================================================


def dunn_bonferroni(groups_data: list[np.ndarray]) -> dict[tuple, float]:
    """
    Dunn's test with Bonferroni correction (post hoc for Kruskal-Wallis),
    with tie correction.
    Returns {(i, j): Bonferroni-adjusted P} for each pair.
    """
    k = len(groups_data)
    n_comparisons = k * (k - 1) // 2
    all_data = np.concatenate(groups_data)
    n = len(all_data)
    ranks = stats.rankdata(all_data)

    # Tie correction
    _, counts = np.unique(all_data, return_counts=True)
    tie_correction = np.sum(counts**3 - counts) / (12 * (n - 1)) if n > 1 else 0.0

    # Mean rank of each group
    group_sizes = [len(g) for g in groups_data]
    group_mean_ranks = []
    idx = 0
    for size in group_sizes:
        group_mean_ranks.append(ranks[idx : idx + size].mean())
        idx += size

    pairwise = {}
    for i, j in combinations(range(k), 2):
        ni, nj = group_sizes[i], group_sizes[j]
        sigma = np.sqrt((n * (n + 1) / 12 - tie_correction) * (1 / ni + 1 / nj))
        z = (group_mean_ranks[i] - group_mean_ranks[j]) / (sigma + 1e-12)
        p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
        p_bonf = min(p_raw * n_comparisons, 1.0)
        pairwise[(i, j)] = float(p_bonf)
    return pairwise


def monte_carlo_fisher_pvalue(
    data: pd.DataFrame, feature: str, group_col: str
) -> float:
    """
    rxc Fisher exact test by Monte Carlo permutation.
    Permutes the group labels, builds the null distribution of the chi-squared statistic and returns the permutation P value.
    """
    valid = data[feature].notna()
    feat_vals = data.loc[valid, feature].values
    group_vals = data.loc[valid, group_col].values

    obs_table = pd.crosstab(feat_vals, group_vals)
    obs_stat, *_ = chi2_contingency(obs_table)

    rng = np.random.default_rng(MONTE_CARLO_SEED)
    count_extreme = 0
    for _ in range(MONTE_CARLO_N):
        perm = rng.permutation(group_vals)
        pt = pd.crosstab(feat_vals, perm)
        # Add a small constant to avoid numerical problems with all-zero columns
        ps, *_ = chi2_contingency(pt.values + 1e-10)
        if ps >= obs_stat:
            count_extreme += 1
    return count_extreme / MONTE_CARLO_N


def check_fisher_needed(contingency_table: pd.DataFrame) -> tuple[bool, str]:
    """
    Decide whether Fisher's exact test is needed.
    Returns (needed, reason).
    """
    _, _, _, expected = chi2_contingency(contingency_table)
    is_2x2 = contingency_table.shape == (2, 2)
    n_cells = expected.size
    n_small = int((expected < 5).sum())
    min_expected = expected.min()

    if is_2x2:
        if min_expected < FISHER_2X2_EXPECTED:
            return (
                True,
                f"2x2 table: minimum expected count {min_expected:.2f} < {FISHER_2X2_EXPECTED}",
            )
        return False, ""
    else:
        if min_expected < FISHER_RXC_MIN_ANY:
            return True, f"rxc table: expected count {min_expected:.2f} < 1"
        if n_small / n_cells > FISHER_RXC_PROPORTION:
            return (
                True,
                f"rxc table: {n_small}/{n_cells} cells with expected count < 5 (>{FISHER_RXC_PROPORTION:.0%})",
            )
        return False, ""


def compare_groups(
    data: pd.DataFrame,
    feature: str,
    group_col: str,
    var_type: str,
    is_normal: bool = None,
) -> tuple[str, float, str]:
    """
    Compare the groups.
    Returns (test name, P value, post hoc note).
    """
    groups = sorted(data[group_col].unique())
    k = len(groups)
    groups_data = [data[data[group_col] == g][feature].dropna().values for g in groups]

    posthoc_note = ""

    # Rules A/B: continuous variables
    if var_type == "Continuous":
        if is_normal:
            # Rule A: normal -> t test / ANOVA with Tukey HSD
            if k == 2:
                _, p = ttest_ind(groups_data[0], groups_data[1])
                method = "Independent t-test"
            else:
                _, p = f_oneway(*groups_data)
                method = "One-way ANOVA"
                if p < 0.05:
                    all_vals = np.concatenate(groups_data)
                    all_labels = np.concatenate(
                        [[g] * len(gd) for g, gd in zip(groups, groups_data)]
                    )
                    pairwise_tukeyhsd(all_vals, all_labels)  # run for the record
                    posthoc_note = "Post-hoc: Tukey HSD"
        else:
            # Rule B: non-normal -> Mann-Whitney U / Kruskal-Wallis with Dunn
            if k == 2:
                _, p = mannwhitneyu(
                    groups_data[0], groups_data[1], alternative="two-sided"
                )
                method = "Mann-Whitney U test"
            else:
                _, p = kruskal(*groups_data)
                method = "Kruskal-Wallis test"
                if p < 0.05:
                    dunn_bonferroni(groups_data)  # run for the record
                    posthoc_note = "Post-hoc: Dunn test (Bonferroni)"

    # Rule D: ordinal variables
    elif var_type == "Ordinal":
        # Use rank tests, never the chi-squared test
        if k == 2:
            _, p = mannwhitneyu(groups_data[0], groups_data[1], alternative="two-sided")
            method = "Mann-Whitney U test"
        else:
            _, p = kruskal(*groups_data)
            method = "Kruskal-Wallis test"
            if p < 0.05:
                dunn_bonferroni(groups_data)
                posthoc_note = "Post-hoc: Dunn test (Bonferroni)"

    # Rule C: nominal variables
    elif var_type == "Nominal":
        contingency = pd.crosstab(
            data[feature].dropna(), data.loc[data[feature].notna(), group_col]
        )
        use_fisher, fisher_reason = check_fisher_needed(contingency)

        if use_fisher:
            if contingency.shape == (2, 2):
                _, p = fisher_exact(contingency)
                method = "Fisher's exact test"
            else:
                # rxc: Monte Carlo simulation
                p = monte_carlo_fisher_pvalue(data, feature, group_col)
                method = "Fisher's exact test (Monte Carlo)"
        else:
            _, p, _, _ = chi2_contingency(contingency)
            method = "Chi-square test"

    else:
        raise ValueError(f"Unknown variable type: {var_type}")

    return method, float(p), posthoc_note


# Alias used inside the functions above
FISHER_2X2_EXPECTED = FISHER_2X2_MIN_EXPECTED


# ================================================================
# (7) Table footnote
# ================================================================


def generate_footnotes(used_methods: dict) -> str:
    """Build the footnote describing the statistical methods."""
    lines = [
        "**Statistical footnotes:**",
        "",
        "Continuous variables were assessed for normality within each group using the "
        "Shapiro-Wilk test (n ≤ 50) or the Kolmogorov-Smirnov test with Lilliefors "
        "correction (n > 50); significance threshold was set at p < 0.05.",
    ]

    if used_methods.get("continuous_normal"):
        lines.append(
            "Normally distributed continuous variables are presented as Mean ± SD "
            "and compared using the independent-samples t-test (two groups) or "
            "one-way ANOVA followed by Tukey's HSD post-hoc test (≥ 3 groups)."
        )
    if used_methods.get("continuous_skewed"):
        lines.append(
            "Non-normally distributed continuous variables are presented as "
            "Median (Q1, Q3) and compared using the Mann-Whitney U test (two groups) "
            "or Kruskal-Wallis test followed by Dunn's test with Bonferroni correction "
            "(≥ 3 groups)."
        )
    if used_methods.get("nominal"):
        lines.append(
            "Nominal (unordered) categorical variables are presented as n (%) and "
            "compared using the Chi-square test, or Fisher's exact test when any "
            "expected cell frequency < 5 in 2×2 tables, or when > 20% of cells had "
            "expected frequency < 5 (or any cell < 1) in r×c tables; "
            "Monte Carlo permutation simulation (10,000 iterations) was applied for "
            "sparse r×c contingency tables."
        )
    if used_methods.get("ordinal"):
        lines.append(
            "Ordinal variables are presented as n (%) per category. "
            "The Chi-square test was not applied to ordinal data; "
            "group comparisons used the Mann-Whitney U test (two groups) "
            "or Kruskal-Wallis test with Dunn's post-hoc test (Bonferroni correction; "
            "≥ 3 groups)."
        )

    lines += [
        "",
        "Multiple comparisons were adjusted using the Benjamini-Hochberg false "
        "discovery rate (FDR) method (α = 0.05).",
        "",
        "All statistical analyses were performed using Python 3 with the following "
        "libraries: SciPy (v≥1.10), statsmodels (v≥0.14), pandas (v≥2.0), "
        "and NumPy (v≥1.24).",
        "",
        "Each participant contributes one baseline visit; observations are independent at the "
        "participant level. Univariable analyses remain exploratory and are not used for feature selection.",
        "",
        "Abbreviations: SD, standard deviation; Q1/Q3, first/third quartile; "
        "n, number of participants with available data; FDR, false discovery rate.",
    ]
    return "\n".join(lines)


# ================================================================
# (8) Univariable logistic GLM
# ================================================================


def fit_univariable_glm(
    y: pd.Series,
    x: pd.Series,
    feat_name: str,
) -> tuple[float, float, float, float, str]:
    """
    Fit a univariable logistic GLM.
    Returns (or_val, ci_low, ci_high, p_val, status_desc).
    Returns NaN with a status note if the fit does not converge or shows separation or extreme estimates.
    """
    try:
        # Design matrix with a constant
        X = sm.add_constant(x.astype(float), has_constant="add")
        y_num = y.astype(float)

        # Check the variance
        if x.nunique() <= 1:
            return np.nan, np.nan, np.nan, np.nan, "Zero variance"

        glm = sm.GLM(y_num, X, family=sm.families.Binomial())
        res = glm.fit()

        if hasattr(res, "converged") and not res.converged:
            print(f"  ⚠  {feat_name} GLM fitting did not converge!")
            return np.nan, np.nan, np.nan, np.nan, "Did not converge"

        param = float(res.params.iloc[1])
        se = float(res.bse.iloc[1])
        p_val = float(res.pvalues.iloc[1])

        # Check that the estimates are finite
        if (
            np.isnan(param)
            or np.isinf(param)
            or np.isnan(se)
            or np.isinf(se)
            or se <= 0
        ):
            print(
                f"  ⚠  {feat_name} non-finite parameter or SE (param={param}, se={se})"
            )
            return np.nan, np.nan, np.nan, np.nan, "Non-finite parameter or SE"

        # Check for separation or very large standard errors
        if se > 10.0 or abs(param) > 20.0:
            print(
                f"  ⚠  {feat_name} potential complete separation: param={param:.2f}, se={se:.2f}"
            )
            return np.nan, np.nan, np.nan, np.nan, "Potential separation (extreme SE)"

        or_val = float(np.exp(param))
        ci_low = float(np.exp(param - 1.96 * se))
        ci_high = float(np.exp(param + 1.96 * se))

        if not (ci_low <= or_val <= ci_high):
            print(
                f"  ⚠  {feat_name} abnormal CI order: {ci_low} <= {or_val} <= {ci_high}"
            )
            return np.nan, np.nan, np.nan, np.nan, "Abnormal CI"

        return or_val, ci_low, ci_high, p_val, "OK"
    except Exception as exc:
        print(f"  ⚠  {feat_name} GLM error: {exc}")
        return np.nan, np.nan, np.nan, np.nan, f"Error: {exc}"


def analyze_cohort(
    file_path: str, output_file: str, cohort_name: str = "Cohort"
) -> pd.DataFrame:
    """Run the stratified description and univariable logistic GLMs for one dataset and save a CSV file."""
    print(f"\n{'=' * 65}")
    print(f"[{cohort_name}] Reading data: {file_path}")
    print(f"{'=' * 65}")

    if not os.path.exists(file_path):
        print(f"Error: data file not found: {file_path}")
        return None

    data = pd.read_csv(file_path)

    if GROUP_COL not in data.columns:
        print(f'Error: column "{GROUP_COL}" not found; check the data file')
        return None

    # Feature columns (excluding the target and the first 5 columns)
    all_columns = [c for c in data.columns if c != GROUP_COL]
    feature_columns = all_columns[5:]

    groups = sorted(data[GROUP_COL].unique())
    group_labels = [GROUP_LABELS.get(g, f"Group {g}") for g in groups]

    n_total = len(data)
    n_subjects = data["PATNO"].nunique()
    print("Sample size:")
    print(f"  Participants: {n_subjects}")
    print(f"  Visit records: {n_total}")
    for g, lab in zip(groups, group_labels):
        print(f"  {lab} ({GROUP_COL}={g}): {len(data[data[GROUP_COL] == g])}")
    print(f"Features to analyse: {len(feature_columns)}")

    # Step 1: variable types and normality
    print("\n" + "=" * 65)
    print(f"[{cohort_name} step 1] Variable types and univariable models")
    print("=" * 65)

    var_type_dict = {}
    normality_dict = {}

    for feat in feature_columns:
        missing_rate = data[feat].isnull().mean()
        if missing_rate > MISSING_SKIP_THRESHOLD:
            print(f"\nWarning: {feat}: missing rate {missing_rate:.1%} > 50%, skipped")
            continue

        vtype, reason = get_variable_type(data[feat], feat)
        var_type_dict[feat] = vtype
        display = FEATURE_NAME_MAPPING.get(feat, feat)

        if vtype == "Continuous":
            is_normal, _ = check_normality(data, feat, GROUP_COL)
            normality_dict[feat] = is_normal

    # Step 2: univariable model for each variable
    y = data[GROUP_COL]
    table_rows = []
    raw_p_values = []
    parent_row_indices = []

    # Sample-size row
    n_row = {
        "Variable": "n",
        "Overall": str(n_total),
        "Univariable OR (95% CI)": "",
        "P Value": "",
        "Adjusted P Value": "",
    }
    for g, lab in zip(groups, group_labels):
        n_row[lab] = str(len(data[data[GROUP_COL] == g]))
    table_rows.append(n_row)

    pat_row = {
        "Variable": "Participants",
        "Overall": str(data["PATNO"].nunique()),
        "Univariable OR (95% CI)": "",
        "P Value": "",
        "Adjusted P Value": "",
    }
    for g, lab in zip(groups, group_labels):
        pat_row[lab] = str(data[data[GROUP_COL] == g]["PATNO"].nunique())
    table_rows.append(pat_row)

    for feat in feature_columns:
        if feat not in var_type_dict:
            continue

        vtype = var_type_dict[feat]
        is_normal = normality_dict.get(feat, False)
        display = FEATURE_NAME_MAPPING.get(feat, feat)
        cfg = FEATURE_OR_CONFIG.get(feat, {})

        # Rescale where configured (e.g. LEDD per 100 mg)
        scale = cfg.get("scale", 1.0)
        x_series = data[feat].astype(float) / scale

        # Fit the univariable model
        or_val, ci_l, ci_u, p_val, status = fit_univariable_glm(
            y=y,
            x=x_series,
            feat_name=feat,
        )

        or_ci_str = format_or_ci(or_val, ci_l, ci_u)
        raw_p_values.append(p_val)
        parent_row_idx = len(table_rows)
        parent_row_indices.append(parent_row_idx)

        # Continuous variable: one row
        if vtype == "Continuous":
            row = {
                "Variable": display,
                "Overall": describe_continuous(
                    data[feat].dropna().values, is_normal, feat=feat
                ),
                "Univariable OR (95% CI)": or_ci_str,
                "P Value": p_val,
                "Adjusted P Value": "",
            }
            for g, lab in zip(groups, group_labels):
                arr = data[data[GROUP_COL] == g][feat].dropna().values
                row[lab] = describe_continuous(arr, is_normal, feat=feat)
            table_rows.append(row)

        # Categorical or ordinal variable: one row per level
        else:
            parent = {
                "Variable": display,
                "Overall": "",
                "Univariable OR (95% CI)": or_ci_str,
                "P Value": p_val,
                "Adjusted P Value": "",
            }
            for lab in group_labels:
                parent[lab] = ""
            table_rows.append(parent)

            # Level rows
            level_labels_map = cfg.get("level_labels", {})
            for lr in describe_categorical_levels(data, feat, GROUP_COL):
                lvl = lr["level"]
                lvl_str = level_labels_map.get(lvl, str(lvl))
                sub = {
                    "Variable": f"  {lvl_str}",
                    "Overall": lr["overall"],
                    "Univariable OR (95% CI)": "",
                    "P Value": "",
                    "Adjusted P Value": "",
                }
                for g, lab in zip(groups, group_labels):
                    sub[lab] = lr[f"group_{g}"]
                table_rows.append(sub)

    # Step 3: Benjamini-Hochberg FDR correction within the dataset
    valid_p_indices = [
        (idx, p)
        for idx, p in zip(parent_row_indices, raw_p_values)
        if not np.isnan(p) and not np.isinf(p)
    ]
    if valid_p_indices:
        print(
            f"\n[{cohort_name} FDR] Benjamini-Hochberg correction of {len(valid_p_indices)} valid P values..."
        )
        p_to_adjust = [p for _, p in valid_p_indices]
        _, p_adjusted = fdrcorrection(p_to_adjust, alpha=FDR_ALPHA)

        for (row_idx, _), p_adj in zip(valid_p_indices, p_adjusted):
            table_rows[row_idx]["Adjusted P Value"] = format_p(p_adj)

    # Format the unadjusted P values
    for row in table_rows:
        if isinstance(row.get("P Value"), float):
            row["P Value"] = format_p(row["P Value"])

    # Build and write the table
    col_order = [
        "Variable",
        "Overall",
        group_labels[0],
        group_labels[1],
        "Univariable OR (95% CI)",
        "P Value",
        "Adjusted P Value",
    ]
    results_df = pd.DataFrame(table_rows, columns=col_order)

    # Print a Markdown table
    print("\n\n" + "=" * 65)
    print(f"[{cohort_name} Univariable Association & Characteristics Table]")
    print("=" * 65 + "\n")
    print(markdown_table(results_df))

    # Print the significant results
    print("\n" + "=" * 65)
    print(f"[{cohort_name} significant results] (FDR-adjusted P < 0.05)")
    print("=" * 65)
    for i, row in enumerate(table_rows):
        if i in parent_row_indices and row.get("Adjusted P Value"):
            p_str = row["Adjusted P Value"]
            try:
                p_val = 0.0001 if p_str == "<0.001" else float(p_str)
                if p_val < 0.05:
                    var = row["Variable"].strip()
                    or_str = row["Univariable OR (95% CI)"]
                    print(f"  {var}: OR={or_str}, p_adj={p_str}")
            except (ValueError, TypeError):
                pass

    # Save the CSV file
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n[OK] {cohort_name} results saved to: {output_file}")

    return results_df


def summarize_main_table(
    train_path: str = "PPMI_7_train.csv",
    test_path: str = "PPMI_7_test.csv",
    output_file: str = "step_09_characteristics_stats_main.csv",
) -> pd.DataFrame:
    """Build main Table 1: overall characteristics of the training and test sets, without hypothesis tests."""
    print(f"\n{'=' * 65}")
    print("[Main Table 1] Describing the training and test sets...")
    print(f"{'=' * 65}")

    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)

    def get_stats(df: pd.DataFrame) -> dict[str, str]:
        stats = {}
        # 3.1 Dataset structure
        n_pats = df["PATNO"].nunique()
        n_records = len(df)
        stats["Participants"] = str(n_pats)

        # 3.2 Key baseline characteristics
        # Age
        arr = df["age_at_visit"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["Age"] = f"{np.median(arr):.2f} ({q1:.2f}, {q3:.2f})"

        # Sex (Male)
        n_male = int((df["SEX"] == 1).sum())
        stats["Sex (Male)"] = f"{n_male} ({n_male / n_records * 100:.1f}%)"

        # Education
        arr = df["EDUCYRS"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["Education"] = f"{np.median(arr):.0f} ({q1:.0f}, {q3:.0f})"

        # FH-PD
        n_fh = int((df["fampd_bin"] == 1).sum())
        stats["FH-PD"] = f"{n_fh} ({n_fh / n_records * 100:.1f}%)"

        # Duration
        arr = df["duration"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["Duration"] = f"{np.median(arr):.2f} ({q1:.2f}, {q3:.2f})"

        # On medication
        n_med = int((df["PDTRTMNT"] == 1).sum())
        stats["On medication"] = f"{n_med} ({n_med / n_records * 100:.1f}%)"

        # LEDD (mg/day)
        arr = df["LEDD"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["LEDD (mg/day)"] = f"{np.median(arr):.0f} ({q1:.0f}, {q3:.0f})"

        # PIGD (mean score, typically reported to 1 decimal place)
        arr = df["pigd"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["PIGD"] = f"{np.median(arr):.1f} ({q1:.1f}, {q3:.1f})"

        # UPDRS-III
        arr = df["updrs3_score"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["UPDRS-III"] = f"{np.median(arr):.1f} ({q1:.1f}, {q3:.1f})"

        # GDS (integer scale 0-15)
        arr = df["gds"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["GDS"] = f"{np.median(arr):.0f} ({q1:.0f}, {q3:.0f})"

        # RBDSQ (integer scale 0-13)
        arr = df["rem"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["RBDSQ"] = f"{np.median(arr):.0f} ({q1:.0f}, {q3:.0f})"

        # SCOPA-AUT (integer scale 0-69)
        arr = df["scopa"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["SCOPA-AUT"] = f"{np.median(arr):.0f} ({q1:.0f}, {q3:.0f})"

        return stats

    s_tr = get_stats(tr)
    s_te = get_stats(te)

    main_df = pd.DataFrame(
        {
            "Characteristic": list(s_tr.keys()),
            f"Training set (n = {len(tr)})": list(s_tr.values()),
            f"Validation set (n = {len(te)})": list(s_te.values()),
        }
    )

    print("\n" + "=" * 65)
    print("[Table 1 Content Preview]")
    print("=" * 65 + "\n")
    print(markdown_table(main_df))

    main_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Main Table 1 saved to: {output_file}")

    return main_df


def main():
    # 1. Training set
    df_train = analyze_cohort(
        file_path="PPMI_7_train.csv",
        output_file="step_09_characteristics_stats_train.csv",
        cohort_name="Training Set",
    )

    # 2. Test set
    df_test = analyze_cohort(
        file_path="PPMI_7_test.csv",
        output_file="step_09_characteristics_stats_test.csv",
        cohort_name="Validation Set",
    )

    # 3. Main Table 1
    df_main = summarize_main_table(
        train_path="PPMI_7_train.csv",
        test_path="PPMI_7_test.csv",
        output_file="step_09_characteristics_stats_main.csv",
    )

    print("\n" + "=" * 65)
    print("[SUCCESS] Characteristics and univariable associations finished. Outputs:")
    print("  1. step_09_characteristics_stats_train.csv (Table S10, training set)")
    print("  2. step_09_characteristics_stats_test.csv (Table S10, test set)")
    print("  3. step_09_characteristics_stats_main.csv (main Table 1)")
    print("=" * 65)

    return df_train, df_test, df_main


if __name__ == "__main__":
    results = main()

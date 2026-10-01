"""
基线单访视特征描述与单变量关联分析。

输出包括：
  - 训练集和受试者隔离验证集的总体描述（正文 Table 1）
  - 按 PD-NC / PD-MCI 分层的受试者层面描述（补充 Table S10）
  - 单变量 logistic GLM 的 OR 与模型标准 95% CI，
    以及在每个数据集内分别进行 Benjamini-Hochberg 校正的 P 值

单变量关联分析仅用于描述/探索，不参与 LASSO 特征选择。
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
# ① 配置区 —— 修改此处即可调整分析行为
# ================================================================

# 分组列名与显示标签
GROUP_COL = "MCI"
GROUP_LABELS = {0: "PD-NC", 1: "PD-MCI"}

# 变量类型手动覆盖（优先级高于自动检测）
# 可选值：'Continuous'（连续）/ 'Nominal'（无序分类）/ 'Ordinal'（有序分类）
VARIABLE_TYPE_OVERRIDE: dict[str, str] = {
    "SEX": "Nominal",  # 性别：无序二分类
    "NHY": "Ordinal",  # H&Y 分期：有序分类
    "APOE_e4": "Ordinal",  # APOE ε4：有序等级变量
    "quip": "Continuous",  # QUIP 冲动控制障碍量表总分
}

# 特征显示名称映射（包含全部 31 个候选特征）
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

# 连续变量有效数字精度定制（整数量表/剂量去 .00，临床评分 1 位，浮点特征 2 位）
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

# 特征单变量 OR 模型配置与子类标签
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

# 正态性判断阈值
NORMALITY_ALPHA = 0.05

# 卡方 → Fisher 切换阈值
FISHER_2X2_MIN_EXPECTED = 5  # 2×2：任一格期望频数 < 5 则改用 Fisher
FISHER_RXC_PROPORTION = 0.20  # r×c：超过 20% 的格子期望频数 < 5
FISHER_RXC_MIN_ANY = 1  # r×c：任一格期望频数 < 1

# Monte Carlo 置换检验模拟次数（r×c Fisher）
MONTE_CARLO_N = 10_000
MONTE_CARLO_SEED = 42

# 缺失率超过此阈值时跳过该变量
MISSING_SKIP_THRESHOLD = 0.50

# FDR 校正阈值
FDR_ALPHA = 0.05


# ================================================================
# ② 工具函数
# ================================================================


def format_p(p: float) -> str:
    """格式化 p 值：极小值写为 <0.001，其余保留 3 位小数"""
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "NE"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_or_ci(or_val: float, ci_low: float, ci_high: float) -> str:
    """格式化 OR (95% CI)：保留 2 位小数，使用标准短横线或 en-dash"""
    if any(
        v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
        for v in [or_val, ci_low, ci_high]
    ):
        return "NE"
    return f"{or_val:.2f} ({ci_low:.2f}-{ci_high:.2f})"


def markdown_table(df: pd.DataFrame) -> str:
    """将 DataFrame 转为 Markdown 表格（不依赖 tabulate）"""
    cols = df.columns.tolist()
    # 计算每列最大宽度
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
# ③ Step 1：变量类型判断
# ================================================================


def auto_detect_type(series: pd.Series) -> tuple[str, str]:
    """
    自动判断变量类型
    返回 (类型, 判断理由)
    """
    data = series.dropna()
    n_unique = data.nunique()

    # 判断是否全为整数值
    all_int = all(
        isinstance(x, (int, np.integer)) or (isinstance(x, float) and x.is_integer())
        for x in data
    )

    if n_unique <= 2:
        return "Nominal", f"唯一值数量为 {n_unique}（≤2），判断为二分类无序变量"
    elif n_unique <= 8 and all_int:
        return "Nominal", (
            f"唯一值数量为 {n_unique}（≤8）且全为整数，自动判断为无序分类变量；"
            '若实为有序分类，请在 VARIABLE_TYPE_OVERRIDE 中手动指定 "Ordinal"'
        )
    else:
        return (
            "Continuous",
            f"唯一值数量为 {n_unique}（>8）或包含非整数，判断为连续变量",
        )


def get_variable_type(series: pd.Series, col_name: str) -> tuple[str, str]:
    """获取变量类型（手动覆盖优先）"""
    if col_name in VARIABLE_TYPE_OVERRIDE:
        vtype = VARIABLE_TYPE_OVERRIDE[col_name]
        return vtype, f'手动指定为 "{vtype}"（VARIABLE_TYPE_OVERRIDE）'
    return auto_detect_type(series)


# ================================================================
# ④ Step 2：正态性检验
# ================================================================


def normality_test_single(arr: np.ndarray) -> tuple[float, str]:
    """
    对单组数据进行正态性检验
    n ≤ 50 → Shapiro-Wilk；n > 50 → KS-Lilliefors
    返回 (p 值, 检验方法名)
    """
    n = len(arr)
    if n < 3:
        return 1.0, "N/A（样本量不足）"
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
    对各组分别做正态性检验，全部组 p ≥ 0.05 才认为正态
    返回 (is_normal, 详细描述)
    """
    groups = sorted(data[group_col].unique())
    details = []

    for g in groups:
        arr = data[data[group_col] == g][feature].dropna().values
        p, met = normality_test_single(arr)
        details.append(f"Group {g} [{met}, p={p:.3f}, n={len(arr)}]")

    # 重新提取 p 值列表做判断
    all_normal = True
    for g in groups:
        arr = data[data[group_col] == g][feature].dropna().values
        p, _ = normality_test_single(arr)
        if p < NORMALITY_ALPHA:
            all_normal = False
            break

    return all_normal, "; ".join(details)


# ================================================================
# ⑤ Step 3：描述统计
# ================================================================


def describe_continuous(arr: np.ndarray, is_normal: bool, feat: str = None) -> str:
    """连续变量描述：根据临床特征定制精度（0位/1位/2位）"""
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
    分类/有序变量按等级展开描述
    每个等级返回一个字典，包含 overall 及各组的 'n (%)' 字符串
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
# ⑥ Step 3：组间比较检验
# ================================================================


def dunn_bonferroni(groups_data: list[np.ndarray]) -> dict[tuple, float]:
    """
    Dunn 检验（Bonferroni 校正）—— Kruskal-Wallis 事后检验
    含并列秩修正
    返回 {(i, j): bonferroni_p} 两两比较 p 值字典
    """
    k = len(groups_data)
    n_comparisons = k * (k - 1) // 2
    all_data = np.concatenate(groups_data)
    n = len(all_data)
    ranks = stats.rankdata(all_data)

    # 并列修正项
    _, counts = np.unique(all_data, return_counts=True)
    tie_correction = np.sum(counts**3 - counts) / (12 * (n - 1)) if n > 1 else 0.0

    # 各组平均秩
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
    r×c Fisher 精确检验（Monte Carlo 置换模拟）
    对分组标签进行随机置换，计算卡方统计量的零分布，返回置换 p 值
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
        # 添加微小值避免全零列导致的数值问题
        ps, *_ = chi2_contingency(pt.values + 1e-10)
        if ps >= obs_stat:
            count_extreme += 1
    return count_extreme / MONTE_CARLO_N


def check_fisher_needed(contingency_table: pd.DataFrame) -> tuple[bool, str]:
    """
    判断是否需要改用 Fisher 精确检验
    返回 (需要, 原因说明)
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
                f"2×2 表中最小期望频数 {min_expected:.2f} < {FISHER_2X2_EXPECTED}",
            )
        return False, ""
    else:
        if min_expected < FISHER_RXC_MIN_ANY:
            return True, f"r×c 表中存在期望频数 {min_expected:.2f} < 1"
        if n_small / n_cells > FISHER_RXC_PROPORTION:
            return (
                True,
                f"r×c 表中 {n_small}/{n_cells} 个格子期望频数 < 5（>{FISHER_RXC_PROPORTION:.0%}）",
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
    执行组间比较
    返回 (检验方法名, p 值, 事后检验说明)
    """
    groups = sorted(data[group_col].unique())
    k = len(groups)
    groups_data = [data[data[group_col] == g][feature].dropna().values for g in groups]

    posthoc_note = ""

    # ── 规则 A / B：连续变量 ──────────────────────────────────────
    if var_type == "Continuous":
        if is_normal:
            # 规则 A：正态 → t 检验 / ANOVA + Tukey HSD
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
                    pairwise_tukeyhsd(all_vals, all_labels)  # 执行（结果供记录）
                    posthoc_note = "Post-hoc: Tukey HSD"
        else:
            # 规则 B：非正态 → Mann-Whitney U / Kruskal-Wallis + Dunn
            if k == 2:
                _, p = mannwhitneyu(
                    groups_data[0], groups_data[1], alternative="two-sided"
                )
                method = "Mann-Whitney U test"
            else:
                _, p = kruskal(*groups_data)
                method = "Kruskal-Wallis test"
                if p < 0.05:
                    dunn_bonferroni(groups_data)  # 执行（结果供记录）
                    posthoc_note = "Post-hoc: Dunn test (Bonferroni)"

    # ── 规则 D：有序分类变量 ──────────────────────────────────────
    elif var_type == "Ordinal":
        # 绝不使用卡方检验；使用秩检验
        if k == 2:
            _, p = mannwhitneyu(groups_data[0], groups_data[1], alternative="two-sided")
            method = "Mann-Whitney U test"
        else:
            _, p = kruskal(*groups_data)
            method = "Kruskal-Wallis test"
            if p < 0.05:
                dunn_bonferroni(groups_data)
                posthoc_note = "Post-hoc: Dunn test (Bonferroni)"

    # ── 规则 C：无序分类变量 ──────────────────────────────────────
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
                # r×c：Monte Carlo 模拟
                p = monte_carlo_fisher_pvalue(data, feature, group_col)
                method = "Fisher's exact test (Monte Carlo)"
        else:
            _, p, _, _ = chi2_contingency(contingency)
            method = "Chi-square test"

    else:
        raise ValueError(f"未知变量类型：{var_type}")

    return method, float(p), posthoc_note


# 修正拼写错误：统一变量名
FISHER_2X2_EXPECTED = FISHER_2X2_MIN_EXPECTED  # 兼容内部引用


# ================================================================
# ⑦ 脚注生成
# ================================================================


def generate_footnotes(used_methods: dict) -> str:
    """生成 SCI 规范的统计学方法脚注"""
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
# ⑧ 单变量 Logistic GLM
# ================================================================


def fit_univariable_glm(
    y: pd.Series,
    x: pd.Series,
    feat_name: str,
) -> tuple[float, float, float, float, str]:
    """
    拟合单变量 logistic GLM。
    返回 (or_val, ci_low, ci_high, p_val, status_desc)
    如果未收敛或存在完全分离/极端估计，返回 NaN 并标注状态。
    """
    try:
        # 数据集准备与常数项
        X = sm.add_constant(x.astype(float), has_constant="add")
        y_num = y.astype(float)

        # 检查方差
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

        # 检查是否有限
        if (
            np.isnan(param)
            or np.isinf(param)
            or np.isnan(se)
            or np.isinf(se)
            or se <= 0
        ):
            print(f"  ⚠  {feat_name} non-finite parameter or SE (param={param}, se={se})")
            return np.nan, np.nan, np.nan, np.nan, "Non-finite parameter or SE"

        # 检查完全分离/极大标准误
        if se > 10.0 or abs(param) > 20.0:
            print(f"  ⚠  {feat_name} potential complete separation: param={param:.2f}, se={se:.2f}")
            return np.nan, np.nan, np.nan, np.nan, "Potential separation (extreme SE)"

        or_val = float(np.exp(param))
        ci_low = float(np.exp(param - 1.96 * se))
        ci_high = float(np.exp(param + 1.96 * se))

        if not (ci_low <= or_val <= ci_high):
            print(f"  ⚠  {feat_name} abnormal CI order: {ci_low} <= {or_val} <= {ci_high}")
            return np.nan, np.nan, np.nan, np.nan, "Abnormal CI"

        return or_val, ci_low, ci_high, p_val, "OK"
    except Exception as exc:
        print(f"  ⚠  {feat_name} GLM error: {exc}")
        return np.nan, np.nan, np.nan, np.nan, f"Error: {exc}"


def analyze_cohort(file_path: str, output_file: str, cohort_name: str = "Cohort") -> pd.DataFrame:
    """对指定数据集执行分层描述与单变量 Logistic GLM 分析并保存 CSV 文件"""
    print(f"\n{'=' * 65}")
    print(f"【{cohort_name}】正在读取数据：{file_path}")
    print(f"{'=' * 65}")

    if not os.path.exists(file_path):
        print(f"Error: 数据文件不存在：{file_path}")
        return None

    data = pd.read_csv(file_path)

    if GROUP_COL not in data.columns:
        print(f'Error: 未找到列 "{GROUP_COL}"，请检查数据文件')
        return None

    # 特征列（排除目标列及前 5 列）
    all_columns = [c for c in data.columns if c != GROUP_COL]
    feature_columns = all_columns[5:]

    groups = sorted(data[GROUP_COL].unique())
    group_labels = [GROUP_LABELS.get(g, f"Group {g}") for g in groups]

    n_total = len(data)
    n_subjects = data["PATNO"].nunique()
    print("样本量统计：")
    print(f"  受试者数：{n_subjects}")
    print(f"  访视记录总计：{n_total}")
    for g, lab in zip(groups, group_labels):
        print(f"  {lab}（{GROUP_COL}={g}）：{len(data[data[GROUP_COL] == g])}")
    print(f"待分析特征数：{len(feature_columns)}")

    # ── Step 1：变量类型与正态性判断 ──────────────────────────────
    print("\n" + "=" * 65)
    print(f"【{cohort_name} Step 1】变量类型判断与单变量关联建模")
    print("=" * 65)

    var_type_dict = {}
    normality_dict = {}

    for feat in feature_columns:
        missing_rate = data[feat].isnull().mean()
        if missing_rate > MISSING_SKIP_THRESHOLD:
            print(f"\n⚠  {feat}：缺失率 {missing_rate:.1%} > 50%，跳过")
            continue

        vtype, reason = get_variable_type(data[feat], feat)
        var_type_dict[feat] = vtype
        display = FEATURE_NAME_MAPPING.get(feat, feat)

        if vtype == "Continuous":
            is_normal, _ = check_normality(data, feat, GROUP_COL)
            normality_dict[feat] = is_normal

    # ── Step 2：拟合各变量单变量模型 ─────────────────────────────
    y = data[GROUP_COL]
    table_rows = []
    raw_p_values = []
    parent_row_indices = []

    # 样本量行
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

        # 缩放处理（例如 LEDD 除以 100）
        scale = cfg.get("scale", 1.0)
        x_series = data[feat].astype(float) / scale

        # 拟合单变量模型
        or_val, ci_l, ci_u, p_val, status = fit_univariable_glm(
            y=y,
            x=x_series,
            feat_name=feat,
        )

        or_ci_str = format_or_ci(or_val, ci_l, ci_u)
        raw_p_values.append(p_val)
        parent_row_idx = len(table_rows)
        parent_row_indices.append(parent_row_idx)

        # ── 连续变量：单行 ────────────────────────────────────────
        if vtype == "Continuous":
            row = {
                "Variable": display,
                "Overall": describe_continuous(data[feat].dropna().values, is_normal, feat=feat),
                "Univariable OR (95% CI)": or_ci_str,
                "P Value": p_val,
                "Adjusted P Value": "",
            }
            for g, lab in zip(groups, group_labels):
                arr = data[data[GROUP_COL] == g][feat].dropna().values
                row[lab] = describe_continuous(arr, is_normal, feat=feat)
            table_rows.append(row)

        # ── 分类 / 有序变量：展开各等级 ─────────────────────────────
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

            # 各等级子行
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

    # ── Step 3：数据集内 Benjamini-Hochberg FDR 校正 ──────────────
    valid_p_indices = [
        (idx, p)
        for idx, p in zip(parent_row_indices, raw_p_values)
        if not np.isnan(p) and not np.isinf(p)
    ]
    if valid_p_indices:
        print(f"\n【{cohort_name} FDR 校正】对 {len(valid_p_indices)} 个有效 P 值进行 Benjamini-Hochberg 校正...")
        p_to_adjust = [p for _, p in valid_p_indices]
        _, p_adjusted = fdrcorrection(p_to_adjust, alpha=FDR_ALPHA)

        for (row_idx, _), p_adj in zip(valid_p_indices, p_adjusted):
            table_rows[row_idx]["Adjusted P Value"] = format_p(p_adj)

    # 格式化原始 P 值
    for row in table_rows:
        if isinstance(row.get("P Value"), float):
            row["P Value"] = format_p(row["P Value"])

    # ── 构建并输出 DataFrame ──────────────────────────────────────
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

    # 打印 Markdown 表格
    print("\n\n" + "=" * 65)
    print(f"【{cohort_name} Univariable Association & Characteristics Table】")
    print("=" * 65 + "\n")
    print(markdown_table(results_df))

    # 打印显著结果
    print("\n" + "=" * 65)
    print(f"【{cohort_name} 显著结果汇总】（FDR 校正后 p < 0.05）")
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

    # 保存 CSV 文件
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n[OK] {cohort_name} 结果已保存至：{output_file}")

    return results_df


def summarize_main_table(
    train_path: str = "PPMI_7_train.csv",
    test_path: str = "PPMI_7_test.csv",
    output_file: str = "step_09_characteristics_stats_main.csv",
) -> pd.DataFrame:
    """生成正文 Table 1：训练集和受试者隔离验证集的总体描述表（不含假设检验）"""
    print(f"\n{'=' * 65}")
    print("【正文 Table 1】生成训练集与隔离验证集总体特征描述表...")
    print(f"{'=' * 65}")

    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)

    def get_stats(df: pd.DataFrame) -> dict[str, str]:
        stats = {}
        # 3.1 数据集结构
        n_pats = df["PATNO"].nunique()
        n_records = len(df)
        stats["Participants"] = str(n_pats)

        # 3.2 关键访视层面特征
        # Age
        arr = df["age_at_visit"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["Age"] = (
            f"{np.median(arr):.2f} ({q1:.2f}, {q3:.2f})"
        )

        # Sex (Male)
        n_male = int((df["SEX"] == 1).sum())
        stats["Sex (Male)"] = f"{n_male} ({n_male / n_records * 100:.1f}%)"

        # Education
        arr = df["EDUCYRS"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["Education"] = (
            f"{np.median(arr):.0f} ({q1:.0f}, {q3:.0f})"
        )

        # FH-PD
        n_fh = int((df["fampd_bin"] == 1).sum())
        stats["FH-PD"] = (
            f"{n_fh} ({n_fh / n_records * 100:.1f}%)"
        )

        # Duration
        arr = df["duration"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["Duration"] = (
            f"{np.median(arr):.2f} ({q1:.2f}, {q3:.2f})"
        )

        # On medication
        n_med = int((df["PDTRTMNT"] == 1).sum())
        stats["On medication"] = f"{n_med} ({n_med / n_records * 100:.1f}%)"

        # LEDD (mg/day)
        arr = df["LEDD"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["LEDD (mg/day)"] = (
            f"{np.median(arr):.0f} ({q1:.0f}, {q3:.0f})"
        )

        # PIGD (mean score, typically reported to 1 decimal place)
        arr = df["pigd"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["PIGD"] = f"{np.median(arr):.1f} ({q1:.1f}, {q3:.1f})"

        # UPDRS-III
        arr = df["updrs3_score"].values
        q1, q3 = np.percentile(arr, [25, 75])
        stats["UPDRS-III"] = (
            f"{np.median(arr):.1f} ({q1:.1f}, {q3:.1f})"
        )

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
        stats["SCOPA-AUT"] = (
            f"{np.median(arr):.0f} ({q1:.0f}, {q3:.0f})"
        )

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
    print("【Table 1 Content Preview】")
    print("=" * 65 + "\n")
    print(markdown_table(main_df))

    main_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 正文 Table 1 描述统计已保存至：{output_file}")

    return main_df


def main():
    # 1. 分析训练集
    df_train = analyze_cohort(
        file_path="PPMI_7_train.csv",
        output_file="step_09_characteristics_stats_train.csv",
        cohort_name="Training Set",
    )

    # 2. 分析测试集
    df_test = analyze_cohort(
        file_path="PPMI_7_test.csv",
        output_file="step_09_characteristics_stats_test.csv",
        cohort_name="Validation Set",
    )

    # 3. 生成正文 Table 1
    df_main = summarize_main_table(
        train_path="PPMI_7_train.csv",
        test_path="PPMI_7_test.csv",
        output_file="step_09_characteristics_stats_main.csv",
    )

    print("\n" + "=" * 65)
    print("[SUCCESS] 访视特征与单变量关联分析完成！已输出：")
    print("  1. step_09_characteristics_stats_train.csv (Table S10 训练集)")
    print("  2. step_09_characteristics_stats_test.csv (Table S10 验证集)")
    print("  3. step_09_characteristics_stats_main.csv (正文 Table 1)")
    print("=" * 65)

    return df_train, df_test, df_main


if __name__ == "__main__":
    results = main()

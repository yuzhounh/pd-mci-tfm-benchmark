import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体为 Arial
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

# 读取主数据
df = pd.read_csv('PPMI_7_train.csv')

# 拟分析的全部候选特征列表（后续将只挑选部分用于展示热力图、大部分用于降维去冗余，不直接作为最终模型保留特征）
features = [
    'age_at_visit', 'SEX', 'EDUCYRS', 'fampd_bin', 'BMI', 'agediag', 
    'ageonset', 'sym_tremor', 'sym_rigid', 'sym_brady', 'sym_posins', 
    'sym_other', 'PDTRTMNT', 'LEDD', 'quip', 'ess', 'rem', 'gds', 
    'stai_state', 'stai_trait', 'scopa', 'orthostasis', 'NHY', 'pigd', 
    'td_pigd', 'updrs2_score', 'updrs3_score', 'APOE_e4', 'has_GBA', 
    'has_LRRK2', 'duration'
]

# 读取特征-缩写映射表
_csv_df = pd.read_csv('PPMI_feature_mapping.csv')
feature_name_mapping = dict(zip(_csv_df['feature'], _csv_df['abbreviation']))

# 筛选上述特征的原始值
df_features = df[features]

# 计算斯皮尔曼秩相关系数矩阵 (Spearman Rank Correlation)
# 方法学说明：特征池包含二分类变量 (SEX, fampd_bin, sym_*)、有序等级变量 (NHY) 及非正态连续变量，
# Spearman 秩相关较 Pearson 具有更强的方法学稳健性与适用性。
correlation_matrix = df_features.corr(method='spearman')

# 设定后续相关性展示/分析的阈值（如用于热图展示）
CORRELATION_THRESHOLD = 0.6

# 选出与任意其它特征相关系数绝对值大于等于阈值的特征。
# 实际是用于后续热图展示用，并非最终保留全部，仅作分析和可视化。
def get_features_above_correlation_threshold(corr_matrix, threshold=0.2):
    """
    挑出与任意其它特征相关系数绝对值大于阈值的特征（用于后续相关性分析/热力图展示）
    """
    features_selected = []
    features_unselected = []
    for i, feature in enumerate(corr_matrix.columns):
        # 去除自身后，观察与其他所有特征的相关系数
        correlations = corr_matrix.iloc[i, :].drop(feature)
        if (abs(correlations) >= threshold).any():
            features_selected.append(feature)
        else:
            features_unselected.append(feature)
    return features_selected, features_unselected

# 筛选用于热力图展示的特征
features_for_heatmap, features_with_consistently_low_corr = get_features_above_correlation_threshold(
    correlation_matrix, 
    threshold=CORRELATION_THRESHOLD
)

print("=" * 80)
print(f"可用于相关性热图展示的特征 (与任意其他特征的相关系数绝对值 >= {CORRELATION_THRESHOLD}):")
print("=" * 80)
print(f"\n可纳入展示的特征数量: {len(features_for_heatmap)}")
print(f"始终低相关且未纳入的特征数量: {len(features_with_consistently_low_corr)}")

if features_with_consistently_low_corr:
    print(f"\n以下特征与所有其他特征的相关系数绝对值均 < {CORRELATION_THRESHOLD}（未纳入相关性热图）:")
    for feature in features_with_consistently_low_corr:
        max_corr = correlation_matrix[feature].drop(feature).abs().max()
        print(f"  - {feature:20s} (最大相关系数: {max_corr:.4f})")
else:
    print(f"\n所有特征都至少与一个其他特征有高于阈值的相关性")

print("\n可用于热力图展示的特征（并非最终用于模型的全部特征）:")
for feature in features_for_heatmap:
    max_corr = correlation_matrix[feature].drop(feature).abs().max()
    print(f"  - {feature:20s} (最大相关系数: {max_corr:.4f})")

# 用相关性高于阈值的特征集画相关系数矩阵热图（主要为分析/可视化冗余特征，后续还需进一步去除相关性高的特征以保证特征集互相独立性）
correlation_matrix_filtered = correlation_matrix.loc[features_for_heatmap, features_for_heatmap]

# 行列名用缩写替换，方便排版
correlation_matrix_filtered_renamed = correlation_matrix_filtered.rename(
    index=feature_name_mapping, 
    columns=feature_name_mapping
)

# 做上三角掩码，仅显示下三角
mask = np.triu(np.ones_like(correlation_matrix_filtered_renamed, dtype=bool))

plt.figure(figsize=(6, 5))
sns.heatmap(
    correlation_matrix_filtered_renamed, 
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('step_10_correlation_heatmap.pdf', bbox_inches='tight')

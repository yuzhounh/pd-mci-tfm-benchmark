import pandas as pd


RAW_DATA_FILE = "PPMI_Curated_Data_Cut_Public_20260810.xlsx"
EXPECTED_PD_COHORT = (8546, 1684)
EXPECTED_BASELINE_COHORT = (1684, 1684)

# The 2026-08-10 curated cut renamed four DATSCAN columns. Keep the established
# downstream feature names so the remaining pipeline and manuscript tables use a
# stable interface.
SOURCE_COLUMN_ALIASES = {
    "MIA_LOWPUT_EXPECTED": "mia_lowput_expected",
    "MIA_CAUDATE_BILAT": "MIA_CAUDATE_mean",
    "MIA_PUTAMEN_BILAT": "MIA_PUTAMEN_mean",
    "MIA_STRIATUM_BILAT": "MIA_STRIATUM_mean",
}


def normalize_source_columns(df):
    rename_map = {}
    for source_name, canonical_name in SOURCE_COLUMN_ALIASES.items():
        if source_name in df.columns and canonical_name in df.columns:
            raise RuntimeError(
                f"Raw data contains both {source_name} and {canonical_name}; "
                "the DATSCAN alias is ambiguous"
            )
        if source_name in df.columns:
            rename_map[source_name] = canonical_name
    return df.rename(columns=rename_map)

print("\n开始处理PPMI数据...")

# 步骤1: 读取Excel文件并保存为CSV
print("\n步骤1: 读取Excel文件...")
# 读取Excel文件
df_original = pd.read_excel(RAW_DATA_FILE)
print(f"原始数据形状: {df_original.shape}")

# 保存为CSV
df_original.to_csv("PPMI_0_data.csv", index=False)
print(f"已保存 PPMI_0_data.csv, 形状: {df_original.shape}")

# 仅规范化已确认的上游字段改名；PPMI_0_data.csv 仍保留源文件原始列名。
df_source = normalize_source_columns(df_original)

# 步骤2: 筛选COHORT列为1的行
print("\n步骤2: 筛选COHORT列为1的行...")
# 筛选COHORT为1的行
df_pd = df_source[df_source["COHORT"] == 1].copy()
print(f"筛选后数据形状: {df_pd.shape}")

# 输出被试的数目（PATNO列不重复元素的数目）
num_subjects = df_pd["PATNO"].nunique()
print(f"被试（PATNO唯一数）数量: {num_subjects}")
if (len(df_pd), num_subjects) != EXPECTED_PD_COHORT:
    raise RuntimeError(
        "Locked initial PD cohort changed: "
        f"expected {EXPECTED_PD_COHORT[0]} visits/{EXPECTED_PD_COHORT[1]} subjects, "
        f"got {len(df_pd)} visits/{num_subjects} subjects"
    )

# 单访视分析仅保留基线（BL）。PPMI 当前 cut 中每位 PD 受试者恰有一条 BL 记录。
df_pd = df_pd[df_pd["EVENT_ID"] == "BL"].copy()
baseline_subjects = df_pd["PATNO"].nunique()
if (len(df_pd), baseline_subjects) != EXPECTED_BASELINE_COHORT:
    raise RuntimeError(
        "Locked baseline PD cohort changed: "
        f"expected {EXPECTED_BASELINE_COHORT[0]} rows/"
        f"{EXPECTED_BASELINE_COHORT[1]} subjects, "
        f"got {len(df_pd)} rows/{baseline_subjects} subjects"
    )
if df_pd["PATNO"].duplicated().any():
    raise RuntimeError("Baseline cohort contains duplicate PATNO values")

# 保存单访视 PD 队列
df_pd.to_csv("PPMI_1_PD.csv", index=False)
print(f"已保存 PPMI_1_PD.csv（BL 单访视）, 形状: {df_pd.shape}")

# 步骤3: 读取风险因子配置文件
print("\n步骤3: 处理风险因子特征...")
# 读取风险因子配置
df_risk_config = pd.read_csv("PPMI_initial_feature_sets.csv")
print(f"风险因子配置文件形状: {df_risk_config.shape}")
print("配置文件前几行:")
print(df_risk_config.head())

# 获取flag为1的特征列表
selected_features = df_risk_config[df_risk_config["flag"] == 1]["feature"].tolist()
print(f"选中的特征数量: {len(selected_features)}")
print(f"选中的特征: {selected_features[:10]}...")  # 显示前10个特征

# 检查这些特征在数据中是否存在
available_features = [feat for feat in selected_features if feat in df_pd.columns]
missing_features = [feat for feat in selected_features if feat not in df_pd.columns]

print(f"数据中可用的特征数量: {len(available_features)}")
if missing_features:
    raise RuntimeError(
        "Required features are missing after source-column normalization: "
        f"{missing_features}"
    )

# 提取选中的特征
df_risk_selected = df_pd[available_features].copy()
print(f"提取风险因子后数据形状: {df_risk_selected.shape}")

# 保存结果
df_risk_selected.to_csv("PPMI_2_features.csv", index=False)
print(f"已保存 PPMI_2_features.csv, 形状: {df_risk_selected.shape}")

# 为保持后续步骤的既有输入文件名，另存一份不经跨表回填的相同数据
df_risk_selected.to_csv("PPMI_3_source.csv", index=False)
print(f"已保存 PPMI_3_source.csv（curated 数据副本）, 形状: {df_risk_selected.shape}")

print("\n数据处理完成!")
print("=" * 50)
print("总结:")
print(f"1. PPMI_0_data.csv: {df_original.shape}")
print(f"2. PPMI_1_PD.csv: {df_pd.shape}")
print(f"3. PPMI_2_features.csv: {df_risk_selected.shape}")
print(f"4. PPMI_3_source.csv: {df_risk_selected.shape}")

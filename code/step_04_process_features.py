import pandas as pd
import numpy as np

print("\n=== Step 04: Process Baseline Features & Cohort Filtering ===")

# 读取表格
df = pd.read_csv("PPMI_3_source.csv", low_memory=False)
initial_visits = len(df)
initial_subjects = df["PATNO"].nunique()
print(f"Initial records from PPMI_3_source: {initial_visits} baseline rows, {initial_subjects} subjects")
if (initial_visits, initial_subjects) != (1684, 1684):
    raise RuntimeError(
        "Locked cohort entry changed: "
        f"expected 1684 baseline rows/1684 subjects, got {initial_visits}/{initial_subjects}"
    )
if df["PATNO"].duplicated().any() or not df["EVENT_ID"].eq("BL").all():
    raise RuntimeError("Single-visit invariant failed: expected one BL row per PATNO")

# 协议规则 1: 剔除结局 MCI_testscores 缺失的访视 (不回填、不插补)
if "MCI_testscores" in df.columns:
    missing_mci_mask = df["MCI_testscores"].isna()
    n_missing_mci = missing_mci_mask.sum()
    print(f"Excluded visits with missing MCI_testscores: {n_missing_mci}")
    df = df[~missing_mci_mask].copy()
print(f"After removing missing outcome: {len(df)} rows, {df['PATNO'].nunique()} subjects")
if (len(df), df["PATNO"].nunique()) != (1666, 1666):
    raise RuntimeError(
        "Locked missing-outcome exclusion changed: "
        f"expected 1666 rows/1666 subjects, got {len(df)}/{df['PATNO'].nunique()}"
    )

# 协议规则 2: 剔除明确诊断为 PDD (cogstate == 3.0) 的访视记录
if "cogstate" in df.columns:
    pdd_mask = df["cogstate"] == 3.0
    n_pdd = pdd_mask.sum()
    pdd_subjects = df.loc[pdd_mask, "PATNO"].nunique()
    print(f"Excluded visits with PDD (cogstate == 3.0): {n_pdd} visits across {pdd_subjects} subjects")
    df = df[~pdd_mask].copy()
print(f"After excluding PDD: {len(df)} rows, {df['PATNO'].nunique()} subjects")
if (len(df), df["PATNO"].nunique()) != (1665, 1665):
    raise RuntimeError(
        "Locked PDD exclusion changed: "
        f"expected 1665 rows/1665 subjects, got {len(df)}/{df['PATNO'].nunique()}"
    )

# 筛选出 flag 为 1 的特征，并保留对应列（保持原df中的列顺序）
feature_sets = pd.read_csv("PPMI_feature_sets.csv")
selected_features = feature_sets[feature_sets["flag"] == 1]["feature"].tolist()
existing_selected_features = [col for col in df.columns if col in selected_features]
df = df[existing_selected_features].copy()

# 将 BMI 值超过45的置空
df.loc[df["BMI"] > 45, "BMI"] = pd.NA

# 将被试 41172 的起病年龄置空
# 临床质控说明：受试者 41172 确诊年龄 (agediag) 为 73.83 岁，但原始表格记录的起病年龄 (ageonset) 为 19.62 岁，
# 二者差值超过 54 年，在原发性帕金森病中属于极不合理的生理异常（疑似 PPMI 录入笔误）。
# 将其置为 NA，交由 step_06 基于队列确诊-起病年龄中位差值进行生理合理性修补。
df.loc[df["PATNO"] == 41172, "ageonset"] = pd.NA

# 对 fampd_bin 和 td_pigd 进行重新编码：将 {1, 2} 映射为 {1, 0}
df["fampd_bin"] = df["fampd_bin"].map({1: 1, 2: 0})
df["td_pigd"] = df["td_pigd"].map({1: 1, 2: 0})

# 创建 has_GBA 和 has_LRRK2 二元特征列
df["has_GBA"] = df["subgroup"].str.contains("GBA", case=False, na=False).astype(int)
df["has_LRRK2"] = df["subgroup"].str.contains("LRRK2", case=False, na=False).astype(int)

# 将 MCI_testscores 列移动到表格最右侧并重命名为 MCI
if "MCI_testscores" in df.columns:
    mci_col = df.pop("MCI_testscores")
    df["MCI"] = mci_col

# 最终确保 MCI 无缺失
df = df.dropna(subset=["MCI"]).copy()

# 输出样本统计
print("\n=== Final Cohort Statistics (Locked) ===")
print(f"Final valid baseline rows: {len(df)}")
print(f"Final unique subjects: {df['PATNO'].nunique()}")
print(f"Feature count: {len(df.columns) - 6}")
print("Label distribution (MCI):")
val_counts = df["MCI"].value_counts().to_dict()
val_props = df["MCI"].value_counts(normalize=True).to_dict()
for k in sorted(val_counts.keys()):
    print(f"  Class {int(k)}: {val_counts[k]} ({val_props[k]*100:.2f}%)")

# 保存结果
df.to_csv("PPMI_4_features.csv", index=False)
print(f"\nSaved PPMI_4_features.csv successfully with shape {df.shape}")

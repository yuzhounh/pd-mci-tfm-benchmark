"""Step 04: apply the eligibility rules and recode the candidate features.

Excludes participants without the PD-MCI outcome or with documented dementia,
keeps the flagged features, applies clinical quality-control fixes and recodes
binary variables. Output: PPMI_4_features.csv.
"""

import pandas as pd
import numpy as np

print("\n=== Step 04: Process Baseline Features & Cohort Filtering ===")

# Read the table
df = pd.read_csv("PPMI_3_source.csv", low_memory=False)
initial_visits = len(df)
initial_subjects = df["PATNO"].nunique()
print(
    f"Initial records from PPMI_3_source: {initial_visits} baseline rows, {initial_subjects} subjects"
)
if (initial_visits, initial_subjects) != (1684, 1684):
    raise RuntimeError(
        "Locked cohort entry changed: "
        f"expected 1684 baseline rows/1684 subjects, got {initial_visits}/{initial_subjects}"
    )
if df["PATNO"].duplicated().any() or not df["EVENT_ID"].eq("BL").all():
    raise RuntimeError("Single-visit invariant failed: expected one BL row per PATNO")

# Protocol rule 1: exclude records with missing MCI_testscores (no backfilling or imputation)
if "MCI_testscores" in df.columns:
    missing_mci_mask = df["MCI_testscores"].isna()
    n_missing_mci = missing_mci_mask.sum()
    print(f"Excluded visits with missing MCI_testscores: {n_missing_mci}")
    df = df[~missing_mci_mask].copy()
print(
    f"After removing missing outcome: {len(df)} rows, {df['PATNO'].nunique()} subjects"
)
if (len(df), df["PATNO"].nunique()) != (1666, 1666):
    raise RuntimeError(
        "Locked missing-outcome exclusion changed: "
        f"expected 1666 rows/1666 subjects, got {len(df)}/{df['PATNO'].nunique()}"
    )

# Protocol rule 2: exclude records with a documented PDD diagnosis (cogstate == 3.0)
if "cogstate" in df.columns:
    pdd_mask = df["cogstate"] == 3.0
    n_pdd = pdd_mask.sum()
    pdd_subjects = df.loc[pdd_mask, "PATNO"].nunique()
    print(
        f"Excluded visits with PDD (cogstate == 3.0): {n_pdd} visits across {pdd_subjects} subjects"
    )
    df = df[~pdd_mask].copy()
print(f"After excluding PDD: {len(df)} rows, {df['PATNO'].nunique()} subjects")
if (len(df), df["PATNO"].nunique()) != (1665, 1665):
    raise RuntimeError(
        "Locked PDD exclusion changed: "
        f"expected 1665 rows/1665 subjects, got {len(df)}/{df['PATNO'].nunique()}"
    )

# Keep the features with flag == 1, in their original column order
feature_sets = pd.read_csv("PPMI_feature_sets.csv")
selected_features = feature_sets[feature_sets["flag"] == 1]["feature"].tolist()
existing_selected_features = [col for col in df.columns if col in selected_features]
df = df[existing_selected_features].copy()

# Set BMI values above 45 to missing
df.loc[df["BMI"] > 45, "BMI"] = pd.NA

# Set the age at onset of participant 41172 to missing
# Clinical QC: participant 41172 has age at diagnosis 73.83 years but a recorded age at onset of 19.62 years.
# A gap of more than 54 years is implausible in idiopathic PD (a suspected entry error),
# so it is set to NA and filled in step_06 from the median diagnosis-onset interval.
df.loc[df["PATNO"] == 41172, "ageonset"] = pd.NA

# Recode fampd_bin and td_pigd: {1, 2} -> {1, 0}
df["fampd_bin"] = df["fampd_bin"].map({1: 1, 2: 0})
df["td_pigd"] = df["td_pigd"].map({1: 1, 2: 0})

# Binary has_GBA and has_LRRK2 features
df["has_GBA"] = df["subgroup"].str.contains("GBA", case=False, na=False).astype(int)
df["has_LRRK2"] = df["subgroup"].str.contains("LRRK2", case=False, na=False).astype(int)

# Move MCI_testscores to the last column and rename it MCI
if "MCI_testscores" in df.columns:
    mci_col = df.pop("MCI_testscores")
    df["MCI"] = mci_col

# Make sure MCI has no missing values
df = df.dropna(subset=["MCI"]).copy()

# Print the sample counts
print("\n=== Final Cohort Statistics (Locked) ===")
print(f"Final valid baseline rows: {len(df)}")
print(f"Final unique subjects: {df['PATNO'].nunique()}")
print(f"Feature count: {len(df.columns) - 6}")
print("Label distribution (MCI):")
val_counts = df["MCI"].value_counts().to_dict()
val_props = df["MCI"].value_counts(normalize=True).to_dict()
for k in sorted(val_counts.keys()):
    print(f"  Class {int(k)}: {val_counts[k]} ({val_props[k]*100:.2f}%)")

# Save the results
df.to_csv("PPMI_4_features.csv", index=False)
print(f"\nSaved PPMI_4_features.csv successfully with shape {df.shape}")

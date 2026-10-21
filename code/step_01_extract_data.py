"""Step 01: extract the baseline PD cohort and the initial candidate features.

Reads the PPMI curated data cut, keeps the PD cohort (COHORT == 1) at the
baseline visit, harmonises renamed source columns and writes PPMI_0 to PPMI_3
for the following steps.
"""

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


print("\nProcessing PPMI data...")

# Step 1: read the Excel file and save it as CSV
print("\nStep 1: reading the Excel file...")
# Read the Excel file
df_original = pd.read_excel(RAW_DATA_FILE)
print(f"Raw data shape: {df_original.shape}")

# Save as CSV
df_original.to_csv("PPMI_0_data.csv", index=False)
print(f"Saved PPMI_0_data.csv, shape: {df_original.shape}")

# Only normalise confirmed upstream renames; PPMI_0_data.csv keeps the original column names.
df_source = normalize_source_columns(df_original)

# Step 2: keep rows with COHORT == 1
print("\nStep 2: keeping rows with COHORT == 1...")
# Keep COHORT == 1
df_pd = df_source[df_source["COHORT"] == 1].copy()
print(f"Shape after filtering: {df_pd.shape}")

# Number of participants (unique PATNO)
num_subjects = df_pd["PATNO"].nunique()
print(f"Participants (unique PATNO): {num_subjects}")
if (len(df_pd), num_subjects) != EXPECTED_PD_COHORT:
    raise RuntimeError(
        "Locked initial PD cohort changed: "
        f"expected {EXPECTED_PD_COHORT[0]} visits/{EXPECTED_PD_COHORT[1]} subjects, "
        f"got {len(df_pd)} visits/{num_subjects} subjects"
    )

# Keep only the baseline (BL) visit; each PD participant has exactly one BL record in this cut.
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

# Save the baseline PD cohort
df_pd.to_csv("PPMI_1_PD.csv", index=False)
print(f"Saved PPMI_1_PD.csv (BL visits), shape: {df_pd.shape}")

# Step 3: read the candidate-feature configuration
print("\nStep 3: selecting candidate features...")
# Read the configuration
df_risk_config = pd.read_csv("PPMI_initial_feature_sets.csv")
print(f"Configuration shape: {df_risk_config.shape}")
print("First rows of the configuration:")
print(df_risk_config.head())

# Features with flag == 1
selected_features = df_risk_config[df_risk_config["flag"] == 1]["feature"].tolist()
print(f"Selected features: {len(selected_features)}")
print(f"Selected features: {selected_features[:10]}...")  # first 10

# Check that the features exist in the data
available_features = [feat for feat in selected_features if feat in df_pd.columns]
missing_features = [feat for feat in selected_features if feat not in df_pd.columns]

print(f"Features available in the data: {len(available_features)}")
if missing_features:
    raise RuntimeError(
        "Required features are missing after source-column normalization: "
        f"{missing_features}"
    )

# Extract the selected features
df_risk_selected = df_pd[available_features].copy()
print(f"Shape after feature selection: {df_risk_selected.shape}")

# Save the results
df_risk_selected.to_csv("PPMI_2_features.csv", index=False)
print(f"Saved PPMI_2_features.csv, shape: {df_risk_selected.shape}")

# Save the same data under the input name used by later steps (no cross-table backfilling)
df_risk_selected.to_csv("PPMI_3_source.csv", index=False)
print(
    f"Saved PPMI_3_source.csv (copy of the curated data), shape: {df_risk_selected.shape}"
)

print("\nData processing finished.")
print("=" * 50)
print("Summary:")
print(f"1. PPMI_0_data.csv: {df_original.shape}")
print(f"2. PPMI_1_PD.csv: {df_pd.shape}")
print(f"3. PPMI_2_features.csv: {df_risk_selected.shape}")
print(f"4. PPMI_3_source.csv: {df_risk_selected.shape}")

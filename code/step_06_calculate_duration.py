"""Step 06: derive disease duration at baseline.

Missing age at diagnosis is filled with the training-set median, and missing age
at onset with age at diagnosis minus the training-set median interval between
onset and diagnosis. Outputs: PPMI_6_train.csv and PPMI_6_test.csv.
"""

import pandas as pd
import numpy as np

# Read data
train_df = pd.read_csv("PPMI_5_train.csv")
test_df = pd.read_csv("PPMI_5_test.csv")

print("Training set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("\nTraining set columns:", train_df.columns.tolist())

# Check missing values for agediag and ageonset
print("\n=== Training Set Missing Values ===")
print(f"agediag missing: {train_df['agediag'].isna().sum()}")
print(f"ageonset missing: {train_df['ageonset'].isna().sum()}")

print("\n=== Test Set Missing Values ===")
print(f"agediag missing: {test_df['agediag'].isna().sum()}")
print(f"ageonset missing: {test_df['ageonset'].isna().sum()}")

# Step 1: Fill agediag
# Calculate agediag for each patient in training set (take first non-missing value per patient)
train_agediag_per_patient = train_df.groupby("PATNO")["agediag"].first()
agediag_median = train_agediag_per_patient.median()

print(f"\n=== Step 1: Fill agediag ===")
print(f"Median agediag from training set: {agediag_median}")

# Fill agediag in both training and test sets
train_df["agediag"] = train_df["agediag"].fillna(agediag_median)
test_df["agediag"] = test_df["agediag"].fillna(agediag_median)

print(
    f"After filling - Training set agediag missing: {train_df['agediag'].isna().sum()}"
)
print(f"After filling - Test set agediag missing: {test_df['agediag'].isna().sum()}")


# Step 2: Fill ageonset
# Calculate difference between agediag and ageonset for each patient in training set (only for non-missing values)
def get_patient_diff(group):
    """Get agediag - ageonset difference for each patient"""
    agediag_val = group["agediag"].iloc[0]
    ageonset_val = group["ageonset"].iloc[0]

    if pd.notna(agediag_val) and pd.notna(ageonset_val):
        return agediag_val - ageonset_val
    else:
        return np.nan


# Calculate difference for each patient
train_diff_per_patient = train_df.groupby("PATNO", group_keys=False).apply(
    get_patient_diff, include_groups=False
)
# Remove missing values and calculate median
diff_median = train_diff_per_patient.dropna().median()

print(f"\n=== Step 2: Fill ageonset ===")
print(f"Median difference (agediag - ageonset) from training set: {diff_median}")

# Fill missing ageonset using agediag - diff_median
train_df["ageonset"] = train_df.apply(
    lambda row: (
        row["agediag"] - diff_median if pd.isna(row["ageonset"]) else row["ageonset"]
    ),
    axis=1,
)

test_df["ageonset"] = test_df.apply(
    lambda row: (
        row["agediag"] - diff_median if pd.isna(row["ageonset"]) else row["ageonset"]
    ),
    axis=1,
)

print(
    f"After filling - Training set ageonset missing: {train_df['ageonset'].isna().sum()}"
)
print(f"After filling - Test set ageonset missing: {test_df['ageonset'].isna().sum()}")

# Step 3: Calculate duration
print("\n=== Step 3: Calculate duration ===")
train_df["duration"] = train_df["age_at_visit"] - train_df["ageonset"]
test_df["duration"] = test_df["age_at_visit"] - test_df["ageonset"]

print(
    f"Training set - duration range: [{train_df['duration'].min():.2f}, {train_df['duration'].max():.2f}]"
)
print(
    f"Test set - duration range: [{test_df['duration'].min():.2f}, {test_df['duration'].max():.2f}]"
)

# Insert duration column at second-to-last position (before MCI)
# Get column lists
train_cols = train_df.columns.tolist()
test_cols = test_df.columns.tolist()

# Find MCI position
mci_idx_train = train_cols.index("MCI")
mci_idx_test = test_cols.index("MCI")

# Remove duration column
train_cols.remove("duration")
test_cols.remove("duration")

# Insert duration before MCI
train_cols.insert(mci_idx_train, "duration")
test_cols.insert(mci_idx_test, "duration")

# Reorder columns
train_df = train_df[train_cols]
test_df = test_df[test_cols]

print("\n=== Column order (last 3 columns) ===")
print("Training set:", train_df.columns[-3:].tolist())
print("Test set:", test_df.columns[-3:].tolist())

# Save results
train_output_file = "PPMI_6_train.csv"
test_output_file = "PPMI_6_test.csv"

train_df.to_csv(train_output_file, index=False)
test_df.to_csv(test_output_file, index=False)

print("\n=== Files saved successfully ===")
print(train_output_file)
print(test_output_file)

# Display sample data
print("\n=== Sample data from training set ===")
print(
    train_df[["PATNO", "age_at_visit", "agediag", "ageonset", "duration", "MCI"]].head(
        10
    )
)

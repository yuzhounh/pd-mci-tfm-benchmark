"""Step 07: print the missing-value summary of the training set before imputation."""

import pandas as pd

# Read the CSV file
df = pd.read_csv("PPMI_6_train.csv")

# Missing count of each feature
missing_count = df.isnull().sum()

# Missing percentage of each feature
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Summary table
missing_df = pd.DataFrame(
    {"Missing Count": missing_count, "Missing Percentage (%)": missing_percentage}
)

# Sort by missing percentage, highest first
missing_df = missing_df.sort_values(by="Missing Percentage (%)", ascending=False)

# Show the results
print("=" * 60)
print("Missing values by feature (highest percentage first)")
print("=" * 60)
print(f"Total records: {len(df)}")
print("-" * 60)
print(missing_df)
print("-" * 60)

# Show only features with missing values
missing_features = missing_df[missing_df["Missing Count"] > 0]
if len(missing_features) > 0:
    print(f"\nFeatures with missing values: {len(missing_features)}")
else:
    print("\nNo feature has missing values")

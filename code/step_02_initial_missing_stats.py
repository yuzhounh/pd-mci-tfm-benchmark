import pandas as pd

# Load the dataset
df = pd.read_csv('PPMI_2_features.csv')

# Calculate missing values count and percentage
missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Create summary table
missing_stats = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage (%)': missing_percentage
})

# Sort by missing values count in descending order
missing_stats = missing_stats.sort_values(by='Missing Count', ascending=False)

print("Missing values statistics per column:")
print(missing_stats)

# Save result to CSV file
missing_stats.to_csv('step_02_initial_missing_stats.csv')
print("\nResults saved to step_02_initial_missing_stats.csv")

# Generate PPMI_feature_sets.csv
MISSING_THRESHOLD = 30.0

risk_factors = pd.read_csv('PPMI_initial_feature_sets.csv')
candidate_features = risk_factors.loc[risk_factors['flag'] == 1, 'feature'].tolist()

feature_sets_rows = []
for feat in candidate_features:
    # 明确排除临床诊断变量，不可作为预测自变量
    if feat in ['cogstate']:
        flag = 0
    elif feat in missing_percentage.index:
        pct = missing_percentage[feat]
        flag = 0 if pct > MISSING_THRESHOLD else 1
    else:
        # Feature not present in the data file; treat as fully missing
        pct = 100.0
        flag = 0
    feature_sets_rows.append({'feature': feat, 'flag': flag})

feature_sets_df = pd.DataFrame(feature_sets_rows, columns=['feature', 'flag'])
feature_sets_df.to_csv('PPMI_feature_sets.csv', index=False)
print("\nResults saved to PPMI_feature_sets.csv")

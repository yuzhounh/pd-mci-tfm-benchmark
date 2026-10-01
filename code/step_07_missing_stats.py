import pandas as pd

# 读取CSV文件
df = pd.read_csv('PPMI_6_train.csv')

# 计算每个特征的缺失数量
missing_count = df.isnull().sum()

# 计算每个特征的缺失比例
missing_percentage = (df.isnull().sum() / len(df)) * 100

# 创建汇总DataFrame
missing_df = pd.DataFrame({
    '缺失数量': missing_count,
    '缺失比例(%)': missing_percentage
})

# 按缺失比例从高到低排序
missing_df = missing_df.sort_values(by='缺失比例(%)', ascending=False)

# 显示结果
print("=" * 60)
print("特征缺失统计（按缺失比例从高到低排列）")
print("=" * 60)
print(f"总样本数: {len(df)}")
print("-" * 60)
print(missing_df)
print("-" * 60)

# 只显示有缺失的特征
missing_features = missing_df[missing_df['缺失数量'] > 0]
if len(missing_features) > 0:
    print(f"\n有缺失值的特征数量: {len(missing_features)}")
else:
    print("\n所有特征均无缺失值")
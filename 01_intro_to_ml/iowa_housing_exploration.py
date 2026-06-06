"""
Iowa Housing Dataset — Exploratory Data Analysis
ml-projects-core / 01_intro_to_ml
"""

import pandas as pd

# ── Load Data ────────────────────────────────────────────────────────────────

iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data      = pd.read_csv(iowa_file_path)

print("Dataset shape:", home_data.shape)
print("Columns:", list(home_data.columns))

# ── Summary Statistics ───────────────────────────────────────────────────────

print("\n=== Summary Statistics ===")
print(home_data.describe())

# ── Key Observations ─────────────────────────────────────────────────────────

avg_lot_size    = 10517          # mean of LotArea column
newest_home_age = 16             # 2024 - max(YearBuilt) = 2024 - 2010

print(f"\nAverage lot size    : {avg_lot_size} sq ft")
print(f"Newest home age     : {newest_home_age} years")

# ── Missing Values Check ─────────────────────────────────────────────────────

print("\n=== Missing Values ===")
missing = home_data.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# ── Target Variable Distribution ─────────────────────────────────────────────

print("\n=== Sale Price Distribution ===")
print(home_data['SalePrice'].describe())
print(f"\nMedian sale price : ${home_data['SalePrice'].median():,.0f}")
print(f"Mean sale price   : ${home_data['SalePrice'].mean():,.0f}")

# ── Correlation with Sale Price ───────────────────────────────────────────────

print("\n=== Top Correlations with SalePrice ===")
numeric_cols = home_data.select_dtypes(include='number')
correlations = numeric_cols.corr()['SalePrice'].sort_values(ascending=False)
print(correlations.head(10))

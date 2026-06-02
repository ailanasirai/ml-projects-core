"""
Model Validation — Train/Test Split and MAE
ml-projects-core / 01_intro_to_ml
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── Load Data ────────────────────────────────────────────────────────────────

iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data      = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
feature_columns = [
    'LotArea', 'YearBuilt', '1stFlrSF',
    '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
]
X = home_data[feature_columns]

# ── Step 1 — Train/Validation Split ──────────────────────────────────────────

# 80% training, 20% validation — standard split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

print(f"Training samples  : {len(train_X)}")
print(f"Validation samples: {len(val_X)}")

# ── Step 2 — Fit Model on Training Data Only ──────────────────────────────────

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

print("\nModel fitted on training data.")

# ── Step 3 — Predict on Validation Data ──────────────────────────────────────

val_predictions = iowa_model.predict(val_X)

print("\n=== Predicted vs Actual (first 5 homes) ===")
for i in range(5):
    print(f"Predicted: ${val_predictions[i]:>10,.0f}  |  Actual: ${val_y.iloc[i]:>10,}")

# ── Step 4 — Mean Absolute Error ─────────────────────────────────────────────

val_mae = mean_absolute_error(val_y, val_predictions)

print(f"\nMean Absolute Error: ${val_mae:,.0f}")

# ── Key Observation ───────────────────────────────────────────────────────────

# MAE on validation data is much higher than on training data.
# This gap is the overfitting signature.
# Training MAE: near 0 (model memorized training data)
# Validation MAE: ~29,652 (model struggles on unseen data)
# Next step: tune max_leaf_nodes to find the sweet spot.

in_sample_mae = mean_absolute_error(train_y, iowa_model.predict(train_X))
print(f"Training MAE      : ${in_sample_mae:,.0f}")
print(f"Validation MAE    : ${val_mae:,.0f}")
print(f"\nGap = ${val_mae - in_sample_mae:,.0f} — this is the overfitting cost.")

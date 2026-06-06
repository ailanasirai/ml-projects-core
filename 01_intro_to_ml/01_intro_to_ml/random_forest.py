"""
Random Forest Regressor — Iowa Housing Price Prediction
ml-projects-core / 01_intro_to_ml
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ── Load Data ────────────────────────────────────────────────────────────────

iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data      = pd.read_csv(iowa_file_path)

y        = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X        = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# ── Baseline — Decision Tree ──────────────────────────────────────────────────

dt_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
dt_model.fit(train_X, train_y)
dt_mae = mean_absolute_error(val_y, dt_model.predict(val_X))

print(f"Decision Tree MAE  : {dt_mae:,.0f}")

# ── Random Forest ─────────────────────────────────────────────────────────────

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_mae = mean_absolute_error(val_y, rf_model.predict(val_X))

print(f"Random Forest MAE  : {rf_mae:,.0f}")

# ── Comparison ────────────────────────────────────────────────────────────────

improvement = dt_mae - rf_mae
print(f"\nImprovement        : {improvement:,.0f}")
print(f"Error reduced by   : {(improvement/dt_mae)*100:.1f}%")

# ── Why Random Forest Works Better ───────────────────────────────────────────

print("\n=== Key Insight ===")
print("Decision Tree  : one tree, prone to overfitting")
print("Random Forest  : many trees, each trained on random subset")
print("Final prediction = average of all trees = more stable, lower error")

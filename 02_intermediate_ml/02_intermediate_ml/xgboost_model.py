"""
XGBoost — Gradient Boosting for Tabular Data
ml-projects-core / 02_intermediate_ml
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ── Load Data ────────────────────────────────────────────────────────────────

X      = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

low_cardinality_cols = [c for c in X_train_full.columns if
                        X_train_full[c].nunique() < 10 and X_train_full[c].dtype == "object"]
numeric_cols = [c for c in X_train_full.columns if X_train_full[c].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

# ── Step 1 -- Baseline XGBoost ────────────────────────────────────────────────

my_model_1 = XGBRegressor(random_state=0)
my_model_1.fit(X_train, y_train)

predictions_1 = my_model_1.predict(X_valid)
mae_1 = mean_absolute_error(predictions_1, y_valid)

print(f"Baseline XGBoost MAE   : {mae_1:,.0f}")

# ── Step 2 -- Tuned XGBoost ────────────────────────────────────────────────────

# More trees + lower learning rate = slower but more precise learning
my_model_2 = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=0)
my_model_2.fit(X_train, y_train)

predictions_2 = my_model_2.predict(X_valid)
mae_2 = mean_absolute_error(predictions_2, y_valid)

print(f"Tuned XGBoost MAE      : {mae_2:,.0f}")
print(f"Improvement            : {mae_1 - mae_2:,.0f}")

# ── Step 3 -- Deliberately Broken XGBoost ─────────────────────────────────────

# Too few trees + too high learning rate = underfitting, poor results
my_model_3 = XGBRegressor(n_estimators=1, learning_rate=1, random_state=0)
my_model_3.fit(X_train, y_train)

predictions_3 = my_model_3.predict(X_valid)
mae_3 = mean_absolute_error(predictions_3, y_valid)

print(f"Broken XGBoost MAE     : {mae_3:,.0f}")

# ── Key Insight ────────────────────────────────────────────────────────────────

print("\n=== n_estimators and learning_rate Tradeoff ===")
print("n_estimators    : number of boosting rounds (trees added sequentially)")
print("learning_rate   : how much each tree corrects the previous error")
print("More rounds + smaller learning rate = slower training, better accuracy")
print("Too few rounds or too large a learning rate = underfitting")

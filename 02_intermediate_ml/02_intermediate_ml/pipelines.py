"""
Pipelines — Preprocessing and Modeling in One Step
ml-projects-core / 02_intermediate_ml
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ── Load Data ────────────────────────────────────────────────────────────────

X_full      = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, y, train_size=0.8, test_size=0.2, random_state=0
)

categorical_cols = [c for c in X_train_full.columns if
                    X_train_full[c].nunique() < 10 and X_train_full[c].dtype == "object"]
numerical_cols = [c for c in X_train_full.columns if
                  X_train_full[c].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test  = X_test_full[my_cols].copy()

# ── Baseline Pipeline ──────────────────────────────────────────────────────────

baseline_numerical_transformer = SimpleImputer(strategy='constant')

baseline_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

baseline_preprocessor = ColumnTransformer(transformers=[
    ('num', baseline_numerical_transformer, numerical_cols),
    ('cat', baseline_categorical_transformer, categorical_cols)
])

baseline_model = RandomForestRegressor(n_estimators=100, random_state=0)

baseline_pipeline = Pipeline(steps=[
    ('preprocessor', baseline_preprocessor),
    ('model', baseline_model)
])

baseline_pipeline.fit(X_train, y_train)
baseline_preds = baseline_pipeline.predict(X_valid)
baseline_mae   = mean_absolute_error(y_valid, baseline_preds)

print(f"Baseline MAE : {baseline_mae:,.0f}")

# ── Improved Pipeline ──────────────────────────────────────────────────────────

# Median imputation handles outliers better than constant fill
numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# More trees, more stable predictions
model = RandomForestRegressor(n_estimators=200, random_state=0)

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
score = mean_absolute_error(y_valid, preds)

print(f"Improved MAE : {score:,.0f}")
print(f"Improvement  : {baseline_mae - score:,.0f}")

# ── Test Predictions ──────────────────────────────────────────────────────────

preds_test = my_pipeline.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

print(f"\nsubmission.csv created — {len(output)} predictions")

# ── Why Pipelines Matter ──────────────────────────────────────────────────────

print("\n=== Key Insight ===")
print("Pipeline bundles preprocessing + model into ONE object.")
print("fit() and predict() handle imputation, encoding, and prediction together.")
print("This prevents data leakage and makes code reusable on new data.")

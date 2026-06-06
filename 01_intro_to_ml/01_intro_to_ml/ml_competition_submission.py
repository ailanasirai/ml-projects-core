"""
Machine Learning Competition — Iowa Housing Price Prediction
Kaggle Housing Prices Competition Submission
ml-projects-core / 01_intro_to_ml
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ── Load Training Data ────────────────────────────────────────────────────────

iowa_file_path = '../input/train.csv'
home_data      = pd.read_csv(iowa_file_path)

y        = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X        = home_data[features]

# ── Validation Check ──────────────────────────────────────────────────────────

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))

print(f"Validation MAE: {rf_val_mae:,.0f}")

# ── Train on Full Data ────────────────────────────────────────────────────────

# Once validated, train on all available data for best predictions
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X, y)

print(f"Model trained on {len(X)} samples")

# ── Load Test Data and Predict ────────────────────────────────────────────────

test_data  = pd.read_csv('../input/test.csv')
test_X     = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)

print(f"Predictions generated: {len(test_preds)}")
print(f"Avg predicted price  : ${test_preds.mean():,.0f}")

# ── Generate Submission File ──────────────────────────────────────────────────

output = pd.DataFrame({
    'Id'       : test_data.Id,
    'SalePrice': test_preds
})
output.to_csv('submission.csv', index=False)

print("\nsubmission.csv created successfully")
print(output.head())

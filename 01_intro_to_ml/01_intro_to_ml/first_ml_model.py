"""
First Machine Learning Model — Decision Tree Regressor
ml-projects-core / 01_intro_to_ml
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# ── Load Data ────────────────────────────────────────────────────────────────

iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data      = pd.read_csv(iowa_file_path)

# ── Step 1 — Prediction Target ───────────────────────────────────────────────

y = home_data.SalePrice

print("Target variable: SalePrice")
print(f"Total samples  : {len(y)}")
print(f"Price range    : ${y.min():,} to ${y.max():,}")
print(f"Median price   : ${y.median():,.0f}")

# ── Step 2 — Select Features ─────────────────────────────────────────────────

feature_names = [
    'LotArea',       # total lot size in sq ft
    'YearBuilt',     # year construction completed
    '1stFlrSF',      # first floor sq footage
    '2ndFlrSF',      # second floor sq footage
    'FullBath',      # full bathrooms above ground
    'BedroomAbvGr',  # bedrooms above ground
    'TotRmsAbvGrd'   # total rooms above ground
]

X = home_data[feature_names]

print("\n=== Feature Summary ===")
print(X.describe())
print("\nFirst 5 rows:")
print(X.head())

# ── Step 3 — Define and Fit Model ────────────────────────────────────────────

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(X, y)

print("\nModel fitted successfully.")
print(f"Model type : {type(iowa_model).__name__}")
print(f"Features   : {feature_names}")

# ── Step 4 — Make Predictions ────────────────────────────────────────────────

predictions = iowa_model.predict(X)

print("\n=== Prediction vs Actual (first 5 homes) ===")
for i in range(5):
    print(f"Predicted: ${predictions[i]:>10,.0f}  |  Actual: ${y.iloc[i]:>10,}")

# ── Key Observation ──────────────────────────────────────────────────────────

# Predictions match actuals exactly on training data -- this is overfitting.
# Decision trees memorize training data perfectly.
# Next step: validate on unseen data to get a true accuracy picture.

print("\nNote: Perfect predictions on training data = overfitting.")
print("Model validation on unseen data is the next step.")

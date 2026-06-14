"""
Cross-Validation — Hyperparameter Selection
ml-projects-core / 02_intermediate_ml
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# ── Load Data ────────────────────────────────────────────────────────────────

train_data = pd.read_csv('../input/train.csv', index_col='Id')
test_data  = pd.read_csv('../input/test.csv', index_col='Id')

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

numeric_cols = [c for c in train_data.columns if train_data[c].dtype in ['int64', 'float64']]
X      = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

# ── Cross-Validation Score Function ───────────────────────────────────────────

def get_score(n_estimators):
    """Return average MAE over 3 CV folds for a given number of trees.

    Cross-validation splits the data into multiple folds, trains on
    some, validates on others, and rotates. This gives a more reliable
    error estimate than a single train/validation split.
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
    ])

    scores = -1 * cross_val_score(pipeline, X, y,
                                   cv=3,
                                   scoring='neg_mean_absolute_error')

    return scores.mean()

# ── Test Multiple n_estimators Values ─────────────────────────────────────────

results = {}
for i in range(1, 9):
    n_trees = 50 * i
    mae     = get_score(n_trees)
    results[n_trees] = mae
    print(f"n_estimators={n_trees:>4}  →  MAE={mae:,.0f}")

# ── Find Best Value ────────────────────────────────────────────────────────────

n_estimators_best = min(results, key=results.get)

print(f"\nBest n_estimators : {n_estimators_best}")
print(f"Best MAE          : {results[n_estimators_best]:,.0f}")

# ── Visualize ─────────────────────────────────────────────────────────────────

plt.plot(list(results.keys()), list(results.values()))
plt.xlabel('n_estimators')
plt.ylabel('MAE')
plt.title('Cross-Validation: Trees vs Error')
plt.show()

# ── Key Insight ───────────────────────────────────────────────────────────────

print("\n=== Why Cross-Validation Matters ===")
print("A single train/val split can give a lucky or unlucky result.")
print("Cross-validation averages across multiple splits.")
print("More reliable signal for choosing hyperparameters like n_estimators.")

"""
Underfitting and Overfitting — Decision Tree Optimization
ml-projects-core / 01_intro_to_ml
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# ── Load Data ────────────────────────────────────────────────────────────────

iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data      = pd.read_csv(iowa_file_path)

y        = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X        = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# ── MAE Function ─────────────────────────────────────────────────────────────

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """Return MAE for a Decision Tree with given max_leaf_nodes."""
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_y, preds_val)

# ── Step 1 -- Find Best Tree Size ────────────────────────────────────────────

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

mae_scores = {
    leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y)
    for leaf_size in candidate_max_leaf_nodes
}

print("=== MAE by Tree Size ===")
for size, mae in sorted(mae_scores.items()):
    print(f"  max_leaf_nodes={size:>4}  →  MAE={mae:,.0f}")

best_tree_size = min(mae_scores, key=mae_scores.get)
print(f"\nBest tree size: {best_tree_size}")
print(f"Best MAE      : {mae_scores[best_tree_size]:,.0f}")

# ── Step 2 -- Final Model on All Data ────────────────────────────────────────

# Once best size is found, train on full dataset for deployment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)

print(f"\nFinal model trained on {len(X)} samples")
print(f"max_leaf_nodes = {best_tree_size}")

# ── Key Insight ──────────────────────────────────────────────────────────────

print("\n=== Underfitting vs Overfitting ===")
print("Too few leaves  → underfitting → model too simple → high error")
print("Too many leaves → overfitting  → memorizes training data → high val error")
print("Sweet spot      → best_tree_size captures real patterns without memorizing")

"""
Kaggle - Feature Engineering
Exercise 6: Target Encoding (Final Exercise)
Author: Aila Nasir (@ailanasirai)
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)


def score_dataset(X, y, model=XGBRegressor()):
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score


df = pd.read_csv("../input/fe-course-data/ames.csv")

# --------------------------------------------------
# Explore cardinality of categorical features
# --------------------------------------------------
print(df.select_dtypes(["object"]).nunique())
print(df["SaleType"].value_counts())

# --------------------------------------------------
# Q1: Choose features — Neighborhood (28 categories)
# High cardinality = good target encoding candidate
# --------------------------------------------------

# --------------------------------------------------
# Create encoding and training splits
# --------------------------------------------------
X_encode = df.sample(frac=0.20, random_state=0)
y_encode = X_encode.pop("SalePrice")

X_pretrain = df.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")

# --------------------------------------------------
# Q2: Apply M-Estimate Encoding
# Fit on encoding split ONLY to avoid data leakage
# --------------------------------------------------
encoder = MEstimateEncoder(cols=["Neighborhood"], m=1)

encoder.fit(X_encode, y_encode)

X_train = encoder.transform(X_pretrain, y_train)

# Distribution comparison
feature = encoder.cols
plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=True, hist=False)
ax = sns.distplot(X_train[feature], color='r', ax=ax,
                  hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")
plt.show()

# Score comparison
X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)

print(f"Baseline Score: {score_base:.4f} RMSLE")
print(f"Score with Encoding: {score_new:.4f} RMSLE")

# --------------------------------------------------
# Q3: Demonstrate overfitting with target encoding
# Fitting encoder on SAME data as model = leakage
# --------------------------------------------------
m = 0

X = df.copy()
y = X.pop('SalePrice')

X["Count"] = range(len(X))
X["Count"][1] = 0

encoder = MEstimateEncoder(cols="Count", m=m)
X = encoder.fit_transform(X, y)

score = score_dataset(X, y)
print(f"Overfitting Score: {score:.4f} RMSLE")

plt.figure(dpi=90)
ax = sns.distplot(y, kde=True, hist=False)
ax = sns.distplot(X["Count"], color='r', ax=ax,
                  hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")
plt.show()

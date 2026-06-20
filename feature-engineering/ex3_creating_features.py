"""
Kaggle - Feature Engineering
Exercise 3: Creating Features
Author: Aila Nasir (@ailanasirai)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

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
X = df.copy()
y = X.pop("SalePrice")

# --------------------------------------------------
# Q1: Mathematical Transforms
# --------------------------------------------------
X_1 = pd.DataFrame()

X_1["LivLotRatio"] = df.GrLivArea / df.LotArea
X_1["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
X_1["TotalOutsideSF"] = (df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch
                          + df.Threeseasonporch + df.ScreenPorch)

# --------------------------------------------------
# Q2: Interaction with Categorical (BldgType x GrLivArea)
# --------------------------------------------------
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg")
X_2 = X_2.mul(df.GrLivArea, axis=0)

# --------------------------------------------------
# Q3: Count Feature (how many porch types > 0)
# --------------------------------------------------
X_3 = pd.DataFrame()

X_3["PorchTypes"] = (
    df[["WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
        "Threeseasonporch", "ScreenPorch"]]
    .gt(0)
    .sum(axis=1)
)

# --------------------------------------------------
# Q4: Break Down MSSubClass at first underscore
# --------------------------------------------------
X_4 = pd.DataFrame()

X_4["MSClass"] = df.MSSubClass.str.split("_", n=1).str.get(0)

# --------------------------------------------------
# Q5: Grouped Transform (median GrLivArea per Neighborhood)
# --------------------------------------------------
X_5 = pd.DataFrame()

X_5["MedNhbdArea"] = (
    df.groupby("Neighborhood")["GrLivArea"]
    .transform("median")
)

# --------------------------------------------------
# Final: Join all new features and score
# --------------------------------------------------
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
print("RMSLE with new features:", score_dataset(X_new, y))

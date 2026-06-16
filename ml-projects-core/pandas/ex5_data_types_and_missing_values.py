"""
Kaggle - Pandas
Exercise 5: Data Types and Missing Values
Author: Aila Nasir (@ailanasirai)
"""

import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

# -------------------------------------------------
# Q1: Data type of the points column
# -------------------------------------------------
dtype = reviews.points.dtype
print("dtype of points:", dtype)

# -------------------------------------------------
# Q2: Convert points column entries to strings
# -------------------------------------------------
point_strings = reviews.points.astype(str)
print(point_strings)

# -------------------------------------------------
# Q3: Number of reviews missing a price
# -------------------------------------------------
n_missing_prices = reviews.price.isnull().sum()
print("Missing prices:", n_missing_prices)

# -------------------------------------------------
# Q4: Reviews per region, missing values filled as 'Unknown', sorted desc
# -------------------------------------------------
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)
print(reviews_per_region)

"""
Kaggle - Pandas
Exercise 2: Indexing, Selecting and Assigning
Author: Aila Nasir (@ailanasirai)
"""

import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

print(reviews.head())

# -------------------------------------------------
# Q1: Select description column -> Series
# -------------------------------------------------
desc = reviews["description"]
print(type(desc))  # <class 'pandas.core.series.Series'>

# -------------------------------------------------
# Q2: First value of description column
# -------------------------------------------------
first_description = reviews.description.iloc[0]
print(first_description)

# -------------------------------------------------
# Q3: First row of data
# -------------------------------------------------
first_row = reviews.iloc[0]
print(first_row)

# -------------------------------------------------
# Q4: First 10 values of description column (as Series)
# -------------------------------------------------
first_descriptions = reviews.description.iloc[:10]
print(first_descriptions)

# -------------------------------------------------
# Q5: Rows with index labels 1, 2, 3, 5, 8
# -------------------------------------------------
sample_reviews = reviews.loc[[1, 2, 3, 5, 8]]
print(sample_reviews)

# -------------------------------------------------
# Q6: country, province, region_1, region_2 for rows 0, 1, 10, 100
# -------------------------------------------------
df = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
print(df)

# -------------------------------------------------
# Q7: country, variety for first 100 records
# loc is inclusive -> use :99 to get exactly 100 rows
# -------------------------------------------------
df = reviews.loc[:99, ['country', 'variety']]
print(df)

# -------------------------------------------------
# Q8: Reviews of wines made in Italy
# -------------------------------------------------
italian_wines = reviews[reviews.country == 'Italy']
print(italian_wines)

# -------------------------------------------------
# Q9: Reviews with >= 95 points from Australia or New Zealand
# -------------------------------------------------
top_oceania_wines = reviews[
    (reviews.country.isin(['Australia', 'New Zealand'])) & (reviews.points >= 95)
]
print(top_oceania_wines)

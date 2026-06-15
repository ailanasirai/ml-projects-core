"""
Kaggle - Pandas
Exercise 4: Grouping and Sorting
Author: Aila Nasir (@ailanasirai)
"""

import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

# -------------------------------------------------
# Q1: Number of reviews written by each taster
# -------------------------------------------------
reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
print(reviews_written)

# -------------------------------------------------
# Q2: Best (max) rating for each price, sorted by price ascending
# -------------------------------------------------
best_rating_per_price = reviews.groupby('price').points.max().sort_index()
print(best_rating_per_price)

# -------------------------------------------------
# Q3: Min and max price for each variety
# -------------------------------------------------
price_extremes = reviews.groupby('variety').price.agg([min, max])
print(price_extremes)

# -------------------------------------------------
# Q4: Varieties sorted by min price desc, then max price desc (tiebreak)
# -------------------------------------------------
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)
print(sorted_varieties)

# -------------------------------------------------
# Q5: Average rating given by each reviewer
# -------------------------------------------------
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
print(reviewer_mean_ratings)
print(reviewer_mean_ratings.describe())

# -------------------------------------------------
# Q6: Count of each (country, variety) combination, sorted descending
# -------------------------------------------------
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
print(country_variety_counts)

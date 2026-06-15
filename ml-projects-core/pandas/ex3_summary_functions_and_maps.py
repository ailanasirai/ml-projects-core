"""
Kaggle - Pandas
Exercise 3: Summary Functions and Maps
Author: Aila Nasir (@ailanasirai)
"""

import pandas as pd
pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
print(reviews.head())

# -------------------------------------------------
# Q1: Median of points column
# -------------------------------------------------
median_points = reviews.points.median()
print("Median points:", median_points)

# -------------------------------------------------
# Q2: Unique countries (no duplicates)
# -------------------------------------------------
countries = reviews.country.unique()
print("Countries:", countries)

# -------------------------------------------------
# Q3: Reviews per country
# -------------------------------------------------
reviews_per_country = reviews.country.value_counts()
print(reviews_per_country)

# -------------------------------------------------
# Q4: Price centered around the mean
# -------------------------------------------------
centered_price = reviews.price - reviews.price.mean()
print(centered_price)

# -------------------------------------------------
# Q5: Best bargain wine (highest points-to-price ratio)
# -------------------------------------------------
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']
print("Best bargain wine:", bargain_wine)

# -------------------------------------------------
# Q6: Count of 'tropical' vs 'fruity' in descriptions
# -------------------------------------------------
n_tropical = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_tropical, n_fruity], index=['tropical', 'fruity'])
print(descriptor_counts)

# -------------------------------------------------
# Q7: Star ratings based on points and country
# -------------------------------------------------
def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')
print(star_ratings)

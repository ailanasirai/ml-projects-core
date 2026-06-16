"""
Kaggle - Pandas
Exercise 6: Renaming and Combining
Author: Aila Nasir (@ailanasirai)
"""

import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

# -------------------------------------------------
# Q1: Rename region_1 -> region, region_2 -> locale
# -------------------------------------------------
renamed = reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})
print(renamed.head())

# -------------------------------------------------
# Q2: Rename the index axis to 'wines'
# -------------------------------------------------
reindexed = reviews.rename_axis('wines', axis='rows')
print(reindexed.head())

# -------------------------------------------------
# Q3: Combine gaming and movie subreddit products (vertical stack)
# -------------------------------------------------
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"

movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"

combined_products = pd.concat([gaming_products, movie_products])
print(combined_products.head())

# -------------------------------------------------
# Q4: Combine powerlifting meets and competitors (relational join on MeetID)
# -------------------------------------------------
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")

powerlifting_combined = powerlifting_competitors.set_index("MeetID").join(
    powerlifting_meets.set_index("MeetID")
)
print(powerlifting_combined.head())

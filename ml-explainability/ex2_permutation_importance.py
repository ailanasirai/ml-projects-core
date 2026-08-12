"""
Exercise: Permutation Importance
Dataset: NYC Taxi Fare Prediction
Course: Kaggle - Machine Learning Explainability
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Load, clean, and model a sample of the taxi fare data
# ---------------------------------------------------------------------------
data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)

data = data.query(
    'pickup_latitude > 40.7 and pickup_latitude < 40.8 and '
    'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and '
    'pickup_longitude > -74 and pickup_longitude < -73.9 and '
    'dropoff_longitude > -74 and dropoff_longitude < -73.9 and '
    'fare_amount > 0'
)

y = data.fare_amount

base_features = ['pickup_longitude', 'pickup_latitude',
                  'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=50, random_state=1).fit(train_X, train_y)

# ---------------------------------------------------------------------------
# Question 2: calculate permutation importance for the base model
# ---------------------------------------------------------------------------
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())

# Result: dropoff_latitude and pickup_latitude carried far more weight than
# the longitude features, and passenger_count was effectively irrelevant.

# ---------------------------------------------------------------------------
# Question 4: engineer distance features and recheck importance
# ---------------------------------------------------------------------------
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2 = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
              'dropoff_latitude', 'abs_lat_change', 'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

perm2 = PermutationImportance(second_model, random_state=1).fit(new_val_X, new_val_y)
eli5.show_weights(perm2, feature_names=new_val_X.columns.tolist())

# Result: raw distance traveled (abs_lat_change, abs_lon_change) dominates
# the model's predictions far more than the raw coordinate values do,
# confirming that trip distance -- not location alone -- drives fare price.

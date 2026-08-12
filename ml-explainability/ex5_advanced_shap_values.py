"""
Exercise: Advanced Uses of SHAP Values
Scenario: Hospital patient readmission risk (deeper feature interaction analysis)
Course: Kaggle - Machine Learning Explainability
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Load data and build model on a fixed, curated feature set
# ---------------------------------------------------------------------------
data = pd.read_csv('../input/hospital-readmissions/train.csv')
y = data.readmitted

base_features = ['number_inpatient', 'num_medications', 'number_diagnoses',
                  'num_lab_procedures', 'num_procedures', 'time_in_hospital',
                  'number_outpatient', 'number_emergency', 'gender_Female',
                  'payer_code_?', 'medical_specialty_?', 'diag_1_428',
                  'diag_1_414', 'diabetesMed_Yes', 'A1Cresult_None']

# Some versions of the shap package error when mixing bools and numerics
X = data[base_features].astype(float)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# For speed, calculate SHAP values on a smaller subset of validation data
small_val_X = val_X.iloc[:150]
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)

# ---------------------------------------------------------------------------
# Summary plot across the full curated feature set
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(small_val_X)

shap.summary_plot(shap_values[1], small_val_X)

# ---------------------------------------------------------------------------
# Question 1 & 3: comparing effect range and impact of two binary features
# ---------------------------------------------------------------------------
feature_with_bigger_range_of_effects = 'diag_1_428'
bigger_effect_when_changed = 'diag_1_428'
# diag_1_428 shows a wider spread of SHAP values on the summary plot than
# payer_code_?, and flipping it from 0 to 1 produces a larger swing in
# predicted readmission risk.

# ---------------------------------------------------------------------------
# Question 6: dependence plots to distinguish two "jumbled" features
# ---------------------------------------------------------------------------
shap.dependence_plot('num_medications', shap_values[1], small_val_X)
shap.dependence_plot('num_lab_procedures', shap_values[1], small_val_X)

# Result: num_medications shows a clearer upward trend at higher values --
# more medications tends to push risk higher, on top of a wider overall
# effect range. num_lab_procedures stays comparatively flat and noisy
# across its range, meaning its effect is more evenly distributed rather
# than concentrated at any particular value.

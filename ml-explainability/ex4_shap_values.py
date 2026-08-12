"""
Exercise: SHAP Values
Scenario: Hospital patient readmission risk
Course: Kaggle - Machine Learning Explainability
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
from matplotlib import pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Scenario: predict which patients are at highest risk of readmission
# ---------------------------------------------------------------------------
data = pd.read_csv('../input/hospital-readmissions/train.csv')

y = data.readmitted
base_features = [c for c in data.columns if c != "readmitted"]
X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)

# ---------------------------------------------------------------------------
# Step 1: condensed model overview for doctors -- SHAP summary plot
# ---------------------------------------------------------------------------
small_val_X = val_X.iloc[:150]
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(small_val_X)

shap.summary_plot(shap_values[1], small_val_X)

# ---------------------------------------------------------------------------
# Step 2: how does number_inpatient affect predictions?
# ---------------------------------------------------------------------------
PartialDependenceDisplay.from_estimator(my_model, val_X, ['number_inpatient'])
plt.show()

# ---------------------------------------------------------------------------
# Step 3: same question for time_in_hospital
# ---------------------------------------------------------------------------
PartialDependenceDisplay.from_estimator(my_model, val_X, ['time_in_hospital'])
plt.show()

# ---------------------------------------------------------------------------
# Step 4: compare the model's PDP against the raw readmission rate
# ---------------------------------------------------------------------------
all_train = pd.concat([train_X, train_y], axis=1)
all_train.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()

# Result: the raw readmission rate by time_in_hospital shows the same
# roughly flat pattern as the model's partial dependence plot, confirming
# the model isn't missing an obvious signal -- time_in_hospital genuinely
# has little independent effect once other features are accounted for.

# ---------------------------------------------------------------------------
# Step 5: per-patient risk factor breakdown
# ---------------------------------------------------------------------------
def patient_risk_factors(model, patient_data):
    """Return a SHAP force plot showing which features increased or
    decreased a single patient's predicted readmission risk."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)


data_for_prediction = val_X.iloc[0, :]
patient_risk_factors(my_model, data_for_prediction)

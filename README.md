# ml-projects-core

> Machine learning projects, competition notebooks, and dataset analysis — built with scikit-learn, pandas, and Kaggle.

---

## Overview

This repository documents my progression through applied machine learning. Every file focuses on understanding the reasoning behind each decision — not just producing output.

The learning path follows a deliberate sequence: data exploration → first model → validation → optimization → competition submission → preprocessing → model tuning → leakage detection.

---

## Repository Structure

```
ml-projects-core/
│
├── 01_intro_to_ml/
│   ├── iowa_housing_exploration.py
│   ├── first_ml_model.py
│   ├── underfitting_overfitting.py
│   ├── random_forest.py
│   └── ml_competition_submission.py
│
├── 02_intermediate_ml/
│   ├── intermediate_ml_intro.py
│   ├── missing_values.py
│   ├── categorical_variables.py
│   ├── pipelines.py
│   ├── cross_validation.py
│   ├── xgboost_model.py
│   └── data_leakage.py
│
└── README.md
```

All exercise notebooks are published on Kaggle and linked from each LinkedIn post documenting that exercise. The `.py` files here are clean, standalone versions of the same work — written for reference and reuse outside the Kaggle environment.

---

## Module 01 — Intro to Machine Learning

### `iowa_housing_exploration.py`
Exploratory analysis on the Iowa housing dataset before touching any model. 1,460 samples, 79 features, one target: SalePrice.

Key observations documented:
- Average lot size is 10,517 sq ft but heavily skewed — mean alone is misleading
- Newest home is 16 years old — no recent construction captured in data
- Missing values in PoolQC, MiscFeature, Alley — understanding why data is missing matters more than just filling it

### `first_ml_model.py`
First end-to-end ML model built from scratch. Covers feature selection, fitting a Decision Tree Regressor, and generating predictions on training data.

Key concepts:
- Target variable selection — SalePrice
- Feature list — LotArea, YearBuilt, 1stFlrSF, 2ndFlrSF, FullBath, BedroomAbvGr, TotRmsAbvGrd
- fit() and predict() — the core ML loop

### `underfitting_overfitting.py`
Decision Tree optimization by testing six values of max_leaf_nodes: 5, 25, 50, 100, 250, 500. MAE tracked at each size on validation data.

Key insight: validation MAE is what matters, not training MAE. A model that performs perfectly on training data has likely memorized it, not learned from it.

Best tree size found: 100 leaf nodes.

### `random_forest.py`
Replaced Decision Tree with Random Forest. Same data, same features, one line of code changed. Error dropped significantly.

Why it works: instead of one tree making all decisions, hundreds of trees each trained on a random data subset vote together. The average prediction is more stable and more accurate than any single tree.

### `ml_competition_submission.py`
Full competition pipeline — validation, full data retraining, test predictions, submission file generation.

Key insight: validate on a split first to confirm the model works. Then retrain on the complete dataset before generating final predictions.

---

## Module 02 — Intermediate Machine Learning

### `intermediate_ml_intro.py`
Compared five Random Forest configurations using MAE on validation data. Selected the best performing model and generated the first competition submission with this course's dataset structure.

### `missing_values.py`
Compared two strategies for handling missing data: dropping columns versus mean imputation with SimpleImputer.

Key insight: imputation outperformed dropping columns. When missingness is low relative to total rows, imputing preserves useful signal that dropping would discard.

### `categorical_variables.py`
Compared three approaches to encoding categorical features: dropping them, ordinal encoding, and one-hot encoding.

Key insight: one-hot encoding gave the best MAE. Also documented a real-world gotcha — ordinal encoders break if validation data contains categories the training data never saw.

### `pipelines.py`
Bundled imputation, one-hot encoding, and the model into a single object using ColumnTransformer and Pipeline.

Key insight: the value isn't shorter code. fit() and predict() now apply the same preprocessing steps consistently across training and test data, preventing inconsistent transformations and data leakage.

### `cross_validation.py`
Tested eight values of n_estimators (50 to 400) using 3-fold cross-validation, tracking average MAE for each.

Key insight: a single train/validation split can be lucky or unlucky depending on which rows land where. Cross-validation rotates through multiple splits and averages the result for a more reliable signal before locking in a hyperparameter.

### `xgboost_model.py`
Built three XGBoost models with deliberately different configurations: a default baseline, a tuned version with more trees and a lower learning rate, and a version designed to perform worse.

Key insight: n_estimators and learning_rate trade off against each other. More boosting rounds with a smaller learning rate means slower training but better accuracy. Too few rounds or too high a learning rate causes underfitting.

### `data_leakage.py`
Five scenarios designed to break the assumption that low error means a good model — covering target leakage and train-test contamination across shoelace demand, cryptocurrency pricing, surgical infection rates, and housing prices.

Key insight: using a neighborhood's average sale price as a feature seems reasonable until realizing that average is partly built from the exact price being predicted. The moment a result looks too good is exactly when to slow down and ask why.

---

## Kaggle Competition

**Housing Prices Competition for Kaggle Learn Users**

| Item | Detail |
|---|---|
| Dataset | Iowa housing — 1,460 train, 1,459 test samples |
| Target | SalePrice |
| Approaches tried | Decision Tree, Random Forest, Pipelines, XGBoost |
| Metric | Mean Absolute Error |
| Status | Submitted, iterating |

---

## Key Concepts Documented

**Mean Absolute Error**
Average difference between predicted and actual values. Lower is better. Used as the primary evaluation metric throughout.

**Underfitting vs Overfitting**
Too few leaves: model too simple, misses real patterns. Too many leaves: model memorizes training data, fails on new data. Sweet spot: lowest MAE on validation data.

**Train / Validation Split**
Holding out a portion of training data to evaluate model performance before final submission.

**Random Forest vs Decision Tree**
Single tree is prone to overfitting. Random Forest averages across hundreds of trees for more stable, accurate predictions.

**Missing Values and Categorical Encoding**
Imputation generally beats dropping columns. One-hot encoding generally beats ordinal encoding and dropping categorical columns entirely.

**Pipelines**
Bundle preprocessing and modeling into one object so the same transformations apply consistently to training and test data.

**Cross-Validation**
Average performance across multiple data splits gives a more reliable hyperparameter signal than a single split.

**Gradient Boosting (XGBoost)**
Builds models sequentially, each correcting the errors of the previous one. n_estimators and learning_rate must be tuned together.

**Data Leakage**
Target leakage: a feature contains information that only exists after the target is determined. Train-test contamination: preprocessing done on the full dataset before splitting leaks information across the split.

---

## Tech Stack

```
Language      Python 3.11+
Environment   Kaggle Notebooks
Libraries     pandas · scikit-learn · xgboost · matplotlib
Models        DecisionTreeRegressor · RandomForestRegressor · XGBRegressor
Techniques    Pipelines · ColumnTransformer · Cross-Validation
Metric        Mean Absolute Error
```

---

## Roadmap

- [x] Data exploration — Iowa housing dataset
- [x] First ML model — Decision Tree Regressor
- [x] Model validation — train/val split, MAE evaluation
- [x] Underfitting and overfitting — tree size optimization
- [x] Random Forest — improved accuracy over Decision Tree
- [x] First Kaggle competition submission
- [x] Missing values — imputation vs dropping
- [x] Categorical variables — ordinal vs one-hot encoding
- [x] Pipelines — bundled preprocessing and modeling
- [x] Cross-validation — hyperparameter selection
- [x] XGBoost — gradient boosting
- [x] Data leakage — target leakage and train-test contamination
- [ ] pandas — data manipulation and feature engineering
- [ ] Titanic competition — binary classification
- [ ] FlowZint Hackathon — Titanic Survival Predictor web app

---

## Contact

**GitHub** — [github.com/ailanasirai](https://github.com/ailanasirai)
**LinkedIn** — [linkedin.com/in/aila-nasir](https://www.linkedin.com/in/aila-nasir/)
**Kaggle** — [kaggle.com/ailanasirai](https://www.kaggle.com/ailanasirai/code)

---

*This repository is actively maintained. Structure and content evolve as learning progresses.*

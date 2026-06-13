# ml-projects-core

> Machine learning projects, competition notebooks, and dataset analysis — built with scikit-learn, pandas, and Kaggle.

---

## Overview

This repository documents my progression through applied machine learning. Every file focuses on understanding the reasoning behind each decision — not just producing output.

The learning path follows a deliberate sequence: data exploration → first model → validation → optimization → competition submission.

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
├── kaggle_notebooks/
│   ├── exercise-explore-your-data.ipynb
│   ├── exercise-your-first-machine-learning-model.ipynb
│   ├── exercise-underfitting-and-overfitting.ipynb
│   ├── exercise-random-forests.ipynb
│   └── exercise-ml-competitions-submission.ipynb
│
└── README.md
```

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

Key insight: validation MAE is what matters — not training MAE. A model that performs perfectly on training data has likely memorized it, not learned from it.

Best tree size found: 100 leaf nodes.

### `random_forest.py`
Replaced Decision Tree with Random Forest. Same data, same features, one line of code changed. Error dropped significantly.

Why it works: instead of one tree making all decisions, hundreds of trees each trained on a random data subset vote together. The average prediction is more stable and more accurate than any single tree.

### `ml_competition_submission.py`
Full competition pipeline — validation, full data retraining, test predictions, submission file generation.

Key insight: validate on a split first to confirm the model works. Then retrain on the complete dataset before generating final predictions. Holding out validation data at submission means leaving useful signal unused.

---

## Kaggle Competition

**Housing Prices Competition for Kaggle Learn Users**

| Item | Detail |
|---|---|
| Dataset | Iowa housing — 1,460 train, 1,459 test samples |
| Target | SalePrice |
| Features | 7 numeric features |
| Model | RandomForestRegressor |
| Metric | Mean Absolute Error |
| Status | Submitted |

---

## Key Concepts Documented

**Mean Absolute Error**
Average difference between predicted and actual values. Lower is better. Used as the primary evaluation metric throughout.

**Underfitting vs Overfitting**
Too few leaves — model too simple, misses real patterns. Too many leaves — model memorizes training data, fails on new data. Sweet spot: lowest MAE on validation data.

**Train / Validation Split**
Holding out a portion of training data to evaluate model performance before final submission. Standard practice before deploying any model.

**Full Data Retraining**
After validation confirms the model is good, retrain on the complete dataset before generating test predictions. More data means better learned patterns.

**Random Forest vs Decision Tree**
Single tree is prone to overfitting. Random Forest averages across hundreds of trees — more stable, lower error, better generalization.

---

## Tech Stack

```
Language      Python 3.11+
Environment   Kaggle Notebooks
Libraries     pandas · scikit-learn · numpy
Models        DecisionTreeRegressor · RandomForestRegressor
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
- [ ] Intermediate ML — missing values, categorical encoding, pipelines
- [ ] pandas — data manipulation and feature engineering
- [ ] XGBoost — improved competition score
- [ ] Titanic competition — binary classification
- [ ] FlowZint Hackathon — Titanic Survival Predictor web app

---

## Contact

**GitHub** — [github.com/ailanasirai](https://github.com/ailanasirai)
**LinkedIn** — [linkedin.com/in/aila-nasir](https://www.linkedin.com/in/aila-nasir/)
**Kaggle** — [kaggle.com/ailanasirai](https://www.kaggle.com/ailanasirai/code)

---

*This repository is actively maintained. Structure and content evolve as learning progresses.*

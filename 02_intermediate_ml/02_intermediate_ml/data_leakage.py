"""
Data Leakage — Recognizing Hidden Information in Features
ml-projects-core / 02_intermediate_ml
"""

# Data leakage happens when training data contains information about the
# target that will not be available at prediction time. The model looks
# accurate during validation, then fails in production.

# Two types covered here: target leakage and train-test contamination.


# ── Case 1 -- Shoelace Prediction (Target Leakage) ────────────────────────────

# Feature: amount of leather used THIS month
# Target:  number of shoelaces needed THIS month

# Leather used and shoelaces needed are both OUTCOMES of production volume.
# Using "leather used" to predict "shoelaces needed" is circular --
# the feature is only known AFTER the value you're trying to predict
# has effectively already happened.

# Verdict: TARGET LEAKAGE
# Fix: use leather ORDERED before production, not leather USED during it.


# ── Case 2 -- Leather Ordered (Still Depends) ────────────────────────────────

# If "leather ordered" happens BEFORE the prediction is made -> safe to use.
# If ordering happens AFTER production starts (reactive ordering) -> still leaks.

# Verdict: depends on the TIMING of when the feature becomes known
# relative to when the prediction needs to be made.


# ── Case 3 -- Cryptocurrency Price Prediction (Train-Test Contamination) ──────

# Features include: current price, 24h price change, 1h price change.
# These features are extremely close in time to the prediction target
# (price one day ahead).

# A model with average error under $1 on a currency that swings $100/year
# is not "accurate" -- it likely learned that "tomorrow's price is close
# to today's price," which is trivially true and useless for trading.

# Verdict: model is NOT reliable for real trading decisions.
# The low error reflects autocorrelation in price data, not predictive skill.


# ── Case 4 -- Surgeon Infection Rate ──────────────────────────────────────────

# Using each surgeon's historical infection rate (computed across ALL their
# surgeries, including ones in the validation/test set) as a feature.

# TARGET LEAKAGE: if a surgeon's rate is computed using outcomes from
# surgeries that are in the validation set, the model has indirect
# access to validation outcomes during training.

# TRAIN-TEST CONTAMINATION: the same surgeon appears in both train and
# test sets, so information flows between them through the surgeon's
# average rate.

# Fix: compute each surgeon's rate using ONLY their training-set surgeries,
# and recompute per fold during cross-validation.


# ── Case 5 -- Housing Price Prediction ────────────────────────────────────────

# Four candidate features:
# 1. Size of the house (square meters)        -- safe, known at listing time
# 2. Average sales price of homes in the neighborhood  -- LEAKAGE RISK
# 3. Latitude and longitude                   -- safe, fixed property
# 4. Whether the house has a basement         -- safe, fixed property

potential_leakage_feature = 2

print(f"Most likely leakage source: Feature {potential_leakage_feature}")
print("Reason: neighborhood average price is computed FROM sale prices --")
print("including the price of the house being predicted, if it's in the")
print("training set. This indirectly encodes the target into a feature.")


# ── Key Takeaways ──────────────────────────────────────────────────────────────

print("\n=== Two Types of Leakage ===")
print("Target leakage          : a feature contains information that only")
print("                          exists AFTER the target is determined.")
print("Train-test contamination: preprocessing (scaling, imputation, feature")
print("                          engineering) is done on the FULL dataset")
print("                          before splitting into train/validation.")
print("\nBoth produce models that look accurate in testing but fail in production.")

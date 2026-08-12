# ML Explainability

Machine learning models should not feel like black boxes. This module covers the tools to inspect what a trained model actually learned, why it makes the predictions it does, and how much to trust it, using a NYC taxi fare model and a hospital patient readmission model.

## `ex2_permutation_importance.py`
Calculated which features a Random Forest taxi fare model actually relies on by shuffling one feature at a time and measuring how much validation error increases.

Key insight: `dropoff_latitude` and `pickup_latitude` carried far more weight than the longitude features, and `passenger_count` was effectively irrelevant. After engineering `abs_lat_change` and `abs_lon_change` (raw distance traveled), those two features dominated importance scores completely — confirming that **distance**, not raw location, is what actually drives fare price.

## `ex4_shap_values.py`
Built a condensed, doctor-facing model overview for a hospital readmission risk model using a single SHAP summary plot, then used Partial Dependence Plots to isolate how `number_inpatient` and `time_in_hospital` individually affect predictions.

Key insight: comparing a Partial Dependence Plot against the *raw* readmission rate for `time_in_hospital` confirmed the model wasn't missing an obvious signal — the flat pattern held in both the model's view and the real data. Also built a `patient_risk_factors()` function using `shap.force_plot()` to generate a per-patient breakdown of exactly which features raised or lowered that individual's readmission risk.

## `ex5_advanced_shap_values.py`
Went deeper into feature interactions on the same readmission model, comparing `diag_1_428` against `payer_code_?`, and `num_medications` against `num_lab_procedures`.

Key insight: SHAP dependence plots revealed a difference invisible in the summary plot alone — `num_medications` showed a clear upward trend at higher values (more medications, more risk), while `num_lab_procedures` stayed flat and noisy across its whole range, meaning its effect is spread out rather than concentrated at any particular value.

## Certificate

![Certificate](./certificate.png)

# SHAP Per-Feature Analysis for Multiclass XGBoost

##  Purpose of the Analysis

This module provides **feature-level interpretability** for a trained **multiclass XGBoost Intrusion Detection System (IDS)**.
It allows cybersecurity experts to understand **which features influenced each correct classification**, making the model’s reasoning transparent and auditable.

The analysis answers the question:

> “Which network flow features made the model classify a sample as DDoS, Infiltration, etc., and why?”

This helps analysts verify that the model’s logic aligns with known attack behavior — e.g., high `Flow Packets/s` and `SYN Flag Count` may drive DDoS detection, while abnormal `Idle Mean` may indicate Infiltration.

---

##  Analysis Types

The SHAP per-feature analysis is divided into two categories:

```
per-feature_SHAP_analysis/
├── training_analysis/      # How the model learned (TVAE / real data)
└── prediction_analysis/    # How the model predicts (CTGAN data)
```

Each contains subfolders for **multiclass** and **binary_one_vs_all** models.

---

##  Folder Structure

```
ids/
├── datasets/
│   ├── preprocessed/
│   │   └── per_year/
│   │       ├── 2017/
│   │       └── 2018/
│   └── synthetics/
│       ├── oversampling_TVAE/
│       │   ├── 2017/
│       │   ├── 2018/
│       │   └── for_all/
│       └── predict_CTGAN/
│           ├── 2017/
│           ├── 2018/
│           └── for_all/
│
├── xgboost_models/
│   └── multiclass/
│       ├── train_multiclass_xgb.py
│       └── trained_models/
│           ├── 2017/
│           ├── 2017_with_oversampling/
│           ├── 2018/
│           ├── 2018_with_oversampling/
│           ├── for_all/
│           └── for_all_with_oversampling/
│
└── SHAP_analysis/
    └── per-feature_SHAP_analysis/
        └── multiclass/
            ├── prediction_analysis/
            │   ├── 2017/
            │   ├── 2017_with_oversampling/
            │   ├── 2018/
            │   ├── 2018_with_oversampling/
            │   ├── for_all/
            │   └── for_all_with_oversampling/
            │
            └── training_analysis/
                ├── 2017/
                ├── 2017_with_oversampling/
                ├── 2018/
                ├── 2018_with_oversampling/
                ├── for_all/
                └── for_all_with_oversampling/

         
```

---

##  How It Works

1. **Model & Encoder Loading**
   Loads the trained XGBoost model (`xgboost_model.json`) and its corresponding `Label Encoder`.

2. **Dataset Loading**
   Loads and concatenates multiple CSVs from selected folders, cleans `NaN` and `Inf` values.

3. **Prediction**
   The model predicts the class of each network flow record.

4. **SHAP Computation (GPU)**
   SHAP values are computed batch-wise using GPU (`pred_contribs=True`), keeping only the SHAPs of the predicted class and removing the bias term.

5. **Per-Class Aggregation**
   Keeps **true positives** (`Prediction == True_Label`) and computes the **mean SHAP value** per feature for each class.

6. **Results Export**
   Saves one CSV per class with the average SHAP contributions of each feature.

---
## Choosing the Correct Paths

| Analysis Type                 | `LOAD_PATH` (Model Folder)                                                | `BASE_FOLDERS` (Prediction Data)                     | `RESULTS_PATH` (Save SHAP CSVs)                                                                         |
| ----------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **2017**                      | `/ids/xgboost_models/multiclass/trained_models/2017`                      | `['/ids/datasets/synthetics/predict_CTGAN/2017']`    | `/ids/SHAP_analysis/per-feature_SHAP_analysis/multiclass/prediction_analysis/2017`                      |
| **2017_with_oversampling**    | `/ids/xgboost_models/multiclass/trained_models/2017_with_oversampling`    | `['/ids/datasets/synthetics/predict_CTGAN/2017']`    | `/ids/SHAP_analysis/per-feature_SHAP_analysis/multiclass/prediction_analysis/2017_with_oversampling`    |
| **2018**                      | `/ids/xgboost_models/multiclass/trained_models/2018`                      | `['/ids/datasets/synthetics/predict_CTGAN/2018']`    | `/ids/SHAP_analysis/per-feature_SHAP_analysis/multiclass/prediction_analysis/2018`                      |
| **2018_with_oversampling**    | `/ids/xgboost_models/multiclass/trained_models/2018_with_oversampling`    | `['/ids/datasets/synthetics/predict_CTGAN/2018']`    | `/ids/SHAP_analysis/per-feature_SHAP_analysis/multiclass/prediction_analysis/2018_with_oversampling`    |
| **for_all**                   | `/ids/xgboost_models/multiclass/trained_models/for_all`                   | `['/ids/datasets/synthetics/predict_CTGAN/for_all']` | `/ids/SHAP_analysis/per-feature_SHAP_analysis/multiclass/prediction_analysis/for_all`                   |
| **for_all_with_oversampling** | `/ids/xgboost_models/multiclass/trained_models/for_all_with_oversampling` | `['/ids/datasets/synthetics/predict_CTGAN/for_all']` | `/ids/SHAP_analysis/per-feature_SHAP_analysis/multiclass/prediction_analysis/for_all_with_oversampling` |


 *If analyzing model training instead of prediction, replace CTGAN paths with preprocessed or oversampling_TVAE data accordingly.*

---

##  Example Output (CSV)

Each class produces a CSV file such as:

```
SHAP_DDoS.csv
SHAP_Benign.csv
SHAP_Infiltration.csv
```

Each file contains:

```
Feature,Mean_SHAP
Flow Duration,0.1523
Total Fwd Packets,0.0987
Fwd IAT Mean,0.0651
...
```

These represent the **average contribution** of each feature to correct predictions for that class.

---

## Interpretation Context

* **Training Analysis:** explains what the model learned (features that influenced its internal structure).
* **Prediction Analysis:** explains how the model applies that knowledge to new synthetic samples (CTGAN).

The SHAP results allow a security expert to:

* Identify which network traffic patterns the model associates with specific attacks.
* Verify that predictions are based on **legitimate indicators** rather than spurious correlations.
* Document and justify AI-driven intrusion detection decisions.

---

##  Requirements
This project uses a comprehensive environment optimized for **XGBoost**, **SHAP**, and **synthetic data generation (CTGAN/TVAE)**.

### Python Version
* Python >= 3.11

### Installation
All required dependencies are listed in the `requirements.txt` file.  
To install everything, simply run:

```bash
pip install -r requirements.txt


```bash
pip install xgboost pandas numpy joblib
```

---

##  Run

Simply execute the analysis script (example for prediction analysis):

```bash
python xgb_multiclass_shap.py
```

All SHAP results will be saved automatically to the corresponding folder defined in `RESULTS_PATH`.

---

# SHAP Per-Feature Analysis for One-vs-All XGBoost Models

## Overview

This script computes feature-level SHAP contributions for multiple One-vs-All (binary) XGBoost models. It explains how each model correctly classifies a sample, allowing security analysts to understand which network features influenced each correct prediction.

## Purpose

Each One-vs-All model is trained to recognize one attack class versus all others. This script:

* Loads synthetic CTGAN-generated data for prediction.
* Loads one binary XGBoost model per label (e.g., DDoS, Infiltration, FTP, etc.).
* Predicts all classes simultaneously and selects the most confident one.
* For correctly classified samples (True Positives), computes mean SHAP values per feature.
* Exports one CSV per label containing the most influential features.

## How It Works

1. **Load Data**
   Reads synthetic CTGAN prediction datasets (CSV files), merges them, removes NaN/Inf, and shuffles the rows.
2. **Load One-vs-All Models**
   Loads all .json models from the One-vs-All results folder, one per attack label.
3. **Prediction Phase**
   Calculates probabilities for each binary model and selects the label with the highest probability, unless all are below a defined threshold (classified as "Unknown").
4. **SHAP Computation**
   For True Positive samples (where predicted == true label), SHAP values are computed in GPU batches using pred_contribs=True.
5. **Output Generation**
   Saves one CSV per label containing each feature name and its average SHAP contribution (Mean_SHAP).

## Folder Structure

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
│   └── OneVsAll/
│       ├── 2017/
│       ├── 2017_with_oversampling/
│       ├── 2018/
│       ├── 2018_with_oversampling/
│       ├── for_all/
│       └── for_all_with_oversampling/
│
└── SHAP_analysis/
    └── per-feature_SHAP_analysis/
        └── binary_one_vs_all/
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

## Choosing the Correct Paths

| Analysis Type                 | MODELS_DIR (Trained Models)                              | DATA_FOLDERS (Prediction Data)                       | RESULTS_PATH (Save SHAP CSVs)                                                                                  |
| ----------------------------- | -------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **2017**                      | `/ids/xgboost_models/OneVsAll/2017`                      | `['/ids/datasets/synthetics/predict_CTGAN/2017']`    | `/ids/SHAP_analysis/per-feature_SHAP_analysis/binary_one_vs_all/prediction_analysis/2017`                      |
| **2017_with_oversampling**    | `/ids/xgboost_models/OneVsAll/2017_with_oversampling`    | `['/ids/datasets/synthetics/predict_CTGAN/2017']`    | `/ids/SHAP_analysis/per-feature_SHAP_analysis/binary_one_vs_all/prediction_analysis/2017_with_oversampling`    |
| **2018**                      | `/ids/xgboost_models/OneVsAll/2018`                      | `['/ids/datasets/synthetics/predict_CTGAN/2018']`    | `/ids/SHAP_analysis/per-feature_SHAP_analysis/binary_one_vs_all/prediction_analysis/2018`                      |
| **2018_with_oversampling**    | `/ids/xgboost_models/OneVsAll/2018_with_oversampling`    | `['/ids/datasets/synthetics/predict_CTGAN/2018']`    | `/ids/SHAP_analysis/per-feature_SHAP_analysis/binary_one_vs_all/prediction_analysis/2018_with_oversampling`    |
| **for_all**                   | `/ids/xgboost_models/OneVsAll/for_all`                   | `['/ids/datasets/synthetics/predict_CTGAN/for_all']` | `/ids/SHAP_analysis/per-feature_SHAP_analysis/binary_one_vs_all/prediction_analysis/for_all`                   |
| **for_all_with_oversampling** | `/ids/xgboost_models/OneVsAll/for_all_with_oversampling` | `['/ids/datasets/synthetics/predict_CTGAN/for_all']` | `/ids/SHAP_analysis/per-feature_SHAP_analysis/binary_one_vs_all/prediction_analysis/for_all_with_oversampling` |

 *If analyzing model training instead of prediction, replace CTGAN paths with preprocessed or oversampling_TVAE data accordingly.*
 
## Example Output (inside each CSV)

```
Feature,Mean_SHAP
Flow Duration,0.1325
Fwd Packet Length Mean,0.1089
Total Fwd Packets,0.0974
Flow Bytes/s,0.0815
...
```

Each file shows the average SHAP importance of every feature for correctly predicted samples of a given label.

Example output files:

```
SHAP_DDoS.csv
SHAP_Infiltration.csv
SHAP_SQL_Injection.csv
SHAP_Benign.csv
```

## Requirements

All dependencies are included in the project requirements.txt file. Install everything with:

```
pip install -r requirements.txt
```

Main libraries used:

* xgboost==3.0.2 – binary classification (GPU enabled)
* pandas, numpy – data manipulation
* shap==0.48.0 – feature explainability
* ctgan, sdv, torch – synthetic data generation (for CTGAN datasets)

## Run

To execute the SHAP computation:

```
python xgb_one_vs_all_shap.py
```

All SHAP results will be saved automatically to the corresponding folder under RESULTS_PATH.



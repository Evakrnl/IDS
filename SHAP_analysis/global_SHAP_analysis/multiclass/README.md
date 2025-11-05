# Multiclass Global SHAP Analysis

This script (`xgb_multiclass_global_shap.py`) performs a **global SHAP feature importance analysis**
for a trained multiclass XGBoost model. It computes the **average SHAP contribution across all True Positive samples**
from every class combined into a single output CSV file.

---

## Functionality Overview

1. Loads a trained **XGBoost multiclass model** and its corresponding LabelEncoder.
2. Reads and merges CSV data from one or more folders.
3. Cleans NaN and Inf values for stable computation.
4. Makes predictions for all samples.
5. Computes SHAP contributions in GPU batches for efficiency.
6. Extracts SHAP values for the predicted class only.
7. Combines all True Positive SHAP values to calculate a global mean.
8. Saves one CSV file:

   ```
   TP_Global_SHAP_AllClasses.csv
   ```

---

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
│   └── multiclass/
│       └── trained_models/
│           ├── 2017/
│           ├── 2017_with_oversampling/
│           ├── 2018/
│           ├── 2018_with_oversampling/
│           ├── for_all/
│           └── for_all_with_oversampling/
│
└── SHAP_analysis/
    └── global_SHAP_analysis/
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

## Choosing the Correct Paths

| Dataset / Model Type          | `LOAD_PATH` (Model Folder)                                                | `BASE_FOLDERS` (Prediction Data)                     | `RESULTS_PATH` (Save Global SHAP CSV)                                                              |
| ----------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **2017**                      | `/ids/xgboost_models/multiclass/trained_models/2017`                      | `['/ids/datasets/synthetics/predict_CTGAN/2017']`    | `/ids/SHAP_analysis/global_SHAP_analysis/multiclass/prediction_analysis/2017`                      |
| **2017_with_oversampling**    | `/ids/xgboost_models/multiclass/trained_models/2017_with_oversampling`    | `['/ids/datasets/synthetics/predict_CTGAN/2017']`    | `/ids/SHAP_analysis/global_SHAP_analysis/multiclass/prediction_analysis/2017_with_oversampling`    |
| **2018**                      | `/ids/xgboost_models/multiclass/trained_models/2018`                      | `['/ids/datasets/synthetics/predict_CTGAN/2018']`    | `/ids/SHAP_analysis/global_SHAP_analysis/multiclass/prediction_analysis/2018`                      |
| **2018_with_oversampling**    | `/ids/xgboost_models/multiclass/trained_models/2018_with_oversampling`    | `['/ids/datasets/synthetics/predict_CTGAN/2018']`    | `/ids/SHAP_analysis/global_SHAP_analysis/multiclass/prediction_analysis/2018_with_oversampling`    |
| **for_all**                   | `/ids/xgboost_models/multiclass/trained_models/for_all`                   | `['/ids/datasets/synthetics/predict_CTGAN/for_all']` | `/ids/SHAP_analysis/global_SHAP_analysis/multiclass/prediction_analysis/for_all`                   |
| **for_all_with_oversampling** | `/ids/xgboost_models/multiclass/trained_models/for_all_with_oversampling` | `['/ids/datasets/synthetics/predict_CTGAN/for_all']` | `/ids/SHAP_analysis/global_SHAP_analysis/multiclass/prediction_analysis/for_all_with_oversampling` |


 *If analyzing model training instead of prediction, replace CTGAN paths with preprocessed or oversampling_TVAE data accordingly.*
---

## Output

Each run produces one global CSV file:

```
TP_Global_SHAP_AllClasses.csv
```

This file contains two columns:

| Column       | Description                                       |
| ------------ | ------------------------------------------------- |
| **Feature**  | The name of the network feature                   |
| **MeanSHAP** | Average SHAP value across all TP samples combined |

---

## Example Run

```bash
python xgb_multiclass_global_shap.py
```

The global SHAP results are automatically saved to the defined `RESULTS_PATH`.

# One-vs-All Global SHAP Analysis

This script (`xgb_one_vs_all_global_shap.py`) performs a **global SHAP feature importance analysis**
on multiple One-vs-All XGBoost binary models. It computes the **average SHAP contribution across all True Positive samples**
from all classes and outputs a single CSV file representing the global feature importance.

---

## Functionality Overview

1. Loads synthetic **CTGAN-generated** data used for prediction.
2. Loads all **One-vs-All trained XGBoost models** (one per label).
3. Computes class probabilities and selects the best predicted label per sample.
4. Identifies **True Positives (TP)** across all models.
5. Computes SHAP values in **GPU-accelerated batches**.
6. Aggregates all TP SHAP values globally.
7. Saves one CSV file:

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
│   └── binary_one_vs_all/
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

---

## Choosing the Correct Paths

| Dataset / Model Type          | `MODELS_DIR` (One-vs-All Models)                                                 | `DATA_FOLDERS` (Prediction Data)                     | `RESULTS_PATH` (Save Global SHAP CSV)                                                                     |
| ----------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **2017**                      | `/ids/xgboost_models/binary_one_vs_all/trained_models/2017`                      | `['/ids/datasets/synthetics/predict_CTGAN/2017']`    | `/ids/SHAP_analysis/global_SHAP_analysis/binary_one_vs_all/prediction_analysis/2017`                      |
| **2017_with_oversampling**    | `/ids/xgboost_models/binary_one_vs_all/trained_models/2017_with_oversampling`    | `['/ids/datasets/synthetics/predict_CTGAN/2017']`    | `/ids/SHAP_analysis/global_SHAP_analysis/binary_one_vs_all/prediction_analysis/2017_with_oversampling`    |
| **2018**                      | `/ids/xgboost_models/binary_one_vs_all/trained_models/2018`                      | `['/ids/datasets/synthetics/predict_CTGAN/2018']`    | `/ids/SHAP_analysis/global_SHAP_analysis/binary_one_vs_all/prediction_analysis/2018`                      |
| **2018_with_oversampling**    | `/ids/xgboost_models/binary_one_vs_all/trained_models/2018_with_oversampling`    | `['/ids/datasets/synthetics/predict_CTGAN/2018']`    | `/ids/SHAP_analysis/global_SHAP_analysis/binary_one_vs_all/prediction_analysis/2018_with_oversampling`    |
| **for_all**                   | `/ids/xgboost_models/binary_one_vs_all/trained_models/for_all`                   | `['/ids/datasets/synthetics/predict_CTGAN/for_all']` | `/ids/SHAP_analysis/global_SHAP_analysis/binary_one_vs_all/prediction_analysis/for_all`                   |
| **for_all_with_oversampling** | `/ids/xgboost_models/binary_one_vs_all/trained_models/for_all_with_oversampling` | `['/ids/datasets/synthetics/predict_CTGAN/for_all']` | `/ids/SHAP_analysis/global_SHAP_analysis/binary_one_vs_all/prediction_analysis/for_all_with_oversampling` |

 *If analyzing model training instead of prediction, replace CTGAN paths with preprocessed or oversampling_TVAE data accordingly.*
---

## Output

Each run produces a single CSV file:

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
python xgb_one_vs_all_global_shap.py
```

Results are saved automatically to the defined `RESULTS_PATH`.

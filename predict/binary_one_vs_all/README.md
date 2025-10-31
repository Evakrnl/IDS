# One-vs-All XGBoost Prediction

This directory contains the **One-vs-All XGBoost prediction pipeline**, which performs inference using multiple pre-trained binary classifiers — one per attack label — and aggregates their probabilities to produce a **final unified prediction**.

The script evaluates predictions on real or synthetic datasets, computes a **classification report**, and generates a **confusion matrix** for detailed model assessment.

---

## Overview

The prediction workflow is designed to:

* Load multiple **binary XGBoost models** (one per attack label) from subfolders.
* Load one or more **CSV datasets** for evaluation.
* Allow **interactive label selection** (specific or all available datasets).
* Apply a **probability threshold** to classify uncertain samples as `"Unknown"`.
* Produce and save key evaluation artifacts (report, confusion matrix, heatmap).

---

## Script

**File:** `predict_one_vs_all_xgb.py`

This script requires manual configuration of your **data folders**, **model directory**, and **results output path** before running.

Example:

```python
data_folders = [
    "/home/ml1/Documents/ids/datasets/synthetics/predict_CTGAN/2017"
]
models_dir = "/home/ml1/Documents/ids/xgboost_models/binary_one_vs_all/trained_models/2017"
results_path = "/home/ml1/Documents/ids/predict/binary_one_vs_all/2017"
```

---

### Model – Data – Output Mapping

**2017**

* **Model Folder:** `/home/ml1/Documents/ids/xgboost_models/binary_one_vs_all/trained_models/2017`
* **Data Folder:** `/home/ml1/Documents/ids/datasets/synthetics/predict_CTGAN/2017`
* **Output Folder:** `/home/ml1/Documents/ids/predict/binary_one_vs_all/2017`

---

**for_all**

* **Model Folder:** `/home/ml1/Documents/ids/xgboost_models/binary_one_vs_all/trained_models/for_all`
* **Data Folder:** `/home/ml1/Documents/ids/datasets/synthetics/predict_CTGAN/for_all`
* **Output Folder:** `/home/ml1/Documents/ids/predict/binary_one_vs_all/for_all`

---

It is **recommended** to maintain the same folder naming pattern for consistency and automation.

---

## Output Files

After execution, the following results are saved inside the specified `results_path`:

| File                        | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| `classification_report.txt` | Classification report with precision, recall, and F1-score. |
| `confusion_matrix.csv`      | Confusion matrix values in tabular format.                  |
| `confusion_matrix.png`      | Visual heatmap of the confusion matrix.                     |

---

## Usage

1. Open the script `predict_one_vs_all_xgb.py`.
2. Update the following variables near the bottom:

   * `data_folders`
   * `models_dir`
   * `results_path`
3. Run the script:

```bash
python predict_one_vs_all_xgb.py
```

4. When prompted, either:

   * Enter label numbers (e.g., `1,3,5`) to choose specific datasets, or
   * Type `all` to process every available label.

5. The script will load data, perform inference, and save the evaluation results automatically.

---

## Notes

* The **probability threshold** for `"Unknown"` classification is currently **0.2**.
  You can modify it in the `predict_all()` function call:

  ```python
  df, y_true = predict_all(models, df, threshold=0.2)
  ```
* Infinite and NaN values are automatically removed before prediction.
* Each subfolder inside the model directory should contain a `.json` file representing the trained binary model for that label.
* The script supports both **CPU** and **GPU** execution (`device="cuda"`).

---

### Example Folder Hierarchy

```
ids/
├─ xgboost_models/
│  └─ binary_one_vs_all/
│     └─ trained_models/
│        ├─ 2017/
│        │  ├─ FTP/
│        │  │  └─ xgboost_model.json
│        │  ├─ DDoS/
│        │  │  └─ xgboost_model.json
│        │  └─ Infiltration/
│        │     └─ xgboost_model.json
│        └─ for_all/
│
├─ datasets/
│  └─ synthetics/
│     └─ predict_CTGAN/
│        ├─ 2017/
│        └─ for_all/
│
└─ predict/
   └─ binary_one_vs_all/
      ├─ 2017/
      └─ for_all/
```

---

## Requirements

Make sure the following dependencies are installed:

```bash
pip install xgboost pandas numpy seaborn matplotlib scikit-learn
```

---

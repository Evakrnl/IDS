# Multiclass XGBoost Prediction

This directory contains the prediction script and configuration structure used to evaluate pre-trained **multiclass XGBoost models** on **synthetic CTGAN-generated datasets**.

The script performs inference using an existing trained model, computes the **classification report**, and generates a **confusion matrix** for visual and quantitative evaluation.

---

## Overview

The prediction process is designed to:

* Load a pre-trained multiclass **XGBoost model** and its associated **LabelEncoder**.
* Load one or more **synthetic CTGAN datasets** for inference.
* Apply a **probability threshold** to classify uncertain samples as `"Unknown"`.
* Save evaluation artifacts, including the classification report and confusion matrix.

---

## Script

**File:** `predict_multiclass_xgb.py`

This script requires you to manually specify the correct paths before running it.

Example:

```python
results_path = "/home/ml1/Documents/ids/predict/multiclass/2017"
load_path = "/home/ml1/Documents/ids/xgboost_models/multiclass/trained_models/2017"
model_path = os.path.join(load_path, "xgboost_model.json")
encoder_path = os.path.join(load_path, "label_encoder.pkl")
data_folders = ["/home/ml1/Documents/ids/datasets/synthetics/predict_CTGAN/2017"]
```

---

### Model – Data – Output Mapping

**2017**
- **Model Folder:** `/home/ml1/Documents/ids/xgboost_models/multiclass/trained_models/2017`
- **Data Folder:** `/home/ml1/Documents/ids/datasets/synthetics/predict_CTGAN/2017`
- **Output Folder:** `/home/ml1/Documents/ids/predict/multiclass/2017`

---

**2017 (with oversampling)**
- **Model Folder:** `/home/ml1/Documents/ids/xgboost_models/multiclass/trained_models/2017_with_oversampling`
- **Data Folder:** `/home/ml1/Documents/ids/datasets/synthetics/predict_CTGAN/2017`
- **Output Folder:** `/home/ml1/Documents/ids/predict/multiclass/2017_with_oversampling`

---

**for_all**
- **Model Folder:** `/home/ml1/Documents/ids/xgboost_models/multiclass/trained_models/for_all`
- **Data Folder:** `/home/ml1/Documents/ids/datasets/synthetics/predict_CTGAN/for_all`
- **Output Folder:** `/home/ml1/Documents/ids/predict/multiclass/for_all`




It is **recommended** to maintain the same folder naming pattern for clarity and consistency.

---

## Output Files

After execution, the following results are generated inside the `results_path` directory:

| File                   | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `predict_report.txt`   | Classification report with precision, recall, and F1-score. |
| `confusion_matrix.csv` | Raw confusion matrix in tabular format.                     |
| `confusion_matrix.png` | Heatmap visualization of the confusion matrix.              |

---

## Usage

1. Open the script `predict_multiclass_xgb.py`.
2. Update the following variables to match your desired model and dataset configuration:

   * `results_path`
   * `load_path`
   * `data_folders`
3. Run the script:

```bash
python predict_multiclass_xgb.py
```

4. The results will be automatically saved under the corresponding `predict/multiclass/...` folder.

---

## Notes

* The script supports interactive selection of specific labels for prediction or the `"all"` option to evaluate every available label.
* The probability threshold for `"Unknown"` classification is currently set to **0.2**, which can be adjusted inside the script.
* Ensure that the model and encoder files (`xgboost_model.json`, `label_encoder.pkl`) exist in the specified `load_path`.

---

### Example Folder Hierarchy

```
ids/
├─ xgboost_models/
│  └─ multiclass/
│     └─ trained_models/
│        ├─ 2017/
│        │  ├─ xgboost_model.json
│        │  ├─ label_encoder.pkl
│        ├─ 2017_with_oversampling/
│        └─ for_all/
│
├─ datasets/
│  └─ synthetics/
│     └─ predict_CTGAN/
│        ├─ 2017/
│        └─ for_all/
│
└─ predict/
   └─ multiclass/
      ├─ 2017/
      ├─ 2017_with_oversampling/
      └─ for_all/
```

---

## Requirements

Ensure the following dependencies are installed before running the prediction script:

```bash
pip install xgboost pandas numpy seaborn matplotlib scikit-learn joblib
```

---


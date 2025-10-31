# Prediction Module – IDS Project

This directory contains all **prediction scripts** used to evaluate pre-trained XGBoost models on real and synthetic datasets. It supports both **multiclass** and **One-vs-All (binary)** inference modes.

The prediction stage is designed to load previously trained models, perform inference, and generate detailed evaluation metrics such as **classification reports** and **confusion matrices**.

---

## 📁 Structure

```
predict/
├─ multiclass/
│  ├─ 2017/
│  ├─ 2017_with_oversampling/
│  ├─ .../
│  └─ for_all/
│
└─ binary_one_vs_all/
   ├─ 2017/
   ├─ .../
   └─ for_all/
```

Each subdirectory corresponds to a specific model configuration and dataset.
Evaluation outputs (reports and heatmaps) are saved in these folders automatically after prediction.

---

## ⚙️ Scripts

### 🔹 `predict_multiclass_xgb.py`

Performs inference using a **single multiclass XGBoost model** and its corresponding **LabelEncoder**.

**Key features:**

* Loads the model and encoder files (`xgboost_model.json`, `label_encoder.pkl`).
* Loads CTGAN-generated or real datasets.
* Computes the classification report and confusion matrix.
* Supports adjustable probability thresholds for `"Unknown"` predictions.

---

### 🔹 `predict_one_vs_all_xgb.py`

Performs inference using **multiple binary XGBoost models**, one for each attack type.

**Key features:**

* Loads all trained binary models from subfolders.
* Allows interactive selection of specific labels or all available ones.
* Aggregates probabilities across models and assigns the most confident label.
* Classifies uncertain predictions as `"Unknown"` if below a threshold.

---

## 🧾 Output Files

After execution, the following files are generated inside each run's output directory:

| File                        | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| `classification_report.txt` | Classification report with precision, recall, and F1-score. |
| `confusion_matrix.csv`      | Numerical confusion matrix saved as CSV.                    |
| `confusion_matrix.png`      | Heatmap visualization of the confusion matrix.              |

---

##  Usage

1. Choose the correct script (`multiclass` or `binary_one_vs_all`).
2. Update the file paths for:

   * `data_folders`
   * `models_dir`
   * `results_path`
3. Run the script from terminal:

```bash
python predict_multiclass_xgb.py
# or
python predict_one_vs_all_xgb.py
```

4. The results will be saved automatically in the specified output folder.

---

## 📦 requirements.txt

All dependencies needed for running the prediction scripts are listed in the root-level `requirements.txt` file:

```text
xgboost
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install them before running any script:

```bash
pip install -r requirements.txt
```

---

## 💡 Notes

* The probability threshold for `"Unknown"` classification defaults to **0.2**.
* Infinite (`inf`) and missing (`NaN`) values are removed automatically.
* Both CPU and GPU execution are supported (`device="cuda"`).
* Keep consistent folder naming for data, models, and outputs to maintain automation.

---



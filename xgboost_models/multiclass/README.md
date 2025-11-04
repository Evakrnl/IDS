# Multiclass XGBoost Model Training

This directory contains the full pipeline, training outputs, and trained models for the **multiclass XGBoost-based Intrusion Detection System (IDS)**.

The model classifies network flows into multiple attack categories using labeled data from the **CICIDS2017** and **CSECICIDS2018** improved datasets, along with optional **synthetic oversampling** data generated using **TVAE**.

---

## Overview

The multiclass XGBoost model is designed to detect and classify various network intrusion types by learning patterns across multiple real and synthetic datasets.
Training is performed with GPU acceleration (CUDA), enabling fast and large-scale processing.

---

## Data Sources

The model uses data from the following locations:

* **Real data:** `datasets/preprocessed/per_year/2017/` and `datasets/preprocessed/per_year/2018/`
* **Synthetic data (optional oversampling):** `datasets/synthetics/oversampling_TVAE/2017/`, `datasets/synthetics/oversampling_TVAE/2018/`,`datasets/synthetics/oversampling_TVAE/for_all/`

In the `for_all` configuration, both years are combined into a single dataset for global training.

Example configuration:

```python
base_folders = [
    "/ids/datasets/preprocessed/per_year/2017",
    "/ids/datasets/preprocessed/per_year/2018"
]
```
While in the `for_all_with_oversampling` configuration, both years and their synthetics are combined into a single dataset for global training.

Example configuration:

```python
base_folders = [
    "/ids/datasets/preprocessed/per_year/2017",
    "/ids/datasets/preprocessed/per_year/2018",
    "/ids/datasets/synthetics/oversampling_TVAE/for_all"
]
```
---

## Main Script

**File:** `train_multiclass_xgb.py`

This script implements the complete training pipeline:

1. **Load and merge** multiple CSV files from real and synthetic folders.
2. **Clean** data by removing NaN and Inf values.
3. **Encode** labels using scikit-learn's `LabelEncoder` (saved for later inference).
4. **Train** a GPU-accelerated XGBoost model (multi:softmax objective).
5. **Save** all artifacts including model, encoder, classification report, and confusion matrix.

**Run command:**

```bash
python train_multiclass_xgb.py
```

---

## Folder Structure

```
multiclass/
├─ train_multiclass_xgb.py
├─ trained_models/
│  ├─ 2017/
│  ├─ 2017_with_oversampling/
│  ├─ 2018/
│  ├─ 2018_with_oversampling/
│  ├─ for_all/
│  └─ for_all_with_oversampling/
```

Each subfolder under `trained_models/` contains:

* `xgboost_model.json` → Saved model file.
* `label_encoder.pkl` → Saved label encoder mapping.
* `classification_report.txt` → Per-class precision, recall, and F1 metrics.
* `confusion_matrix.png` → Visual confusion matrix with labels sorted alphabetically.

---

## Notes

* Training can include both **real** and **synthetic** (TVAE-generated) samples.
* Labels are automatically sorted **alphabetically** during encoding for consistent visualization and evaluation.
* All training results are saved under `trained_models/` with clear naming based on dataset composition.
* GPU acceleration requires a CUDA-compatible environment with XGBoost compiled for GPU (`device='cuda'`).

---

## Output Artifacts

| File                        | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| `xgboost_model.json`        | Trained multiclass model file.                        |
| `label_encoder.pkl`         | Saved label encoder for prediction consistency.       |
| `classification_report.txt` | Detailed precision, recall, and F1 scores.            |
| `confusion_matrix.png`      | Visual confusion matrix of predicted vs. true labels. |


## Next Stages of the IDS Pipeline

The trained multiclass models are used in subsequent stages of the IDS framework:

### Prediction Phase
The corresponding trained model of each scenario will be reloaded to perform predictions **exclusively on synthetic CTGAN-generated data** that match the same scenario  
(e.g., **2017 model → 2017 synthetic CTGAN samples**).

This allows the evaluation of how well the models generalize to artificial attack data compared to real samples.

### Explainability & Feature Analysis (SHAP)
**SHAP (SHapley Additive exPlanations)** analysis is applied to the trained models to interpret feature importance, understand model decisions, and validate that the learned patterns align with domain expectations.

These stages are implemented in separate scripts within the project and rely on the trained multiclass XGBoost models stored here.

---

### Purpose
These models form the core of the **IDS multiclass classification framework**, enabling network traffic categorization and analysis based on both real and synthetically generated data.

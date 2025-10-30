# Multiclass XGBoost Prediction

This directory contains the prediction script and configuration structure used to evaluate pre-trained **multiclass XGBoost models** on **synthetic CTGAN-generated datasets**.

The script performs inference using an existing trained model, computes the **classification report**, and generates a **confusion matrix** for visual and quantitative evaluation.

---

## Overview

The prediction process is designed to:
- Load a pre-trained multiclass **XGBoost model** and its associated **LabelEncoder**.
- Load one or more **synthetic CTGAN datasets** for inference.
- Apply a **probability threshold** to classify uncertain samples as `"Unknown"`.
- Save evaluation artifacts, including the classification report and confusion matrix.

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

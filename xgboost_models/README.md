# XGBoost Models Overview

This directory contains all XGBoost-based models, training scripts, and results used in the Intrusion Detection System (IDS) project.
It includes both **multiclass** and **one-vs-all (binary)** classifiers trained on real and synthetic network traffic data.

---

## Folder Structure

```
xgboost_models/
├─ multiclass/
│  ├─ train_multiclass_xgb.py
│  └─ trained_models/
│      ├─ 2017/
│      ├─ 2018/
│      ├─ for_all/
│      ├─ 2017_with_oversampling/
│      ├─ 2018_with_oversampling/
│      └─ for_all_with_oversampling/

│
├─ one_vs_all/
│  ├─ train_one_vs_all_xgb.py
│  └─ trained_models/
│      ├─ 2017/
│      ├─ 2018/
│      ├─ for_all/
│      ├─ 2017_with_oversampling/
│      ├─ 2018_with_oversampling/
│      └─ for_all_with_oversampling/
│
└─ requirements.txt
```

---

## Overview

* **Multiclass Models**
  Train a single XGBoost classifier capable of predicting multiple attack categories simultaneously.
  Used for general classification across all network traffic labels.

* **One-vs-All Models**
  Train one binary classifier per label (e.g., `BENIGN` vs All, `DoS_Hulk` vs All, etc.).
  Used for focused evaluation, specialized detection, and explainability (SHAP) per attack type.

* **Requirements**
  The file `requirements.txt` lists the exact Python library versions (e.g., XGBoost, NumPy, pandas, scikit-learn) used during training.

---

## Reproducibility

To ensure that the trained models load and perform correctly, install the same library versions used during development:

```bash
pip install -r requirements.txt
```

> Anyone who wishes to use the uploaded trained models should install the exact dependencies listed in `requirements.txt` to ensure full compatibility with the training environment.

---

## Purpose

These models form the **core of the IDS machine learning framework**, enabling both:

* **Multiclass attack classification**, and
* **Targeted binary detection (one-vs-all)**

They serve as the foundation for the later stages of the IDS pipeline — including prediction on synthetic CTGAN data and SHAP-based explainability.

# SHAP Analysis Overview

This directory contains all scripts and documentation related to the **SHAP (SHapley Additive exPlanations)** analysis
for both **multiclass** and **One-vs-All** XGBoost models. The SHAP framework helps interpret model predictions
by quantifying each feature's contribution to the final decision.

The SHAP analysis in this project is divided into two main categories:

1. **Global SHAP Analysis** – Measures average feature importance across all True Positive samples.
2. **Per-Feature SHAP Analysis** – Examines detailed SHAP distributions per feature and per class.

Each category supports both **multiclass** and **binary (One-vs-All)** model types and includes separate scripts
for training-related and prediction-related analyses.

---

## Folder Structure

```
SHAP_analysis/
├── global_SHAP_analysis/
│   ├── binary_one_vs_all/
│   │   ├── prediction_analysis/
│   │   ├── training_analysis/
│   │   ├── README.md
│   │   └── xgb_one_vs_all_global_shap.py
│   │
│   └── multiclass/
│       ├── prediction_analysis/
│       ├── training_analysis/
│       ├── README.md
│       └── xgb_multiclass_global_shap.py
│
└── per-feature_SHAP_analysis/
    ├── binary_one_vs_all/
    │   ├── prediction_analysis/
    │   ├── training_analysis/
    │   ├── README.md
    │   └── xgb_one_vs_all_shap.py
    │
    └── multiclass/
        ├── prediction_analysis/
        ├── training_analysis/
        ├── README.md
        └── xgb_multiclass_shap.py
```

---

## Analysis Categories

### 1. Global SHAP Analysis

Computes **average SHAP values** across all True Positive samples. The resulting CSV files summarize
the overall feature importance for each dataset and model configuration.

### 2. Per-Feature SHAP Analysis

Calculates **feature-level SHAP distributions** per class. These results allow for a deeper inspection
of how each feature influences predictions for specific attack categories.

---

## Model Variants

Each SHAP analysis type supports the following configurations:

* **2017 / 2018:** Models trained on a specific year of the CICIDS dataset.
* **with_oversampling:** Models trained with synthetic data augmentation (TVAE-generated samples).
* **for_all:** Models trained on combined datasets (2017 + 2018).

---

## Output Files

Each SHAP script outputs one or more `.csv` files containing:

| Column       | Description                                             |
| ------------ | ------------------------------------------------------- |
| **Feature**  | Name of the dataset feature                             |
| **MeanSHAP** | Average SHAP contribution (positive or negative impact) |

---

## Requirements

All dependencies for SHAP analysis scripts are included in the project's `requirements.txt` file.
Ensure that your environment includes the following minimum versions:

```bash
Python >= 3.11
xgboost >= 2.0
pandas >= 2.0
numpy >= 1.24
joblib >= 1.3
```

For a complete list of dependencies (including GPU acceleration and SDV-based synthetic data generation tools),
see the full `requirements.txt` file in the project root directory.

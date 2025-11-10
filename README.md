# Intrusion Detection System (IDS) using XGBoost and GAN-based Synthetic Data

This repository was developed as part of a Bachelor's thesis at the **University of Thessaly**.
It presents a complete Intrusion Detection System (IDS) framework that combines **XGBoost machine learning models** with **deep generative models (CTGAN and TVAE)** for synthetic data generation, oversampling, and explainability analysis (SHAP).

The pipeline covers every stage of the IDS workflow, from dataset preprocessing and feature analysis to model training, synthetic data generation, prediction, and SHAP-based interpretability.

---

## 1. Project Overview

The goal of this project is to design, implement, and analyze an IDS capable of detecting multiple types of network intrusions using real and synthetic data derived from the **CICIDS2017** and **CSE-CIC-IDS2018** datasets.

Main contributions:

* Full preprocessing pipeline for real network flow data.
* Synthetic data generation using **CTGAN** and **TVAE**.
* Multiclass and One-vs-All XGBoost classifiers.
* Evaluation on synthetic (CTGAN) data to test generalization.
* Global and per-feature **SHAP analysis** for explainability.
* KDE visualizations for real vs. synthetic feature distribution comparison.

---

## 2. Folder Structure

```
IDS/
├── data_distribution_kde_plots/     # KDE visualization of real and synthetic feature distributions
├── datasets/                        # Real, preprocessed, and synthetic data
├── preprocessing/                   # Dataset preprocessing scripts and workflows
├── synthesizers/                    # CTGAN and TVAE synthetic data generators
├── xgboost_models/                  # Multiclass and One-vs-All XGBoost training scripts
├── predict/                         # Prediction modules for trained models
├── SHAP_analysis/                   # Explainability and SHAP feature attribution analysis
├── requirements.txt                 # Global dependencies
└── README.md                        # (this file)
```

---

## 3. Complete Pipeline

| Stage                            | Description                                                                                                              | Folder / Script                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| **1. Preprocessing**             | Cleans, normalizes, and splits the raw datasets by label or session. Handles NaN/Inf, duplicates, and label alignment.   | `preprocessing/`               |
| **2. Synthetic Data Generation** | Generates per-label synthetic samples using **CTGAN** or **TVAE**, evaluates quality via KS, RMSD, and duplicate ratios. | `synthesizers/`                |
| **3. KDE Distribution Analysis** | Visual comparison of feature distributions between real and synthetic data.                                              | `data_distribution_kde_plots/` |
| **4. Model Training**            | Trains **XGBoost** classifiers on preprocessed and synthetic data: <br>• Multiclass model <br>• One-vs-All binary models | `xgboost_models/`              |
| **5. Prediction**                | Runs trained models on synthetic CTGAN samples for generalization testing.                                               | `predict/`                     |
| **6. SHAP Explainability**       | Computes global and per-feature SHAP values to interpret model decisions.                                                | `SHAP_analysis/`               |

---

## 4. Module Overview

### `preprocessing/`

Implements two workflows:

* **Per-year preprocessing**: creates per-label CSVs from yearly datasets.
* **Per-session preprocessing**: splits and enumerates CSVs by day/session.

Scripts include label normalization, duplicate handling, and feature statistics.

---

### `datasets/`

Structured into:

* `original/` – Improved datasets (CICIDS2017, CSE-CIC-IDS2018) from CNS2022.
* `preprocessed/` – Clean, ready-to-train subsets (per year or per session).
* `synthetics/` – GAN-generated synthetic samples (`CTGAN`, `TVAE`).

Includes `SCHEMA.md` summarizing dataset columns and datatypes.

---

### `synthesizers/`

Implements synthetic data generation using:

* **CTGAN** (`generate_ctgan.py`) – GAN-based model for conditional tabular data.
* **TVAE** (`generate_tvae.py`) – Variational Autoencoder model for smooth latent generation.

Outputs include:

* Trained models (`.pkl`)
* Evaluation metrics (`.json`)
* Per-feature distance comparisons (`.csv`)

---

### `xgboost_models/`

Contains all XGBoost training pipelines:

* **Multiclass model** (`train_multiclass_xgb.py`)

  * Predicts multiple attack categories simultaneously.
* **One-vs-All model** (`train_one_vs_all_xgb.py`)

  * Trains one binary classifier per label.

Both models support:

* Real and synthetic training data.
* GPU acceleration (`device='cuda'`).
* Export of reports, confusion matrices, and trained models.

---

### `predict/`

Performs inference using the trained models.

* **`predict_multiclass_xgb.py`**: Evaluates a single multiclass model.
* **`predict_one_vs_all_xgb.py`**: Evaluates multiple binary models simultaneously.

Outputs:

* `classification_report.txt`
* `confusion_matrix.csv`
* `confusion_matrix.png`

---

### `SHAP_analysis/`

Contains both **Global** and **Per-Feature** SHAP explainability modules.

* **Global SHAP**: Aggregates SHAP values across all True Positives to measure global feature importance.
* **Per-Feature SHAP**: Computes class-specific feature contributions for interpretability.

Both **multiclass** and **One-vs-All** analyses are supported for:

* `training_analysis` (TVAE data)
* `prediction_analysis` (CTGAN data)

---

### `data_distribution_kde_plots/`

Includes visualization scripts for analyzing feature distribution similarity:

* `kde_plots_per_session.py` – per-day/session KDE plots (real data)
* `kde_plots_per_year.py` – yearly KDE plots (real data)
* `kde_plots_synthetics.py` – KDE plots for synthetic datasets (CTGAN/TVAE)

Used to visually validate the statistical quality of generated synthetic data.

---

## 5. Installation

### Python Version

Requires **Python ≥ 3.11**

### Dependencies

All required packages are listed in the main `requirements.txt` file:

```
pip install -r requirements.txt
```

Submodules (`xgboost_models/`, `predict/`, `synthesizers/`) include their own `requirements.txt` files with exact versions for reproducibility.

---

## 6. Reproducibility Notes

* All paths are relative to the `/ids/` directory.
* Each experiment folder (e.g., `2017`, `2018`, `for_all`) maintains a consistent structure across all modules.
* Trained models (`.json`) and encoders (`.pkl`) are fully compatible with the provided environment.
* For GPU support, ensure XGBoost is compiled with CUDA (`device='cuda'`).
* Synthetic model `.pkl` files are only compatible with the library versions listed in their respective `requirements.txt` files.

---

## 7. Academic Context

This repository was created as part of a **Bachelor’s thesis at the University of Thessaly**, focusing on:

> Intrusion Detection Systems using XGBoost and GAN-based Synthetic Data Generation (CTGAN & TVAE).

It demonstrates a complete end-to-end approach for machine-learning-based network intrusion detection, synthetic data evaluation, and explainable AI using SHAP.

---

## 8. License

The repository is shared for academic and research purposes only.
All dataset references belong to their respective authors and data providers (CICIDS2017, CSE-CIC-IDS2018, and DistriNet KU Leuven).

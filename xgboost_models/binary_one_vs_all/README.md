# One-vs-All (Binary) XGBoost Model Training

This directory contains the training pipeline, outputs, and trained models for the **One‑vs‑All (binary)** XGBoost models used in the Intrusion Detection System (IDS) project.

Each model is a binary classifier trained to distinguish a single target attack label (positive class) from all other traffic (negative class). This approach is useful for focused detection and analysis of individual attack families.

---

## Overview

The One‑vs‑All pipeline loads preprocessed per‑label CSV files (real and optionally synthetic), trains a binary XGBoost model for each selected label, and saves evaluation artifacts. Training uses GPU acceleration when available.

---

## Data Sources

Typical data locations used by the script:

* **Real preprocessed data:** `datasets/preprocessed/per_year/2017/` and `datasets/preprocessed/per_year/2018/`
* **Synthetic oversampling (optional):** `datasets/synthetics/oversampling_TVAE/2017/`, `datasets/synthetics/oversampling_TVAE/2018/`, `datasets/synthetics/oversampling_TVAE/for_all/`

Example `base_folders` configuration (Python):

```python
BASE_FOLDERS = [
    "/ids/datasets/preprocessed/per_year/2018"
]
RESULTS_PATH = "/ids/xgboost_models/binary_one_vs_all/trained_models/2018"
```

You can include any number of folders in `base_folders` (real and/or synthetic). The script concatenates all CSV files found under those folders.

---

## Main Script

**File:** `train_one_vs_all_xgb.py`

Responsibilities:

1. Load & merge multiple CSV files from the configured folders.
2. Clean data (replace `inf` with `NaN` and drop missing rows).
3. Enumerate available labels and allow interactive selection (or `all`).
4. For each selected label, train a binary XGBoost classifier (`label` vs `all`).
5. Save the model and evaluation artifacts per label.

**Run** (simple):

```bash
python train_one_vs_all_xgb.py
```

The script is interactive by default (choose labels via input). It can be easily adapted to accept command‑line arguments if non‑interactive runs are needed.

---

## Naming conventions

To keep artifacts consistent and filesystem‑safe the script uses sanitized label keys when writing files. Filenames follow the pattern:

```
<LABEL_KEY>_model.json
<LABEL_KEY>_classification_report.txt
<LABEL_KEY>_confusion_matrix.png
```

`LABEL_KEY` uses underscore (`_`) as separator (e.g., `DoS_GoldenEye`, `FTP_BruteForce_Patator`, `BENIGN`).

---

## Folder Structure

```
one_vs_all/
├─ train_one_vs_all_xgb.py
├─ trained_models/
│  ├─ 2017/
│  │  ├─ BENIGN/
│  │  │  ├─ BENIGN_model.json
│  │  │  ├─ BENIGN_classification_report.txt
│  │  │  └─ BENIGN_confusion_matrix.png
│  │  ├─ DoS_GoldenEye/
│  │  └─ ...
│  ├─ 2018/
│  └─ for_all_with_oversampling/
```

Each label folder stores the model and evaluation artifacts produced after training that specific one‑vs‑all classifier.

---

## Model & Training details

* XGBoost objective: `binary:logistic` (AUC used as primary eval metric during training).
* Typical hyperparameters (example): `n_estimators=300`, `max_depth=6`, `learning_rate=0.025`, `subsample=0.7`, `colsample_bytree=0.7`.
* Training uses `device='cuda'` when XGBoost is compiled with GPU support. Ensure a CUDA‑enabled environment for accelerated runs.

---

## Output artifacts

| File                                    | Description                                                         |
| ------------------------------------------------------------------------------------------------------------- |
| `<LABEL_KEY>_model.json`                | Trained XGBoost binary model for the specific label.                |
| `<LABEL_KEY>_classification_report.txt` | `classification_report` (precision/recall/F1) for this binary task. |
| `<LABEL_KEY>_confusion_matrix.png`      | Confusion matrix plot (Other vs Target label).                      |

---

## Next stages (pipeline integration)

Saved one‑vs‑all models are consumed in subsequent parts of the IDS pipeline:

* **Prediction:** Models can be reloaded to perform predictions on unseen data (real sessions or synthetic CTGAN/TVAE samples).
* **Explainability (SHAP):** SHAP analyses are applied to binary classifiers to examine feature importance and validate model behaviour for each attack type.

These downstream steps are implemented in separate scripts and expect the trained models and label keys produced here.

---

## Notes & Best Practices

* Keep the `trained_models/` folder free of duplicated filenames; the script creates one subfolder per label to avoid collisions.
* Prefer sanitized label names (underscore separator) so filenames are portable across OS and scripts.
* For full reproducibility, anyone who wishes to use the uploaded trained models should install the exact library versions listed in the provided requirements.txt, ensuring full compatibility with the training environment.

---

**Purpose:**
These binary (one‑vs‑all) models complement the multiclass classifier and enable focused detection, targeted evaluation, and per‑attack explainability within the IDS framework.

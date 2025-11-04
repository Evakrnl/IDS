# CTGAN Synthetic Data Generation

This directory contains all code and generated outputs related to **synthetic data generation using CTGAN**.  
The purpose of this module is to **expand and balance the dataset** by generating realistic, per-label synthetic network flows for intrusion detection research.

---

## Overview

The CTGAN (Conditional Tabular GAN) model learns the distribution of tabular network traffic data and generates synthetic samples that statistically resemble real data.  
Each attack category is modeled **independently**, ensuring that generated samples preserve the statistical properties of their corresponding real data.

---

## Main Script

**File:** `generate_ctgan.py`

This is the main script that handles:
1. Loading real per-label data from preprocessed datasets  
2. Training or re-loading CTGAN models  
3. Generating synthetic samples  
4. Computing distributional similarity metrics (KS, RMSD, etc.)  
5. Saving results (models, metrics, synthetic CSVs)

**Run interactively:**

```bash
python generate_ctgan.py
```
---

## Configuration

Each dataset label has its own configuration for sample size, epochs, and retraining behavior.
These parameters are defined in:

```
synthesizers/ctgan/2017/file_params.json
```

**Example:**

```json
{
    "DoS GoldenEye": { "samples": 100000, "epochs": 500, "retrain": false },
    "BENIGN": { "samples": 100000, "epochs": 100, "retrain": false }
}
```

The script automatically reads and updates this file after every execution.
---

## Folder Structure
```
ctgan/
├─ generate_ctgan.py
├─ 2017/
│  ├─ BENIGN/
│  │  ├─ model_BENIGN.pkl
│  │  ├─ BENIGN_metrics.json
│  │  └─ BENIGN_per_feature_distances.csv
│  ├─ DoS_GoldenEye/
│  └─ file_params.json
├─ 2018/
└─ for_all/
```

---


## CTGAN Synthetic Data Generation
===============================

This directory contains code and outputs related to the generation of synthetic data using CTGAN.

### Structure
----------

* **`generate_ctgan.py`**
  The main script that trains a CTGAN model for each attack category using the real per-label data.

* **2017/**, **2018/**, **for_all/**
  Each folder contains subfolders per label (e.g., `BENIGN`, `DoS_GoldenEye`, etc.), and inside each label folder:

    - `model_<LABEL>.pkl`
      
      The trained CTGAN model for the specific label.
      This file can be reloaded using `CTGANSynthesizer.load()` to generate new synthetic samples
      without retraining.

    - `<LABEL>_metrics.json`
      
      A JSON file with overall similarity metrics between real and synthetic samples, including:
        • Average KS Score  
        • RMSD (Root Mean Square Difference)  
        • Absolute Mean Difference (Overall)  
        • Euclidean Distance  
        • External and Internal Duplicate Ratios  

    - `<LABEL>_per_feature_distances.csv`
      
      A CSV file with per-feature comparisons between real and synthetic data:
        • KS-1 Score per feature (1 - KS statistic)  
        • RMSD per feature  
        • Absolute Mean Difference per feature  

---

## Path Structure Note
--------------------
Synthetic samples (`.csv` files) are storedhere in:
    datasets/synthetics/predict_CTGAN/

Each subfolder (`2017/`, `2018/`, `for_all/`) contains one `.csv` file per attack label.

• Corresponding trained models and evaluation reports (e.g., metrics, per-feature distances)  
  are saved separately inside:
    ctgan/2017/, ctgan/2018/, ctgan/for_all/

This separation keeps **raw synthetic samples** distinct from **training artifacts** and **evaluation files**.

---

## Notes
------
- All models are trained using data from `datasets/preprocessed/per_year`.
- The folder `for_all/` includes runs where categories from both 2017 and 2018 datasets are processed at once.

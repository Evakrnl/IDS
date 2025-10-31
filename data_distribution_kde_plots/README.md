# KDE Feature Distribution Analysis – Overview

This directory contains all scripts and generated results for **KDE (Kernel Density Estimation) feature distribution analysis**. The purpose of this analysis is to **visually assess and compare the distribution of features** across different datasets — both real and synthetic — used in the Intrusion Detection System (IDS) project.

---

##  Objective

The main goal of these plots is to understand **how closely synthetic datasets (CTGAN, TVAE, etc.) reproduce the statistical distributions** of real network traffic features from **CICIDS2017** and **CICIDS2018** datasets.

Through KDE visualization, we can:

* Observe how the values of each feature are distributed across different attack classes.
* Compare the shape and range of feature distributions between **2017 and 2018 real data**.
* Evaluate whether synthetic data generated via **GAN-based models (CTGAN, TVAE)** successfully replicates the natural variability and separation of features found in the real datasets.

This visual inspection helps validate data quality before model training and ensures that generated data is both **statistically sound** and **realistically distributed**.

---

## Directory Structure

```
data_distribution_kde_plots/
├── preprocessed/
│   ├── per_session/
│   │   ├── 2017/
│   │   └── 2018/
│   └── per_year/
│       ├── 2017/
│       └── 2018/
│
└── synthetics/
    ├── ctgan/
    │   ├── 2017/
    │   ├── 2018/
    │   └── for_all/
    └── tvae/
        ├── 2017/
        ├── 2018/
        └── for_all/
```

Each subfolder contains a collection of KDE plots (`.png` files), one for each numeric feature (e.g., `Flow Duration`, `Total Fwd Packets`, `Bwd IAT Mean`). These plots display the feature distributions per label (attack type) and allow comparison across datasets and synthetic generation methods.

---

##  Comparison Focus

| Dataset Type         | Description                                          | Purpose                                       |
| -------------------- | ---------------------------------------------------- | --------------------------------------------- |
| **Real 2017 & 2018** | Original CICIDS network traffic data                 | Baseline for comparison                       |
| **CTGAN Synthetic**  | Data generated using Conditional Tabular GAN         | Tests fidelity of GAN-generated distributions |
| **TVAE Synthetic**   | Data generated using Tabular Variational Autoencoder | Tests alternative generative approach         |

The comparison aims to show **how well each generative model captures feature relationships and density patterns** from the real data.

---

##  Scripts

| Script                     | Description                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| `kde_plots_per_session.py` | Generates KDE plots for preprocessed data per session (daily splits).   |
| `kde_plots_per_year.py`    | Generates KDE plots for data combined by year (CICIDS2017, CICIDS2018). |
| `kde_plots_synthetics.py`  | Generates KDE plots for synthetic datasets (CTGAN & TVAE).              |

Each script merges CSV files, extracts numeric columns, computes KDEs for each feature, and saves the plots as PNGs in their corresponding output folder.

---

##  Interpretation

Each generated figure visualizes the **probability density** of one numeric feature across all labels. By comparing these plots:

* Overlapping distributions suggest similarity in synthetic vs. real data.
* Narrower or shifted distributions may indicate feature imbalance or poor synthetic reproduction.
* Consistency across 2017 and 2018 real data supports dataset reliability.

---

##  Requirements

All scripts rely on the following Python libraries:

```bash
pip install pandas numpy matplotlib scipy
```

---

##  Summary

The KDE feature analysis serves as a **visual validation tool** in the IDS research project. It provides an intuitive understanding of **how data distributions differ between real and synthetic samples**, ensuring that generative models like CTGAN and TVAE faithfully reproduce network flow characteristics before further modeling or detection analysis.

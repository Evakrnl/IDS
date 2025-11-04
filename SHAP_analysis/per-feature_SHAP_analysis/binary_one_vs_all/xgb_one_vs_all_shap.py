#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Compute SHAP Feature Contributions for One-vs-All XGBoost Models
===============================================================================

Description:
------------
This script performs SHAP analysis on multiple One-vs-All XGBoost binary models.
It:
1. Loads synthetic CTGAN-generated prediction data.
2. Loads all One-vs-All trained XGBoost models from a results directory.
3. Computes per-class probabilities and final predicted labels.
4. For each correctly classified sample (True Positive),
   calculates mean SHAP values per feature in GPU-accelerated batches.
5. Saves one CSV per label containing the average SHAP contribution per feature.

Requirements:
-------------
- Python >= 3.11
- Libraries: xgboost, pandas, numpy

Example:
--------
    python xgb_one_vs_all_shap.py
===============================================================================
"""

import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier


# =============================================================================
# Configuration
# =============================================================================

DATA_FOLDERS = [
    ""
]

MODELS_DIR = ""
RESULTS_PATH = ""
os.makedirs(RESULTS_PATH, exist_ok=True)

THRESHOLD = 0.2  # classification probability threshold
DEVICE = "cuda:0"  # GPU device


# =============================================================================
# Helper Functions
# =============================================================================

def load_data(folders):
    """
    Load all CSVs from the specified folders, merge, shuffle, and clean data.

    Parameters
    ----------
    folders : list of str
        List of folder paths containing CSV files.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix for prediction.
    true_labels : np.ndarray
        True label values.
    feature_names : list of str
        List of feature column names.
    """
    print("Reading all CSV files...")
    dfs = []
    for folder in folders:
        all_csvs = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Folder: {folder} | Found {len(all_csvs)} files")
        for csv_file in all_csvs:
            try:
                df_temp = pd.read_csv(csv_file)
                dfs.append(df_temp)
                print(f"  Loaded: {csv_file} ({len(df_temp)} rows)")
            except Exception as e:
                print(f"  Error reading {csv_file}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nFinal dataset: {len(df)} rows")

    # Clean NaN and Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"Removed {before - len(df)} rows with NaN/Inf")

    X = df.drop(columns=["Label"], errors="ignore")
    true_labels = df["Label"].values
    feature_names = list(X.columns)

    return X, true_labels, feature_names


def load_models(models_dir):
    """
    Load all trained One-vs-All XGBoost models from subdirectories.

    Parameters
    ----------
    models_dir : str
        Directory containing subfolders, each with a model JSON file.

    Returns
    -------
    dict
        Mapping of label -> trained XGBClassifier instance.
    """
    print("\nLoading all saved One-vs-All models...")
    models = {}

    for label_folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".json"):
                model_path = os.path.join(folder_path, file)
                try:
                    model = XGBClassifier(tree_method="hist", device=DEVICE)
                    model.load_model(model_path)
                    models[label_folder] = model
                    print(f"  Loaded model: {label_folder}")
                except Exception as e:
                    print(f"  Error loading {model_path}: {e}")

    return models


def compute_shap_for_label(label, model, X, true_labels, final_pred, feature_names):
    """
    Compute SHAP values for True Positive predictions of a specific label.

    Parameters
    ----------
    label : str
        The label corresponding to the One-vs-All model.
    model : XGBClassifier
        Trained binary model for that label.
    X : pd.DataFrame
        Input features.
    true_labels : np.ndarray
        Ground truth labels.
    final_pred : np.ndarray
        Final predicted labels.
    feature_names : list of str
        Feature names for SHAP computation.
    """
    print(f"\nComputing SHAP for {label}...")

    # Select True Positive samples
    mask = (final_pred == label) & (true_labels == label)
    if mask.sum() == 0:
        print(f"  No TP predictions for {label}, skipping.")
        return

    X_pos = X[mask]
    print(f"  Found {len(X_pos)} TP samples for {label}")

    # Batch SHAP computation
    batch_size = 100000
    all_shap = []
    total = len(X_pos)

    for start in range(0, total, batch_size):
        end = start + batch_size
        X_batch = X_pos.iloc[start:end]

        dtest = xgb.DMatrix(X_batch, feature_names=feature_names)
        shap_batch = model.get_booster().predict(dtest, pred_contribs=True)
        shap_batch = shap_batch[:, :-1]  # remove bias term
        all_shap.append(shap_batch)

        print(f"  Batch {start // batch_size + 1}: {len(X_batch)} rows computed")

    shap_values = np.concatenate(all_shap, axis=0)
    mean_shap = pd.DataFrame(shap_values, columns=feature_names).mean().sort_values(ascending=False)

    # Save to CSV
    df_out = pd.DataFrame({
        "Feature": mean_shap.index,
        "Mean_SHAP": mean_shap.values
    })

    output_path = os.path.join(RESULTS_PATH, f"SHAP_{label}.csv")
    df_out.to_csv(output_path, index=False)
    print(f"  SHAP report for {label} saved to: {output_path}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution pipeline."""
    print(f"SHAP results will be saved to: {RESULTS_PATH}")

    # Load data and models
    X, true_labels, feature_names = load_data(DATA_FOLDERS)
    models = load_models(MODELS_DIR)

    # Compute probabilities for all One-vs-All models
    print("\nComputing final predictions with probabilities...")
    probs = {}
    for label, model in models.items():
        probs[label] = model.predict_proba(X)[:, 1]  # probability of class 1

    probs_df = pd.DataFrame(probs)
    max_proba = probs_df.max(axis=1)
    best_label = probs_df.idxmax(axis=1)

    final_pred = np.where(max_proba < THRESHOLD, "Unknown", best_label)

    # Compute SHAP values for each model/label
    for label, model in models.items():
        compute_shap_for_label(label, model, X, true_labels, final_pred, feature_names)

    print("\nAll SHAP computations completed successfully.")


if __name__ == "__main__":
    main()

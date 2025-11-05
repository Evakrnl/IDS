#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Compute Global SHAP Feature Contributions for One-vs-All XGBoost Models
===============================================================================

Description:
------------
This script performs a global SHAP analysis across multiple One-vs-All 
binary XGBoost models. It:
1. Loads CTGAN-generated synthetic prediction data.
2. Loads one trained binary model per label (One-vs-All setup).
3. Computes per-class probabilities and selects the best prediction.
4. Identifies all True Positive (TP) samples across all models.
5. Computes SHAP feature contributions for all TPs in GPU batches.
6. Aggregates and averages SHAP values globally across all classes.
7. Saves one CSV file with global mean SHAP feature contributions.

Requirements:
-------------
- Python >= 3.11
- Libraries: xgboost, pandas, numpy

Example:
--------
    python xgb_one_vs_all_global_shap.py
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

DATA_FOLDERS = [""]
MODELS_DIR = ""
RESULTS_PATH = ""
THRESHOLD = 0.2
BATCH_SIZE = 100000
DEVICE = "cuda:0"

os.makedirs(RESULTS_PATH, exist_ok=True)


# =============================================================================
# Helper Functions
# =============================================================================

def load_data(folders):
    """Load and clean all CSV files from the given folders."""
    print("Loading CSV files...")
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
    feature_names = list(X.columns)
    true_labels = df["Label"].values

    return X, feature_names, true_labels


def load_one_vs_all_models(models_dir, device="cuda:0"):
    """Load all trained One-vs-All models from subdirectories."""
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
                    model = XGBClassifier(tree_method="hist", device=device)
                    model.load_model(model_path)
                    models[label_folder] = model
                    print(f"  Loaded model: {label_folder}")
                except Exception as e:
                    print(f"  Error loading {model_path}: {e}")
    return models


def compute_global_tp_shap(models, X, feature_names, true_labels, threshold, batch_size):
    """Compute global mean SHAP values for all True Positive samples."""
    print("\nComputing Global True Positive SHAP across all classes...")

    # Get predictions for each model
    print("Computing prediction probabilities...")
    probs = {label: model.predict_proba(X)[:, 1] for label, model in models.items()}
    probs_df = pd.DataFrame(probs)

    max_proba = probs_df.max(axis=1)
    best_label = probs_df.idxmax(axis=1)
    final_pred = np.where(max_proba < threshold, "Unknown", best_label)

    tp_shap_all = []

    for label, model in models.items():
        mask = (final_pred == label) & (true_labels == label)
        if mask.sum() == 0:
            continue

        X_pos = X[mask]
        print(f" {label}: {len(X_pos)} TP samples")

        booster = model.get_booster()
        all_shap = []

        total = len(X_pos)
        for start in range(0, total, batch_size):
            end = start + batch_size
            X_batch = X_pos.iloc[start:end]
            dtest = xgb.DMatrix(X_batch, feature_names=feature_names)
            shap_batch = booster.predict(dtest, pred_contribs=True)
            shap_batch = shap_batch[:, :-1]  # remove bias
            all_shap.append(shap_batch)

        if all_shap:
            shap_values = np.concatenate(all_shap, axis=0)
            tp_shap_all.append(shap_values)

    if not tp_shap_all:
        print("No True Positives found in any class.")
        return None

    tp_shap_all = np.concatenate(tp_shap_all, axis=0)
    mean_shap = pd.DataFrame(tp_shap_all, columns=feature_names).mean().sort_values(ascending=False)

    df_out = pd.DataFrame({
        "Feature": mean_shap.index,
        "Mean_SHAP": mean_shap.values
    })

    return df_out


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution pipeline."""
    print(f"Global SHAP results will be saved to: {RESULTS_PATH}")

    # Load data and models
    X, feature_names, true_labels = load_data(DATA_FOLDERS)
    models = load_one_vs_all_models(MODELS_DIR, DEVICE)

    # Compute global SHAP
    df_global = compute_global_tp_shap(models, X, feature_names, true_labels,
                                       THRESHOLD, BATCH_SIZE)

    if df_global is not None:
        output_path = os.path.join(RESULTS_PATH, "TP_Global_SHAP_AllClasses.csv")
        df_global.to_csv(output_path, index=False)
        print(f"\nGlobal TP SHAP saved successfully to: {output_path}")


if __name__ == "__main__":
    main()

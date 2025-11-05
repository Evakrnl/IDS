#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Compute Global SHAP Feature Contributions from a Trained XGBoost Multiclass Model
===============================================================================

Description:
------------
This script:
1. Loads a trained XGBoost multiclass model and its corresponding LabelEncoder.
2. Reads CSV data from one or more folders, merges and cleans them.
3. Performs predictions for all samples.
4. Computes SHAP feature contributions in GPU-accelerated batches.
5. Extracts SHAP values for the predicted class only.
6. Calculates global mean SHAP values for all True Positive samples combined.
7. Saves a single CSV file containing the globally averaged SHAP contributions.

Requirements:
-------------
- Python >= 3.11
- Libraries: xgboost, pandas, numpy, joblib

Example:
--------
    python xgb_multiclass_global_shap.py
===============================================================================
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from xgboost import XGBClassifier


# =============================================================================
# Configuration
# =============================================================================

RESULTS_PATH = ""
LOAD_PATH = ""
MODEL_PATH = os.path.join(LOAD_PATH, "xgboost_model.json")
ENCODER_PATH = os.path.join(LOAD_PATH, "label_encoder.pkl")
BASE_FOLDERS = [""]
BATCH_SIZE = 100000
DEVICE = "cuda:0"


# =============================================================================
# Helper Functions
# =============================================================================

def load_model_and_encoder(model_path: str, encoder_path: str, device: str = "cuda:0"):
    """
    Load a trained XGBoost model and its corresponding LabelEncoder.

    Parameters
    ----------
    model_path : str
        Path to the saved XGBoost model (.json).
    encoder_path : str
        Path to the saved Label Encoder (.pkl).
    device : str, optional
        XGBoost device to use (e.g., "cuda:0" or "cpu").

    Returns
    -------
    model : XGBClassifier
        The loaded XGBoost classifier.
    label_encoder : LabelEncoder
        The corresponding label encoder.
    """
    print("Loading XGBoost model and LabelEncoder...")

    model = XGBClassifier(tree_method="hist", device=device)
    model.load_model(model_path)
    label_encoder = joblib.load(encoder_path)

    booster = model.get_booster()
    booster.set_param({"tree_method": "hist", "device": device})

    print("Model and encoder successfully loaded.")
    return model, label_encoder


def load_and_clean_data(folders):
    """
    Load all CSV files from the specified folders and clean NaN/Inf values.

    Parameters
    ----------
    folders : list of str
        List of folder paths containing CSV files.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame ready for prediction.
    """
    print("Loading CSV data...")
    dfs = []

    for folder in folders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        for csv_file in csv_files:
            try:
                df_temp = pd.read_csv(csv_file)
                dfs.append(df_temp)
                print(f"Loaded: {csv_file} ({len(df_temp)} rows)")
            except Exception as exc:
                print(f" Error loading {csv_file}: {exc}")

    if not dfs:
        raise FileNotFoundError("No CSV files found in the given folders.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows for prediction: {len(df)}")

    # Replace Inf and drop NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    print(f"Removed {before - len(df)} rows with NaN/Inf")

    return df


def compute_shap_batches(booster, X, batch_size):
    """
    Compute SHAP contributions in GPU batches to reduce memory usage.

    Parameters
    ----------
    booster : xgb.Booster
        The trained booster extracted from the model.
    X : pd.DataFrame
        Feature DataFrame.
    batch_size : int
        Number of samples per batch.

    Returns
    -------
    np.ndarray
        SHAP contributions (n_samples, n_classes, n_features + 1)
    """
    print("Computing SHAP contributions (GPU batches)...")

    total = len(X)
    num_batches = (total // batch_size) + (1 if total % batch_size > 0 else 0)

    all_shap = []
    start = 0
    for i in range(num_batches):
        end = min(start + batch_size, total)
        X_batch = X.iloc[start:end]

        dtest = xgb.DMatrix(X_batch, feature_names=list(X.columns))
        shap_batch = booster.predict(dtest, pred_contribs=True)
        all_shap.append(shap_batch)

        print(f" Computed batch {i + 1}/{num_batches}: {len(X_batch)} rows")
        start = end

    shap_values = np.concatenate(all_shap, axis=0)
    print("SHAP computation complete.")
    return shap_values


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution pipeline for global SHAP computation."""
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Load model and encoder
    model, label_encoder = load_model_and_encoder(MODEL_PATH, ENCODER_PATH, DEVICE)

    # Load and prepare data
    df = load_and_clean_data(BASE_FOLDERS)
    X = df.drop(columns=["Label"], errors="ignore")

    # Predictions
    print("Running predictions...")
    y_pred = model.predict(X)
    print("Predictions completed.")

    # Compute SHAP values in batches
    booster = model.get_booster()
    shap_values = compute_shap_batches(booster, X, BATCH_SIZE)
    shap_values = shap_values[:, :, :-1]  # remove bias

    # Extract SHAP values for predicted class
    n_samples = shap_values.shape[0]
    idx = np.arange(n_samples)
    shap_pred = shap_values[idx, y_pred, :]

    # Create SHAP DataFrame
    shap_df = pd.DataFrame(shap_pred, columns=X.columns)
    shap_df["Prediction"] = y_pred
    shap_df["True_Label"] = label_encoder.transform(df["Label"].values)

    # Compute global True Positive SHAP
    print("\nComputing Global True Positive SHAP (all classes combined)...")
    tp_all = shap_df[shap_df["Prediction"] == shap_df["True_Label"]].copy()
    shap_only_all = tp_all.drop(columns=["Prediction", "True_Label"])
    mean_shap_all = shap_only_all.mean().sort_values(ascending=False)

    df_global_tp = pd.DataFrame({
        "Feature": mean_shap_all.index,
        "Mean_SHAP": mean_shap_all.values
    })

    output_path = os.path.join(RESULTS_PATH, "TP_Global_SHAP_AllClasses.csv")
    df_global_tp.to_csv(output_path, index=False)
    print(f"Global TP SHAP saved successfully to: {output_path}")


if __name__ == "__main__":
    main()

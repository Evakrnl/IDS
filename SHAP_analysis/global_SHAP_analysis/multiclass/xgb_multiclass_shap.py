#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Compute SHAP-like Feature Contributions from a Trained XGBoost Multiclass Model
===============================================================================

Description:
------------
This script:
1. Loads a trained XGBoost JSON model and LabelEncoder.
2. Reads CSV files from one or more folders, concatenates them, and cleans NaN/Inf.
3. Makes predictions on all rows using the trained model.
4. Computes SHAP-like feature contributions in GPU-friendly batches.
5. Extracts contributions for the predicted class only.
6. Calculates the mean SHAP values per class and saves them as CSV files.
7. Optionally includes the true label (if present in the data).

Requirements:
-------------
- Python >= 3.11
- Libraries: xgboost, pandas, numpy, joblib

Example:
--------
Simply run the script:
    python xgb_multiclass_shap.py
===============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBClassifier


# =============================================================================
# Configuration
# =============================================================================

RESULTS_PATH = "/home/ml1/Documents/ids/SHAP_analysis/per-feature_SHAP_analysis/multiclass/..."
LOAD_PATH = "/home/ml1/Documents/ids/xgboost_models/multiclass/trained_models/..."
MODEL_PATH = os.path.join(LOAD_PATH, "xgboost_model.json")
ENCODER_PATH = os.path.join(LOAD_PATH, "label_encoder.pkl")
BASE_FOLDERS = ["/home/ml1/Documents/ids/datasets/preprocessed/per_year/..."]
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
    label_encoder : Label Encoder
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

    # Replace Inf values and drop NaN rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"After cleaning: {len(df)} rows")

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
    full_batches = total // batch_size
    remainder = total % batch_size

    batch_sizes = [batch_size] * full_batches
    if remainder > 0:
        batch_sizes.append(remainder)

    all_shap = []
    start = 0
    for i, size in enumerate(batch_sizes, start=1):
        end = start + size
        X_batch = X.iloc[start:end]
        dtest = xgb.DMatrix(X_batch, feature_names=list(X.columns))

        shap_batch = booster.predict(dtest, pred_contribs=True)
        all_shap.append(shap_batch)

        print(f" Computed batch {i}/{len(batch_sizes)}: {size} rows")
        start = end

    shap_values = np.concatenate(all_shap, axis=0)
    print("SHAP computation complete.")
    return shap_values


def extract_predicted_class_shap(shap_values, y_pred):
    """
    Extract SHAP values corresponding to the predicted class and remove bias term.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values with shape (n_samples, n_classes, n_features + 1).
    y_pred : np.ndarray
        Predicted class indices for each sample.

    Returns
    -------
    np.ndarray
        SHAP values for the predicted class only (n_samples, n_features).
    """
    shap_no_bias = shap_values[:, :, :-1]  # remove bias term
    n_samples = shap_no_bias.shape[0]
    idx = np.arange(n_samples)
    shap_pred = shap_no_bias[idx, y_pred, :]
    return shap_pred


def group_true_positives(shap_df):
    """
    Group SHAP results by predicted class, keeping only true positives.

    Parameters
    ----------
    shap_df : pd.DataFrame
        DataFrame with SHAP values, including 'Prediction' and 'True_Label'.

    Returns
    -------
    dict
        Mapping of class_id -> DataFrame of true positives.
    """
    groups = {}
    for class_id in np.unique(shap_df["Prediction"].values):
        df_class = shap_df[
            (shap_df["Prediction"] == class_id) & (shap_df["True_Label"] == class_id)
        ].copy()
        groups[class_id] = df_class
    return groups


def compute_mean_shap_per_class(groups, label_encoder, results_path):
    """
    Compute mean SHAP values for each class and save to CSV.

    Parameters
    ----------
    groups : dict
        Mapping from class_id -> DataFrame with true positives.
    label_encoder : LabelEncoder
        Fitted encoder for decoding class IDs.
    results_path : str
        Directory to save output CSV files.
    """
    os.makedirs(results_path, exist_ok=True)

    for class_id, df_class in groups.items():
        if df_class.empty:
            continue

        shap_only = df_class.drop(columns=["Prediction", "True_Label"], errors="ignore")
        mean_shap = shap_only.mean().sort_values(ascending=False)

        df_means = pd.DataFrame({
            "Feature": mean_shap.index,
            "Mean_SHAP": mean_shap.values
        })

        class_name = label_encoder.inverse_transform([class_id])[0]
        output_path = os.path.join(results_path, f"SHAP_{class_name}.csv")
        df_means.to_csv(output_path, index=False)
        print(f"Saved: {output_path} ({len(df_class)} samples)")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution pipeline."""
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # Load model and encoder
    model, label_encoder = load_model_and_encoder(MODEL_PATH, ENCODER_PATH, DEVICE)

    # Load and clean data
    df = load_and_clean_data(BASE_FOLDERS)

    # Split features and (optional) labels
    X = df.drop(columns=["Label"], errors="ignore")
    y_true = df["Label"] if "Label" in df.columns else None

    # Predict
    print("Running predictions...")
    y_pred = model.predict(X)
    print("Predictions completed.")

    # Compute SHAP values in batches
    booster = model.get_booster()
    shap_values = compute_shap_batches(booster, X, BATCH_SIZE)

    # Extract SHAP for predicted class only
    shap_pred = extract_predicted_class_shap(shap_values, y_pred)

    # Create SHAP DataFrame
    shap_df = pd.DataFrame(shap_pred, columns=X.columns)
    shap_df["Prediction"] = y_pred

    if y_true is not None:
        try:
            shap_df["True_Label"] = label_encoder.transform(y_true.values)
        except Exception:
            shap_df["True_Label"] = np.nan
    else:
        shap_df["True_Label"] = np.nan

    # Group by class (true positives only)
    tp_groups = group_true_positives(shap_df)
    for cid, df_class in tp_groups.items():
        cname = label_encoder.inverse_transform([cid])[0]
        print(f"Class {cname}: {len(df_class)} true-positive samples")

    # Compute mean SHAP per class
    compute_mean_shap_per_class(tp_groups, label_encoder, RESULTS_PATH)

    print("All SHAP computations completed successfully.")


if __name__ == "__main__":
    main()

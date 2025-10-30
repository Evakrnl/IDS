#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multiclass XGBoost Training Pipeline
====================================

This script trains an XGBoost classifier on labeled network flow data
for multiclass intrusion detection.

The training pipeline includes:
- Loading and merging multiple CSV files from one or more folders
- Handling missing or infinite values
- Label encoding with saved mapping for future predictions
- Training the model using GPU (CUDA) with customizable hyperparameters
- Saving the model, label encoder, classification report, and confusion matrix

Workflow:
---------
1. Collect and shuffle all .csv files from defined input folders.
2. Drop NaN and Inf values to ensure clean training input.
3. Encode string labels using scikit-learn's LabelEncoder and save mapping.
4. Train the XGBoost model using GPU acceleration (device='cuda').
5. Save the model and all evaluation artifacts (report, confusion matrix).

Outputs:
--------
- Trained model (`xgboost_model.json`)
- Label encoder (`label_encoder.pkl`)
- Classification report (`classification_report.txt`)
- Confusion matrix (`confusion_matrix.png`)

Configuration:
--------------
- All paths are defined in the `PATHS` section of this script.
- Modify `base_folders` and `results_path` to match your dataset and output structure.

Usage:
------
From the command line:

    python train_multiclass_xgb.py
"""

import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ========================
# PATHS
# ========================
base_folders = ["/home/ml1/Documents/ids/datasets/preprocessed/per_year/2017","/home/ml1/Documents/ids/datasets/preprocessed/per_year/2018","/home/ml1/Documents/ids/datasets/synthetics/oversampling_TVAE/for_all"]
results_path = "/home/ml1/Documents/ids/xgboost_models/multiclass/trained_models/for_all_with_oversampling"
os.makedirs(results_path, exist_ok=True)


# ========================
# FUNCTIONS
# ========================
def load_and_concatenate_csvs(folders: list[str]) -> pd.DataFrame:
    dfs = []
    for folder in folders:
        all_csvs = glob.glob(os.path.join(folder, "*.csv"))
        print(f"\nFolder: {folder} | Found {len(all_csvs)} files")

        for csv_file in all_csvs:
            try:
                df_temp = pd.read_csv(csv_file)
                dfs.append(df_temp)
                print(f" Loaded: {os.path.basename(csv_file)} ({len(df_temp)} rows)")
            except Exception as e:
                print(f" Error reading {csv_file}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def encode_labels(df: pd.DataFrame, results_path: str) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    joblib.dump(le, os.path.join(results_path, "label_encoder.pkl"))

    print("\nLabel Mapping:")
    for i, label in enumerate(le.classes_):
        print(f" {i}: {label}")

    return y, le


def train_model(X: pd.DataFrame, y: np.ndarray, num_classes: int) -> XGBClassifier:
    model = XGBClassifier(
        n_jobs=-1,
        use_label_encoder=False,
        device="cuda",
        tree_method="hist",
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=num_classes,
        n_estimators=500,
        max_depth=12,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=1.0,
        verbosity=1
    )
    model.fit(X, y, eval_set=[(X, y)], verbose=True)
    return model


def save_classification_report(y_true, y_pred, encoder, path):
    report = classification_report(encoder.inverse_transform(y_true), encoder.inverse_transform(y_pred))
    with open(path, "w") as f:
        f.write(report)


def save_confusion_matrix(y_true, y_pred, encoder, path):
    cm = confusion_matrix(encoder.inverse_transform(y_true), encoder.inverse_transform(y_pred), labels=encoder.classes_)
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                annot_kws={"size": 8.5})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print(f"\nResults will be saved to: {results_path}")
    df = load_and_concatenate_csvs(base_folders)
    print(f"\nTotal samples: {len(df)}")

    print("\nCleaning data (removing NaN and Inf)...")
    df = clean_data(df)
    print(f" After cleaning: {len(df)} samples")

    print("\nPreparing features and labels...")
    X = df.drop(columns=["Label"])
    y, encoder = encode_labels(df, results_path)

    print("\nTraining XGBoost model...")
    model = train_model(X, y, num_classes=len(encoder.classes_))

    model.save_model(os.path.join(results_path, "xgboost_model.json"))
    print("Model saved.")

    print("\nGenerating predictions and evaluation artifacts...")
    y_pred = model.predict(X)

    save_classification_report(y, y_pred, encoder, os.path.join(results_path, "classification_report.txt"))
    print("Classification report saved.")

    save_confusion_matrix(y, y_pred, encoder, os.path.join(results_path, "confusion_matrix.png"))
    print("Confusion matrix saved.")

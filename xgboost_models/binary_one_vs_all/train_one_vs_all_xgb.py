#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-vs-All XGBoost Training Pipeline
====================================

This script trains **binary (one-vs-all)** XGBoost models for each selected attack category
using preprocessed network flow data.

Each model learns to distinguish one target attack type (positive class)
against all other traffic types (negative class).

The training process includes:
- Loading and merging multiple CSV files
- Cleaning (removing NaN/Inf values)
- Interactive label selection
- Model training for each selected label
- Saving all results (model, report, confusion matrix)

Outputs:
--------
For each label, a dedicated results folder is created containing:
- `<LABEL>_model.json`
- `<LABEL>_classification_report.txt`
- `<LABEL>_confusion_matrix.png`

Usage:
------
Run from the command line:
    python train_one_vs_all_xgb.py
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ==============================
# CONFIGURATION
# ==============================
BASE_FOLDERS = [
    "/home/ml1/Documents/Diplomatiki/Datasets/perYear/2018"
]
RESULTS_PATH = "/home/ml1/Documents/Diplomatiki/TrainModels/OneVsAll/2018/results"
os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"\nResults will be saved to: {RESULTS_PATH}")


# ==============================
# FUNCTIONS
# ==============================
def load_and_merge_csvs(folders: list[str]) -> pd.DataFrame:
    """Load and merge all CSV files from the given list of folders."""
    dfs = []
    for folder in folders:
        all_csvs = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Folder: {folder} | Found {len(all_csvs)} files")
        for csv_file in all_csvs:
            try:
                df_temp = pd.read_csv(csv_file)
                dfs.append(df_temp)
                print(f" Loaded: {os.path.basename(csv_file)} ({len(df_temp)} rows)")
            except Exception as e:
                print(f" Error reading {csv_file}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nTotal combined samples: {len(combined_df)}")
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values with NaN and drop missing rows."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f" After cleaning: {len(df)} samples")
    return df


def select_labels(df: pd.DataFrame) -> list[str]:
    """Display available labels and allow the user to select one or more for training."""
    labels = df["Label"].unique()
    print(f"\nFound {len(labels)} unique labels:")
    for i, lbl in enumerate(labels, 1):
        print(f" {i}. {lbl}")

    valid = False
    while not valid:
        choice = input("\nEnter label numbers (e.g., 1,3,5) or 'all' to select all: ").strip().lower()
        if choice in ["all", "όλα", "ola", ""]:
            return labels

        indices = []
        for x in choice.split(","):
            x = x.strip()
            if x.isdigit():
                idx = int(x) - 1
                if 0 <= idx < len(labels):
                    indices.append(idx)
                else:
                    print(f" '{x}' is out of range.")
            else:
                print(f" '{x}' is not a valid number.")

        if indices:
            valid = True
            return [labels[i] for i in indices]
        else:
            print(" Invalid input, please try again.")


def train_one_vs_all(X: pd.DataFrame, y: pd.Series, target_label: str, results_dir: str):
    """
    Train a binary XGBoost model for the given label vs all others.
    Saves the model, classification report, and confusion matrix.
    """
    print(f"\nTraining One-vs-All model for: {target_label}")
    y_binary = np.where(y == target_label, 1, 0)

    model = XGBClassifier(
        n_estimators=300,
        verbosity=1,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=6,
        min_child_weight=1.0,
        gamma=0.3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2,
        reg_lambda=3,
        scale_pos_weight=1.6,
        learning_rate=0.025,
        use_label_encoder=False
    )

    model.fit(X, y_binary, eval_set=[(X, y_binary)], verbose=True)

    label_key = target_label.replace(" - ", "_").replace(" ", "_")
    save_dir = os.path.join(results_dir, target_label)
    os.makedirs(save_dir, exist_ok=True)

    # Model
    model_path = os.path.join(save_dir, f"{label_key}_model.json")
    model.save_model(model_path)
    print(f" Model saved at: {model_path}")

    # Predictions
    y_pred = model.predict(X)

    # Classification report
    report_path = os.path.join(save_dir, f"{label_key}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(y_binary, y_pred))
    print(f" Classification report saved at: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_binary, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Other", target_label],
        yticklabels=["Other", target_label],
        cbar=True,
        annot_kws={"size": 10}
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix — {target_label}")
    plt.tight_layout()

    cm_path = os.path.join(save_dir, f"{label_key}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f" Confusion matrix saved at: {cm_path}")


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("\n=== One-vs-All XGBoost Training Pipeline ===")
    df = load_and_merge_csvs(BASE_FOLDERS)
    df = clean_data(df)

    X = df.drop(columns=["Label"])
    y = df["Label"]

    selected_labels = select_labels(df)
    print("\nSelected labels:")
    for lbl in selected_labels:
        print(" -", lbl)

    for label in selected_labels:
        train_one_vs_all(X, y, label, RESULTS_PATH)

    print("\nAll One-vs-All model trainings completed successfully.")

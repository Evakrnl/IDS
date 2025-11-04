#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multiclass XGBoost Prediction Script
====================================

This script performs predictions using a pre-trained multiclass XGBoost model 
on one or more CTGAN-generated synthetic datasets.

It supports:
- Selecting specific labels for prediction
- Merging multiple label datasets
- Applying a probability threshold ("Unknown" assignment)
- Generating a confusion matrix and classification report
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder




# ============================================================
# Utility Functions
# ============================================================

def load_model_and_encoder(model_path: str, encoder_path: str) -> tuple[XGBClassifier, LabelEncoder]:

    """
    Load a trained XGBoost model and its corresponding LabelEncoder.

    Args:
        model_path (str): Path to the saved XGBoost model (.json)
        encoder_path (str): Path to the saved LabelEncoder (.pkl)

    Returns:
        tuple: (XGBClassifier model, LabelEncoder)
    """
    print("Loading XGBoost model and LabelEncoder...")
    model = XGBClassifier(tree_method="hist", device="cuda")
    model.load_model(model_path)
    encoder = joblib.load(encoder_path)
    print("Model and encoder loaded successfully.\n")
    return model, encoder


def load_csv_datasets(folders: list[str]) -> dict[str, list[pd.DataFrame]]:
    """
    Load all CSV datasets from the provided folders and group them by label.

    Args:
        folders (list[str]): List of folder paths containing .csv files.

    Returns:
        dict: Dictionary where keys are labels and values are lists of DataFrames.
    """
    combined_files = {}
    for folder in folders:
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        print(f"Folder: {folder} | Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            label = os.path.splitext(os.path.basename(csv_file))[0]
            try:
                df_temp = pd.read_csv(csv_file)
                combined_files.setdefault(label, []).append(df_temp)
                print(f" Loaded: {label} ({len(df_temp)} rows)")
            except Exception as e:
                print(f" Error loading {csv_file}: {e}")
    return combined_files


def select_labels(available_labels: list[str]) -> list[str]:
    """
    Prompt the user to select one or more labels for prediction.

    Args:
        available_labels (list[str]): List of all detected labels.

    Returns:
        list[str]: List of user-selected labels.
    """
    print("\nDetected labels:")
    for i, lbl in enumerate(available_labels, 1):
        print(f"{i}. {lbl}")

    while True:
        choice = input("\nEnter label numbers (e.g., 1,3,5) or 'all' for all: ").strip().lower()
        if choice in ["all", ""]:
            return available_labels

        indices = []
        valid = True
        for x in choice.split(","):
            x = x.strip()
            if x.isdigit():
                idx = int(x) - 1
                if 0 <= idx < len(available_labels):
                    indices.append(idx)
                else:
                    print(f"'{x}' is out of range.")
                    valid = False
            else:
                print(f"'{x}' is not a valid number.")
                valid = False

        if valid and indices:
            return [available_labels[i] for i in indices]
        else:
            print("Invalid selection. Please try again.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by replacing infinite values with NaN and removing missing rows.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def make_predictions(model, label_encoder, X: pd.DataFrame, threshold: float = 0.2) -> list[str]:
    """
    Perform model predictions with a probability threshold for assigning 'Unknown'.

    Args:
        model (XGBClassifier): Trained XGBoost model.
        label_encoder: Corresponding LabelEncoder.
        X (pd.DataFrame): Input feature matrix.
        threshold (float): Minimum probability required to assign a class.

    Returns:
        list[str]: Predicted class labels or 'Unknown'.
    """
    y_proba = model.predict_proba(X)
    best_idx = np.argmax(y_proba, axis=1)
    best_proba = np.max(y_proba, axis=1)

    y_pred_labels = [
        label_encoder.classes_[idx] if prob >= threshold else "Unknown"
        for idx, prob in zip(best_idx, best_proba)
    ]
    return y_pred_labels


def save_evaluation_results(y_true, y_pred, results_path: str):
    """
    Generate and save a classification report and confusion matrix.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        results_path (str): Output directory to save results.
    """
    report_classes = list(set(y_true))
    report = classification_report(y_true, y_pred, labels=report_classes, zero_division=0)

    report_path = os.path.join(results_path, "predict_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved: {report_path}")

    cm_classes = list(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=cm_classes)

    cm_df = pd.DataFrame(cm, index=cm_classes, columns=cm_classes)
    cm_csv_path = os.path.join(results_path, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path, index=True)
    print("Confusion matrix saved as CSV.")

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=cm_classes,
        yticklabels=cm_classes,
        cbar=True,
        annot_kws={"size": 8.5}
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_png_path = os.path.join(results_path, "confusion_matrix.png")
    plt.savefig(cm_png_path)
    plt.close()
    print("Confusion matrix saved as image.\n")


# ============================================================
# Main Execution
# ============================================================

def main():
    """Main execution function for multiclass XGBoost prediction."""
    # Paths (to be filled before execution)
    results_path = ""
    load_path = ""
    model_path = os.path.join(load_path, "xgboost_model.json")
    encoder_path = os.path.join(load_path, "label_encoder.pkl")
    data_folders = [""]

    os.makedirs(results_path, exist_ok=True)

    # Load model and encoder
    model, label_encoder = load_model_and_encoder(model_path, encoder_path)

    # Load datasets
    combined_files = load_csv_datasets(data_folders)
    available_labels = list(combined_files.keys())
    selected_labels = select_labels(available_labels)

    print("\nSelected labels:")
    for lbl in selected_labels:
        print(" -", lbl)

    # Merge and clean
    dfs = [pd.concat(combined_files[label], ignore_index=True) for label in selected_labels]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nFinal dataset: {len(df)} rows before cleaning")

    df = clean_dataframe(df)
    print(f"After cleaning: {len(df)} valid samples")

    X = df.drop(columns=["Label"], errors="ignore")

    # Predictions
    print("\nRunning predictions...")
    y_pred = make_predictions(model, label_encoder, X, threshold=0.2)

    # Evaluation
    if "Label" in df.columns:
        y_true = df["Label"].values
        save_evaluation_results(y_true, y_pred, results_path)
    else:
        print("No 'Label' column found — skipping evaluation.\n")

    # Unknown count summary
    unknown_count = y_pred.count("Unknown")
    print(f"Total Unknown predictions: {unknown_count} ({unknown_count / len(y_pred) * 100:.2f}%)")
    print("\n Prediction process completed successfully.")


if __name__ == "__main__":
    main()

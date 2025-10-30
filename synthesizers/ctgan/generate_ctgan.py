#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTGAN Synthetic Data Generator

This script trains or loads per-label CTGAN models, generates synthetic samples,
and evaluates similarity between real and synthetic data using:
- Kolmogorov–Smirnov (KS) score per numeric feature
- Root Mean Squared Difference (RMSD) on scaled features
- Absolute mean difference (scaled)
- Duplicate ratios (external against real data, and internal within synthetic data)

Interactive flow:
1) Scan configured data directories for CSV files containing a 'Label' column.
2) Merge files per label and show available labels to select.
3) For each selected label, optionally retrain a CTGAN model and choose sample count and epochs.
4) Save synthetic data, per-feature distances, and overall metrics to disk.

Notes
-----
- Paths are configured below in the PATH SETTINGS section.
- Docstrings follow the NumPy style for compatibility with common doc tools.


Usage
-----
Run interactively:
    python generate_ctgan.py
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sklearn.preprocessing import MinMaxScaler


# ----------------------------- PATH SETTINGS -----------------------------

PARAMS_FILE = Path(".../file_params.json")
DATA_DIRS = [
    Path("...")
]
OUTPUT_PATH = Path("...")
SYNTHETIC_SAVE_DIR = Path("...")

LABEL_COLUMN = "Label"


# ----------------------------- HELPER FUNCTIONS -----------------------------

def detect_device() -> str:
    """
    Detect whether CUDA is available and print the selected device.

    Returns
    -------
    str
        The device identifier: 'cuda' if a GPU is available, else 'cpu'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def ensure_dirs() -> None:
    """
    Ensure that required output directories exist.

    Creates the directories defined by OUTPUT_PATH and SYNTHETIC_SAVE_DIR.
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def collect_csv_paths(dirs: List[Path]) -> List[Path]:
    """
    Collect CSV file paths from the given directories.

    Parameters
    ----------
    dirs : list[pathlib.Path]
        A list of directories to search for CSV files.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of CSV file paths discovered in the provided directories.
    """
    csvs: List[Path] = []
    for d in dirs:
        if d.is_dir():
            csvs.extend(sorted([p for p in d.iterdir() if p.suffix.lower() == ".csv"]))
        else:
            print(f"Warning: Folder not found: {d}")
    return csvs


def load_labelled_dfs(csv_paths: List[Path]) -> Dict[str, pd.DataFrame]:
    """
    Read CSV files, group them by label, and merge per-label DataFrames.

    Each CSV is expected to contain a column named according to LABEL_COLUMN.
    The label is assumed to be constant within a file; the first row value is used.

    Parameters
    ----------
    csv_paths : list[pathlib.Path]
        Paths to CSV files to load.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from label name to a concatenated DataFrame for that label.
    """
    grouped: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            if LABEL_COLUMN not in df.columns:
                print(f"Skipping file {path.name}: missing '{LABEL_COLUMN}' column.")
                continue
            label = str(df[LABEL_COLUMN].iloc[0])
            grouped[label].append(df)
        except Exception as exc:
            print(f"Error reading {path}: {exc}")

    merged: Dict[str, pd.DataFrame] = {}
    for label, parts in grouped.items():
        merged[label] = pd.concat(parts, ignore_index=True)

    return merged


def prompt_select_labels(label_names: List[str]) -> List[str]:
    """
    Prompt the user to select labels by index in an interactive session.

    Parameters
    ----------
    label_names : list[str]
        The available label names.

    Returns
    -------
    list[str]
        The subset of label names chosen by the user.
    """
    print("\nSelect labels by their indices separated by commas (e.g., 0,2,4)")
    while True:
        selection_raw = input("Selection: ").strip()
        chosen: List[int] = []
        try:
            for token in selection_raw.split(","):
                token = token.strip()
                if token.isdigit():
                    idx = int(token)
                    if 0 <= idx < len(label_names):
                        chosen.append(idx)
            if chosen:
                return [label_names[i] for i in chosen]
            print("No valid numbers selected.")
        except Exception:
            print("Invalid input. Try again.")


def read_params(params_path: Path) -> Dict[str, dict]:
    """
    Read model parameter configurations from a JSON file if it exists.

    Parameters
    ----------
    params_path : pathlib.Path
        Path to the JSON file containing saved parameters.

    Returns
    -------
    dict[str, dict]
        A mapping of label to parameter dict, or an empty dict if unavailable.
    """
    if params_path.exists():
        try:
            with params_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def write_params(params_path: Path, params: Dict[str, dict]) -> None:
    """
    Write model parameter configurations to a JSON file.

    Parameters
    ----------
    params_path : pathlib.Path
        Output JSON path.
    params : dict[str, dict]
        The parameters to persist.
    """
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)
    print("Settings saved to fileParams.json")


def choose_ctgan_hyperparams(n_rows: int) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], int]:
    """
    Choose CTGAN hyperparameters based on dataset size.

    Parameters
    ----------
    n_rows : int
        Number of rows in the training DataFrame.

    Returns
    -------
    tuple
        A tuple of (pac, generator_dim, discriminator_dim, batch_size).
    """
    if n_rows < 100:
        pac, gdim, ddim, bs = 1, (32,), (32,), min(4, n_rows)
    elif n_rows < 1_000:
        pac, gdim, ddim, bs = 2, (128, 128), (128, 128), 128
    elif n_rows < 10_000:
        pac, gdim, ddim, bs = 4, (256, 256), (256, 256), 256
    elif n_rows < 100_000:
        pac, gdim, ddim, bs = 6, (256, 256, 128), (256, 256, 128), 512
    elif n_rows < 1_000_000:
        pac, gdim, ddim, bs = 8, (512, 512, 256), (512, 512, 256), 1024
    else:
        pac, gdim, ddim, bs = 10, (1024, 1024, 512), (1024, 1024, 512), 2048

    # Ensure batch size is a multiple of pac
    if bs % pac != 0:
        bs = (bs // pac) * pac or pac
    return pac, gdim, ddim, bs


def compute_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """
    Compute distribution similarity metrics between real and synthetic data.

    Metrics include:
    - KS Score Avg: average of (1 - KS statistic) across numeric columns
    - RMSD Mean: mean root-mean-squared difference on MinMax-scaled features
    - Absolute Mean Difference Overall: mean absolute difference of scaled means
    - Euclidean: L2 norm between scaled means

    Parameters
    ----------
    real_df : pandas.DataFrame
        The real dataset.
    synth_df : pandas.DataFrame
        The synthetic dataset.
    numeric_cols : list[str]
        Names of numeric columns to compare.

    Returns
    -------
    dict[str, float]
        A dictionary with metric names and values.
    """
    ks_scores = [1 - ks_2samp(real_df[col], synth_df[col])[0] for col in numeric_cols]
    ks_score_avg_all = round(float(np.mean(ks_scores)), 6)

    scaler = MinMaxScaler()
    real_scaled = scaler.fit_transform(real_df[numeric_cols])
    synth_scaled = scaler.transform(synth_df[numeric_cols])

    mean_real = real_scaled.mean(axis=0)
    mean_synth = synth_scaled.mean(axis=0)
    var_real = real_scaled.var(axis=0)
    var_synth = synth_scaled.var(axis=0)

    mean_squared_diff = var_real + var_synth + (mean_real - mean_synth) ** 2
    root_mean_squared_diff = np.sqrt(mean_squared_diff)
    abs_diff_per_feature = np.abs(mean_real - mean_synth)
    euclidean_overall = float(np.linalg.norm(mean_real - mean_synth, ord=2))

    metrics = {
        "KS Score Avg": ks_score_avg_all,
        "RMSD Mean": round(float(root_mean_squared_diff.mean()), 6),
        "Absolute Mean Difference Overall": round(float(abs_diff_per_feature.mean()), 6),
        "Euclidean": round(euclidean_overall, 6),
    }
    return metrics


def check_duplicates(real_df_no_label: pd.DataFrame, synth_df_no_label: pd.DataFrame) -> Tuple[float, float, int, int]:
    """
    Compute external and internal duplicate ratios for synthetic data.

    External duplicates: rows in synthetic data that also appear in the real data.
    Internal duplicates: duplicate rows within the synthetic data itself.

    Parameters
    ----------
    real_df_no_label : pandas.DataFrame
        Real data without the label column and duplicates removed for matching.
    synth_df_no_label : pandas.DataFrame
        Synthetic data without the label column.

    Returns
    -------
    tuple
        (external_dup_ratio, internal_dup_ratio, external_dup_count, internal_dup_count)
    """
    merged = synth_df_no_label.merge(real_df_no_label.drop_duplicates(), how="inner")
    external_dup_count = len(merged)
    external_dup_ratio = round(external_dup_count / max(len(synth_df_no_label), 1), 6)

    internal_dup_count = synth_df_no_label.duplicated().sum()
    internal_dup_ratio = round(internal_dup_count / max(len(synth_df_no_label), 1), 6)

    return external_dup_ratio, internal_dup_ratio, external_dup_count, internal_dup_count


# ----------------------------- MAIN EXECUTION -----------------------------

def main() -> None:
    """
    Entry point for the interactive CTGAN generation and evaluation workflow.

    Steps
    -----
    1. Detect device and ensure output directories exist.
    2. Load and merge CSV files per label.
    3. Prompt the user to select labels to process.
    4. For each selected label:
       - Configure training parameters (retrain, samples, epochs).
       - Train or load the CTGAN model.
       - Generate synthetic samples and save them to disk.
       - Compute and save per-feature distances and summary metrics.
    """
    detect_device()
    ensure_dirs()

    csv_paths = collect_csv_paths(DATA_DIRS)
    dfs_by_label = load_labelled_dfs(csv_paths)

    if not dfs_by_label:
        print("No valid CSV files with a 'Label' column found.")
        return

    print("\nDataFrames by Label:")
    label_names = list(dfs_by_label.keys())
    for i, label in enumerate(label_names):
        print(f"[{i}] Label '{label}' -> {len(dfs_by_label[label])} rows")

    selected_labels = prompt_select_labels(label_names)
    print(f"\nSelected labels: {selected_labels}")

    existing_params = read_params(PARAMS_FILE)
    file_params = dict(existing_params)

    # Collect per-label parameters interactively and persist them
    for label in selected_labels:
        tag = str(label).strip().replace(" - ", "_").replace("-", "_").replace(" ", "_")
        output_per_label = OUTPUT_PATH / tag
        output_per_label.mkdir(parents=True, exist_ok=True)
        model_path = output_per_label / f"model_{tag}.pkl"

        if model_path.exists():
            retrain = input(f"Retrain model for label {label}? (yes/no): ").strip().lower()
            retrain_flag = retrain in {"yes", "y", "true", "1"}
        else:
            print(f"No saved model for {label}. Training will be performed.")
            retrain_flag = True

        try:
            samples = int(input(f"How many synthetic samples for label {label}: ").strip())
        except ValueError:
            samples = 100000
            print("Invalid input. Using 100000 samples.")

        if retrain_flag:
            try:
                epochs = int(input(f"How many epochs for label {label}: ").strip())
            except ValueError:
                epochs = 300
                print("Invalid input. Using 300 epochs.")
            file_params[str(label)] = {"samples": samples, "epochs": epochs, "retrain": True}
        else:
            epochs = existing_params.get(str(label), {}).get("epochs")
            if epochs is None:
                print(f"No saved epochs for {label}. Please set manually or default to 300.")
                try:
                    epochs = int(input(f"Epochs for label {label}: ").strip())
                except ValueError:
                    epochs = 300
                    print("Invalid input. Using 300 epochs.")
            file_params[str(label)] = {"samples": samples, "epochs": epochs, "retrain": False}

    write_params(PARAMS_FILE, file_params)

    # Process each selected label
    for label in selected_labels:
        key = str(label)
        tag = str(label).strip().replace(" - ", "_").replace("-", "_").replace(" ", "_")

        output_per_label = OUTPUT_PATH / tag
        model_path = output_per_label / f"model_{tag}.pkl"
        synthetic_path = SYNTHETIC_SAVE_DIR / f"{key}.csv"

        df = dfs_by_label[label].copy()
        # Replace infs with NaN and drop rows that contain NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"DataFrame does not contain '{LABEL_COLUMN}' column.")

        df_data = df.drop(columns=[LABEL_COLUMN])

        params = file_params.get(key, {})
        synthetic_samples = int(params.get("samples", 100000))
        epochs = int(params.get("epochs", 300))
        train_new_model = bool(params.get("retrain", True))

        if not train_new_model and not model_path.exists():
            print(f"Model for {label} not found, forcing retraining.")
            train_new_model = True

        print(f"Using parameters for label {label}: {synthetic_samples} samples, {epochs} epochs")

        # Build SDV metadata from the training DataFrame
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_data)

        # Train or load CTGAN
        if train_new_model:
            print(f"\nTraining label {label} with {len(df)} rows")
            pac, gen_dim, disc_dim, batch_size = choose_ctgan_hyperparams(len(df_data))
            print(f"Training for {epochs} epochs with batch size {batch_size} on {len(df_data):,} rows.")

            synth = CTGANSynthesizer(
                metadata,
                epochs=epochs,
                batch_size=batch_size,
                pac=pac,
                generator_dim=gen_dim,
                discriminator_dim=disc_dim,
                enforce_min_max_values=True,
                enforce_rounding=True,
            )
            start = time.time()
            synth.fit(df_data)
            elapsed = round(time.time() - start, 2)
            print(f"Training completed in {elapsed} seconds with model {synth.__class__.__name__}")
            synth.save(model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Using existing model from: {model_path}")
            synth = CTGANSynthesizer.load(model_path)

        # Generate synthetic data
        synthetic_df = synth.sample(synthetic_samples)
        synthetic_df[LABEL_COLUMN] = key
        synthetic_df.to_csv(synthetic_path, index=False)
        print(f"Saved {len(synthetic_df)} synthetic samples to: {synthetic_path}")

        # Compute metrics and per-feature distances
        try:
            numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()
            metrics = compute_metrics(df_data, synthetic_df, numeric_cols)

            scaler = MinMaxScaler()
            real_scaled = scaler.fit_transform(df_data[numeric_cols])
            synth_scaled = scaler.transform(synthetic_df[numeric_cols])

            mean_real = real_scaled.mean(axis=0)
            mean_synth = synth_scaled.mean(axis=0)
            var_real = real_scaled.var(axis=0)
            var_synth = synth_scaled.var(axis=0)
            mean_squared_diff = var_real + var_synth + (mean_real - mean_synth) ** 2
            root_mean_squared_diff = np.sqrt(mean_squared_diff)
            abs_diff_per_feature = np.abs(mean_real - mean_synth)

            per_feature_df = pd.DataFrame(
                {
                    "feature": numeric_cols,
                    "KS-1": [1 - ks_2samp(df_data[c], synthetic_df[c])[0] for c in numeric_cols],
                    "RMSD": root_mean_squared_diff,
                    "Absolute Mean Difference": abs_diff_per_feature,
                }
            )
            per_feature_path = output_per_label / f"{tag}_per_feature_distances.csv"
            per_feature_df.to_csv(per_feature_path, index=False)
            print(f"Per-feature distances saved to: {per_feature_path}")

            real_no_label = df_data.drop(columns=[LABEL_COLUMN], errors="ignore")
            synth_no_label = synthetic_df.drop(columns=[LABEL_COLUMN], errors="ignore")
            ext_dup_ratio, int_dup_ratio, ext_dup_count, int_dup_count = check_duplicates(real_no_label, synth_no_label)
            print(f"External duplicates: {ext_dup_count} ({ext_dup_ratio:.2%})")
            print(f"Internal duplicates: {int_dup_count} ({int_dup_ratio:.2%})")

            metrics.update({
                "Duplicate Ratio": ext_dup_ratio,
                "Internal Duplicate Ratio": int_dup_ratio,
            })
        except Exception as exc:
            print(f"Error computing metrics: {exc}")
            metrics = {
                "KS Score Avg": None,
                "RMSD Mean": None,
                "Absolute Mean Difference Overall": None,
                "Euclidean": None,
                "Duplicate Ratio": None,
                "Internal Duplicate Ratio": None,
            }

        metrics_json = output_per_label / f"{tag}_metrics.json"
        with metrics_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"Metrics saved to: {metrics_json}")


if __name__ == "__main__":
    main() 
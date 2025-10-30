import os
import pandas as pd
import numpy as np
from pathlib import Path


def clean_dataset_folder(input_folder: str | Path, columns_to_drop: list[str]) -> None:
    """
    Clean all CSV files in the given folder:
    - Drop unnecessary columns
    - Replace inf/-inf with NaN and remove affected rows
    - Save cleaned data back to the same files
    - Log cleaning statistics to a log file in the folder

    Parameters
    ----------
    input_folder : str | Path
        Folder containing the CSV files (will be searched recursively).
    columns_to_drop : list[str]
        Column names to remove if present in each CSV file.
    """
    input_folder = Path(input_folder)
    log_file_path = input_folder / "cleaning_log.txt"

    # Initialize log file
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Cleaning log:\n\n")

    # Walk through all subfolders and CSV files
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if not filename.endswith(".csv"):
                continue

            filepath = Path(root) / filename
            print(f"Processing: {filepath}")

            try:
                # Load file
                df = pd.read_csv(filepath, low_memory=False)
                initial_rows = len(df)

                # Drop selected columns (only if they exist)
                dropped_cols = [col for col in columns_to_drop if col in df.columns]
                df = df.drop(columns=dropped_cols, errors="ignore")

                # Replace inf/-inf with NaN, count them, and remove NaN rows
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                num_nans = df.isna().any(axis=1).sum()
                df.dropna(inplace=True)

                final_rows = len(df)

                # Save cleaned file (overwrite original)
                df.to_csv(filepath, index=False)
                print(f" Saved: {filepath}")

                # Log cleaning info
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"File: {filepath}\n")
                    log_file.write(f" - Initial rows: {initial_rows}\n")
                    log_file.write(f" - Columns removed: {dropped_cols}\n")
                    log_file.write(f" - Rows with NaN (from inf): {num_nans}\n")
                    log_file.write(f" - Final rows: {final_rows}\n\n")

            except Exception as e:
                print(f" Error in {filename}: {e}")
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f" Error in {filepath}: {e}\n\n")


if __name__ == "__main__":
    # Configuration
    input_folder = "..."

    columns_to_drop = [
        "Attempted Category", "id", "Flow ID",
        "Src IP", "Src Port", "Dst IP", "Timestamp"
    ]

    clean_dataset_folder(input_folder, columns_to_drop)

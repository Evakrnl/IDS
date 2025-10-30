import os
import re
import pandas as pd
from pathlib import Path


def make_filename_safe(label: str) -> str:
    """
    Return the label as a valid filename by replacing only illegal characters.
    """
    return re.sub(r'[\\/:\*\?"<>\|\n\r\t]', "_", str(label).strip())


def group_files_by_label(input_folder: str | Path, output_folder: str | Path, chunksize: int = 1_000_000) -> None:
    """
    Split each CSV file by 'Label' and save one CSV per label.
    The filename will be identical to the label name (only illegal filesystem characters replaced).
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue

        filepath = input_folder / filename
        print(f"Reading: {filename}")

        try:
            for chunk in pd.read_csv(filepath, chunksize=chunksize):
                if "Label" not in chunk.columns:
                    print(f"Skipped {filename}: missing 'Label' column.")
                    continue

                # Group rows by unique label
                for label in chunk["Label"].dropna().unique():
                    label_df = chunk.loc[chunk["Label"] == label]
                    safe_name = make_filename_safe(label)
                    out_path = output_folder / f"{safe_name}.csv"

                    # Append rows to per-label file
                    label_df.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

            print(f"Completed: {filename}")

        except Exception as e:
            print(f"Error in file {filename}: {e}")


if __name__ == "__main__":
    # Folders
    inputFolder = "..."
    outputFolder = "..."

    group_files_by_label(inputFolder, outputFolder)

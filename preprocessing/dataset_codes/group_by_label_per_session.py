import os
import re
import pandas as pd
from pathlib import Path


def make_filename_safe(text: str) -> str:
    """
    Return `text` unchanged except for replacing illegal filesystem characters,
    so it can be used safely as a filename.
    """
    return re.sub(r'[\\/:\*\?"<>\|\n\r\t]', "_", str(text).strip())


def split_csvs_into_label_files(input_folder: str | Path, output_folder: str | Path) -> None:
    """
    For each CSV in `input_folder`:
      - Create a subfolder under `output_folder` named after the CSV (without .csv).
      - Split the dataframe by 'Label' and save one CSV per label inside that subfolder.
      - Write per-file logs and a general log under `output_folder/general_log.txt`.

    Notes
    -----
    - The output filename is identical to the label value (only illegal characters are replaced).
    - If a CSV lacks a 'Label' column, it is skipped and logged.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # General log for all files
    general_log_path = output_folder / "general_log.txt"
    with general_log_path.open("w", encoding="utf-8") as general_log:
        general_log.write("General split log:\n\n")

    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue

        filepath = input_folder / filename
        print(f"Processing: {filename}")

        try:
            # Load CSV
            df = pd.read_csv(filepath, low_memory=False)

            # Create subfolder named after the input CSV (without extension)
            base_name = Path(filename).stem
            output_subfolder = output_folder / base_name
            output_subfolder.mkdir(parents=True, exist_ok=True)

            # Per-file log inside the subfolder
            log_file_path = output_subfolder / "log.txt"
            with log_file_path.open("w", encoding="utf-8") as log_file:
                log_file.write(f"Log for {filename}\n\n")

            # Ensure 'Label' column exists
            if "Label" not in df.columns:
                msg = f"The file {filename} does not contain a 'Label' column.\n"
                print(msg.strip())
                with log_file_path.open("a", encoding="utf-8") as log_file, \
                     general_log_path.open("a", encoding="utf-8") as general_log:
                    log_file.write(msg)
                    general_log.write(msg)
                continue

            # Split by label and save one CSV per label
            for label, group in df.groupby("Label"):
                safe_label = make_filename_safe(label)
                output_path = output_subfolder / f"{safe_label}.csv"

                group.to_csv(output_path, index=False)
                print(f"Saved: {output_path}")

                # Log to per-file and general logs
                msg = f"Label: {label} → Rows: {len(group)}\n"
                with log_file_path.open("a", encoding="utf-8") as log_file, \
                     general_log_path.open("a", encoding="utf-8") as general_log:
                    log_file.write(msg)
                    general_log.write(f"{filename} - {msg}")

        except Exception as e:
            msg = f"Error with {filename}: {e}\n"
            print(msg.strip())
            with general_log_path.open("a", encoding="utf-8") as general_log:
                general_log.write(msg)


if __name__ == "__main__":
    # Input/Output folders
    input_folder = "/home/ml1/Desktop/Datasets1/Original/CICIDS2017_improved"
    output_folder = "/home/ml1/Desktop/Datasets1/groupPerFile/2017"

    split_csvs_into_label_files(input_folder, output_folder)

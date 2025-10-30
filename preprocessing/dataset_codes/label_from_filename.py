import os
import re
from pathlib import Path
import pandas as pd


def make_filename_safe(text: str) -> str:
    """Keep the name as is, only replacing illegal filesystem characters."""
    return re.sub(r'[\\/:\*\?"<>\|\n\r\t]', "_", str(text).strip())


def enumerate_first_appearance(root_folder: str | Path) -> None:
    """
    For each .csv file inside `root_folder` (and its subfolders):
      - Detect the base name (without .csv)
      - Maintain a global counter per base name
      - Rename each file to <name>-<number>.csv
      - Set the 'Label' column to match <name>-<number>
    """
    root = Path(root_folder)

    # Collect all CSV files in stable order
    csv_paths = []
    for current_dir, _, files in os.walk(root):
        current_dir = Path(current_dir)
        for f in sorted(files):
            if f.endswith(".csv"):
                csv_paths.append(current_dir / f)

    counters: dict[str, int] = {}

    for p in csv_paths:
        try:
            base_name = Path(p.stem).name  # e.g. "Benign" or "Botnet Ares"
            base_name = make_filename_safe(base_name)
            counters[base_name] = counters.get(base_name, 0) + 1
            idx = counters[base_name]

            new_stem = f"{base_name}-{idx}"
            new_path = p.with_name(f"{new_stem}.csv")

            # Update the Label column
            df = pd.read_csv(p, low_memory=False)
            df["Label"] = new_stem

            tmp = p.with_suffix(".tmp.csv")
            df.to_csv(tmp, index=False)
            os.replace(tmp, new_path)

            # Remove old file if renamed
            if new_path != p and p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

            print(f"{p} -> {new_path}  (Label = {new_stem})")

        except Exception as e:
            print(f"Error with {p}: {e}")


if __name__ == "__main__":
    ROOT = "..."  # Replace with your actual path
    enumerate_first_appearance(ROOT)

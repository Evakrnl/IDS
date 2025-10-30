import os
import pandas as pd
from pathlib import Path


def generate_normalized_statistics(input_dir: str | Path, stats_dir: str | Path) -> None:
    """
    For each CSV in the input folder:
      - Keep only numeric columns.
      - Apply Min-Max normalization (0–1).
      - Compute descriptive statistics.
      - Sort so that features with mean = 0 appear last.
      - Save each result in the stats output folder.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing the input CSV files (one per label).
    stats_dir : str | Path
        Directory where the output statistics files will be saved.
    """
    input_dir = Path(input_dir)
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file in csv_files:
        input_path = input_dir / file
        print(f"Processing file: {input_path}")

        # Load data
        df = pd.read_csv(input_path)

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=["number"])

        # Min-Max normalization (0–1)
        normalized_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())

        # Compute statistics
        stats = normalized_df.describe().transpose()

        # Sort so that features with mean = 0 appear last
        stats["mean_is_zero"] = stats["mean"] == 0
        stats = stats.sort_values(by="mean_is_zero").drop(columns=["mean_is_zero"])

        # Save results
        output_path = stats_dir / f"{input_path.stem}_stats.csv"
        stats.to_csv(output_path, index=True)
        print(f"Statistics saved to: {output_path}")


if __name__ == "__main__":
    input_dir = "..."
    stats_dir = ".../stats"
    generate_normalized_statistics(input_dir, stats_dir)

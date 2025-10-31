"""
KDE Feature Distribution Plotter (Per Session)
---------------------------------------------
This script generates kernel density estimation (KDE) plots for each numeric feature
across all labeled sessions (days) from the CICIDS2017 and CICIDS2018 datasets.

Each dataset folder contains subfolders per session/day.
Each day folder includes multiple CSV files with a 'Label' column.
The script merges all CSVs, retains the day information, and produces one KDE plot per feature.

Example Configuration:
----------------------
INPUT_FOLDERS = {
    "2017": "/datasets/preprocessed/per_session/2017",
    "2018": "/datasets/preprocessed/per_session/2018",
}
OUTPUT_FOLDER = "/data_distribution_kde_plots/preprocessed/per_session"
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# === Configuration ===
INPUT_FOLDERS = {
    "2017": "/datasets/preprocessed/per_session/2017",
    "2018": "/datasets/preprocessed/per_session/2018"
}
OUTPUT_FOLDER = "/data_distribution_kde_plots/preprocessed/per_session"
BINS = 10  # number of intervals on the X-axis

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_session_data(base_folder: str) -> pd.DataFrame:
    """
    Load and merge all CSV files from subfolders (days) within a session directory.

    Parameters
    ----------
    base_folder : str
        Path to the yearly 'per_session' dataset folder (e.g., 2017 or 2018).

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing all session CSVs with a new column '_day_'
        indicating the source day folder.
    """
    all_dfs = []
    for day_folder in os.listdir(base_folder):
        day_path = os.path.join(base_folder, day_folder)
        if not os.path.isdir(day_path):
            continue

        for filename in os.listdir(day_path):
            if not filename.endswith(".csv"):
                continue
            try:
                file_path = os.path.join(day_path, filename)
                df = pd.read_csv(file_path)
                if "Label" in df.columns:
                    df["_day_"] = day_folder
                    all_dfs.append(df)
            except Exception as e:
                print(f" Error in {filename}: {e}")

    if not all_dfs:
        print(" No valid CSVs with 'Label' found.")
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def plot_feature_kde(data: pd.DataFrame, output_dir: str, dataset_name: str) -> None:
    """
    Generate and save KDE plots for all numeric features grouped by label and day.

    Parameters
    ----------
    data : pd.DataFrame
        The merged dataset containing numeric features, 'Label', and '_day_'.
    output_dir : str
        Path to save generated KDE plots.
    dataset_name : str
        The dataset identifier (e.g., '2017' or '2018').
    """
    numeric_df = data.select_dtypes(include=["number"])
    labels = sorted(data["Label"].unique())

    for col in numeric_df.columns:
        if col in ["Dst Port", "Protocol"]:
            continue

        labels_count = len(labels)
        cols_per_row = 5
        rows_count = (labels_count // cols_per_row) + int(labels_count % cols_per_row > 0)

        fig, axes = plt.subplots(rows_count, cols_per_row, figsize=(20, 4 * rows_count))
        axes = axes.flatten()

        for i, label in enumerate(labels):
            ax = axes[i]
            subset = data[data["Label"] == label]
            series = subset[col].dropna()

            if series.empty:
                ax.set_visible(False)
                continue

            min_val, max_val = series.min(), series.max()
            day_value = subset["_day_"].iloc[0] if "_day_" in subset.columns else "N/A"

            if min_val == max_val:
                const_val = f"{int(min_val):_}"
                ax.text(0.5, 0.5, f"Constant Value: {const_val}",
                        ha="center", va="center")
                ax.set_title(f"{label} ({day_value})", fontsize=9)
                ax.axis("off")
                continue

            kde = gaussian_kde(series)
            x_grid = np.linspace(min_val, max_val, 1000)
            density = kde(x_grid)

            ax.fill_between(x_grid, density, color="blue", alpha=0.3)
            ax.set_title(f"{label} ({day_value})", fontsize=9)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")

            # Define X-axis bins
            bin_edges = np.linspace(min_val, max_val, BINS + 1)
            ax.set_xticks(bin_edges)
            ax.set_xticklabels([f"{int(edge):_}" for edge in bin_edges],
                               rotation=45, ha="right")

            # Format Y-axis ticks
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([format(y, ".3f") for y in ax.get_yticks()])

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Feature Distribution: {col} ({dataset_name})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Safe file name
        safe_col = col.replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_col}_KDE.png")
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"  Saved KDE for feature: {col}")


def main() -> None:
    """Main entry point for generating KDE plots per session and year."""
    for dataset, folder in INPUT_FOLDERS.items():
        output_img_dir = os.path.join(OUTPUT_FOLDER, dataset)
        os.makedirs(output_img_dir, exist_ok=True)

        print(f"\nProcessing dataset: {dataset}")
        data = load_session_data(folder)

        if data.empty:
            print(" Skipping empty dataset.")
            continue

        print(f"  Total samples: {len(data)}")
        plot_feature_kde(data, output_img_dir, dataset)


if __name__ == "__main__":
    main()

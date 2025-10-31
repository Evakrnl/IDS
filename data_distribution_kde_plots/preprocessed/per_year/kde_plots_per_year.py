"""
KDE Feature Distribution Plotter
--------------------------------
This script generates kernel density estimation (KDE) plots for each numeric feature
across multiple labeled datasets (e.g., CICIDS2017 and CICIDS2018).

Each dataset folder contains several CSV files with a "Label" column.
For each numeric feature, the script produces one figure containing subplots — one per label —
showing the distribution of that feature.


"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# === Configuration ===
INPUT_FOLDERS = {
    "2017": "/datasets/preprocessed/per_year/2017",
    "2018": "/datasets/preprocessed/per_year/2018",
}

OUTPUT_FOLDER = "/data_distribution_kde_plots/preprocessed/per_year"
BINS = 10  # number of intervals on the X-axis

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_and_merge_csvs(folder_path: str) -> pd.DataFrame:
    """
    Load and merge all CSV files from a given folder that contain a 'Label' column.

    Parameters
    ----------
    folder_path : str
        Directory containing CSV files.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame of all valid CSVs in the folder.
    """
    dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                if "Label" in df.columns:
                    dfs.append(df)
            except Exception as e:
                print(f" Error reading {filename}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def plot_kde_per_feature(data: pd.DataFrame, output_dir: str, dataset_name: str) -> None:
    """
    Generate KDE plots for each numeric feature grouped by label.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing numeric columns and 'Label'.
    output_dir : str
        Folder to save the generated plots.
    dataset_name : str
        Dataset identifier (e.g., '2017' or '2018').
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
            series = data.loc[data["Label"] == label, col].dropna()

            if series.empty:
                ax.set_visible(False)
                continue

            min_val, max_val = series.min(), series.max()
            if min_val == max_val:
                ax.text(0.5, 0.5, f"Constant Value: {int(min_val):_}",
                        ha="center", va="center")
                ax.set_title(label, fontsize=9)
                ax.axis("off")
                continue

            # KDE Calculation
            kde = gaussian_kde(series)
            x_grid = np.linspace(min_val, max_val, 1000)
            density = kde(x_grid)

            ax.fill_between(x_grid, density, color="blue", alpha=0.3)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")

            # X-axis bins
            bin_edges = np.linspace(min_val, max_val, BINS + 1)
            ax.set_xticks(bin_edges)
            ax.set_xticklabels([f"{int(edge):_}" for edge in bin_edges],
                               rotation=45, ha="right")

            # Format Y-axis
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([format(y, ".3f") for y in ax.get_yticks()])

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Feature Distribution: {col} ({dataset_name})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        safe_col = col.replace("/", "_")
        output_path = os.path.join(output_dir, f"{safe_col}_KDE.png")
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"  Saved KDE plot for feature: {col}")


def main() -> None:
    """Main entry point for generating KDE plots across datasets."""
    for dataset, folder in INPUT_FOLDERS.items():
        output_img_dir = os.path.join(OUTPUT_FOLDER, dataset)
        os.makedirs(output_img_dir, exist_ok=True)

        print(f"\nProcessing dataset: {dataset}")
        data = load_and_merge_csvs(folder)
        if data.empty:
            print("  No valid CSV files found.")
            continue

        print(f"  Total samples: {len(data)}")
        plot_kde_per_feature(data, output_img_dir, dataset)


if __name__ == "__main__":
    main()

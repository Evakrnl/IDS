import os
import pandas as pd
from pathlib import Path


def count_labels(folder_path: str | Path, output_file: str | Path) -> None:
    """
    Count occurrences of each label category in all CSV files inside a folder.

    Parameters
    ----------
    folder_path : str | Path
        Path to the folder containing CSV files.
    output_file : str | Path
        Path to the text file where results will be saved.
    """
    folder_path = Path(folder_path)
    output_file = Path(output_file)

    # Create the parent folder for output file if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out:
        for filename in os.listdir(folder_path):
            if not filename.endswith(".csv"):
                continue

            file_path = folder_path / filename

            try:
                # Read only the column that contains "label" (case- and space-insensitive)
                df = pd.read_csv(file_path, usecols=lambda col: "label" in col.strip().lower())

                # Clean up column names
                df.columns = [col.strip() for col in df.columns]

                # Ensure the exact column name "Label" exists
                if "Label" not in df.columns:
                    out.write(f"\nFile '{filename}' does not contain column 'Label'.\n")
                    continue

                # Count how many times each label appears
                value_counts = df["Label"].value_counts()

                # Write the results to the output file
                out.write(f"\nFile: {filename}\n")
                out.write("Label categories:\n")
                for label, count in value_counts.items():
                    out.write(f"  - {label}: {count}\n")

                print(f"Completed: {filename}")

            except Exception as e:
                out.write(f"\nError in file {filename}: {e}\n")

    print(f"\nAll results saved to: '{output_file}'")


if __name__ == "__main__":
    folder_path = "..."      # Replace with your folder path
    output_file = ".../count.txt"
    count_labels(folder_path, output_file)

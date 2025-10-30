import os
import csv
from collections import Counter
from pathlib import Path


def list_features_and_duplicates(folder_path: str | Path, output_file: str | Path) -> None:
    """
    For each CSV in `folder_path`, read the header safely and:
      - list all feature names,
      - detect and list duplicate feature names,
      - write results to `output_file`.

    Notes
    -----
    - Uses csv.reader to correctly handle commas within quoted fields.
    - Uses 'utf-8-sig' to gracefully handle BOM if present.
    """
    folder_path = Path(folder_path)
    output_file = Path(output_file)

    with output_file.open("w", encoding="utf-8") as out:
        for filename in os.listdir(folder_path):
            if not filename.endswith(".csv"):
                continue

            file_path = folder_path / filename
            try:
                # Read only the header row, robust to quoted commas
                with file_path.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    header = [h.strip() for h in header]

                duplicates = [feat for feat, count in Counter(header).items() if count > 1]

                # Write results
                out.write(f"\nFile: {filename}\n")
                out.write("Features:\n")
                for feat in header:
                    out.write(f"  - {feat}\n")

                if duplicates:
                    out.write("Duplicate features found:\n")
                    for dup in duplicates:
                        out.write(f"  - {dup}\n")
                else:
                    out.write("No duplicate features found.\n")

            except Exception as e:
                out.write(f"\nError in file {filename}: {e}\n")

    print(f"Done. Results saved to '{output_file}'")


if __name__ == "__main__":
    # Configure paths here
    folder_path = "..."
    output_file = ".../dublicate_features_check.txt"
    list_features_and_duplicates(folder_path, output_file)

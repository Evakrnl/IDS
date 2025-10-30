#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path


def sample_large_csv(
    input_file: str | Path,
    output_file: str | Path,
    target_rows: int = 5_000_000,
    chunk_size: int = 1_000_000,
    seed: int = 42,
):
    """
    Randomly select exactly `target_rows` rows from a huge CSV without loading it fully into memory.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    print(f"Counting total rows in {input_file} ...")
    # Step 1: Count all rows (excluding header)
    with input_file.open("r", encoding="utf-8", newline="") as f:
        total_rows = sum(1 for _ in f) - 1
    if total_rows <= 0:
        raise ValueError("No data rows found in file.")

    print(f"Total rows: {total_rows:,}")
    if total_rows <= target_rows:
        print("File smaller than target size, copying entire file.")
        df = pd.read_csv(input_file)
        df.to_csv(output_file, index=False)
        return

    # Step 2: Choose random indices
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(total_rows, size=target_rows, replace=False))

    print(f"Selecting {target_rows:,} random rows (seed={seed})...")

    # Step 3: Stream and save selected rows
    header_written = False
    seen_start = 0

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        m = len(chunk)
        left = np.searchsorted(picked, seen_start, side="left")
        right = np.searchsorted(picked, seen_start + m, side="left")
        local = picked[left:right] - seen_start
        if local.size > 0:
            out = chunk.iloc[local]
            out.to_csv(output_file, mode="a", index=False, header=not header_written)
            header_written = True
        seen_start += m

    print(f"Done! Saved exactly {target_rows:,} random rows to {output_file}")


if __name__ == "__main__":
    # === Edit these paths before running ===
    input_path = "/path/to/your/huge.csv"      # <-- put your CSV path here
    output_path = "/path/to/output/sample.csv" # <-- where to save the sample

    sample_large_csv(
        input_file=input_path,
        output_file=output_path,
        target_rows=5_000_000,
        chunk_size=1_000_000,
        seed=42,
    )

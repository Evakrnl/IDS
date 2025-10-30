#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rename_files_normalize.py

Recursively walks through a folder and renames all files according to these rules:
- Replace '-' and spaces with underscores ('_')
- Rename 'classificationReport' → 'classification_report'
- Rename 'confusionMatrix' → 'confusion_matrix'

Usage:
    Just set the 'root_path' variable below and run the script:
        python rename_files_normalize.py
"""

import os
import re
from pathlib import Path

# =======================================
# 🔧 CONFIGURATION — SET YOUR FOLDER PATH
# =======================================
root_path = "/home/ml1/Documents/ids/xgboost_models/binary_one_vs_all/trained_models/for_all_with_oversampling"

# =======================================
# 🚀 FUNCTION DEFINITIONS
# =======================================
def normalize_name(name: str) -> str:
    """Return normalized filename based on naming rules."""
    p = Path(name)
    stem = p.stem
    suffix = "".join(p.suffixes)

    # Specific replacements
    stem = stem.replace("classificationReport", "classification_report")
    stem = stem.replace("confusionMatrix", "confusion_matrix")

    # Replace dashes and spaces with underscores
    stem = stem.replace(" - ", "_").replace("-", "_").replace(" ", "_")

    # Collapse multiple underscores
    stem = re.sub(r"_+", "_", stem).strip("_")

    return f"{stem}{suffix}"


def rename_files_in_folder(root: str):
    """Walk through all subdirectories and rename files following normalization rules."""
    root = Path(root)
    if not root.exists():
        print(f"❌ Folder not found: {root}")
        return

    print(f"\n🔍 Searching files in: {root}")
    renamed = 0

    for dirpath, _, filenames in os.walk(root):
        dirpath = Path(dirpath)
        for fname in filenames:
            old_path = dirpath / fname
            new_name = normalize_name(fname)

            if new_name != fname:
                new_path = dirpath / new_name
                try:
                    os.rename(old_path, new_path)
                    print(f"✅ {fname} → {new_name}")
                    renamed += 1
                except Exception as e:
                    print(f"⚠️ Error renaming {fname}: {e}")

    if renamed == 0:
        print("\nℹ️ No files needed renaming.")
    else:
        print(f"\n✅ Done! {renamed} files were renamed successfully.")


# =======================================
# 🏁 MAIN EXECUTION
# =======================================
if __name__ == "__main__":
    rename_files_in_folder(root_path)

# quick_schema.py
import pandas as pd
from pathlib import Path

# === ΒΑΛΕ ΕΔΩ ΤΟ ΜΟΝΟΠΑΤΙ ΤΟΥ CSV ===
CSV_PATH = Path("/home/ml1/Documents/ids/datasets/original/CSECICIDS2018_improved_grouped/Infiltration - Communication Victim Attacker.csv")

# === ΜΗΝ ΠΕΙΡΑΖΕΙΣ ΑΠΟ ΚΑΤΩ ===
TYPE_MAP = {
    "int64": "int", "Int64": "int",
    "float64": "float", "Float64": "float",
    "boolean": "bool", "bool": "bool",
    "string": "string", "object": "string",
    "category": "string", "datetime64[ns]": "datetime",
}

def infer_type(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(200)
        if not sample.empty:
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.8:
                return "datetime"
    return TYPE_MAP.get(str(s.dtype), str(s.dtype))

def main():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, nrows=5000).convert_dtypes()
    print("\n# Schema")
    print("Column Name | Type (είδος)")
    print("--- | ---")
    for col in df.columns:
        print(f"{col} | {infer_type(df[col])}")

if __name__ == "__main__":
    main()

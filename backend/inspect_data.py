"""
Run this from your backend/ folder to inspect all datasets.
python inspect_data.py
"""
import pandas as pd
import os

DATA_DIR = "data"

files = {
    "xray_metadata":    ("xray_metadata.csv",       "csv"),
    "clinical_data":    ("clinical_data.csv",        "csv"),
    "clinical_symptoms":("clinical_symptoms.csv",    "csv"),
    "DST_SAMPLES":      ("DST_SAMPLES.parquet",      "parquet"),
    "PREDICTIONS":      ("PREDICTIONS.parquet",      "parquet"),
    "ct_data":          ("ct_data.parquet",          "parquet"),
}

for name, (filename, ftype) in files.items():
    path = os.path.join(DATA_DIR, filename)
    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print(f"{'='*60}")
    try:
        if ftype == "csv":
            df = pd.read_csv(path, nrows=2)
        else:
            df = pd.read_parquet(path)
            df = df.head(2)

        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nSample row:")
        print(df.iloc[0].to_dict())
    except Exception as e:
        print(f"ERROR reading {filename}: {e}")

print("\n\nDone!")
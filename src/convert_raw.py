from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
SOURCE = RAW_DIR / "online_retail_II.xlsx"
TARGET = RAW_DIR / "online_retail_II.csv"

sheets = pd.read_excel(SOURCE, sheet_name=None)

for name, df in sheets.items():
    print(f"{name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

combined = pd.concat(
    [df.assign(source_sheet=name) for name, df in sheets.items()],
    ignore_index=True,
)

print(f"Combined: {combined.shape[0]:,} rows")
print(combined.columns.tolist())

combined.to_csv(TARGET, index=False)
print(f"Saved to {TARGET}")
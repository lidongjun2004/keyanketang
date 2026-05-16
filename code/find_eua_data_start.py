import sys
from pathlib import Path
import pandas

DATA_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/datastreamm")


def find_time_series_start(file_path):
    print(f"\n{'=' * 80}")
    print(f"FILE: {file_path.name}")
    print(f"{'=' * 80}")
    raw = pandas.read_excel(file_path, sheet_name=0, header=None)
    print(f"  total shape: {raw.shape}")
    for index, row in raw.iterrows():
        cell = row[0]
        if isinstance(cell, str) and ("日期" in cell or "Date" in cell or "Trade Date" in cell):
            print(f"  >>> found header at row {index}: {row.tolist()}")
            print(f"  next 5 data rows:")
            print(raw.iloc[index + 1: index + 6].to_string(max_cols=10))
            return index
    print("  [no 日期/Date header found, scanning rows 20-80 for first datetime]")
    for index in range(20, min(80, len(raw))):
        cell = raw.iloc[index, 0]
        if isinstance(cell, pandas.Timestamp):
            print(f"  >>> first datetime appears at row {index}: {raw.iloc[index].tolist()}")
            print(f"  preceding 3 rows:")
            print(raw.iloc[max(0, index - 3): index].to_string(max_cols=10))
            return index


for year in [22, 23, 24, 25]:
    find_time_series_start(DATA_ROOT / f"EUA FUTURE DEC {year}价格历史_20260126_1449.xlsx")

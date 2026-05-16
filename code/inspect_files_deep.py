import sys
from pathlib import Path
import pandas

DATA_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/datastreamm")


def deep_inspect_eua(file_path):
    print(f"\n{'=' * 80}")
    print(f"FILE: {file_path.name}")
    print(f"{'=' * 80}")
    raw = pandas.read_excel(file_path, sheet_name=0, header=None)
    print(f"  total raw shape: {raw.shape}")
    print(f"  first 12 rows (all columns):")
    print(raw.head(12).to_string(max_cols=15, max_colwidth=30))
    print(f"\n  last 5 rows:")
    print(raw.tail(5).to_string(max_cols=15, max_colwidth=30))


def deep_inspect_msci(file_path):
    print(f"\n{'=' * 80}")
    print(f"FILE: {file_path.name}")
    print(f"{'=' * 80}")
    raw = pandas.read_excel(file_path, sheet_name=0, header=None)
    print(f"  total raw shape: {raw.shape}")
    print(f"  first 5 rows:")
    print(raw.head(5).to_string(max_cols=10, max_colwidth=40))
    print(f"\n  last 5 rows:")
    print(raw.tail(5).to_string(max_cols=10, max_colwidth=40))


for year in [22, 23, 24, 25]:
    deep_inspect_eua(DATA_ROOT / f"EUA FUTURE DEC {year}价格历史_20260126_1449.xlsx")

deep_inspect_msci(DATA_ROOT / "新建 Microsoft Excel 工作表 (3).xlsx")

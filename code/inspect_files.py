import sys
from pathlib import Path
import pandas

DATA_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/datastreamm")


def inspect_xlsx(file_path, max_rows=5):
    print(f"\n{'=' * 80}")
    print(f"FILE: {file_path.name}")
    print(f"{'=' * 80}")
    try:
        excel = pandas.ExcelFile(file_path)
        print(f"  sheets: {excel.sheet_names}")
        for sheet_name in excel.sheet_names:
            try:
                frame = pandas.read_excel(file_path, sheet_name=sheet_name, nrows=20)
                print(f"\n  --- sheet: {sheet_name} ---")
                print(f"  shape (first 20 rows): {frame.shape}")
                print(f"  columns: {list(frame.columns)[:15]}")
                print(f"  head:")
                print(frame.head(max_rows).to_string(max_cols=8))
            except Exception as error:
                print(f"  [error reading sheet {sheet_name}]: {error}")
    except Exception as error:
        print(f"  [error opening file]: {error}")


targets = [
    DATA_ROOT / "EUA FUTURE DEC 22价格历史_20260126_1449.xlsx",
    DATA_ROOT / "EUA FUTURE DEC 23价格历史_20260126_1449.xlsx",
    DATA_ROOT / "EUA FUTURE DEC 24价格历史_20260126_1449.xlsx",
    DATA_ROOT / "EUA FUTURE DEC 25价格历史_20260126_1449.xlsx",
    DATA_ROOT / "新建 Microsoft Excel 工作表 (3).xlsx",
]

for target in targets:
    inspect_xlsx(target)

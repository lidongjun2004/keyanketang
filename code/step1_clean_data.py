from pathlib import Path
import numpy
import pandas

DATA_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/datastreamm")
OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")
OUTPUT_ROOT.mkdir(exist_ok=True)

SAMPLE_START = "2018-01-01"
SAMPLE_END = "2025-12-15"

EUA_HEADER_ROW = {22: 31, 23: 32, 24: 32, 25: 27}
EUA_COLUMNS = [
    "date", "close", "change", "change_pct",
    "open", "low", "high", "volume", "open_interest", "bid", "ask",
]
ROLL_RULE = {
    22: ("2018-01-01", "2022-11-30"),
    23: ("2022-12-01", "2023-11-30"),
    24: ("2023-12-01", "2024-11-30"),
    25: ("2024-12-01", "2025-12-15"),
}

CONTROL_FILES = {
    "china_10y": ("Economic Indicator_China Treasury bond yield, 10-year_10 Mar 2025.xlsx", 1),
    "eu_pmi": ("Economic Indicator_Euro Zone HCOB Mfg Flash PMI_5 Mar 2025.xlsx", 7),
    "eu_ip": ("Economic Indicator_Euro Zone Industrial Production Index, Standardized, SA, Index_5 Mar 2025.xlsx", 1),
}


def load_eua_contract(year):
    file_path = DATA_ROOT / f"EUA FUTURE DEC {year}价格历史_20260126_1449.xlsx"
    raw = pandas.read_excel(
        file_path,
        sheet_name=0,
        header=None,
        skiprows=EUA_HEADER_ROW[year] + 1,
    )
    raw = raw.iloc[:, : len(EUA_COLUMNS)]
    raw.columns = EUA_COLUMNS
    raw["date"] = pandas.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).copy()
    raw["close"] = pandas.to_numeric(raw["close"], errors="coerce")
    return raw[["date", "close"]].sort_values("date").reset_index(drop=True)


def build_eua_continuous():
    contracts = {year: load_eua_contract(year) for year in [22, 23, 24, 25]}
    pieces = []
    for year, (start, end) in ROLL_RULE.items():
        slice_frame = contracts[year][
            (contracts[year]["date"] >= start) & (contracts[year]["date"] <= end)
        ].copy()
        slice_frame["contract"] = f"Dec{year}"
        pieces.append(slice_frame)
    continuous = pandas.concat(pieces, ignore_index=True).sort_values("date")
    continuous = continuous.drop_duplicates(subset=["date"], keep="last")
    continuous = continuous.rename(columns={"close": "eua_close"})
    return continuous.reset_index(drop=True)


def load_msci():
    file_path = DATA_ROOT / "新建 Microsoft Excel 工作表 (3).xlsx"
    raw = pandas.read_excel(file_path, sheet_name=0, header=None, skiprows=2)
    frame = pandas.DataFrame({
        "date": pandas.to_datetime(raw.iloc[:, 0], errors="coerce"),
        "msci_materials": pandas.to_numeric(raw.iloc[:, 1], errors="coerce"),
        "msci_energy": pandas.to_numeric(raw.iloc[:, 3], errors="coerce"),
    })
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_economic_indicator(file_name, value_column_index, value_column_name):
    import datetime
    file_path = DATA_ROOT / "datastream" / file_name
    raw = pandas.read_excel(file_path, sheet_name=0, header=None)
    data_start = None
    for index in range(10, len(raw)):
        cell = raw.iloc[index, 0]
        if isinstance(cell, (pandas.Timestamp, datetime.datetime, datetime.date)):
            data_start = index
            break
    if data_start is None:
        raise ValueError(f"no datetime found in {file_name}")
    frame = pandas.DataFrame({
        "date": pandas.to_datetime(raw.iloc[data_start:, 0], errors="coerce"),
        value_column_name: pandas.to_numeric(raw.iloc[data_start:, value_column_index], errors="coerce"),
    })
    frame = frame.dropna(subset=["date", value_column_name])
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def compute_log_return(series):
    return numpy.log(series / series.shift(1))


def main():
    print(">>> step 1: build EUA continuous main contract")
    eua = build_eua_continuous()
    print(f"    rows: {len(eua)}, range: {eua['date'].min().date()} ~ {eua['date'].max().date()}")
    print(f"    contract counts: {eua['contract'].value_counts().to_dict()}")

    print("\n>>> step 2: load MSCI WORLD ENERGY / MATERIALS")
    msci = load_msci()
    print(f"    rows: {len(msci)}, range: {msci['date'].min().date()} ~ {msci['date'].max().date()}")

    print("\n>>> step 3: load monthly control variables")
    controls = {}
    for column_name, (file_name, value_column_index) in CONTROL_FILES.items():
        frame = load_economic_indicator(file_name, value_column_index, column_name)
        print(f"    {column_name}: rows={len(frame)}, range={frame['date'].min().date()} ~ {frame['date'].max().date()}")
        controls[column_name] = frame

    print("\n>>> step 4: merge daily series (EUA + MSCI), inner join on date")
    daily = pandas.merge(eua[["date", "eua_close"]], msci, on="date", how="inner")
    daily = daily[(daily["date"] >= SAMPLE_START) & (daily["date"] <= SAMPLE_END)].sort_values("date").reset_index(drop=True)

    print("\n>>> step 5: forward-fill monthly controls to daily (max 31 days)")
    for column_name, frame in controls.items():
        daily = pandas.merge_asof(
            daily.sort_values("date"),
            frame.sort_values("date"),
            on="date",
            direction="backward",
            tolerance=pandas.Timedelta(days=45),
        )

    print("\n>>> step 6: compute log returns and first differences")
    daily["eua_return"] = compute_log_return(daily["eua_close"])
    daily["msci_energy_return"] = compute_log_return(daily["msci_energy"])
    daily["msci_materials_return"] = compute_log_return(daily["msci_materials"])
    daily["china_10y_diff"] = daily["china_10y"].diff()
    daily["eu_pmi_diff"] = daily["eu_pmi"].diff()
    daily["eu_ip_diff"] = daily["eu_ip"].diff()

    daily = daily.dropna(subset=["eua_return", "msci_energy_return", "msci_materials_return"]).reset_index(drop=True)

    print(f"\n>>> final master_data shape: {daily.shape}")
    print(f"    sample period: {daily['date'].min().date()} ~ {daily['date'].max().date()}")
    print(f"    columns: {list(daily.columns)}")
    print(f"\n    missing per column:")
    print(daily.isna().sum().to_string())
    print(f"\n    head:")
    print(daily.head(3).to_string())
    print(f"\n    tail:")
    print(daily.tail(3).to_string())

    output_path = OUTPUT_ROOT / "master_data.csv"
    daily.to_csv(output_path, index=False)
    print(f"\n>>> saved to {output_path}")


if __name__ == "__main__":
    main()

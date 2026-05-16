from pathlib import Path
import numpy
import pandas
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings("ignore")

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")


def granger_full_grid(data, cause, effect, max_lag=10):
    subset = data[[effect, cause]].dropna()
    result = grangercausalitytests(subset, maxlag=max_lag, verbose=False)
    rows = []
    for lag in range(1, max_lag + 1):
        f_test = result[lag][0]["ssr_ftest"]
        chi2_test = result[lag][0]["ssr_chi2test"]
        rows.append({
            "lag": lag,
            "F_stat": f_test[0],
            "F_pvalue": f_test[1],
            "chi2_stat": chi2_test[0],
            "chi2_pvalue": chi2_test[1],
        })
    return pandas.DataFrame(rows)


def split_subperiods(data):
    sub_periods = {
        "全样本 2018-2025": (data["date"].min(), data["date"].max()),
        "样本前期 2018-2020": (pandas.Timestamp("2018-01-01"), pandas.Timestamp("2020-12-31")),
        "样本中期 2021-2022": (pandas.Timestamp("2021-01-01"), pandas.Timestamp("2022-12-31")),
        "样本后期 2023-2025": (pandas.Timestamp("2023-01-01"), pandas.Timestamp("2025-12-15")),
    }
    return sub_periods


def main():
    data = pandas.read_csv(OUTPUT_ROOT / "master_data.csv", parse_dates=["date"])
    print(f">>> loaded master_data: {data.shape}")

    print("\n" + "=" * 80)
    print("ROBUSTNESS CHECK 1: Granger causality with extended lag (1-10)")
    print("=" * 80)
    for cause, effect in [
        ("eua_return", "msci_energy_return"),
        ("msci_energy_return", "eua_return"),
        ("eua_return", "msci_materials_return"),
        ("msci_materials_return", "eua_return"),
    ]:
        print(f"\n--- {cause} -> {effect} ---")
        result = granger_full_grid(data, cause, effect, max_lag=10)
        print(result.to_string(float_format=lambda value: f"{value:.4f}"))

    print("\n" + "=" * 80)
    print("ROBUSTNESS CHECK 2: subperiod stability of '股→碳' direction")
    print("=" * 80)
    subperiods = split_subperiods(data)
    summary_rows = []
    for period_name, (start, end) in subperiods.items():
        slice_data = data[(data["date"] >= start) & (data["date"] <= end)]
        print(f"\n--- {period_name} ({slice_data['date'].min().date()} ~ {slice_data['date'].max().date()}, n={len(slice_data)}) ---")
        for cause, effect in [
            ("msci_energy_return", "eua_return"),
            ("msci_materials_return", "eua_return"),
            ("eua_return", "msci_energy_return"),
            ("eua_return", "msci_materials_return"),
        ]:
            subset = slice_data[[effect, cause]].dropna()
            if len(subset) < 30:
                continue
            try:
                result = grangercausalitytests(subset, maxlag=5, verbose=False)
                min_pvalue = min(result[lag][0]["ssr_ftest"][1] for lag in range(1, 6))
                best_lag = min(range(1, 6), key=lambda lag: result[lag][0]["ssr_ftest"][1])
                significant = "✓" if min_pvalue < 0.05 else "✗"
                summary_rows.append({
                    "period": period_name,
                    "cause": cause,
                    "effect": effect,
                    "best_lag": best_lag,
                    "min_p_value": min_pvalue,
                    "significant_5pct": significant,
                })
                print(f"    {cause:25s} -> {effect:25s} | best_lag={best_lag}, min_p={min_pvalue:.4f} {significant}")
            except Exception as error:
                print(f"    {cause} -> {effect}: error {error}")

    summary_frame = pandas.DataFrame(summary_rows)
    summary_frame.to_csv(OUTPUT_ROOT / "granger_robustness_subperiods.csv", index=False)
    print(f"\n>>> saved granger_robustness_subperiods.csv")

    print("\n" + "=" * 80)
    print("ROBUSTNESS CHECK 3: VAR model order selection + Granger via VAR")
    print("=" * 80)
    var_data = data[["eua_return", "msci_energy_return", "msci_materials_return"]].dropna()
    var_model = VAR(var_data)
    print("    VAR lag order selection:")
    print(var_model.select_order(maxlags=10).summary())
    fitted = var_model.fit(maxlags=10, ic="aic")
    print(f"\n    selected lag order (AIC): {fitted.k_ar}")

    for cause in ["msci_energy_return", "msci_materials_return"]:
        granger_result = fitted.test_causality("eua_return", causing=cause, kind="f")
        print(f"\n    [VAR-Granger] {cause} -> eua_return: F={granger_result.test_statistic:.4f}, p={granger_result.pvalue:.4f}, conclusion: {granger_result.conclusion}")

    for cause in ["eua_return"]:
        for effect in ["msci_energy_return", "msci_materials_return"]:
            granger_result = fitted.test_causality(effect, causing=cause, kind="f")
            print(f"    [VAR-Granger] {cause} -> {effect}: F={granger_result.test_statistic:.4f}, p={granger_result.pvalue:.4f}, conclusion: {granger_result.conclusion}")

    print("\n" + "=" * 80)
    print("ROBUSTNESS CHECK 4: directional check on |returns| (volatility spillover)")
    print("=" * 80)
    abs_data = pandas.DataFrame({
        "date": data["date"],
        "eua_abs": data["eua_return"].abs(),
        "energy_abs": data["msci_energy_return"].abs(),
        "materials_abs": data["msci_materials_return"].abs(),
    })
    for cause, effect in [
        ("energy_abs", "eua_abs"),
        ("eua_abs", "energy_abs"),
        ("materials_abs", "eua_abs"),
        ("eua_abs", "materials_abs"),
    ]:
        subset = abs_data[[effect, cause]].dropna()
        result = grangercausalitytests(subset, maxlag=5, verbose=False)
        min_pvalue = min(result[lag][0]["ssr_ftest"][1] for lag in range(1, 6))
        best_lag = min(range(1, 6), key=lambda lag: result[lag][0]["ssr_ftest"][1])
        significant = "✓" if min_pvalue < 0.05 else "✗"
        print(f"    {cause:18s} -> {effect:18s} | best_lag={best_lag}, min_p={min_pvalue:.4f} {significant}")


if __name__ == "__main__":
    main()

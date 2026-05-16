from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy
import pandas
from scipy import stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Heiti TC", "STHeiti", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")
FIGS_ROOT = OUTPUT_ROOT / "figs"
FIGS_ROOT.mkdir(exist_ok=True)

RETURN_COLUMNS = ["eua_return", "msci_energy_return", "msci_materials_return"]
RETURN_LABELS = {
    "eua_return": "EUA 碳期货",
    "msci_energy_return": "MSCI 能源",
    "msci_materials_return": "MSCI 材料",
}


def descriptive_statistics(data):
    rows = []
    for column in RETURN_COLUMNS:
        series = data[column].dropna()
        jb_statistic, jb_pvalue = stats.jarque_bera(series)
        adf_result = adfuller(series, autolag="AIC")
        rows.append({
            "variable": RETURN_LABELS[column],
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "skewness": stats.skew(series),
            "kurtosis": stats.kurtosis(series),
            "JB_stat": jb_statistic,
            "JB_pvalue": jb_pvalue,
            "ADF_stat": adf_result[0],
            "ADF_pvalue": adf_result[1],
        })
    return pandas.DataFrame(rows)


def rolling_correlation(data, window=60):
    correlations = pandas.DataFrame({"date": data["date"]})
    correlations["corr_eua_energy"] = data["eua_return"].rolling(window).corr(data["msci_energy_return"])
    correlations["corr_eua_materials"] = data["eua_return"].rolling(window).corr(data["msci_materials_return"])
    correlations["corr_energy_materials"] = data["msci_energy_return"].rolling(window).corr(data["msci_materials_return"])
    return correlations


def granger_causality(data, max_lag=5):
    rows = []
    pairs = [
        ("eua_return", "msci_energy_return"),
        ("msci_energy_return", "eua_return"),
        ("eua_return", "msci_materials_return"),
        ("msci_materials_return", "eua_return"),
    ]
    for cause, effect in pairs:
        subset = data[[effect, cause]].dropna()
        result = grangercausalitytests(subset, maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            f_test = result[lag][0]["ssr_ftest"]
            rows.append({
                "cause": RETURN_LABELS[cause],
                "effect": RETURN_LABELS[effect],
                "lag": lag,
                "F_stat": f_test[0],
                "p_value": f_test[1],
                "significant_5pct": f_test[1] < 0.05,
            })
    return pandas.DataFrame(rows)


def plot_rolling_correlation(correlations):
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(correlations["date"], correlations["corr_eua_energy"], label="EUA - MSCI 能源", color="#d62728", linewidth=1.0)
    axis.plot(correlations["date"], correlations["corr_eua_materials"], label="EUA - MSCI 材料", color="#2ca02c", linewidth=1.0)
    axis.plot(correlations["date"], correlations["corr_energy_materials"], label="MSCI 能源 - 材料", color="#1f77b4", linewidth=0.8, alpha=0.6)
    axis.axhline(0, color="black", linestyle="--", alpha=0.4)
    axis.set_title("60 日滚动相关系数 (碳—股—行业内)", fontsize=13)
    axis.set_xlabel("日期")
    axis.set_ylabel("相关系数")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left", fontsize=10)
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_5_1_rolling_correlation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


def plot_return_distribution(data):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, column in zip(axes, RETURN_COLUMNS):
        series = data[column].dropna()
        axis.hist(series, bins=80, density=True, alpha=0.6, color="#1f77b4", label="实际分布")
        x_range = numpy.linspace(series.min(), series.max(), 200)
        normal_pdf = stats.norm.pdf(x_range, series.mean(), series.std())
        axis.plot(x_range, normal_pdf, color="#d62728", linewidth=1.5, label="正态分布")
        axis.set_title(f"{RETURN_LABELS[column]} 收益率分布", fontsize=11)
        axis.set_xlabel("日度对数收益率")
        axis.set_ylabel("密度")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_4_2_return_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


def main():
    data = pandas.read_csv(OUTPUT_ROOT / "master_data.csv", parse_dates=["date"])
    print(f">>> loaded master_data: {data.shape}")

    print("\n>>> step 2.1: descriptive statistics + ADF + JB")
    desc = descriptive_statistics(data)
    print(desc.to_string(float_format=lambda value: f"{value:.4f}"))
    desc.to_csv(OUTPUT_ROOT / "desc_stats.csv", index=False)

    print("\n>>> step 2.2: 60-day rolling correlation")
    correlations = rolling_correlation(data, window=60)
    print(f"    mean correlation:")
    for col in ["corr_eua_energy", "corr_eua_materials", "corr_energy_materials"]:
        mean_value = correlations[col].mean()
        print(f"      {col}: mean={mean_value:.4f}, std={correlations[col].std():.4f}")
    correlations.to_csv(OUTPUT_ROOT / "rolling_correlation.csv", index=False)

    print("\n>>> step 2.3: Granger causality test (max_lag=5)")
    granger = granger_causality(data, max_lag=5)
    print(granger.to_string(float_format=lambda value: f"{value:.4f}"))
    granger.to_csv(OUTPUT_ROOT / "granger_causality.csv", index=False)

    print("\n>>> step 2.4: figures")
    plot_rolling_correlation(correlations)
    plot_return_distribution(data)

    print("\n>>> Step 2 done.")


if __name__ == "__main__":
    main()

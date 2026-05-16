from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy
import pandas

plt.rcParams["font.sans-serif"] = ["Heiti TC", "STHeiti", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")
FIGS_ROOT = OUTPUT_ROOT / "figs"
FIGS_ROOT.mkdir(exist_ok=True)

EVENTS = [
    ("2020-03-15", "COVID-19 全球爆发"),
    ("2021-09-15", "欧洲能源危机"),
    ("2022-02-24", "俄乌冲突"),
    ("2022-08-15", "EUA 创历史新高"),
    ("2023-10-07", "巴以冲突"),
]


def main():
    data = pandas.read_csv(OUTPUT_ROOT / "master_data.csv", parse_dates=["date"])
    print(f">>> loaded master_data: {data.shape}")

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axis_top = axes[0]
    axis_top_right = axis_top.twinx()
    line1 = axis_top.plot(data["date"], data["eua_close"], color="#d62728", linewidth=1.2, label="EUA Future (€/tCO2)")
    line2 = axis_top_right.plot(data["date"], data["msci_energy"], color="#1f77b4", linewidth=1.0, label="MSCI World Energy")
    line3 = axis_top_right.plot(data["date"], data["msci_materials"], color="#2ca02c", linewidth=1.0, alpha=0.7, label="MSCI World Materials")
    axis_top.set_ylabel("EUA Price (€/tCO2)", color="#d62728")
    axis_top_right.set_ylabel("MSCI Index Level", color="#1f77b4")
    axis_top.tick_params(axis="y", labelcolor="#d62728")
    axis_top_right.tick_params(axis="y", labelcolor="#1f77b4")
    axis_top.set_title("EUA 碳期货 vs MSCI 高碳行业股票指数 (2018-2025)", fontsize=13)
    axis_top.grid(alpha=0.3)

    for event_date, label in EVENTS:
        event_timestamp = pandas.Timestamp(event_date)
        if data["date"].min() <= event_timestamp <= data["date"].max():
            axis_top.axvline(event_timestamp, color="gray", linestyle="--", alpha=0.5)
            axis_top.text(event_timestamp, axis_top.get_ylim()[1] * 0.95, label,
                          rotation=90, fontsize=8, va="top", ha="right", alpha=0.7)

    lines = line1 + line2 + line3
    axis_top.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=9)

    axis_bottom = axes[1]
    axis_bottom.plot(data["date"], data["eua_return"], color="#d62728", linewidth=0.5, alpha=0.7, label="EUA log return")
    axis_bottom.plot(data["date"], data["msci_energy_return"], color="#1f77b4", linewidth=0.5, alpha=0.6, label="MSCI Energy log return")
    axis_bottom.set_ylabel("Daily Log Return")
    axis_bottom.set_xlabel("Date")
    axis_bottom.set_title("日度对数收益率走势 (波动率聚集可见)", fontsize=12)
    axis_bottom.grid(alpha=0.3)
    axis_bottom.legend(loc="upper left", fontsize=9)
    axis_bottom.xaxis.set_major_locator(mdates.YearLocator())
    axis_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_4_1_price_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f">>> saved {output_path}")
    plt.close()


if __name__ == "__main__":
    main()

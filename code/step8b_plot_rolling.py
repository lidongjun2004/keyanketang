from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import pandas

plt.rcParams["font.sans-serif"] = ["Heiti TC", "STHeiti", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")
FIGS_ROOT = OUTPUT_ROOT / "figs"

TARGET_LABEL = {
    "eua_return": "EUA 碳期货",
    "msci_energy_return": "MSCI 能源股",
    "msci_materials_return": "MSCI 材料股",
}


def main():
    summary = pandas.read_csv(OUTPUT_ROOT / "rolling_forecast_summary.csv")
    figure, axis = plt.subplots(figsize=(13, 5))
    bar_data = []
    labels = []
    colors = []
    for target in ["eua_return", "msci_energy_return", "msci_materials_return"]:
        for tau in [0.05, 0.10, 0.50, 0.90, 0.95]:
            row = summary[(summary["target"] == target) & (summary["tau"] == tau)]
            if len(row) == 0:
                continue
            improvement = row["ERNN_relative_improvement"].iloc[0]
            bar_data.append(improvement)
            labels.append(f"{TARGET_LABEL[target][:6]}\nτ={tau}")
            colors.append("#2ca02c" if improvement > 0 else "#d62728")

    x_positions = numpy.arange(len(bar_data))
    bars = axis.bar(x_positions, bar_data, color=colors, edgecolor="black", linewidth=0.5)
    axis.axhline(0, color="black", linewidth=1.0)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("ERNN 相对 L-ER 的改进 (%, 越大越好)")
    axis.set_title("图 6.3: 滚动窗口预测中 ERNN 相对 L-ER 的改进幅度\n(window=1000 days, refit every 30 days; 绿色=ERNN 胜，红色=ERNN 负)", fontsize=11)
    axis.grid(alpha=0.3, axis="y")
    for bar, value in zip(bars, bar_data):
        axis.text(bar.get_x() + bar.get_width() / 2, value + (0.3 if value > 0 else -0.6),
                  f"{value:+.1f}%", ha="center", fontsize=7)
    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_6_3_rolling_ernn_vs_ler.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


if __name__ == "__main__":
    main()

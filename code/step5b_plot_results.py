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

MODEL_COLORS = {
    "L-ER": "#7f7f7f",
    "QRNN": "#1f77b4",
    "QRF": "#2ca02c",
    "Q-GBM": "#ff7f0e",
    "ERNN": "#d62728",
}

DIRECTION_LABEL = {
    "eua_return": "目标: EUA 碳期货收益率",
    "msci_energy_return": "目标: MSCI 能源股收益率",
    "msci_materials_return": "目标: MSCI 材料股收益率",
}


def plot_pinball_loss_comparison():
    results = pandas.read_csv(OUTPUT_ROOT / "main_results.csv")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for axis, target in zip(axes, ["eua_return", "msci_energy_return", "msci_materials_return"]):
        subset = results[results["target"] == target]
        for model_name in ["L-ER", "QRNN", "QRF", "Q-GBM", "ERNN"]:
            model_subset = subset[subset["model"] == model_name].sort_values("tau")
            line_width = 2.5 if model_name == "ERNN" else 1.2
            marker_style = "o" if model_name == "ERNN" else "."
            axis.plot(model_subset["tau"], model_subset["pinball_loss"] * 1000,
                      label=model_name, color=MODEL_COLORS[model_name],
                      linewidth=line_width, marker=marker_style, markersize=7 if model_name == "ERNN" else 5)
        axis.set_title(DIRECTION_LABEL[target], fontsize=11)
        axis.set_xlabel(r"分位水平 $\tau$")
        axis.set_ylabel("Pinball Loss (×1000)")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper center", fontsize=8, ncol=5)
    plt.suptitle("图 5.2: 五种模型在不同 τ 下的预测损失对比 (test set)", fontsize=12, y=1.02)
    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_5_2_pinball_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


def plot_quantile_r_squared():
    results = pandas.read_csv(OUTPUT_ROOT / "main_results.csv")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for axis, target in zip(axes, ["eua_return", "msci_energy_return", "msci_materials_return"]):
        subset = results[results["target"] == target]
        x_labels = sorted(subset["tau"].unique())
        x_positions = numpy.arange(len(x_labels))
        bar_width = 0.16
        for offset_index, model_name in enumerate(["L-ER", "QRNN", "QRF", "Q-GBM", "ERNN"]):
            model_subset = subset[subset["model"] == model_name].sort_values("tau")
            offset_pos = x_positions + (offset_index - 2) * bar_width
            axis.bar(offset_pos, model_subset["quantile_r_squared"].values,
                     width=bar_width, label=model_name, color=MODEL_COLORS[model_name],
                     edgecolor="black", linewidth=0.4 if model_name != "ERNN" else 1.2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(DIRECTION_LABEL[target], fontsize=11)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"τ={tau}" for tau in x_labels])
        axis.set_ylabel(r"Quantile $R^2$")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(loc="upper right", fontsize=8, ncol=5)
    plt.suptitle("图 5.3: Quantile R² 对比 (大于 0 = 优于无条件分位数 baseline)", fontsize=12, y=1.02)
    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_5_3_quantile_r2_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


def plot_expectile_r_squared():
    results = pandas.read_csv(OUTPUT_ROOT / "main_results.csv")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for axis, target in zip(axes, ["eua_return", "msci_energy_return", "msci_materials_return"]):
        subset = results[results["target"] == target]
        x_labels = sorted(subset["tau"].unique())
        x_positions = numpy.arange(len(x_labels))
        bar_width = 0.16
        for offset_index, model_name in enumerate(["L-ER", "QRNN", "QRF", "Q-GBM", "ERNN"]):
            model_subset = subset[subset["model"] == model_name].sort_values("tau")
            offset_pos = x_positions + (offset_index - 2) * bar_width
            axis.bar(offset_pos, model_subset["expectile_r_squared"].values,
                     width=bar_width, label=model_name, color=MODEL_COLORS[model_name],
                     edgecolor="black", linewidth=0.4 if model_name != "ERNN" else 1.2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(DIRECTION_LABEL[target], fontsize=11)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"τ={tau}" for tau in x_labels], fontsize=8)
        axis.set_ylabel(r"Expectile $R^2$")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(loc="upper right", fontsize=8, ncol=5)
    plt.suptitle("图 5.6: Expectile R² 对比（ERNN 原生平方损失口径，大于 0 = 优于无条件 expectile baseline）", fontsize=12, y=1.02)
    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_5_6_expectile_r2_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


def plot_dm_test_heatmap(loss_metric, output_filename, fig_title):
    dm_data = pandas.read_csv(OUTPUT_ROOT / "dm_test.csv")
    dm_data = dm_data[dm_data["loss_metric"] == loss_metric]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    for axis, target in zip(axes, ["eua_return", "msci_energy_return", "msci_materials_return"]):
        subset = dm_data[dm_data["target"] == target]
        pivot_dm = subset.pivot(index="benchmark", columns="tau", values="DM_stat")
        pivot_p = subset.pivot(index="benchmark", columns="tau", values="p_value")
        pivot_dm = pivot_dm.reindex(["L-ER", "QRNN", "QRF", "Q-GBM"])
        pivot_p = pivot_p.reindex(["L-ER", "QRNN", "QRF", "Q-GBM"])
        image = axis.imshow(pivot_dm.values, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
        axis.set_xticks(range(len(pivot_dm.columns)))
        axis.set_xticklabels([f"τ={tau}" for tau in pivot_dm.columns], fontsize=8)
        axis.set_yticks(range(len(pivot_dm.index)))
        axis.set_yticklabels(pivot_dm.index)
        axis.set_title(DIRECTION_LABEL[target], fontsize=10)
        for i_index, benchmark in enumerate(pivot_dm.index):
            for j_index, tau in enumerate(pivot_dm.columns):
                value = pivot_dm.iloc[i_index, j_index]
                p_value = pivot_p.iloc[i_index, j_index]
                marker = "*" if p_value < 0.05 else ""
                axis.text(j_index, i_index, f"{value:.2f}{marker}", ha="center", va="center", fontsize=8,
                          color="white" if abs(value) > 2.5 else "black")
        plt.colorbar(image, ax=axis, fraction=0.04)
    plt.suptitle(fig_title, fontsize=12, y=1.04)
    plt.tight_layout()
    output_path = FIGS_ROOT / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


def main():
    plot_pinball_loss_comparison()
    plot_quantile_r_squared()
    plot_expectile_r_squared()
    plot_dm_test_heatmap(
        "pinball", "fig_5_4_dm_test_heatmap.png",
        "图 5.4: DM 检验统计量（Pinball 损失口径，正值 = ERNN 优于 baseline；* 代表 p<0.05）",
    )
    plot_dm_test_heatmap(
        "expectile", "fig_5_7_dm_test_expectile_heatmap.png",
        "图 5.7: DM 检验统计量（Expectile 平方损失口径，正值 = ERNN 优于 baseline；* 代表 p<0.05）",
    )


if __name__ == "__main__":
    main()

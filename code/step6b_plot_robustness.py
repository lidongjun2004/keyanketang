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


def plot_hidden_dim_heatmap():
    data = pandas.read_csv(OUTPUT_ROOT / "robustness_b_hidden_dim.csv")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, target in zip(axes, ["eua_return", "msci_energy_return", "msci_materials_return"]):
        subset = data[data["target"] == target]
        pivot = subset.pivot(index="hidden_dim", columns="tau", values="quantile_r_squared")
        image = axis.imshow(pivot.values, cmap="RdYlGn", vmin=-0.05, vmax=0.25, aspect="auto")
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels([f"τ={tau}" for tau in pivot.columns])
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels([f"J={j}" for j in pivot.index])
        axis.set_title(TARGET_LABEL[target], fontsize=11)
        for i_index, hidden_dim in enumerate(pivot.index):
            for j_index, tau in enumerate(pivot.columns):
                value = pivot.iloc[i_index, j_index]
                axis.text(j_index, i_index, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(image, ax=axis, fraction=0.04)
    plt.suptitle("图 6.1: ERNN 不同隐层节点数 J 的 Quantile R² 对比 (稳健性 B)", fontsize=12, y=1.02)
    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_6_1_hidden_dim_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


def plot_robustness_comparison():
    main_data = pandas.read_csv(OUTPUT_ROOT / "main_results.csv")
    main_subset = main_data[main_data["model"] == "ERNN"][["target", "tau", "pinball_loss", "quantile_r_squared"]].copy()
    main_subset["scenario"] = "主回归"

    a_data = pandas.read_csv(OUTPUT_ROOT / "robustness_a_exclude_covid.csv")
    a_subset = a_data[a_data["model"] == "ERNN"][["target", "tau", "pinball_loss", "quantile_r_squared"]].copy()
    a_subset["scenario"] = "稳健性 A: 剔除 COVID"

    c_data = pandas.read_csv(OUTPUT_ROOT / "robustness_c_add_controls.csv")
    c_subset = c_data[c_data["model"] == "ERNN"][["target", "tau", "pinball_loss", "quantile_r_squared"]].copy()
    c_subset["scenario"] = "稳健性 C: 加宏观控制变量"

    combined = pandas.concat([main_subset, a_subset, c_subset], ignore_index=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    bar_width = 0.27
    colors = {"主回归": "#d62728", "稳健性 A: 剔除 COVID": "#ff9896", "稳健性 C: 加宏观控制变量": "#1f77b4"}
    for axis, target in zip(axes, ["eua_return", "msci_energy_return", "msci_materials_return"]):
        subset = combined[combined["target"] == target]
        x_labels = sorted(subset["tau"].unique())
        x_positions = numpy.arange(len(x_labels))
        for offset_index, scenario_name in enumerate(["主回归", "稳健性 A: 剔除 COVID", "稳健性 C: 加宏观控制变量"]):
            scenario_subset = subset[subset["scenario"] == scenario_name].sort_values("tau")
            offset_pos = x_positions + (offset_index - 1) * bar_width
            axis.bar(offset_pos, scenario_subset["quantile_r_squared"].values,
                     width=bar_width, label=scenario_name, color=colors[scenario_name],
                     edgecolor="black", linewidth=0.4)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(TARGET_LABEL[target], fontsize=11)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([f"τ={tau}" for tau in x_labels])
        axis.set_ylabel(r"ERNN Quantile $R^2$")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(loc="upper right", fontsize=8)
    plt.suptitle("图 6.2: ERNN Quantile R² 在三种稳健性场景下的对比", fontsize=12, y=1.02)
    plt.tight_layout()
    output_path = FIGS_ROOT / "fig_6_2_robustness_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path}")


if __name__ == "__main__":
    plot_hidden_dim_heatmap()
    plot_robustness_comparison()

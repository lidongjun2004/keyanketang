from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy
import pandas
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step3_ernn_model import fit_ernn, predict_ernn
from step5_main_experiment import build_lagged_features, standardize

plt.rcParams["font.sans-serif"] = ["Heiti TC", "STHeiti", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")
FIGS_ROOT = OUTPUT_ROOT / "figs"

LAG_LIST = [1, 2, 3, 5]
EXPECTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TRAIN_RATIO = 0.8
RETURN_COLUMNS = ["eua_return", "msci_energy_return", "msci_materials_return"]

TARGET_LABEL = {
    "eua_return": "EUA 碳期货",
    "msci_energy_return": "MSCI 能源股",
    "msci_materials_return": "MSCI 材料股",
}


def plot_quantile_coverage_one_target(target_column, dates_test, predictions_by_tau, targets_test, output_filename):
    figure, axis = plt.subplots(figsize=(15, 6))
    axis.plot(dates_test, targets_test, color="black", linewidth=0.6, alpha=0.7, label="实际收益率")
    axis.fill_between(dates_test, predictions_by_tau[0.05], predictions_by_tau[0.95],
                      color="#1f77b4", alpha=0.12, label="ERNN 预测 90% 区间 (τ=0.05~0.95)")
    axis.fill_between(dates_test, predictions_by_tau[0.10], predictions_by_tau[0.90],
                      color="#1f77b4", alpha=0.20, label="ERNN 预测 80% 区间 (τ=0.10~0.90)")
    axis.fill_between(dates_test, predictions_by_tau[0.25], predictions_by_tau[0.75],
                      color="#1f77b4", alpha=0.30, label="ERNN 预测 50% 区间 (τ=0.25~0.75)")
    axis.plot(dates_test, predictions_by_tau[0.50], color="#d62728", linewidth=1.0, label="ERNN 预测中位数")

    coverage_outside_5_95 = ((targets_test < predictions_by_tau[0.05]) | (targets_test > predictions_by_tau[0.95])).mean()
    coverage_outside_10_90 = ((targets_test < predictions_by_tau[0.10]) | (targets_test > predictions_by_tau[0.90])).mean()
    coverage_outside_25_75 = ((targets_test < predictions_by_tau[0.25]) | (targets_test > predictions_by_tau[0.75])).mean()
    axis.set_title(f"{TARGET_LABEL[target_column]} ERNN 分位预测覆盖 (test set)\n"
                   f"实际越界比例: 90% 区间={coverage_outside_5_95 * 100:.1f}% (理论 10%) | "
                   f"80% 区间={coverage_outside_10_90 * 100:.1f}% (理论 20%) | "
                   f"50% 区间={coverage_outside_25_75 * 100:.1f}% (理论 50%)",
                   fontsize=11)
    axis.set_xlabel("日期")
    axis.set_ylabel("日度对数收益率")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left", fontsize=9)
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    output_path = FIGS_ROOT / output_filename
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f">>> saved {output_path} | coverage 90={coverage_outside_5_95 * 100:.1f}% 80={coverage_outside_10_90 * 100:.1f}% 50={coverage_outside_25_75 * 100:.1f}%")
    return coverage_outside_5_95, coverage_outside_10_90, coverage_outside_25_75


def main():
    data = pandas.read_csv(OUTPUT_ROOT / "master_data.csv", parse_dates=["date"])
    print(f">>> loaded master_data: {data.shape}")

    coverage_records = []
    for target_column in RETURN_COLUMNS:
        print(f"\n>>> training ERNN at 7 taus for target={target_column}")
        combined = build_lagged_features(data, target_column, LAG_LIST)
        date_series = data["date"].iloc[len(data) - len(combined):].reset_index(drop=True)
        feature_columns = [col for col in combined.columns if col != "target"]
        features_array = combined[feature_columns].to_numpy(dtype=numpy.float32)
        targets_array = combined["target"].to_numpy(dtype=numpy.float32)

        split_index = int(len(combined) * TRAIN_RATIO)
        features_train_raw, features_test_raw = features_array[:split_index], features_array[split_index:]
        targets_train, targets_test = targets_array[:split_index], targets_array[split_index:]
        dates_test = date_series.iloc[split_index:].reset_index(drop=True)
        features_train, features_test = standardize(features_train_raw, features_test_raw)

        predictions_by_tau = {}
        for expectile_level in EXPECTILE_LEVELS:
            model, _ = fit_ernn(features_train, targets_train, expectile_level, hidden_dimension=5, regularization=1e-3)
            predictions_by_tau[expectile_level] = predict_ernn(model, features_test)

        sort_idx = numpy.argsort([predictions_by_tau[tau].mean() for tau in EXPECTILE_LEVELS])
        sorted_taus = [EXPECTILE_LEVELS[i] for i in sort_idx]
        if sorted_taus != EXPECTILE_LEVELS:
            print(f"    [WARN] monotonicity violated, sorted taus: {sorted_taus}")

        cov_90, cov_80, cov_50 = plot_quantile_coverage_one_target(
            target_column, dates_test, predictions_by_tau, targets_test,
            output_filename=f"fig_5_5_coverage_{target_column}.png",
        )
        coverage_records.append({
            "target": target_column,
            "coverage_outside_90pct_actual": cov_90,
            "coverage_outside_90pct_theory": 0.10,
            "coverage_outside_80pct_actual": cov_80,
            "coverage_outside_80pct_theory": 0.20,
            "coverage_outside_50pct_actual": cov_50,
            "coverage_outside_50pct_theory": 0.50,
        })

    coverage_frame = pandas.DataFrame(coverage_records)
    coverage_frame.to_csv(OUTPUT_ROOT / "coverage_test.csv", index=False)
    print(f"\n>>> saved coverage_test.csv")
    print(coverage_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

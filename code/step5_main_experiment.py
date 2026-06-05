from pathlib import Path
import time
import numpy
import pandas
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step3_ernn_model import (
    fit_ernn, predict_ernn, expectile_loss_numpy, quantile_loss, unconditional_expectile,
)
from step4_baselines import (
    fit_linear_expectile_regression, predict_linear_expectile_regression,
    fit_qrnn, predict_qrnn,
    fit_qrf, predict_qrf,
    fit_qgbm, predict_qgbm,
)

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")

LAG_LIST = [1, 2, 3, 5]
EXPECTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
TRAIN_RATIO = 0.8
RETURN_COLUMNS = ["eua_return", "msci_energy_return", "msci_materials_return"]


def build_lagged_features(data, target_column, lag_list):
    feature_frames = []
    for column in RETURN_COLUMNS:
        for lag in lag_list:
            feature_frames.append(data[column].shift(lag).rename(f"{column}_lag{lag}"))
    features = pandas.concat(feature_frames, axis=1)
    target = data[target_column].rename("target")
    combined = pandas.concat([features, target], axis=1).dropna()
    return combined


def standardize(features_train, features_test):
    mean = features_train.mean(axis=0)
    std = features_train.std(axis=0) + 1e-8
    return (features_train - mean) / std, (features_test - mean) / std


def diebold_mariano_test(loss_a, loss_b, max_lag=5):
    diff = loss_a - loss_b
    n = len(diff)
    diff_mean = diff.mean()
    diff_var = diff.var(ddof=1)
    if diff_var <= 0:
        return float("nan"), float("nan")
    autocov = numpy.array([numpy.cov(diff[lag:], diff[:n - lag], ddof=1)[0, 1] for lag in range(1, max_lag + 1)])
    long_run_var = diff_var + 2 * sum(autocov)
    if long_run_var <= 0:
        long_run_var = diff_var
    dm_statistic = diff_mean / numpy.sqrt(long_run_var / n)
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))
    return dm_statistic, p_value


def run_one_model(model_name, features_train, targets_train, features_test, targets_test, expectile_level):
    if model_name == "ERNN":
        model, _ = fit_ernn(features_train, targets_train, expectile_level, hidden_dimension=5, regularization=1e-3)
        predictions = predict_ernn(model, features_test)
    elif model_name == "L-ER":
        weights = fit_linear_expectile_regression(features_train, targets_train, expectile_level)
        predictions = predict_linear_expectile_regression(weights, features_test)
    elif model_name == "QRNN":
        model = fit_qrnn(features_train, targets_train, expectile_level, hidden_dimension=5, regularization=1e-3)
        predictions = predict_qrnn(model, features_test)
    elif model_name == "QRF":
        model = fit_qrf(features_train, targets_train)
        predictions = predict_qrf(model, features_test, expectile_level)
    elif model_name == "Q-GBM":
        model = fit_qgbm(features_train, targets_train, expectile_level)
        predictions = predict_qgbm(model, features_test)
    else:
        raise ValueError(f"unknown model {model_name}")
    return predictions


def quantile_r_squared(predictions, targets, expectile_level, train_targets):
    unconditional_quantile = numpy.quantile(train_targets, expectile_level)
    model_loss = quantile_loss(predictions, targets, expectile_level)
    baseline_loss = quantile_loss(numpy.full_like(targets, unconditional_quantile), targets, expectile_level)
    return 1.0 - model_loss / baseline_loss


def expectile_r_squared(predictions, targets, expectile_level, train_targets):
    """基于非对称平方损失的 R²，对偶于 quantile_r_squared。

    baseline 为训练集无条件 τ-expectile。这是 ERNN 的"原生"拟合优度指标——
    ERNN 训练时最小化的正是非对称平方损失，故用同一损失评价才符合模型逻辑。
    """
    unconditional = unconditional_expectile(train_targets, expectile_level)
    model_loss = expectile_loss_numpy(predictions, targets, expectile_level)
    baseline_loss = expectile_loss_numpy(numpy.full_like(targets, unconditional), targets, expectile_level)
    return 1.0 - model_loss / baseline_loss


def pinball_per_observation(predictions, targets, expectile_level):
    residuals = targets - predictions
    return numpy.maximum(expectile_level * residuals, (expectile_level - 1) * residuals)


def expectile_per_observation(predictions, targets, expectile_level):
    residuals = targets - predictions
    weights = numpy.where(residuals < 0, 1.0 - expectile_level, expectile_level)
    return weights * residuals ** 2


def main():
    data = pandas.read_csv(OUTPUT_ROOT / "master_data.csv", parse_dates=["date"])
    print(f">>> loaded master_data: {data.shape}")

    directions = [
        ("eua_return", "股+碳 -> 碳"),
        ("msci_energy_return", "碳+股 -> 股能源"),
        ("msci_materials_return", "碳+股 -> 股材料"),
    ]
    model_names = ["L-ER", "QRNN", "QRF", "Q-GBM", "ERNN"]

    all_results = []
    dm_records = []

    for target_column, direction_label in directions:
        print(f"\n{'=' * 80}")
        print(f"DIRECTION: {direction_label} (target = {target_column})")
        print('=' * 80)

        combined = build_lagged_features(data, target_column, LAG_LIST)
        feature_columns = [col for col in combined.columns if col != "target"]
        features_array = combined[feature_columns].to_numpy(dtype=numpy.float32)
        targets_array = combined["target"].to_numpy(dtype=numpy.float32)

        split_index = int(len(combined) * TRAIN_RATIO)
        features_train_raw, features_test_raw = features_array[:split_index], features_array[split_index:]
        targets_train, targets_test = targets_array[:split_index], targets_array[split_index:]
        features_train, features_test = standardize(features_train_raw, features_test_raw)
        print(f"    train: {features_train.shape}, test: {features_test.shape}")

        for expectile_level in EXPECTILE_LEVELS:
            print(f"\n    --- tau = {expectile_level} ---")
            tau_pinball_obs = {}
            tau_expectile_obs = {}
            for model_name in model_names:
                start = time.time()
                predictions = run_one_model(model_name, features_train, targets_train, features_test, targets_test, expectile_level)
                elapsed = time.time() - start
                pinball = quantile_loss(predictions, targets_test, expectile_level)
                expectile = expectile_loss_numpy(predictions, targets_test, expectile_level)
                qr2 = quantile_r_squared(predictions, targets_test, expectile_level, targets_train)
                er2 = expectile_r_squared(predictions, targets_test, expectile_level, targets_train)
                rmse = float(numpy.sqrt(numpy.mean((targets_test - predictions) ** 2)))
                mae = float(numpy.mean(numpy.abs(targets_test - predictions)))
                tau_pinball_obs[model_name] = pinball_per_observation(predictions, targets_test, expectile_level)
                tau_expectile_obs[model_name] = expectile_per_observation(predictions, targets_test, expectile_level)
                all_results.append({
                    "direction": direction_label,
                    "target": target_column,
                    "tau": expectile_level,
                    "model": model_name,
                    "pred_mean": predictions.mean(),
                    "pinball_loss": pinball,
                    "expectile_loss": expectile,
                    "quantile_r_squared": qr2,
                    "expectile_r_squared": er2,
                    "rmse": rmse,
                    "mae": mae,
                    "elapsed_seconds": elapsed,
                })
                print(f"      {model_name:6s} | pinball={pinball:.5f} | expect={expectile:.6f} | QR2={qr2:+.4f} | ER2={er2:+.4f} | RMSE={rmse:.4f} | t={elapsed:.1f}s")

            for benchmark in ["L-ER", "QRNN", "QRF", "Q-GBM"]:
                for loss_metric, obs_dict in (("pinball", tau_pinball_obs), ("expectile", tau_expectile_obs)):
                    dm_stat, p_value = diebold_mariano_test(obs_dict[benchmark], obs_dict["ERNN"])
                    dm_records.append({
                        "direction": direction_label,
                        "target": target_column,
                        "tau": expectile_level,
                        "benchmark": benchmark,
                        "competitor": "ERNN",
                        "loss_metric": loss_metric,
                        "DM_stat": dm_stat,
                        "p_value": p_value,
                        "ernn_better": (dm_stat > 0) and (p_value < 0.1),
                    })

    result_frame = pandas.DataFrame(all_results)
    dm_frame = pandas.DataFrame(dm_records)
    result_frame.to_csv(OUTPUT_ROOT / "main_results.csv", index=False)
    dm_frame.to_csv(OUTPUT_ROOT / "dm_test.csv", index=False)
    print(f"\n>>> saved main_results.csv and dm_test.csv")

    print(f"\n{'=' * 80}")
    print("SUMMARY: best model per (direction, tau) by PINBALL loss (分位数损失主场)")
    print('=' * 80)
    summary_pinball = result_frame.loc[result_frame.groupby(["direction", "tau"])["pinball_loss"].idxmin()]
    print(summary_pinball[["direction", "tau", "model", "pinball_loss", "quantile_r_squared"]].to_string(index=False))

    print(f"\n{'=' * 80}")
    print("SUMMARY: best model per (direction, tau) by EXPECTILE loss (符合 ERNN 模型逻辑)")
    print('=' * 80)
    summary_expectile = result_frame.loc[result_frame.groupby(["direction", "tau"])["expectile_loss"].idxmin()]
    print(summary_expectile[["direction", "tau", "model", "expectile_loss", "expectile_r_squared"]].to_string(index=False))

    print(f"\n{'=' * 80}")
    print("DM TEST: ERNN vs each baseline (positive DM = ERNN better), 两种损失口径")
    print('=' * 80)
    for loss_metric in ["pinball", "expectile"]:
        subset = dm_frame[dm_frame["loss_metric"] == loss_metric]
        win_count = subset.groupby("benchmark")["ernn_better"].sum().to_dict()
        n_cells = subset.groupby("benchmark").size().max()
        print(f"  [{loss_metric:9s}] ERNN 显著胜出计数: {win_count} (各 / {n_cells})")


if __name__ == "__main__":
    main()

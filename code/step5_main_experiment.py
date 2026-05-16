from pathlib import Path
import time
import numpy
import pandas
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step3_ernn_model import (
    fit_ernn, predict_ernn, expectile_loss_numpy, quantile_loss,
)
from step4_baselines import (
    fit_linear_expectile_regression, predict_linear_expectile_regression,
    fit_qrnn, predict_qrnn,
    fit_qrf, predict_qrf,
    fit_qgbm, predict_qgbm,
)

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")

LAG_LIST = [1, 2, 3, 5]
EXPECTILE_LEVELS = [0.05, 0.1, 0.5, 0.9, 0.95]
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
    pinball_records = {}

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
            tau_results = {}
            tau_pinball = {}
            for model_name in model_names:
                start = time.time()
                predictions = run_one_model(model_name, features_train, targets_train, features_test, targets_test, expectile_level)
                elapsed = time.time() - start
                pinball = quantile_loss(predictions, targets_test, expectile_level)
                expectile = expectile_loss_numpy(predictions, targets_test, expectile_level)
                qr2 = quantile_r_squared(predictions, targets_test, expectile_level, targets_train)
                pinball_per_obs = numpy.maximum(expectile_level * (targets_test - predictions), (expectile_level - 1) * (targets_test - predictions))
                tau_pinball[model_name] = pinball_per_obs
                tau_results[model_name] = {"pred_mean": predictions.mean(), "pinball": pinball, "expectile": expectile, "QR2": qr2, "elapsed_s": elapsed}
                all_results.append({
                    "direction": direction_label,
                    "target": target_column,
                    "tau": expectile_level,
                    "model": model_name,
                    "pred_mean": predictions.mean(),
                    "pinball_loss": pinball,
                    "expectile_loss": expectile,
                    "quantile_r_squared": qr2,
                    "elapsed_seconds": elapsed,
                })
                print(f"      {model_name:6s} | pinball={pinball:.5f} | expect={expectile:.6f} | QR2={qr2:+.4f} | t={elapsed:.1f}s | pred_mean={predictions.mean():+.4f}")

            for benchmark in ["L-ER", "QRNN", "QRF", "Q-GBM"]:
                dm_stat, p_value = diebold_mariano_test(tau_pinball[benchmark], tau_pinball["ERNN"])
                dm_records.append({
                    "direction": direction_label,
                    "target": target_column,
                    "tau": expectile_level,
                    "benchmark": benchmark,
                    "competitor": "ERNN",
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
    print("SUMMARY: best model per (direction, tau) by pinball loss")
    print('=' * 80)
    summary = result_frame.loc[result_frame.groupby(["direction", "tau"])["pinball_loss"].idxmin()]
    print(summary[["direction", "tau", "model", "pinball_loss", "quantile_r_squared"]].to_string(index=False))

    print(f"\n{'=' * 80}")
    print("DM TEST: ERNN vs each baseline (positive DM = ERNN better)")
    print('=' * 80)
    print(dm_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()

from pathlib import Path
import numpy
import pandas
import sys
sys.path.insert(0, str(Path(__file__).parent))
from step3_ernn_model import (
    fit_ernn, predict_ernn, expectile_loss_numpy, quantile_loss,
    select_hyperparameters,
)

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")

LAG_LIST = [1, 2, 3, 5]
EXPECTILE_LEVELS = [0.05, 0.1, 0.5, 0.9, 0.95]
TRAIN_RATIO = 0.8


def build_lagged_features(data, target_column, return_columns, lag_list):
    feature_frames = []
    for column in return_columns:
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


def main():
    data = pandas.read_csv(OUTPUT_ROOT / "master_data.csv", parse_dates=["date"])
    print(f">>> loaded master_data: {data.shape}")

    return_columns = ["eua_return", "msci_energy_return", "msci_materials_return"]

    print("\n>>> direction 1: predict next EUA return (股 + 碳 -> 碳)")
    combined = build_lagged_features(data, "eua_return", return_columns, LAG_LIST)
    print(f"    features after lag: {combined.shape}")
    feature_columns = [col for col in combined.columns if col != "target"]
    features_array = combined[feature_columns].to_numpy(dtype=numpy.float32)
    targets_array = combined["target"].to_numpy(dtype=numpy.float32)

    split_index = int(len(combined) * TRAIN_RATIO)
    features_train_raw = features_array[:split_index]
    features_test_raw = features_array[split_index:]
    targets_train = targets_array[:split_index]
    targets_test = targets_array[split_index:]

    features_train, features_test = standardize(features_train_raw, features_test_raw)
    print(f"    train: {features_train.shape}, test: {features_test.shape}")

    print("\n>>> running ERNN at five expectile levels (no hyperparam search, J=5, lambda=1e-3)")
    rows = []
    for expectile_level in EXPECTILE_LEVELS:
        model, in_sample_loss = fit_ernn(
            features_train, targets_train, expectile_level,
            hidden_dimension=5, regularization=1e-3, verbose=False,
        )
        predictions_test = predict_ernn(model, features_test)
        predictions_train = predict_ernn(model, features_train)

        in_loss_expectile = expectile_loss_numpy(predictions_train, targets_train, expectile_level)
        out_loss_expectile = expectile_loss_numpy(predictions_test, targets_test, expectile_level)
        out_loss_quantile_proxy = quantile_loss(predictions_test, targets_test, expectile_level)

        rows.append({
            "tau": expectile_level,
            "in_sample_expectile_loss": in_loss_expectile,
            "out_sample_expectile_loss": out_loss_expectile,
            "out_sample_pinball_loss": out_loss_quantile_proxy,
            "pred_mean": predictions_test.mean(),
            "pred_std": predictions_test.std(),
            "actual_mean": targets_test.mean(),
            "actual_quantile_at_tau": numpy.quantile(targets_test, expectile_level),
        })
        print(f"    tau={expectile_level:.2f} | in_expectile_loss={in_loss_expectile:.6f} | out_expectile_loss={out_loss_expectile:.6f} | pred_mean={predictions_test.mean():.5f}")

    result_frame = pandas.DataFrame(rows)
    print("\n>>> summary:")
    print(result_frame.to_string(float_format=lambda value: f"{value:.5f}"))
    result_frame.to_csv(OUTPUT_ROOT / "ernn_minimal_pipeline.csv", index=False)
    print(f"\n>>> saved ernn_minimal_pipeline.csv")

    print("\n>>> sanity check: predicted expectiles should be monotonic in tau")
    for index in range(len(rows) - 1):
        assert rows[index]["pred_mean"] <= rows[index + 1]["pred_mean"] + 1e-3, f"monotonicity violated between tau={rows[index]['tau']} and tau={rows[index + 1]['tau']}"
    print("    monotonicity OK")


if __name__ == "__main__":
    main()

from pathlib import Path
import time
import numpy
import pandas
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step3_ernn_model import fit_ernn, predict_ernn, expectile_loss_numpy, quantile_loss
from step4_baselines import (
    fit_linear_expectile_regression, predict_linear_expectile_regression,
    fit_qrnn, predict_qrnn,
    fit_qrf, predict_qrf,
    fit_qgbm, predict_qgbm,
)
from step5_main_experiment import build_lagged_features, standardize, run_one_model

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")

LAG_LIST = [1, 2, 3, 5]
EXPECTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
RETURN_COLUMNS = ["eua_return", "msci_energy_return", "msci_materials_return"]
MODEL_NAMES = ["L-ER", "ERNN"]

ROLLING_WINDOW = 1000
REFIT_FREQUENCY = 30


def rolling_forecast_one_target(data, target_column):
    print(f"\n>>> rolling forecast for target = {target_column}")
    combined = build_lagged_features(data, target_column, LAG_LIST)
    feature_columns = [col for col in combined.columns if col != "target"]
    features_array = combined[feature_columns].to_numpy(dtype=numpy.float32)
    targets_array = combined["target"].to_numpy(dtype=numpy.float32)
    sample_size = len(combined)

    rolling_records = []
    test_start = ROLLING_WINDOW
    print(f"    sample_size={sample_size}, rolling_window={ROLLING_WINDOW}, refit_every={REFIT_FREQUENCY} days")

    cached_models = {}

    for current_index in range(test_start, sample_size):
        train_start = current_index - ROLLING_WINDOW
        train_end = current_index
        if (current_index - test_start) % REFIT_FREQUENCY == 0:
            features_train_raw = features_array[train_start:train_end]
            targets_train = targets_array[train_start:train_end]
            train_mean = features_train_raw.mean(axis=0)
            train_std = features_train_raw.std(axis=0) + 1e-8
            features_train = (features_train_raw - train_mean) / train_std
            cached_models = {}
            for model_name in MODEL_NAMES:
                for expectile_level in EXPECTILE_LEVELS:
                    if model_name == "ERNN":
                        model, _ = fit_ernn(features_train, targets_train, expectile_level, hidden_dimension=5, regularization=1e-3)
                        cached_models[(model_name, expectile_level)] = ("ernn", model)
                    elif model_name == "L-ER":
                        weights = fit_linear_expectile_regression(features_train, targets_train, expectile_level)
                        cached_models[(model_name, expectile_level)] = ("ler", weights)
            cached_train_mean = train_mean
            cached_train_std = train_std

        feature_today = (features_array[current_index:current_index + 1] - cached_train_mean) / cached_train_std
        target_today = targets_array[current_index]

        for model_name in MODEL_NAMES:
            for expectile_level in EXPECTILE_LEVELS:
                kind, payload = cached_models[(model_name, expectile_level)]
                if kind == "ernn":
                    pred = predict_ernn(payload, feature_today)[0]
                elif kind == "ler":
                    pred = predict_linear_expectile_regression(payload, feature_today)[0]
                residual = target_today - pred
                pinball = max(expectile_level * residual, (expectile_level - 1) * residual)
                expectile = (expectile_level if residual >= 0 else 1.0 - expectile_level) * residual ** 2
                rolling_records.append({
                    "target": target_column,
                    "index": current_index,
                    "model": model_name,
                    "tau": expectile_level,
                    "prediction": float(pred),
                    "actual": float(target_today),
                    "pinball": float(pinball),
                    "expectile": float(expectile),
                })

        if (current_index - test_start) % 200 == 0:
            print(f"    progress: {current_index - test_start}/{sample_size - test_start}")

    return pandas.DataFrame(rolling_records)


def main():
    data = pandas.read_csv(OUTPUT_ROOT / "master_data.csv", parse_dates=["date"])
    print(f">>> loaded master_data: {data.shape}")

    all_records = []
    start_time = time.time()
    for target_column in RETURN_COLUMNS:
        records = rolling_forecast_one_target(data, target_column)
        all_records.append(records)
    rolling_frame = pandas.concat(all_records, ignore_index=True)
    rolling_frame.to_csv(OUTPUT_ROOT / "rolling_forecast.csv", index=False)
    elapsed = time.time() - start_time
    print(f"\n>>> total time: {elapsed:.1f}s, saved rolling_forecast.csv ({len(rolling_frame)} rows)")

    print("\n" + "=" * 80)
    print("ROLLING FORECAST SUMMARY: average loss per (target, tau, model), 双损失口径")
    print("=" * 80)
    pinball_pivot = rolling_frame.groupby(["target", "tau", "model"])["pinball"].mean().reset_index()
    pinball_pivot = pinball_pivot.pivot_table(index=["target", "tau"], columns="model", values="pinball")
    pinball_pivot["pinball_ERNN_minus_LER"] = pinball_pivot["ERNN"] - pinball_pivot["L-ER"]
    pinball_pivot["pinball_ERNN_rel_improve_pct"] = (pinball_pivot["L-ER"] - pinball_pivot["ERNN"]) / pinball_pivot["L-ER"] * 100

    expectile_pivot = rolling_frame.groupby(["target", "tau", "model"])["expectile"].mean().reset_index()
    expectile_pivot = expectile_pivot.pivot_table(index=["target", "tau"], columns="model", values="expectile")
    expectile_pivot["expectile_ERNN_rel_improve_pct"] = (expectile_pivot["L-ER"] - expectile_pivot["ERNN"]) / expectile_pivot["L-ER"] * 100

    summary = pinball_pivot.copy()
    summary["expectile_ERNN_rel_improve_pct"] = expectile_pivot["expectile_ERNN_rel_improve_pct"]
    print(summary.to_string(float_format=lambda v: f"{v:.6f}"))
    summary.reset_index().to_csv(OUTPUT_ROOT / "rolling_forecast_summary.csv", index=False)
    print(f"\n>>> saved rolling_forecast_summary.csv")


if __name__ == "__main__":
    main()

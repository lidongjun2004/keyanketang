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
from step5_main_experiment import diebold_mariano_test, build_lagged_features, standardize, run_one_model, quantile_r_squared, expectile_r_squared

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")

LAG_LIST = [1, 2, 3, 5]
EXPECTILE_LEVELS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
TRAIN_RATIO = 0.8
RETURN_COLUMNS = ["eua_return", "msci_energy_return", "msci_materials_return"]
CONTROL_DIFF_COLUMNS = ["china_10y_diff", "eu_pmi_diff", "eu_ip_diff"]
MODEL_NAMES = ["L-ER", "QRNN", "QRF", "Q-GBM", "ERNN"]


def evaluate_one_split(features_train, targets_train, features_test, targets_test, expectile_level):
    rows = []
    pinballs = {}
    for model_name in MODEL_NAMES:
        predictions = run_one_model(model_name, features_train, targets_train, features_test, targets_test, expectile_level)
        pinball = quantile_loss(predictions, targets_test, expectile_level)
        expectile = expectile_loss_numpy(predictions, targets_test, expectile_level)
        qr2 = quantile_r_squared(predictions, targets_test, expectile_level, targets_train)
        er2 = expectile_r_squared(predictions, targets_test, expectile_level, targets_train)
        pinball_per_obs = numpy.maximum(expectile_level * (targets_test - predictions), (expectile_level - 1) * (targets_test - predictions))
        pinballs[model_name] = pinball_per_obs
        rows.append({
            "model": model_name,
            "tau": expectile_level,
            "pinball_loss": pinball,
            "expectile_loss": expectile,
            "quantile_r_squared": qr2,
            "expectile_r_squared": er2,
        })
    return rows, pinballs


def robustness_a_exclude_covid(data):
    print("\n" + "=" * 80)
    print("ROBUSTNESS A: exclude 2020-Q1 (COVID extreme volatility period)")
    print("=" * 80)
    excluded = data[~((data["date"] >= "2020-01-01") & (data["date"] <= "2020-06-30"))].reset_index(drop=True)
    print(f"    rows after exclusion: {len(excluded)} (original: {len(data)})")
    return run_main_grid(excluded, label="exclude_covid")


def robustness_b_hidden_dim(data):
    print("\n" + "=" * 80)
    print("ROBUSTNESS B: ERNN hidden dim grid (J in {3, 5, 7, 10})")
    print("=" * 80)
    rows = []
    for target_column in RETURN_COLUMNS:
        combined = build_lagged_features(data, target_column, LAG_LIST)
        feature_columns = [col for col in combined.columns if col != "target"]
        features_array = combined[feature_columns].to_numpy(dtype=numpy.float32)
        targets_array = combined["target"].to_numpy(dtype=numpy.float32)
        split_index = int(len(combined) * TRAIN_RATIO)
        features_train_raw, features_test_raw = features_array[:split_index], features_array[split_index:]
        targets_train, targets_test = targets_array[:split_index], targets_array[split_index:]
        features_train, features_test = standardize(features_train_raw, features_test_raw)

        for hidden_dim in [3, 5, 7, 10]:
            for expectile_level in EXPECTILE_LEVELS:
                model, _ = fit_ernn(features_train, targets_train, expectile_level, hidden_dimension=hidden_dim, regularization=1e-3)
                predictions = predict_ernn(model, features_test)
                pinball = quantile_loss(predictions, targets_test, expectile_level)
                qr2 = quantile_r_squared(predictions, targets_test, expectile_level, targets_train)
                er2 = expectile_r_squared(predictions, targets_test, expectile_level, targets_train)
                rows.append({
                    "target": target_column,
                    "hidden_dim": hidden_dim,
                    "tau": expectile_level,
                    "pinball_loss": pinball,
                    "quantile_r_squared": qr2,
                    "expectile_r_squared": er2,
                })
                print(f"    target={target_column[:20]:20s} | J={hidden_dim:2d} | tau={expectile_level} | pinball={pinball:.6f} | QR2={qr2:+.4f}")
    return pandas.DataFrame(rows)


def robustness_c_add_controls(data):
    print("\n" + "=" * 80)
    print("ROBUSTNESS C: add macro control variables (china_10y_diff, eu_pmi_diff, eu_ip_diff)")
    print("=" * 80)
    enriched = data.dropna(subset=CONTROL_DIFF_COLUMNS).reset_index(drop=True)
    print(f"    rows after dropna for controls: {len(enriched)} (original: {len(data)})")

    rows = []
    dm_records = []
    for target_column in RETURN_COLUMNS:
        feature_frames = []
        for column in RETURN_COLUMNS:
            for lag in LAG_LIST:
                feature_frames.append(enriched[column].shift(lag).rename(f"{column}_lag{lag}"))
        for column in CONTROL_DIFF_COLUMNS:
            for lag in [1, 2]:
                feature_frames.append(enriched[column].shift(lag).rename(f"{column}_lag{lag}"))
        features = pandas.concat(feature_frames, axis=1)
        target = enriched[target_column].rename("target")
        combined = pandas.concat([features, target], axis=1).dropna()

        feature_columns = [col for col in combined.columns if col != "target"]
        features_array = combined[feature_columns].to_numpy(dtype=numpy.float32)
        targets_array = combined["target"].to_numpy(dtype=numpy.float32)
        split_index = int(len(combined) * TRAIN_RATIO)
        features_train_raw, features_test_raw = features_array[:split_index], features_array[split_index:]
        targets_train, targets_test = targets_array[:split_index], targets_array[split_index:]
        features_train, features_test = standardize(features_train_raw, features_test_raw)

        print(f"\n  --- target = {target_column} (n_features = {features_train.shape[1]}, train={features_train.shape[0]}) ---")
        for expectile_level in EXPECTILE_LEVELS:
            split_rows, pinballs = evaluate_one_split(features_train, targets_train, features_test, targets_test, expectile_level)
            for row in split_rows:
                row["target"] = target_column
                rows.append(row)
                print(f"    tau={expectile_level} | {row['model']:6s} | pinball={row['pinball_loss']:.6f} | QR2={row['quantile_r_squared']:+.4f}")
            for benchmark in ["L-ER", "QRNN", "QRF", "Q-GBM"]:
                dm_stat, p_value = diebold_mariano_test(pinballs[benchmark], pinballs["ERNN"])
                dm_records.append({
                    "target": target_column,
                    "tau": expectile_level,
                    "benchmark": benchmark,
                    "DM_stat": dm_stat,
                    "p_value": p_value,
                    "ernn_better": (dm_stat > 0) and (p_value < 0.1),
                })
    return pandas.DataFrame(rows), pandas.DataFrame(dm_records)


def run_main_grid(data, label):
    rows = []
    dm_records = []
    for target_column in RETURN_COLUMNS:
        combined = build_lagged_features(data, target_column, LAG_LIST)
        feature_columns = [col for col in combined.columns if col != "target"]
        features_array = combined[feature_columns].to_numpy(dtype=numpy.float32)
        targets_array = combined["target"].to_numpy(dtype=numpy.float32)
        split_index = int(len(combined) * TRAIN_RATIO)
        features_train_raw, features_test_raw = features_array[:split_index], features_array[split_index:]
        targets_train, targets_test = targets_array[:split_index], targets_array[split_index:]
        features_train, features_test = standardize(features_train_raw, features_test_raw)

        print(f"\n  --- {label} | target = {target_column} (train={features_train.shape[0]}, test={features_test.shape[0]}) ---")
        for expectile_level in EXPECTILE_LEVELS:
            split_rows, pinballs = evaluate_one_split(features_train, targets_train, features_test, targets_test, expectile_level)
            for row in split_rows:
                row["target"] = target_column
                row["label"] = label
                rows.append(row)
                print(f"    tau={expectile_level} | {row['model']:6s} | pinball={row['pinball_loss']:.6f} | QR2={row['quantile_r_squared']:+.4f}")
            for benchmark in ["L-ER", "QRNN", "QRF", "Q-GBM"]:
                dm_stat, p_value = diebold_mariano_test(pinballs[benchmark], pinballs["ERNN"])
                dm_records.append({
                    "label": label,
                    "target": target_column,
                    "tau": expectile_level,
                    "benchmark": benchmark,
                    "DM_stat": dm_stat,
                    "p_value": p_value,
                    "ernn_better": (dm_stat > 0) and (p_value < 0.1),
                })
    return pandas.DataFrame(rows), pandas.DataFrame(dm_records)


def main():
    data = pandas.read_csv(OUTPUT_ROOT / "master_data.csv", parse_dates=["date"])
    print(f">>> loaded master_data: {data.shape}")

    print("\n>>> Robustness A: exclude COVID")
    results_a, dm_a = robustness_a_exclude_covid(data)
    results_a.to_csv(OUTPUT_ROOT / "robustness_a_exclude_covid.csv", index=False)
    dm_a.to_csv(OUTPUT_ROOT / "robustness_a_dm.csv", index=False)
    print(f">>> saved robustness_a_*.csv")

    print("\n>>> Robustness B: hidden dim grid")
    results_b = robustness_b_hidden_dim(data)
    results_b.to_csv(OUTPUT_ROOT / "robustness_b_hidden_dim.csv", index=False)
    print(f">>> saved robustness_b_hidden_dim.csv")

    print("\n>>> Robustness C: add controls")
    results_c, dm_c = robustness_c_add_controls(data)
    results_c.to_csv(OUTPUT_ROOT / "robustness_c_add_controls.csv", index=False)
    dm_c.to_csv(OUTPUT_ROOT / "robustness_c_dm.csv", index=False)
    print(f">>> saved robustness_c_*.csv")

    print("\n" + "=" * 80)
    print("FINAL SUMMARIES")
    print("=" * 80)

    print("\n[A] Best model per (target, tau) when COVID excluded:")
    summary_a = results_a.loc[results_a.groupby(["target", "tau"])["pinball_loss"].idxmin()]
    print(summary_a[["target", "tau", "model", "pinball_loss", "quantile_r_squared"]].to_string(index=False))

    print("\n[B] Best ERNN hidden_dim per (target, tau):")
    summary_b = results_b.loc[results_b.groupby(["target", "tau"])["pinball_loss"].idxmin()]
    print(summary_b[["target", "tau", "hidden_dim", "pinball_loss", "quantile_r_squared"]].to_string(index=False))

    print("\n[C] Best model per (target, tau) with controls:")
    summary_c = results_c.loc[results_c.groupby(["target", "tau"])["pinball_loss"].idxmin()]
    print(summary_c[["target", "tau", "model", "pinball_loss", "quantile_r_squared"]].to_string(index=False))

    print("\n[A vs main] DM check: how often does ERNN still beat each baseline when COVID excluded?")
    win_count_a = dm_a.groupby("benchmark")["ernn_better"].sum().to_dict()
    print(f"    {win_count_a} (out of 21 = 3 targets × 7 taus per benchmark)")

    print("\n[C vs main] DM check: how often does ERNN beat each baseline with macro controls?")
    win_count_c = dm_c.groupby("benchmark")["ernn_better"].sum().to_dict()
    print(f"    {win_count_c} (out of 21)")


if __name__ == "__main__":
    main()

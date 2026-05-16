from pathlib import Path
import numpy
import pandas
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from sklearn.ensemble import GradientBoostingRegressor
from quantile_forest import RandomForestQuantileRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))
from step3_ernn_model import (
    fit_ernn, predict_ernn, expectile_loss_numpy, quantile_loss,
)


TORCH_DEVICE = torch.device("cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)


def fit_linear_expectile_regression(features_train, targets_train, expectile_level, max_iterations=2000, learning_rate=1e-2, regularization=1e-4):
    weights = torch.zeros(features_train.shape[1] + 1, dtype=torch.float32, requires_grad=True)
    features_with_bias = torch.cat([
        torch.ones(features_train.shape[0], 1, dtype=torch.float32),
        torch.tensor(features_train, dtype=torch.float32),
    ], dim=1)
    targets_tensor = torch.tensor(targets_train, dtype=torch.float32)
    optimizer = torch.optim.Adam([weights], lr=learning_rate, weight_decay=regularization)

    for iteration in range(max_iterations):
        optimizer.zero_grad()
        predictions = features_with_bias @ weights
        residuals = targets_tensor - predictions
        sample_weights = torch.where(residuals < 0, 1.0 - expectile_level, expectile_level)
        loss = torch.mean(sample_weights * residuals.pow(2))
        loss.backward()
        optimizer.step()
    return weights.detach().numpy()


def predict_linear_expectile_regression(weights, features):
    features_with_bias = numpy.concatenate([
        numpy.ones((features.shape[0], 1), dtype=numpy.float32),
        features.astype(numpy.float32),
    ], axis=1)
    return features_with_bias @ weights


class QuantileRegressionNeuralNetwork(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dimension, hidden_dimension)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(hidden_dimension, 1)

    def forward(self, features):
        return self.output_layer(self.activation(self.hidden_layer(features))).squeeze(-1)


def quantile_loss_torch(predictions, targets, quantile_level):
    residuals = targets - predictions
    return torch.mean(torch.maximum(quantile_level * residuals, (quantile_level - 1) * residuals))


def fit_qrnn(features_train, targets_train, quantile_level, hidden_dimension=5, regularization=1e-3, max_epochs=2000, patience=200):
    model = QuantileRegressionNeuralNetwork(features_train.shape[1], hidden_dimension).to(TORCH_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=regularization)
    features_tensor = torch.tensor(features_train, dtype=torch.float32, device=TORCH_DEVICE)
    targets_tensor = torch.tensor(targets_train, dtype=torch.float32, device=TORCH_DEVICE)

    best_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        predictions = model(features_tensor)
        loss = quantile_loss_torch(predictions, targets_tensor, quantile_level)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        if loss_value < best_loss - 1e-6:
            best_loss = loss_value
            best_state_dict = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    model.load_state_dict(best_state_dict)
    model.eval()
    return model


def predict_qrnn(model, features):
    features_tensor = torch.tensor(features, dtype=torch.float32, device=TORCH_DEVICE)
    with torch.no_grad():
        return model(features_tensor).cpu().numpy()


def fit_qrf(features_train, targets_train, n_estimators=200, max_depth=10):
    model = RandomForestQuantileRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(features_train, targets_train)
    return model


def predict_qrf(model, features, quantile_level):
    return model.predict(features, quantiles=[quantile_level]).reshape(-1)


def fit_qgbm(features_train, targets_train, quantile_level, n_estimators=200, max_depth=4, learning_rate=0.05):
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile_level,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_SEED,
    )
    model.fit(features_train, targets_train)
    return model


def predict_qgbm(model, features):
    return model.predict(features)


def smoke_test():
    print(">>> Baseline smoke test on synthetic data")
    rng = numpy.random.default_rng(RANDOM_SEED)
    sample_size = 800
    feature = rng.normal(size=(sample_size, 2)).astype(numpy.float32)
    target = (numpy.sin(2 * feature[:, 0]) + 0.5 * feature[:, 1] + rng.normal(size=sample_size) * 0.5).astype(numpy.float32)
    features_train, targets_train = feature[:600], target[:600]
    features_test, targets_test = feature[600:], target[600:]

    for tau in [0.1, 0.5, 0.9]:
        print(f"\n  tau={tau}")
        weights = fit_linear_expectile_regression(features_train, targets_train, tau)
        predictions = predict_linear_expectile_regression(weights, features_test)
        print(f"    L-ER  | pred_mean={predictions.mean():.3f}, expectile_loss={expectile_loss_numpy(predictions, targets_test, tau):.4f}")

        qrnn_model = fit_qrnn(features_train, targets_train, tau)
        predictions = predict_qrnn(qrnn_model, features_test)
        print(f"    QRNN  | pred_mean={predictions.mean():.3f}, pinball_loss={quantile_loss(predictions, targets_test, tau):.4f}")

        qrf_model = fit_qrf(features_train, targets_train)
        predictions = predict_qrf(qrf_model, features_test, tau)
        print(f"    QRF   | pred_mean={predictions.mean():.3f}, pinball_loss={quantile_loss(predictions, targets_test, tau):.4f}")

        qgbm_model = fit_qgbm(features_train, targets_train, tau)
        predictions = predict_qgbm(qgbm_model, features_test)
        print(f"    Q-GBM | pred_mean={predictions.mean():.3f}, pinball_loss={quantile_loss(predictions, targets_test, tau):.4f}")


if __name__ == "__main__":
    smoke_test()

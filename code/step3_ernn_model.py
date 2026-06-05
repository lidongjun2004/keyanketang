from pathlib import Path
import math
import numpy
import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUTPUT_ROOT = Path("/Users/minimax/workplace/personal/college/curriculum/junior/科研课堂/output")
TORCH_DEVICE = torch.device("cpu")
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)


class ExpectileRegressionNeuralNetwork(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dimension, hidden_dimension)
        self.activation = nn.Tanh()
        self.output_layer = nn.Linear(hidden_dimension, 1)

    def forward(self, features):
        return self.output_layer(self.activation(self.hidden_layer(features))).squeeze(-1)


def expectile_loss(predictions, targets, expectile_level):
    residuals = targets - predictions
    weights = torch.where(residuals < 0, 1.0 - expectile_level, expectile_level)
    return torch.mean(weights * residuals.pow(2))


def fit_ernn(features_train, targets_train, expectile_level, hidden_dimension, regularization, learning_rate=1e-3, max_epochs=2000, patience=200, verbose=False):
    model = ExpectileRegressionNeuralNetwork(features_train.shape[1], hidden_dimension).to(TORCH_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    features_tensor = torch.tensor(features_train, dtype=torch.float32, device=TORCH_DEVICE)
    targets_tensor = torch.tensor(targets_train, dtype=torch.float32, device=TORCH_DEVICE)

    best_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        predictions = model(features_tensor)
        loss = expectile_loss(predictions, targets_tensor, expectile_level)
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
                if verbose:
                    print(f"      early stop at epoch {epoch}, best_loss={best_loss:.6f}")
                break
        if verbose and epoch % 200 == 0:
            print(f"      epoch {epoch}, loss={loss_value:.6f}")

    model.load_state_dict(best_state_dict)
    model.eval()
    return model, best_loss


def predict_ernn(model, features):
    features_tensor = torch.tensor(features, dtype=torch.float32, device=TORCH_DEVICE)
    with torch.no_grad():
        return model(features_tensor).cpu().numpy()


def compute_bic(in_sample_loss, sample_size, input_dimension, hidden_dimension):
    parameter_count = (input_dimension + 2) * hidden_dimension + 1
    return math.log(in_sample_loss + 1e-12) + 0.5 * math.log(sample_size) / sample_size * parameter_count


def select_hyperparameters(features_train, targets_train, expectile_level, hidden_grid, regularization_grid, verbose=True):
    sample_size = features_train.shape[0]
    input_dimension = features_train.shape[1]
    best = {"hidden": None, "regularization": None, "bic": float("inf"), "loss": None}
    if verbose:
        print(f"    [tau={expectile_level}] BIC grid search over {len(hidden_grid)} x {len(regularization_grid)}")
    for hidden_dimension in hidden_grid:
        for regularization in regularization_grid:
            _, in_sample_loss = fit_ernn(
                features_train, targets_train, expectile_level,
                hidden_dimension, regularization, verbose=False,
            )
            bic = compute_bic(in_sample_loss, sample_size, input_dimension, hidden_dimension)
            if bic < best["bic"]:
                best.update({"hidden": hidden_dimension, "regularization": regularization, "bic": bic, "loss": in_sample_loss})
            if verbose:
                print(f"      J={hidden_dimension}, lambda={regularization:.5f} -> loss={in_sample_loss:.6f}, BIC={bic:.4f}")
    if verbose:
        print(f"    [tau={expectile_level}] best: J={best['hidden']}, lambda={best['regularization']:.5f}, BIC={best['bic']:.4f}")
    return best


def quantile_loss(predictions, targets, quantile_level):
    residuals = targets - predictions
    return numpy.mean(numpy.maximum(quantile_level * residuals, (quantile_level - 1) * residuals))


def expectile_loss_numpy(predictions, targets, expectile_level):
    residuals = targets - predictions
    weights = numpy.where(residuals < 0, 1.0 - expectile_level, expectile_level)
    return numpy.mean(weights * residuals ** 2)


def unconditional_expectile(targets, expectile_level, iterations=200, tolerance=1e-12):
    """求样本无条件 τ-expectile：最小化非对称平方损失的常数预测。

    与无条件分位数 numpy.quantile(.) 对偶，用于构造 Expectile R² 的 baseline。
    采用 Newey & Powell (1987) 的不动点迭代：mu = sum(w*y)/sum(w)，
    w = tau if y>=mu else (1-tau)。该迭代单调收敛。
    """
    targets = numpy.asarray(targets, dtype=numpy.float64)
    mu = targets.mean()
    for _ in range(iterations):
        weights = numpy.where(targets >= mu, expectile_level, 1.0 - expectile_level)
        new_mu = numpy.sum(weights * targets) / numpy.sum(weights)
        if abs(new_mu - mu) < tolerance:
            mu = new_mu
            break
        mu = new_mu
    return float(mu)


def quick_smoke_test():
    print(">>> ERNN smoke test on synthetic data")
    rng = numpy.random.default_rng(RANDOM_SEED)
    sample_size = 800
    feature = rng.normal(size=(sample_size, 1)).astype(numpy.float32)
    noise = rng.normal(size=sample_size).astype(numpy.float32)
    target = numpy.sin(2 * feature[:, 0]) + 2 * numpy.exp(-16 * feature[:, 0] ** 2) + noise
    features_train, targets_train = feature[:600], target[:600]
    features_test, targets_test = feature[600:], target[600:]
    for expectile_level in [0.1, 0.5, 0.9]:
        model, in_sample_loss = fit_ernn(
            features_train, targets_train, expectile_level,
            hidden_dimension=5, regularization=1e-4, verbose=False,
        )
        predictions = predict_ernn(model, features_test)
        out_loss = expectile_loss_numpy(predictions, targets_test, expectile_level)
        print(f"    tau={expectile_level} | in-loss={in_sample_loss:.4f}, out-loss={out_loss:.4f}, pred mean={predictions.mean():.3f}")


if __name__ == "__main__":
    quick_smoke_test()

import numpy as np
from typing import Dict


def mean_absolute_percent_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    return np.mean(np.abs((y - y_pred) / y))


def mean_absolute_scaled_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    # https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
    avg_error = np.mean(np.abs(y - y_pred))
    avg_diff = np.mean(np.abs(y[1:] - y[:-1]))
    return avg_error / avg_diff


def mean_squared_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.square(y - y_pred))


def mean_square_logarithmic_error(
    y: np.ndarray, y_pred: np.ndarray, base: float = np.e
) -> float:
    # Recommended by Ethan Caballero.
    # See: https://insideaiml.com/blog/MeanSquared-Logarithmic-Error-Loss-1035
    assert np.all(y > 0.0)
    assert np.all(y_pred >= 0.0)
    log_base_y = np.log(y + 1.05) / np.log(base)
    log_base_y_pred = np.log(y_pred + 1.05) / np.log(base)
    return np.mean(np.square(log_base_y - log_base_y_pred))


def root_mean_squared_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y=y, y_pred=y_pred))


def score_predictive_model(
    predictive_model, x: np.ndarray, y: np.ndarray
) -> Dict[str, float]:
    y_pred = predictive_model.predict(x=x)
    predictive_model_scores_results = {}
    return predictive_model_scores_results


def symmetric_mean_absolute_percent_error(y: np.ndarray, y_pred: np.ndarray) -> float:
    # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    numerator = np.abs(y - y_pred)
    denominator = (np.abs(y) + np.abs(y_pred)) / 2.0
    return np.mean(numerator / denominator)


ALL_METRICS_DICT = {
    "mean_absolute_percent_error": mean_absolute_percent_error,
    "mean_absolute_scaled_error": mean_absolute_scaled_error,
    "mean_squared_error": mean_squared_error,
    "mean_square_logarithmic_error": mean_square_logarithmic_error,
    "root_mean_squared_error": root_mean_squared_error,
    "symmetric_mean_absolute_percent_error": symmetric_mean_absolute_percent_error,
}

import json
import random
from typing import Any

import joblib
import numpy as np
import torch
from safetensors.torch import save_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from torch import nn


def set_all_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): The seed value to set for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, rounding: int | None = None
) -> dict[str, float]:
    """
    Calculate evaluation metrics for regression, binary, or multiclass classification.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values (discrete for classification).
        rounding (Optional[int]): Decimal rounding for metrics.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }

    # Optional rounding
    if rounding is not None:
        metrics = {name: round(value, rounding) for name, value in metrics.items()}

    return metrics


def organize_in_out_sample_metrics(
    insample_metrics: dict[str, float], outsample_metrics: dict[str, float]
) -> dict[str, float]:
    """
    Organize in-sample and out-of-sample metrics with prefixes.

    Args:
        insample_metrics (dict[str, float]): In-sample metrics dictionary.
        outsample_metrics (dict[str, float]): Out-of-sample metrics dictionary.

    Returns:
        dict: Dictionary containing in-sample and out-of-sample metrics.
    """
    insample_metrics = {f"train_{k}": v for k, v in insample_metrics.items()}
    outsample_metrics = {f"test_{k}": v for k, v in outsample_metrics.items()}
    return {**insample_metrics, **outsample_metrics}


def save_best_model(model: nn.Module | Pipeline, filename: str) -> None:
    """
    Save the best model to a file.

    Args:
        model (ModelType): The trained model to save.
        filename (str): Name of the file to save the model.
    """
    if isinstance(model, nn.Module):
        save_model(model, f"{filename}.safetensors")
    else:
        with open(f"{filename}.joblib", "wb") as f:
            joblib.dump(model, f)


def save_predictions(y_pred: np.ndarray, filename: str) -> None:
    """
    Save predictions to a file.

    Args:
        y_pred (np.ndarray): Predicted values.
        filename (str): Name of the file to save the predictions.
    """
    np.save(filename, y_pred)


def save_json(data: dict[str, Any], filename: str) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): Dictionary containing the data to save.
        filename (str): Name of the file to save the data.
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

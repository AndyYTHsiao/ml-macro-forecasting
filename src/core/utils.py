import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Any, Optional
from safetensors.torch import save_model
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rounding: Optional[int] = None,
    task: str = "regression",  # "regression", "binary", or "multiclass"
    y_proba: Optional[np.ndarray] = None,  # needed for ROC-AUC
) -> dict:
    """
    Calculate evaluation metrics for regression, binary, or multiclass classification.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values (discrete for classification).
        rounding (Optional[int]): Decimal rounding for metrics.
        task (str): Type of task: "regression", "binary", or "multiclass".
        y_proba (Optional[np.ndarray]): Predicted probabilities (for ROC-AUC).

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {}

    if task == "regression":
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = root_mean_squared_error(y_true, y_pred)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)

    elif task == "binary":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    elif task == "multiclass":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision_macro"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_weighted"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        if y_proba is not None:
            try:
                metrics["roc_auc_macro"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
            except ValueError:
                pass  # in case probabilities don't cover all classes

    else:
        raise ValueError(f"Unsupported task type: {task}")

    # Optional rounding
    if rounding is not None:
        metrics = {k: round(v, rounding) for k, v in metrics.items()}

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

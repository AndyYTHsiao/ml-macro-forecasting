import torch.nn as nn
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from typing import Type
from optuna.samplers import BaseSampler, TPESampler, CmaEsSampler
from ..models.pytorch_models import MLPModel, LSTMModel

MODEL_REGISTRY = {
    "linear": {
        "sklearn": {
            "regression": ElasticNet,
            "binary": LogisticRegression,
            "multiclass": LogisticRegression,
        },
        "requires_scaling": True,
    },
    "random_forest": {
        "sklearn": {
            "regression": RandomForestRegressor,
            "binary": RandomForestClassifier,
            "multiclass": RandomForestClassifier,
        },
    },
    "svm": {
        "sklearn": {
            "regression": SVR,
            "binary": SVC,
            "multiclass": SVC,
        },
        "requires_scaling": True,
    },
    "xgboost": {
        "sklearn": {
            "regression": XGBRegressor,
            "binary": XGBClassifier,
            "multiclass": XGBClassifier,
        }
    },
    "mlp": {
        "torch": {
            "regression": MLPModel,
            "binary": MLPModel,
            "multiclass": MLPModel,
        }
    },
    "lstm": {
        "torch": {
            "regression": LSTMModel,
            "binary": LSTMModel,
            "multiclass": LSTMModel,
        }
    },
}


SAMPLER_REGISTRY = {
    "linear": {"regression": TPESampler, "classification": TPESampler},
    "random_forest": {"regression": TPESampler, "classification": TPESampler},
    "svm": {"regression": CmaEsSampler, "classification": CmaEsSampler},
    "xgboost": {"regression": TPESampler, "classification": TPESampler},
    "mlp": {"regression": TPESampler, "classification": TPESampler},
    "lstm": {"regression": TPESampler, "classification": TPESampler},
}


def requires_scaling(model_type: str) -> bool:
    """
    Check if the model type requires feature scaling.

    Args:
        model_type (str): Type of model, e.g., "linear", "random_forest", "svm", "xgboost", "mlp", "lstm"

    Returns:
        bool: True if the model requires scaling, False otherwise.
    """
    return MODEL_REGISTRY.get(model_type, {}).get("requires_scaling", False)


def get_model_class(
    model_type: str, framework: str, task: str
) -> Type[BaseEstimator] | Type[nn.Module]:
    """
    Look up the model class from the registry.

    Args:
        model_type (str): Type of model, e.g., "linear", "random_forest", "svm", "xgboost", "mlp", "lstm"
        framework (str): Framework, e.g., "sklearn", "xgboost", "torch"
        task (str): Task type, e.g., "regression", "binary", "multiclass"

    Returns:
        Type[BaseEstimator] | Type[nn.Module]: The model class.
    """
    try:
        return MODEL_REGISTRY[model_type][framework][task]
    except KeyError:
        raise ValueError(f"No model found for {model_type=}, {framework=}, {task=}")


def get_sampler(model_type: str, seed: int = 42) -> BaseSampler:
    """
    Get the Optuna sampler based on the model type.

    Args:
        model_type (str): Type of model, e.g., "linear", "random_forest", "svm", "xgboost", "mlp", "lstm"
        seed (int): Random seed for the sampler. Default is 42.

    Returns:
        BaseSampler: An instance of an Optuna sampler.
    """
    if model_type == "svm":
        return CmaEsSampler(seed=seed, sigma0=1.0, warn_independent_sampling=False)
    elif model_type == "random_forest":
        return TPESampler(seed=seed, multivariate=True, constant_liar=True)
    elif model_type in ["xgboost", "linear"]:
        return TPESampler(seed=seed, multivariate=True, n_startup_trials=20)
    else:
        return TPESampler(seed=seed)

from optuna.samplers import BaseSampler, CmaEsSampler, TPESampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from torch import nn
from xgboost import XGBRegressor

from ..models.pytorch_models import LSTMModel, MLPModel

MODEL_REGISTRY = {
    "linear": {"sklearn": ElasticNet, "requires_scaling": True},
    "random_forest": {"sklearn": RandomForestRegressor},
    "svm": {"sklearn": SVR, "requires_scaling": True},
    "xgboost": {"sklearn": XGBRegressor},
    "mlp": {"torch": MLPModel, "requires_scaling": True},
    "lstm": {"torch": LSTMModel, "requires_scaling": True},
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
    model_type: str, framework: str
) -> type[BaseEstimator] | type[nn.Module]:
    """
    Look up the model class from the registry.

    Args:
        model_type (str): Type of model, e.g., "linear", "random_forest", "svm", "xgboost", "mlp", "lstm"
        framework (str): Framework, e.g., "sklearn", "xgboost", "torch"

    Returns:
        type[BaseEstimator] | type[nn.Module]: The model class.
    """
    try:
        return MODEL_REGISTRY[model_type][framework]
    except KeyError as exc:
        raise ValueError(f"No model found for {model_type=}, {framework=}") from exc


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

    if model_type == "random_forest":
        return TPESampler(seed=seed, multivariate=True, constant_liar=True)

    if model_type in {"xgboost", "linear"}:
        return TPESampler(seed=seed, multivariate=True, n_startup_trials=20)

    return TPESampler(seed=seed)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..core.registry import requires_scaling


def set_sklearn_pipeline(
    model_cls: Pipeline, params: dict, model_type: str
) -> Pipeline:
    """
    Returns a sklearn Pipeline with optional scaling.

    Parameters:
        model_cls (Pipeline): The sklearn model class to instantiate
        params (dict): Model hyperparameters
        model_type (str): One of ["linear", "random_forest", "svm", "xgboost"]

    Returns:
        Pipeline: sklearn pipeline with initialized model
    """
    steps = []
    if requires_scaling(model_type):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model_cls(**params)))
    return Pipeline(steps)

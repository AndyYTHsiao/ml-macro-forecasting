from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..core.registry import requires_scaling


def set_sklearn_pipeline(
    model_cls: Pipeline, params: dict, model_type: str, task: str = "regression"
) -> Pipeline:
    """
    Returns a sklearn Pipeline with optional scaling.

    Parameters:
        model_cls (Pipeline): The sklearn model class to instantiate
        params (dict): Model hyperparameters
        model_type (str): One of ["linear", "random_forest", "svm", "xgboost"]
        task (str): Type of task - "regression", "binary", or "multiclass".

    Returns:
        Pipeline: sklearn pipeline with initialized model
    """
    # For linear classification, enforce elastic-net penalty if specified
    if (task == "binary" or task == "multiclass") and model_type == "linear":
        if "penalty" not in params:
            params["penalty"] = "elasticnet"
        if "solver" not in params:
            params["solver"] = "saga"  # saga required for elasticnet penalty

    # Wrap with StandardScaler if required
    if requires_scaling(model_type):
        return Pipeline([("scaler", StandardScaler()), ("model", model_cls(**params))])
    else:
        return Pipeline([("model", model_cls(**params))])

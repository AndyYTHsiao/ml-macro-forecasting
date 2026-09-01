from typing import Any

import optuna


def _en_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of ElasticNet using Optuna.
    """
    return {
        "alpha": trial.suggest_float("alpha", 1e-2, 10, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.3, 1.0, log=True),
        "max_iter": 2000,
    }


def _rf_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of RandomForestRegressor using Optuna.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 600, step=100),
        "criterion": trial.suggest_categorical(
            "criterion", ["squared_error", "friedman_mse"]
        ),
        "max_depth": trial.suggest_int("max_depth", 4, 6),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 6),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5]
        ),
    }


def _svr_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of SVR using Optuna.
    """
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    config = {
        "C": trial.suggest_float("C", 0.05, 1.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-2, 0.8, log=True),
        "kernel": kernel,
    }
    if kernel == "rbf":
        config["gamma"] = trial.suggest_float("gamma", 1e-2, 2.0, log=True)
    return config


def _xgb_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of XGBRegressor using Optuna.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "lambda": trial.suggest_float("lambda", 0.1, 50.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "early_stopping_rounds": 10,
    }


def _mlp_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of MLPModel using Optuna.
    """
    n_layers = trial.suggest_int("n_layers", 1, 2)
    hidden_size = trial.suggest_categorical("hidden_size", [4, 8, 16, 32])
    hidden_sizes = [hidden_size] * n_layers
    return {
        "hidden_sizes": hidden_sizes,
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def _lstm_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of LSTMModel using Optuna.
    Ensures dropout = 0.0 when num_layers = 1.
    """
    num_layers = trial.suggest_int("num_layers", 1, 2)

    # Only sample dropout if num_layers > 1
    dropout = (
        0.0 if num_layers == 1 else trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    )

    return {
        "num_layers": num_layers,
        "hidden_size": trial.suggest_categorical("hidden_size", [4, 8, 16, 32]),
        "lr": trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2]),
        "dropout": dropout,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


TUNING_CONFIG = {
    "linear": _en_config,
    "random_forest": _rf_config,
    "xgboost": _xgb_config,
    "svm": _svr_config,
    "mlp": _mlp_config,
    "lstm": _lstm_config,
}

import optuna
from typing import Any


def _en_reg_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of ElasticNet using Optuna.
    """
    return {
        "alpha": trial.suggest_float("alpha", 1e-2, 10, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.3, 1.0, log=True),
        "max_iter": 2000,
    }


def _logistic_clf_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of LogisticRegression using Optuna.
    """
    penalty = trial.suggest_categorical("penalty", [None, "l1", "l2", "elasticnet"])
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "penalty": penalty,
        "solver": trial.suggest_categorical(
            "solver", ["lbfgs", "saga"]
        ),  # saga required for elasticnet penalty
        "l1_ratio": (
            trial.suggest_float("l1_ratio", 0.0, 1.0)
            if penalty == "elasticnet"
            else None
        ),
        "max_iter": trial.suggest_categorical("max_iter", [100, 200, 300]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _rf_reg_config(trial: optuna.Trial) -> dict[str, Any]:
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


def _rf_clf_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of RandomForestClassifier using Optuna.
    """
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 150, 200]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 1, 3, 5, 7]),
        "max_features": trial.suggest_categorical(
            "max_features", [1, "sqrt", "log2", None]
        ),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
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


def _svc_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of SVC using Optuna.
    """
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def _xgb_reg_config(trial: optuna.Trial) -> dict[str, Any]:
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


def _xgb_clf_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of XGBClassifier using Optuna.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "early_stopping_rounds": trial.suggest_int(
            "early_stopping_rounds", 5, 10, step=5
        ),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "lambda": trial.suggest_float("lambda", 1e-2, 10.0, log=True),
        "use_label_encoder": False,
    }


def _mlp_reg_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of MLPModel using Optuna.
    """
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    hidden_sizes = [hidden_size] * n_layers
    return {
        "hidden_sizes": hidden_sizes,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def _mlp_clf_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Hyperparameter search space for MLP classification tasks.
    """
    n_layers = trial.suggest_int("n_layers", 1, 3)
    return {
        "hidden_sizes": [
            trial.suggest_categorical(f"hidden_size_{i}", [32, 64, 128])
            for i in range(n_layers)
        ],
        "lr": trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def _lstm_reg_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Configuration for tuning hyperparameters of LSTMModel using Optuna.
    Ensures dropout = 0.0 when num_layers = 1.
    """
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # Only sample dropout if num_layers > 1
    dropout = (
        0.0 if num_layers == 1 else trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
    )

    return {
        "num_layers": num_layers,
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "lr": trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2]),
        "dropout": dropout,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def _lstm_clf_config(trial: optuna.Trial) -> dict[str, Any]:
    """
    Hyperparameter search space for LSTM classification tasks.
    """
    return {
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "lr": trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


TUNING_CONFIG = {
    "linear": {"regression": _en_reg_config, "classification": _logistic_clf_config},
    "random_forest": {"regression": _rf_reg_config, "classification": _rf_clf_config},
    "xgboost": {"regression": _xgb_reg_config, "classification": _xgb_clf_config},
    "svm": {"regression": _svr_config, "classification": _svc_config},
    "mlp": {"regression": _mlp_reg_config, "classification": _mlp_clf_config},
    "lstm": {"regression": _lstm_reg_config, "classification": _lstm_clf_config},
}

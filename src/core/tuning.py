from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch import nn, optim
from torch.utils.data import DataLoader

from ..dataprep.preprocessing import set_up_dataloader
from ..models.pytorch_models import set_pytorch_model
from ..models.sklearn_models import set_sklearn_pipeline
from .callback import TrainingCallback
from .registry import get_model_class, get_sampler
from .training import training_loop
from .utils import calculate_metrics
from ..core.registry import requires_scaling


class Tuner:
    def __init__(
        self,
        model_type: str,
        framework: str = "sklearn",
        input_size: int | None = None,
        output_size: int | None = None,
        tuning_config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the Tuner with the model type and tuning configuration.

        Args:
            model_type (str): Type of model to tune (e.g., "linear", "random_forest", "xgboost", "svm", "mlp", "lstm").
            framework (str): Framework to use ("sklearn" or "torch").
            tuning_config (dict[str, Any] | None): Custom tuning configuration. If None, uses the default TUNING_CONFIG.
            input_size (int | None): Input feature size for PyTorch models. If None, inferred during training.
            output_size (int | None): Output size for PyTorch models. If None, inferred during training.
            device (torch.device | None): Device to use for PyTorch models. If None, uses CUDA if available, else CPU.
        """
        self.model_type = model_type
        self.framework = framework
        if tuning_config is None:
            from .search_spaces import TUNING_CONFIG

            self.tuning_config = TUNING_CONFIG
        else:
            self.tuning_config = tuning_config

        self.model_cls = get_model_class(self.model_type, self.framework)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _get_tuning_config(self, trial: optuna.Trial) -> dict[str, Any]:
        try:
            config = self.tuning_config[self.model_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Available types: {list(self.tuning_config)}"
            ) from exc

        return config(trial)

    def _infer_input_output_size(self, train_loader: DataLoader) -> None:
        if self.input_size is None or self.output_size is None:
            X_batch, y_batch = next(iter(train_loader))
            if self.input_size is None:
                self.input_size = X_batch.shape[-1]
            if self.output_size is None:
                self.output_size = 1 if y_batch.ndim == 1 else y_batch.shape[-1]

    def _sklearn_objective(
        self,
        params: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv_splitter: KFold | TimeSeriesSplit,
        scoring: str = "neg_mean_squared_error",
        cv_splits: int = 5,
    ) -> float:
        """
        Define the objective function for sklearn models using time series cross-validation.

        Args:
            params (dict): Hyperparameters for the model.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target values.
            model_type (str): Type of model to tune.
            cv_splitter (KFold | TimeSeriesSplit): Cross-validation splitter.
            scoring (str): Scoring metric for cross-validation.
            cv_splits (int): Number of cross-validation splits.

        Returns:
            float: The mean cross-validation score to minimize.
        """
        model = set_sklearn_pipeline(
            model_cls=self.model_cls,
            params=params,
            model_type=self.model_type,
        )
        cv = cv_splitter(n_splits=cv_splits)
        if "early_stopping_rounds" in params and hasattr(
            model.named_steps["model"], "fit"
        ):
            return self._manual_cv_with_early_stopping(
                model=model, X_train=X_train, y_train=y_train, cv=cv
            )

        val_scores_mean = cross_val_score(
            model, X_train, y_train, cv=cv, n_jobs=-1, scoring=scoring
        )
        return -np.mean(val_scores_mean)

    def _manual_cv_with_early_stopping(
        self,
        model: Pipeline,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: KFold | TimeSeriesSplit,
    ) -> float:
        """
        Custom CV loop for models that require early stopping (e.g., XGBoost).

        Args:
            model (Pipeline): sklearn pipeline
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target values
            cv (KFold | TimeSeriesSplit): Cross-validation splitter
        """
        val_scores = []
        for train_idx, val_idx in cv.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            model.fit(
                X_train_fold,
                y_train_fold,
                model__eval_set=[(X_val_fold, y_val_fold)],
                model__verbose=False,
            )

            preds = model.predict(X_val_fold)
            val_scores.append(calculate_metrics(y_val_fold, preds)["mse"])

        return np.mean(val_scores)

    def _pytorch_objective(
        self,
        params: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimizer_cls: Callable[[nn.Module], optim.Optimizer] | None = None,
        cv_splitter: KFold | TimeSeriesSplit | None = None,
        n_epochs: int = 50,
        cv_splits: int = 5,
        patience: int = 5,
        min_epochs: int = 1,
        min_delta: float = 0.0,
    ) -> float:
        """
        Tune hyperparameters for a PyTorch MLP model using Optuna with time series cross-validation.

        Args:
            mlp_model (MLPModel): The MLP model class to be tuned.
            X_train (torch.Tensor): Training features as a PyTorch tensor.
            y_train (torch.Tensor): Training target values as a PyTorch tensor.
            callback (TrainingCallback | None): Optional callback for early stopping.
            batch_size (int): Batch size for training.
            n_epochs (int): Number of epochs for training.
            n_trials (int): Number of Optuna trials.
            direction (str): Direction of optimization ('minimize' or 'maximize').

        Returns:
            float: The best objective value.
        """
        if cv_splitter is None:
            cv = TimeSeriesSplit(n_splits=cv_splits)
        else:
            cv = cv_splitter

        loss_per_fold = []
        batch_size = params.get("batch_size", 32)
        lr = params.get("lr", 1e-3)
        loss_fn = nn.MSELoss()
        for train_idx, val_idx in cv.split(X_train):
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            if requires_scaling(self.model_type):
                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_fold)

                X_val_fold = scaler.transform(X_val_fold)

            train_loader, val_loader = set_up_dataloader(
                X_train_tensor=torch.tensor(X_train_fold, dtype=torch.float32),
                y_train_tensor=torch.tensor(y_train_fold, dtype=torch.float32),
                X_test_tensor=torch.tensor(X_val_fold, dtype=torch.float32),
                y_test_tensor=torch.tensor(y_val_fold, dtype=torch.float32),
                batch_size=batch_size,
                device=self.device,
            )
            # --- Training ---
            self._infer_input_output_size(train_loader)

            model_params = {
                k: params[k] for k in params if k not in {"lr", "batch_size"}
            }
            self.model = set_pytorch_model(
                self.model_cls,
                model_params,
                input_size=self.input_size,
                output_size=self.output_size,
                model_type=self.model_type,
            ).to(self.device)
            optimizer = (optimizer_cls or optim.Adam)(self.model.parameters(), lr=lr)
            callback = TrainingCallback(
                model=self.model,
                patience=patience,
                min_epochs=min_epochs,
                min_delta=min_delta,
            )
            loss = training_loop(
                self.model,
                optimizer,
                loss_fn,
                self.device,
                train_loader,
                val_loader=val_loader,
                callback=callback,
                n_epochs=n_epochs,
                return_preds=False,
            )
            loss_per_fold.append(loss)

        return np.mean(loss_per_fold)

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs,
    ) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial (optuna.Trial): An Optuna trial object.
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target values.
            **kwargs: Additional arguments for the training function.

        Returns:
            float: The objective value to minimize.
        """
        params = self._get_tuning_config(trial)
        if self.framework == "sklearn":
            return self._sklearn_objective(
                params,
                X_train,
                y_train,
                cv_splitter=kwargs.get("cv_splitter", TimeSeriesSplit),
                scoring=kwargs.get("scoring", "neg_mean_squared_error"),
                cv_splits=kwargs.get("cv_splits", 5),
            )
        elif self.framework == "torch":
            return self._pytorch_objective(
                params,
                X_train,
                y_train,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported framework: {self.framework}. Choose 'sklearn' or 'torch'."
            )

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 50,
        seed: int = 42,
        direction: str = "minimize",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training target values.
            n_trials (int): Number of Optuna trials.
            direction (str): Direction of optimization ('minimize' or 'maximize').
            seed (int | None): Random seed for reproducibility.
            **kwargs: Additional arguments for the objective function.

        Returns:
            dict[str, Any]: The best hyperparameters found by Optuna.
        """
        study = optuna.create_study(
            direction=direction,
            sampler=get_sampler(self.model_type, seed=seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )

        # NOTE: Setting n_jobs > 1 makes the optimization non-deterministic
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, **kwargs),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        return study.best_params

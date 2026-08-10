from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ..models.pytorch_models import set_pytorch_model
from ..models.sklearn_models import set_sklearn_pipeline
from .callback import TrainingCallback
from .registry import get_model_class
from .utils import (
    calculate_metrics,
    organize_in_out_sample_metrics,
    save_best_model,
    save_json,
    save_predictions,
)


class Trainer:
    def __init__(
        self,
        model_type: str,
        params: dict,
        framework: str = "sklearn",
        input_size: int | None = None,
        output_size: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Hybrid Trainer that supports sklearn pipelines and PyTorch models.

        Args:
            model_type (str): Type of model to initialize.
            params (dict): Dictionary containing model parameters.
            framework (str): Framework to use - "sklearn" or "torch".
            input_size (Optional[int]): Number of input features (required for PyTorch models).
            output_size (Optional[int]): Number of output features (required for PyTorch models).
            device (Optional[torch.device]): Device to run the training on (CPU or GPU).
        """
        self.model_type = model_type
        self.params = params
        self.framework = framework
        if self.framework not in {"sklearn", "torch"}:
            raise ValueError(f"Unsupported framework: {self.framework}")

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_cls = get_model_class(self.model_type, self.framework)
        self.model = None
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = None
        self.loss_fn = None

        # Outputs
        self.train_preds = None
        self.test_preds = None
        self.metrics = None

    def _infer_input_output_size(self, train_loader: DataLoader) -> None:
        """
        Infer input and output size from the training DataLoader.

        Args:
            train_loader (DataLoader): DataLoader for training data.
        """
        if self.input_size is None or self.output_size is None:
            X_batch, y_batch = next(iter(train_loader))
            if self.input_size is None:
                self.input_size = X_batch.shape[-1]
            if self.output_size is None:
                self.output_size = 1 if y_batch.ndim == 1 else y_batch.shape[-1]

    def _train_sklearn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """
        Train sklearn models (with optional early stopping for e.g. XGBoost).

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
        """
        # Early stopping is used inside temporal CV during tuning.  The final
        # estimator is fitted on all training observations without consulting
        # the test set.
        final_params = {
            key: value
            for key, value in self.params.items()
            if key != "early_stopping_rounds"
        }
        self.model = set_sklearn_pipeline(self.model_cls, final_params, self.model_type)
        self.model.fit(X_train, y_train)

    def _train_torch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer_cls: Callable[[nn.Module], optim.Optimizer] | None = None,
        loss_fn: nn.Module | None = None,
        n_epochs: int = 50,
        min_epochs: int = 1,
        patience: int = 5,
        min_delta: float = 0.0,
        logger_name: str = "trainer",
        metric_mode: str = "min",
        enable_early_stopping: bool = True,
        enable_logging: bool = False,
    ) -> None:
        """
        Train PyTorch models.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader | None): DataLoader for validation data. If None, no validation is performed.
            callback (TrainingCallback | None): Callback for early stopping and logging.
            n_epochs (int): Maximum number of epochs to train.
            min_epochs (int): Minimum number of epochs before early stopping can occur.
            device (torch.device): Device to run the training on (CPU or GPU).
            return_preds (bool): If True, return predictions on the validation set.
        """
        # Wrap into DataLoaders
        self.model = set_pytorch_model(
            self.model_cls,
            self.params,
            self.input_size,
            self.output_size,
            self.model_type,
        ).to(self.device)
        self.optimizer = (optimizer_cls or optim.Adam)(
            self.model.parameters(), lr=self.params.get("lr", 1e-3)
        )
        self.loss_fn = loss_fn or nn.MSELoss()
        self.callback = TrainingCallback(
            self.model,
            patience=patience,
            min_epochs=min_epochs,
            min_delta=min_delta,
            logger_name=logger_name,
            metric_mode=metric_mode,
            enable_early_stopping=enable_early_stopping,
            enable_logging=enable_logging,
        )
        _ = training_loop(
            self.model,
            self.optimizer,
            self.loss_fn,
            self.device,
            train_loader,
            val_loader,
            self.callback,
            n_epochs=n_epochs,
            return_preds=False,
        )

    def train(
        self,
        train_data: tuple[np.ndarray, np.ndarray] | DataLoader,
        validation_data: tuple[np.ndarray, np.ndarray] | DataLoader | None = None,
        optimizer_cls: Callable[[nn.Module], optim.Optimizer] | None = None,
        loss_fn: nn.Module | None = None,
        min_delta: float = 0.0,
        n_epochs: int = 50,
        patience: int = 5,
        min_epochs: int = 1,
        logger_name: str = "trainer",
        metric_mode: str = "min",
        enable_early_stopping: bool = True,
        enable_logging: bool = False,
    ) -> None:
        """
        Train the model based on the specified framework.

        Args:
            train_data (tuple[np.ndarray, np.ndarray] | DataLoader): Training data.
            validation_data: Optional validation data for PyTorch early stopping. This must not be the held-out test set.
            optimizer_cls (Optional[Callable[[nn.Module], optim.Optimizer]]): Optimizer class for PyTorch models.
            loss_fn (Optional[nn.Module]): Loss function for PyTorch models.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            n_epochs (int): Maximum number of epochs to train.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_epochs (int): Minimum number of epochs to train before considering early stopping.
            logger_name (str): Name for the logger.
            metric_mode (str): One of {"min", "max"}. In "min" mode, training will stop when the quantity monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing.
            enable_early_stopping (bool): Whether to enable early stopping.
            enable_logging (bool): Whether to enable logging during training.
        """
        if self.framework == "sklearn":
            self._train_sklearn(train_data[0], train_data[1])
        elif self.framework == "torch":
            self._infer_input_output_size(train_data)
            self._train_torch(
                train_data,
                validation_data,
                optimizer_cls=optimizer_cls,
                loss_fn=loss_fn,
                n_epochs=n_epochs,
                min_epochs=min_epochs,
                patience=patience,
                min_delta=min_delta,
                logger_name=logger_name,
                metric_mode=metric_mode,
                enable_early_stopping=enable_early_stopping,
                enable_logging=enable_logging,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rounding: int | None = None,
    ) -> None:
        """
        Generate in-sample and out-of-sample predictions based on task type.
        """
        if self.framework == "sklearn":
            self.train_preds = self.model.predict(X_train)
            self.test_preds = self.model.predict(X_test)
        else:
            self.model.eval()
            with torch.no_grad():
                train_tensor = torch.tensor(
                    X_train, dtype=torch.float32, device=self.device
                )
                test_tensor = torch.tensor(
                    X_test, dtype=torch.float32, device=self.device
                )

                self.train_preds = self.model(train_tensor).squeeze(-1).cpu().numpy()
                self.test_preds = self.model(test_tensor).squeeze(-1).cpu().numpy()

        # calculate metrics
        insample_metrics = calculate_metrics(
            y_train, self.train_preds, rounding=rounding
        )
        outsample_metrics = calculate_metrics(
            y_test, self.test_preds, rounding=rounding
        )
        self.metrics = organize_in_out_sample_metrics(
            insample_metrics, outsample_metrics
        )

    def save_results(
        self,
        output_dir: Path,
        suffix: str | None = None,
    ) -> None:
        """
        Save the best model, predictions, configuration, and metrics to the experiment directory.

        Args:
            output_dir (Path): The directory where the experiment outputs will be saved.
            suffix (Optional[str]): Suffix to append to filenames.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix_str = f"_{suffix}" if suffix else ""
        base_name = f"{self.model_type}{suffix_str}"
        save_best_model(self.model, output_dir / base_name)
        save_predictions(self.test_preds, output_dir / f"{base_name}_preds.npy")
        save_json(self.params, output_dir / f"{base_name}_config.json")
        save_json(self.metrics, output_dir / f"{base_name}_results.json")


def training_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    callback: TrainingCallback | None = None,
    n_epochs: int = 50,
    return_preds: bool = False,
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """
    General training loop that works for both CV (with validation) and final training.
    If val_loader is None, no validation is performed (only train loss is tracked).

    Args:
        model (nn.Module): The PyTorch model to train.
        optimizer (optim.Optimizer): The optimizer for training.
        loss_fn (nn.Module): The loss function.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader | None): DataLoader for validation data. If None, no validation is performed.
        callback (TrainingCallback | None): Callback for early stopping and logging.
        n_epochs (int): Maximum number of epochs to train.
        device (torch.device): Device to run the training on (CPU or GPU).
        return_preds (bool): If True, return predictions on the validation set.

    Returns:
        float | tuple[float, np.ndarray, np.ndarray]: Best validation loss (and predictions and targets if return_preds is True).
    """
    best_loss = float("inf")

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        n_train = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            batch_n = x_batch.size(0)

            optimizer.zero_grad()
            outputs = model(x_batch).squeeze(-1)

            assert outputs.shape == y_batch.shape, (
                f"{outputs.shape=} != {y_batch.shape=}"
            )

            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_n
            n_train += batch_n

        train_loss /= n_train

        # --- Validation ---
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val = 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    batch_n = x_batch.size(0)

                    outputs = model(x_batch).squeeze(-1)
                    assert outputs.shape == y_batch.shape, (
                        f"{outputs.shape=} != {y_batch.shape=}"
                    )

                    loss = loss_fn(outputs, y_batch)
                    val_loss += loss.item() * batch_n
                    n_val += batch_n

            val_loss /= n_val

            # --- Early stopping on validation loss ---
            if callback and callback.should_stop(epoch, val_loss):
                break

        else:
            # --- Early stopping on training loss ---
            if callback and callback.should_stop(epoch, train_loss):
                break

    # --- Restore best model ---
    if callback and callback.get_best_model_state_dict is not None:
        callback.restore_best_model()
        best_loss = callback.get_best_score
    else:
        best_loss = val_loss if val_loader is not None else train_loss

    # --- Return predictions from best model ---
    if return_preds and val_loader is not None:
        model.eval()

        preds = []
        targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(x_batch).squeeze(-1)

                preds.append(outputs.cpu())
                targets.append(y_batch.cpu())

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()

        return best_loss, preds, targets

    return best_loss

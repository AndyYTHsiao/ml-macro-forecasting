import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional
from torch.utils.data import DataLoader
from .utils import (
    save_best_model,
    save_json,
    save_predictions,
    calculate_metrics,
    organize_in_out_sample_metrics,
)
from .callback import TrainingCallback
from .registry import get_model_class
from ..models.sklearn_models import set_sklearn_pipeline
from ..models.pytorch_models import set_pytorch_model, set_loss_fn, logit_to_output


class Trainer:
    def __init__(
        self,
        model_type: str,
        params: dict,
        framework: str = "sklearn",
        task: str = "regression",
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Hybrid Trainer that supports sklearn pipelines and PyTorch models.

        Args:
            model_type (str): Type of model to initialize.
            params (dict): Dictionary containing model parameters.
            framework (str): Framework to use - "sklearn" or "torch".
            task (str): Type of task - "regression", "binary", or "multiclass".
            input_size (Optional[int]): Number of input features (required for PyTorch models).
            output_size (Optional[int]): Number of output features (required for PyTorch models).
            device (Optional[torch.device]): Device to run the training on (CPU or GPU).
        """
        self.model_type = model_type
        self.params = params
        self.framework = framework
        if self.framework not in {"sklearn", "torch"}:
            raise ValueError(f"Unsupported framework: {self.framework}")

        self.task = task
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_cls = get_model_class(self.model_type, self.framework, self.task)
        self.model = None
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = None
        self.loss_fn = None

        # Outputs
        self.train_preds = None
        self.test_preds = None
        self.train_classes = None
        self.test_classes = None
        self.train_probs = None
        self.test_probs = None
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
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        Train sklearn models (with optional early stopping for e.g. XGBoost).

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training targets.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing targets.
        """
        self.model = set_sklearn_pipeline(
            self.model_cls, self.params, self.model_type, self.task
        )

        if "early_stopping_rounds" in self.params and hasattr(
            self.model.named_steps["model"], "fit"
        ):
            self.model.fit(
                X_train,
                y_train,
                model__eval_set=[(X_test, y_test)],
                model__verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

    def _train_torch(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer_cls: Optional[Callable[[nn.Module], optim.Optimizer]] = None,
        loss_fn: Optional[nn.Module] = None,
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
            self.task,
        )
        self.optimizer = (optimizer_cls or optim.Adam)(
            self.model.parameters(), lr=self.params.get("lr", 1e-3)
        )
        self.loss_fn = loss_fn or set_loss_fn(task=self.task)
        self.callback = TrainingCallback(
            self.model,
            patience=patience,
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
            train_loader,
            test_loader,
            self.callback,
            n_epochs=n_epochs,
            min_epochs=min_epochs,
            device=self.device,
            return_preds=False,
        )

    def train(
        self,
        train_data: tuple[np.ndarray, np.ndarray] | DataLoader,
        test_data: tuple[np.ndarray, np.ndarray] | DataLoader,
        optimizer_cls: Optional[Callable[[nn.Module], optim.Optimizer]] = None,
        loss_fn: Optional[nn.Module] = None,
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
            test_data (tuple[np.ndarray, np.ndarray] | DataLoader): Testing data.
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
            self._train_sklearn(
                train_data[0], train_data[1], test_data[0], test_data[1]
            )
        elif self.framework == "torch":
            self._infer_input_output_size(train_data)
            self._train_torch(
                train_data,
                test_data,
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
        rounding: Optional[int] = None,
    ) -> None:
        """
        Generate in-sample and out-of-sample predictions based on task type.
        """
        if self.framework == "sklearn":
            self.train_preds = self.model.predict(X_train)
            self.test_preds = self.model.predict(X_test)
        elif self.framework == "torch":
            self.model.eval()
            with torch.no_grad():
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(
                    self.device
                )
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(
                    self.device
                )

                train_output = self.model(X_train_tensor)
                test_output = self.model(X_test_tensor)

                if self.task == "regression":
                    self.train_preds = train_output.squeeze().cpu().numpy()
                    self.test_preds = test_output.squeeze().cpu().numpy()

                elif self.task == "binary":
                    # Assume model outputs raw logits -> apply sigmoid
                    train_probs = torch.sigmoid(train_output)
                    test_probs = torch.sigmoid(test_output)

                    self.train_preds = train_probs.squeeze().cpu().numpy()
                    self.test_preds = test_probs.squeeze().cpu().numpy()

                    # Optionally also store hard class predictions (0 or 1)
                    self.train_classes = (self.train_preds >= 0.5).astype(int)
                    self.test_classes = (self.test_preds >= 0.5).astype(int)

                elif self.task == "multiclass":
                    # Assume model outputs raw logits -> apply softmax
                    train_probs = torch.softmax(train_output, dim=1)
                    test_probs = torch.softmax(test_output, dim=1)

                    self.train_preds = train_probs.cpu().numpy()
                    self.test_preds = test_probs.cpu().numpy()

                    # Optionally also store predicted class labels
                    self.train_classes = train_probs.argmax(dim=1).cpu().numpy()
                    self.test_classes = test_probs.argmax(dim=1).cpu().numpy()
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

        # calculate metrics
        insample_metrics = calculate_metrics(
            y_train, self.train_preds, rounding=rounding, task=self.task
        )
        outsample_metrics = calculate_metrics(
            y_test, self.test_preds, rounding=rounding, task=self.task
        )
        self.metrics = organize_in_out_sample_metrics(
            insample_metrics, outsample_metrics
        )

    def save_results(
        self,
        output_dir: Path,
        suffix: Optional[str] = None,
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
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    callback: TrainingCallback | None = None,
    n_epochs: int = 50,
    min_epochs: int = 1,
    device: torch.device = torch.device("cpu"),
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
        min_epochs (int): Minimum number of epochs before early stopping can occur.
        device (torch.device): Device to run the training on (CPU or GPU).
        return_preds (bool): If True, return predictions on the validation set.

    Returns:
        float | tuple[float, np.ndarray, np.ndarray]: Best validation loss (and predictions and targets if return_preds is True).
    """
    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            outputs = logit_to_output(outputs, task=model.task)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        val_loss = None
        fold_preds, fold_targets = [], []
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    outputs = logit_to_output(outputs, task=model.task)
                    val_loss_batch = loss_fn(outputs, y_batch).item()
                    val_loss += val_loss_batch
                    if return_preds:
                        fold_preds.append(outputs.detach().cpu())
                        fold_targets.append(y_batch.detach().cpu())
            val_loss /= len(val_loader)

            # --- Track best model by validation loss ---
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())

            # --- Early stopping on validation loss ---
            if (
                callback
                and epoch >= min_epochs
                and callback.should_stop(epoch, val_loss)
            ):
                break

        else:
            # --- No validation: track best model by training loss ---
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = deepcopy(model.state_dict())

            # --- Early stopping on training loss ---
            if (
                callback
                and epoch >= min_epochs
                and callback.should_stop(epoch, train_loss)
            ):
                break

    # --- Restore best model ---
    if best_state is not None:
        model.load_state_dict(best_state)
        if callback:
            callback.restore_best_model()

    # --- Return ---
    if return_preds and val_loader is not None:
        preds = torch.cat(fold_preds).numpy()
        targets = torch.cat(fold_targets).numpy()
        return best_loss, preds, targets
    else:
        return best_loss

import logging
import os
from copy import deepcopy

from torch import nn


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_epochs: int = 1,
        min_delta: float = 0.0,
        logger_name: str = "training",
        metric_mode: str = "min",
        enable_early_stopping: bool = True,
        enable_logging: bool = False,
    ):
        """
        Initialize the early stopping mechanism.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_epochs (int): Minimum number of epochs to run before considering early stopping.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            logger_name (str): Name of the log file if logging is enabled.
            metric_mode (str): Mode for the monitored metric, either 'min' or 'max'.
            enable_early_stopping (bool): Whether to enable early stopping.
            enable_logging (bool): Whether to log messages during training.
        """
        if patience <= 0:
            raise ValueError("patience must be positive.")

        if min_epochs < 1:
            raise ValueError("min_epochs must be at least 1.")

        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")

        if metric_mode not in ["min", "max"]:
            raise ValueError("metric_mode must be either 'min' or 'max'.")

        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.metric_mode = metric_mode
        self.counter = 0
        self.enable_early_stopping = enable_early_stopping

        # Check if logging is required
        self.logger = get_logger(logger_name=logger_name) if enable_logging else None

        # Initialize best values
        self.best_epoch = None
        self.best_score = float("inf") if metric_mode == "min" else float("-inf")
        self.best_model_state_dict = None

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_epoch = None
        self.best_model_state_dict = None
        self.best_score = float("inf") if self.metric_mode == "min" else float("-inf")

    def __call__(self, epoch: int, score: float, model: nn.Module) -> bool:
        """
        Check if the training should be stopped based on the current score.

        Args:
            epoch (int): Current epoch number.
            score (float): The current validation score to evaluate.
            model (nn.Module): The model being trained.

        Returns:
            should_stop (bool): True if training should stop, False otherwise.
        """
        # Skip early stopping if not enabled
        if not self.enable_early_stopping:
            if self.logger:
                self.logger.info(
                    f"Early stopping is disabled. Epoch {epoch + 1} | score: {score:.4f}"
                )
            return False

        score_improved = (
            score < self.best_score - self.min_delta
            if self.metric_mode == "min"
            else score > self.best_score + self.min_delta
        )

        # Reset the counter and save the best model, loss, and epoch if the loss improved
        if score_improved:
            self.counter = 0
            self.best_score = score
            self.best_epoch = epoch
            self.best_model_state_dict = deepcopy(model.state_dict())

            if self.logger:
                self.logger.info(
                    f"Epoch {epoch + 1} | Score improved | score: {score:.4f} | Counter reset."
                )

        # Counter is not incremented if minimum epochs have not been reached
        elif epoch + 1 < self.min_epochs:
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch + 1} | No improvement | score: {score:.4f} | Minimum epochs not reached."
                )

        else:
            self.counter += 1
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch + 1} | No improvement | score: {score:.4f} | Counter: {self.counter}/{self.patience}"
                )

        # Early stopping is triggered if the counter exceeds the patience
        if self.counter >= self.patience:
            if self.logger:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. Best score: {self.best_score:.4f} at epoch {self.best_epoch + 1}."
                )
            return True

        return False


class TrainingCallback:
    def __init__(
        self,
        model: nn.Module,
        patience: int = 5,
        min_epochs: int = 1,
        min_delta: float = 0.0,
        logger_name: str = "training",
        metric_mode: str = "min",
        enable_early_stopping: bool = True,
        enable_logging: bool = False,
    ):
        """
        Initialize the TrainingCallback with a model and early stopping parameters.

        Args:
            model: The model to be trained.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_epochs (int): Minimum number of epochs to run before considering early stopping.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            logger_name (str): Name of the log file if logging is enabled.
            metric_mode (str): Mode for the monitored metric, either 'min' or 'max'.
            enable_early_stopping (bool): Whether to enable early stopping.
            enable_logging (bool): Whether to log messages during training.
        """
        self.model = model
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_epochs=min_epochs,
            min_delta=min_delta,
            logger_name=logger_name,
            metric_mode=metric_mode,
            enable_early_stopping=enable_early_stopping,
            enable_logging=enable_logging,
        )

    def should_stop(self, epoch: int, score: float) -> bool:
        """
        Check if training should continue or stop based on the current score.

        Args:
            epoch (int): Current epoch number.
            score (float): The current score value to evaluate.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        return self.early_stopping(epoch, score, self.model)

    @property
    def get_best_model_state_dict(self) -> dict | None:
        """
        Get the best model state based on early stopping criteria.

        Returns:
            dict: State dict of the best model.
        """
        return self.early_stopping.best_model_state_dict

    @property
    def get_best_epoch(self) -> int | None:
        """
        Get the epoch at which the best model state was saved.

        Returns:
            int: The epoch number of the best model state.
        """
        return self.early_stopping.best_epoch

    @property
    def get_best_score(self) -> float:
        """
        Get the best score value recorded during training.

        Returns:
            float: The best score value.
        """
        return self.early_stopping.best_score

    def restore_best_model(self) -> None:
        """
        Restore the model to the best state recorded during training.
        This method updates the model's state dict to the best one found.
        """
        if self.early_stopping.best_model_state_dict is None:
            raise RuntimeError("No best model has been recorded yet.")

        self.model.load_state_dict(self.early_stopping.best_model_state_dict)

    def log(self, msg: str) -> None:
        """
        Log messages using the logger associated with the early stopping callback.
        This method is used only when logging is enabled.

        Args:
            msg (str): The message to log.

        Raises:
            ValueError: If logging is not enabled for this callback.
        """
        logger = self.early_stopping.logger

        if not logger:
            raise ValueError("Logging is not enabled for this callback.")

        logger.info(msg)

    def enable_logging(self, enable: bool, logger_name: str = "training") -> None:
        self.early_stopping.logger = get_logger(logger_name) if enable else None

    def set_early_stopping_enabled(self, enabled: bool = True) -> None:
        """Enable or disable early stopping."""
        self.early_stopping.enable_early_stopping = enabled

    def reset(self, model: nn.Module = None) -> None:
        """
        Reset the early stopping counter and best score.
        Optionally update the model to be tracked.

        Args:
            model (nn.Module, optional): New model to track. If not provided, the existing model is used.
        """
        if model is not None:
            self.model = model

        self.early_stopping.reset()


def get_logger(
    logger_name: str = __name__,
    log_dir: str = "./logs",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    file_handler_mode: str = "w",
    level: int = logging.INFO,
    add_console_handler: bool = False,
    add_file_handler: bool = True,
) -> logging.Logger:
    # Create log directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{logger_name}.log")

    # Create or get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False  # prevent duplicate logs from root logger

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    if add_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    # File handler
    if add_file_handler:
        file_handler = logging.FileHandler(log_path, mode=file_handler_mode)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger

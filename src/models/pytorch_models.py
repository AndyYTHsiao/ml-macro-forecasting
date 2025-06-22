import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        task: str = "regression",
    ) -> None:
        """
        A simple Multi-Layer Perceptron (MLP) model.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list[int]): List of hidden layer sizes.
            output_size (int): Number of output features.
            task (str): Type of task - "regression", "binary", or "multiclass".
        """
        super().__init__()
        assert task in {"regression", "binary", "multiclass"}, "Invalid task type"
        self.task = task

        layers = []
        for hidden_size in hidden_sizes:
            fc = nn.Linear(input_size, hidden_size)
            nn.init.xavier_uniform_(fc.weight)
            layers.append(fc)
            layers.append(nn.ReLU())
            input_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(input_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor after applying the model and activation function based on the task.
        """
        return self.model(x)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        task: str = "regression",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        """
        A Long Short-Term Memory (LSTM) model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            output_size (int): Number of output features.
            task (str): Type of task - "regression", "binary", or "multiclass".
            dropout (float): Dropout probability for the LSTM layers.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
        """
        super().__init__()
        assert task in {"regression", "binary", "multiclass"}, "Invalid task type"
        self.task = task
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        fc_input_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            torch.Tensor: Output tensor after applying the model and activation function based on the task.
        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out[:, -1, :])  # (B, output_size)

        return logits


def logit_to_output(logits: torch.Tensor, task: str) -> torch.Tensor:
    """
    Convert logits to final output based on the task.

    Args:
        logits (torch.Tensor): Raw output from the model.
        task (str): Type of task - "regression", "binary", or "multiclass".

    Returns:
        torch.Tensor: Final output after applying the appropriate activation function.
    """
    if task == "binary":
        outputs = torch.sigmoid(logits)
    elif task == "multiclass":
        outputs = F.softmax(logits, dim=1)
    else:
        outputs = logits.squeeze(-1)

    return outputs


def validate_params(params: dict, required: set[str]) -> None:
    """
    Validate that all required hyperparameters are present in the params dictionary.

    Args:
        params (dict): Dictionary of model parameters.
        required (set[str]): Set of required parameter keys.
    """
    missing = required - params.keys()
    if missing:
        raise ValueError(f"Missing hyperparams: {missing}")


def set_pytorch_model(
    model_cls: nn.Module,
    params: dict,
    input_size: int,
    output_size: int,
    model_type: str = "mlp",
    task: str = "regression",
) -> nn.Module:
    """
    Set up a PyTorch model based on the provided parameters.

    Args:
        model_cls (nn.Module): The PyTorch model class to instantiate.
        params (dict): Dictionary containing model parameters.
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        model_type (str): Type of model to initialize - "mlp" or "lstm".
        task (str): Type of task - "regression", "binary", or "multiclass".
        return_logits (bool): If True, model returns raw logits.

    Returns:
        nn.Module: Initialized PyTorch model.
    """
    if model_type == "mlp":
        required_keys = {"hidden_sizes"}
        validate_params(params, required_keys)

        return model_cls(
            input_size=input_size,
            hidden_sizes=params["hidden_sizes"],
            output_size=output_size,
            task=task,
        )
    elif model_type == "lstm":
        required_keys = {
            "hidden_size",
            "num_layers",
        }
        validate_params(params, required_keys)

        return model_cls(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            output_size=output_size,
            task=task,
            dropout=params.get("dropout", 0.0),
            bidirectional=params.get("bidirectional", False),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def set_loss_fn(task: str = "regression") -> nn.Module:
    """
    Set up the appropriate loss function based on the task.

    Args:
        task (str): Type of task - "regression", "binary", or "multiclass".

    Returns:
        nn.Module: Appropriate loss function.
    """
    if task == "regression":
        return nn.MSELoss()
    elif task == "binary":
        return nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task: {task}")

import torch
from torch import nn


class MLPModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
    ) -> None:
        """
        A simple Multi-Layer Perceptron (MLP) model.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list[int]): List of hidden layer sizes.
            output_size (int): Number of output features.
        """
        super().__init__()

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
            torch.Tensor: Output tensor after applying the model and activation function.
        """
        return self.model(x)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
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
            dropout (float): Dropout probability for the LSTM layers.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
        """
        super().__init__()

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
            torch.Tensor: Output tensor after applying the model and activation function.
        """
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)

        return self.fc(lstm_out[:, -1, :])


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
) -> nn.Module:
    """
    Set up a PyTorch model based on the provided parameters.

    Args:
        model_cls (nn.Module): The PyTorch model class to instantiate.
        params (dict): Dictionary containing model parameters.
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        model_type (str): Type of model to initialize - "mlp" or "lstm".

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
            dropout=params.get("dropout", 0.0),
            bidirectional=params.get("bidirectional", False),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

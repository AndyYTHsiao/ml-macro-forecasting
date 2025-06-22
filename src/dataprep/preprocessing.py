import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_regression


def correlation_clustering(
    x: pd.DataFrame, mi_scores: pd.Series, threshold: float = 0.9
) -> pd.DataFrame:
    """
    Removes redundant features by correlation clustering.
    Keeps the feature with the highest mutual information in each cluster.

    Parameters:
        x (pd.DataFrame): Feature matrix.
        mi_scores (pd.Series): Mutual information scores (indexed by feature name).
        threshold (float): Correlation threshold for redundancy (default=0.9).

    Returns:
        pd.DataFrame: Reduced feature matrix.
    """
    # Compute correlation matrix
    corr_matrix = x.corr().abs()

    # Features to keep
    kept = []
    dropped = set()

    # Sort features by mutual information (high → low)
    sorted_features = mi_scores.sort_values(ascending=False).index

    for f in sorted_features:
        if f in dropped:
            continue
        kept.append(f)

        # Drop all highly correlated features
        high_corr = corr_matrix.index[
            (corr_matrix[f] > threshold) & (corr_matrix.index != f)
        ]
        dropped.update(high_corr)

    return x[kept]


def calculate_mutual_information(
    x: pd.DataFrame, y: pd.Series, top_k: int | None = None, threshold: float = 0.1
) -> pd.DataFrame:
    """
    Calculate mutual information with the target variable.

    Parameters:
        x (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target variable.
        threshold (float): Minimum mutual information value to keep a feature.

    Returns:
        pd.DataFrame: DataFrame with features that have mutual information above the threshold.
    """
    mi_scores = mutual_info_regression(x, y)
    mi_scores = pd.Series(mi_scores, index=x.columns)

    if top_k is None:
        return mi_scores.sort_values(ascending=False)[mi_scores > threshold]
    else:
        return mi_scores.sort_values(ascending=False)[mi_scores > threshold].head(top_k)


def load_and_filter_data(
    data_path: str,
    target_col: str,
    mi_threshold: float,
    corr_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess data from a CSV file.

    Parameters:
        data_path (str): Path to the CSV file containing the dataset.
        target_col (str): Name of the target column in the dataset.
        mi_threshold (float): Threshold for mutual information feature selection.
        corr_threshold (float): Threshold for correlation clustering.

    Returns:
        tuple[np.ndarray, np.ndarray]: Preprocessed data (X, y).
    """
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    y = data[target_col].values
    X = data.drop(columns=[target_col])
    mi_scores = calculate_mutual_information(X, y, threshold=mi_threshold)
    X = X[mi_scores.index]
    X = correlation_clustering(X, mi_scores, threshold=corr_threshold).values

    return X, y


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 12,
    forecast_horizon: int = 1,
    gap: int = 0,
    return_type: str = "np",
    flatten: bool = False,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Create input-output sequences for time series forecasting.

    Args:
        X (np.ndarray): Input features, shape (n_samples, n_features).
        y (np.ndarray): Target variable, shape (n_samples,).
        lookback (int): Number of past observations to use as input.
        forecast_horizon (int): Number of future observations to predict.
        gap (int): Steps between input window and forecast start.
        return_type (str): 'np' for numpy arrays, 'pt' for PyTorch tensors.
        flatten (bool): If True, flatten windows to 2D (tabular form).
                        Needed for sklearn/MLP; False for LSTM.

    Returns:
        Xs (np.ndarray or torch.Tensor): Inputs.
            - Shape (n_samples, lookback, n_features) if flatten=False
            - Shape (n_samples, lookback * n_features) if flatten=True
        ys (np.ndarray or torch.Tensor): Outputs.
            - Shape (n_samples, forecast_horizon)
    """
    n = len(X)

    if n != len(y):
        raise ValueError("X and y must have the same length.")
    if n <= lookback + gap + forecast_horizon:
        raise ValueError("Not enough data for given lookback/gap/horizon.")

    Xs, ys = [], []
    n_samples = n - lookback - gap - forecast_horizon + 1

    for i in range(n_samples):
        x_seq = X[i : i + lookback]  # (lookback, n_features)
        y_seq = y[
            i + lookback + gap : i + lookback + gap + forecast_horizon
        ]  # (forecast_horizon,)

        if flatten:
            # flatten 2D lookback × features window into 1D
            x_seq = x_seq.reshape(-1)

        Xs.append(x_seq)
        ys.append(y_seq)

    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # Squeeze target shape for single-step forecasting
    if forecast_horizon == 1:
        ys = ys.squeeze(-1)  # (n_samples,)

    # Convert to torch if needed
    if return_type == "pt":
        Xs = torch.tensor(Xs, dtype=torch.float32)
        ys = torch.tensor(ys, dtype=torch.float32)

    return Xs, ys


def set_up_dataloader(
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """
    Set up DataLoaders for training and testing.

    Args:
        X_train (np.ndarray): Training features, shape (n_samples, n_features).
        y_train (np.ndarray): Training target, shape (n_samples,).
        X_test (np.ndarray): Testing features, shape (n_samples, n_features).
        y_test (np.ndarray): Testing target, shape (n_samples,).
        batch_size (int): Batch size for DataLoader.
        device (torch.device): Device to which tensors should be moved.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(
        device
    )
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=False,  # keep time order
        pin_memory=True if device.type == "cuda" else False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True if device.type == "cuda" else False,
    )

    return train_loader, test_loader

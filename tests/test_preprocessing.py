import numpy as np
import pandas as pd
import pytest
import torch

from src.dataprep.preprocessing import (
    FeatureSelector,
    calculate_mutual_information,
    correlation_clustering,
    create_sequences,
    load_and_filter_data,
    load_data,
    set_up_dataloader,
)


def test_load_data_sorts_dates_and_separates_target(tmp_path):
    data_path = tmp_path / "observations.csv"
    pd.DataFrame(
        {
            "date": ["2024-03-31", "2024-01-31", "2024-02-29"],
            "feature": [3.0, 1.0, 2.0],
            "target": [30.0, 10.0, 20.0],
        }
    ).set_index("date").to_csv(data_path)

    features, target = load_data(str(data_path), "target")

    assert features.index.is_monotonic_increasing
    assert features.columns.tolist() == ["feature"]
    np.testing.assert_array_equal(features["feature"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(target, [10.0, 20.0, 30.0])


def test_load_data_rejects_unknown_target(tmp_path):
    data_path = tmp_path / "observations.csv"
    pd.DataFrame({"feature": [1.0]}, index=["2024-01-31"]).to_csv(data_path)

    with pytest.raises(ValueError, match="Target column 'target' was not found"):
        load_data(str(data_path), "target")


def test_correlation_clustering_keeps_most_informative_feature():
    frame = pd.DataFrame(
        {
            "strong": [1.0, 2.0, 3.0, 4.0],
            "duplicate": [2.0, 4.0, 6.0, 8.0],
            "independent": [0.0, 1.0, 0.0, 1.0],
        }
    )
    scores = pd.Series({"strong": 0.8, "duplicate": 0.4, "independent": 0.6})

    selected = correlation_clustering(frame, scores, threshold=0.9)

    assert selected.columns.tolist() == ["strong", "independent"]


def test_feature_selector_treats_series_values_positionally(monkeypatch):
    frame = pd.DataFrame(
        {"useful": [1.0, 2.0, 3.0], "unused": [3.0, 2.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    target = pd.Series([10.0, 20.0, 30.0])

    def fake_mutual_information(x, y, **kwargs):
        assert y.index.equals(x.index)
        np.testing.assert_array_equal(y, target)
        return pd.Series({"useful": 1.0})

    monkeypatch.setattr(
        "src.dataprep.preprocessing.calculate_mutual_information",
        fake_mutual_information,
    )
    selector = FeatureSelector()

    transformed = selector.fit_transform(frame, target)

    assert selector.selected_features_ == ["useful"]
    np.testing.assert_array_equal(transformed[:, 0], frame["useful"])


def test_feature_selector_requires_matching_rows_and_fit():
    selector = FeatureSelector()
    frame = pd.DataFrame({"feature": [1.0, 2.0]})

    with pytest.raises(ValueError, match="same length"):
        selector.fit(frame, np.array([1.0]))
    with pytest.raises(RuntimeError, match="must be fitted"):
        selector.transform(frame)


def test_feature_selector_rejects_empty_mutual_information_result(monkeypatch):
    monkeypatch.setattr(
        "src.dataprep.preprocessing.calculate_mutual_information",
        lambda *args, **kwargs: pd.Series(dtype=float),
    )

    with pytest.raises(ValueError, match="No features passed"):
        FeatureSelector().fit(pd.DataFrame({"feature": [1.0]}), np.array([1.0]))


def test_calculate_mutual_information_applies_threshold_and_top_k(monkeypatch):
    monkeypatch.setattr(
        "src.dataprep.preprocessing.mutual_info_regression",
        lambda *args, **kwargs: np.array([0.2, 0.8, 0.5]),
    )
    frame = pd.DataFrame({"low": [1], "high": [2], "middle": [3]})
    target = pd.Series([1])

    scores = calculate_mutual_information(frame, target, threshold=0.3, top_k=1)

    assert scores.to_dict() == {"high": 0.8}


def test_load_and_filter_data_fits_only_leading_rows(tmp_path, monkeypatch):
    data_path = tmp_path / "observations.csv"
    frame = pd.DataFrame(
        {"feature": np.arange(6), "target": np.arange(10, 16)},
        index=pd.date_range("2024-01-01", periods=6),
    )
    frame.to_csv(data_path)
    fitted_indices = []

    def fake_fit(self, x, y):
        fitted_indices.extend(x.index)
        self.selected_features_ = ["feature"]
        return self

    monkeypatch.setattr(FeatureSelector, "fit", fake_fit)

    features, target = load_and_filter_data(
        str(data_path), "target", fit_size=4, mi_threshold=0.1, corr_threshold=0.9
    )

    assert fitted_indices == frame.index[:4].tolist()
    np.testing.assert_array_equal(features[:, 0], frame["feature"])
    np.testing.assert_array_equal(target, frame["target"])


@pytest.mark.parametrize("fit_size", [0, 4])
def test_load_and_filter_data_validates_fit_size(tmp_path, fit_size):
    data_path = tmp_path / "observations.csv"
    pd.DataFrame(
        {"feature": [1, 2, 3], "target": [2, 3, 4]},
        index=pd.date_range("2024-01-01", periods=3),
    ).to_csv(data_path)

    with pytest.raises(ValueError, match="fit_size must be between"):
        load_and_filter_data(str(data_path), "target", fit_size, 0.1, 0.9)


def test_create_sequences_respects_lookback_gap_horizon_and_flattening():
    features = np.arange(12).reshape(6, 2)
    target = np.arange(100, 106)

    sequences, outcomes = create_sequences(
        features, target, lookback=2, gap=1, forecast_horizon=2, flatten=True
    )

    assert sequences.shape == (2, 4)
    assert outcomes.shape == (2, 2)
    np.testing.assert_array_equal(sequences[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(outcomes[0], [103, 104])


def test_create_sequences_can_return_tensors_and_squeeze_one_step_target():
    sequences, outcomes = create_sequences(
        np.arange(8).reshape(4, 2),
        np.arange(4),
        lookback=2,
        return_type="pt",
    )

    assert isinstance(sequences, torch.Tensor)
    assert isinstance(outcomes, torch.Tensor)
    assert sequences.shape == (2, 2, 2)
    assert outcomes.shape == (2,)
    assert sequences.dtype == outcomes.dtype == torch.float32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lookback": 0}, "must be positive"),
        ({"gap": -1}, "cannot be negative"),
        ({"return_type": "list"}, "either 'np' or 'pt'"),
        ({"lookback": 4}, "Not enough data"),
    ],
)
def test_create_sequences_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        create_sequences(np.ones((4, 2)), np.ones(4), **kwargs)


def test_set_up_dataloader_moves_batches_to_device():
    device = torch.device("cpu")
    train_loader, test_loader = set_up_dataloader(
        torch.ones((4, 2)),
        torch.arange(4, dtype=torch.float32),
        torch.zeros((2, 2)),
        torch.arange(2, dtype=torch.float32),
        batch_size=2,
        device=device,
    )

    train_features, train_target = next(iter(train_loader))
    test_features, test_target = next(iter(test_loader))
    assert train_features.device == train_target.device == device
    assert test_features.device == test_target.device == device
    assert len(train_loader.dataset) == 4
    assert len(test_loader.dataset) == 2

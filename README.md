# ML Macro Forecasting

This repository contains the implementation used to compare machine-learning models for forecasting macroeconomic indicators.
The pipeline is regression only and supports Elastic Net, random forests, support-vector regression,
XGBoost, MLP, and LSTM.
Model selection uses time-series cross-validation, and the final evaluation reserves the latest 20%
of observations as a chronological test set.

## Data

The datasets are in [`data/`](./data).
Each country/indicator CSV contains a monthly date index, a target, and engineered predictors.
Analyst forecasts from Bloomberg are provided separately for benchmarking.
See the [`data` documentation](./data/README.md) for sources and collection details.

## Requirements and setup

- Python 3.13 or newer
- [`uv`](https://docs.astral.sh/uv/)

To install the development tools as well, use `uv sync --dev`.

```bash
git clone https://github.com/AndyYTHsiao/ml-macro-forecasting.git
cd ml-macro-forecasting
uv sync --dev
```

## Run an experiment

The executable example in [`src/run_experiments.py`](./src/run_experiments.py) defaults to the Hong Kong unemployment dataset and an MLP.
It performs feature selection using only the training period, tunes hyperparameters with expanding
window cross-validation, trains on the full training sample, evaluates on the held-out test period, and writes artifacts beneath `outputs/`.


```bash
uv run python -m src.run_experiments
```

The default run performs 100 Optuna trials and can take some time.
Before running, edit the configuration block near the top of `src/run_experiments.py` to choose the dataset, target, lookback windows, trial count, and training epochs.

Generated artifacts are grouped by dataset, framework, and model and include:

- the fitted model (`.joblib` for scikit-learn or `.safetensors` for PyTorch),
- held-out predictions (`.npy`),
- selected hyperparameters (`.json`), and
- train/test regression metrics (`.json`).

## Pipeline

The forecasting workflow is:

1. Load observations in chronological order.
2. Split the data into training and test periods.
3. Fit feature selection on the training period only, then apply the fitted selector to both periods.   This prevents the held-out target from affecting feature selection.
4. Create lagged sequences independently for the training and test periods.
5. Tune hyperparameters with time-series cross-validation on training data.
6. Fit the selected model without consulting the test period.
7. Evaluate once on the held-out test period.

The following condensed example demonstrates the sklearn workflow:

```python
from pathlib import Path
from src.core.training import Trainer
from src.core.tuning import Tuner

tuner = Tuner(model_type=model_type, framework=framework)

X, y = load_data(
    data_path="./data/hk_unemp.csv",
    target_col="Total unemployment rate",
)

split_index = int(len(X) * 0.8)
X_train_frame, X_test_frame = X.iloc[:split_index], X.iloc[split_index:]
y_train = y.iloc[:split_index].to_numpy()
y_test = y.iloc[split_index:].to_numpy()

selector = FeatureSelector(mi_threshold=0.2, corr_threshold=0.9)
X_train = selector.fit_transform(X_train_frame, y_train)
X_test = selector.transform(X_test_frame)

X_train, y_train = create_sequences(X_train, y_train, lookback=12, flatten=True)
X_test, y_test = create_sequences(X_test, y_test, lookback=12, flatten=True)

tuner = Tuner(model_type="svm", framework="sklearn")
best_params = tuner.tune(
    X_train=X_train,
    y_train=y_train,
    n_trials=100,
    cv_splits=5,
)

# Model training
trainer = Trainer(
    model_type="svm",
    params=best_params,
    framework="sklearn",
)
trainer.train((X_train, y_train))
trainer.evaluate(X_train, y_train, X_test, y_test, rounding=4)
trainer.save_results(Path("./outputs/sklearn/svm"))
```

## Tests

```
uv run --with pytest pytest
```
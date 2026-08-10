# Overview

This repository contains the refined implementation of my thesis which investigates the performance of machine learning models in forecasting key macroeconomic indicators.

# Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment and dependencies. Follow [the offical guide](https://docs.astral.sh/uv/getting-started/installation/) to install `uv` on your device. Once installed, you can initialize this project with the following steps.

1. **Clone the repo:**
    ```bash
    git clone https://github.com/AndyYTHsiao/ml-macro-forecasting.git
    cd ml-macro-forecasting
    ```

2. **Sync the environments:**
    ```bash
    uv sync --dev
    ```

    This command automatically creates a virtual environment and installs all dependencies specified in [`pyproject.toml`](./pyproject.toml).

# Quick start

A sample implementation is available at [run_experiments.py](./src/run_experiments.py).

```bash
# With uv
uv run python -m src.run_experiments

# Without uv
python -m src.run_experiments
```

If you want to tune and train the model on your own, you can follow the following steps.

```python
from pathlib import Path
from src.core.tuning import Tuner
from src.core.training import Trainer

# Assume you have prepared the training and test datasets

# Basic setting
model_type = "svm"
framework = "sklearn"
n_trials = 100
cv_splits = 3
output_dir = Path("./outputs")

# Model tuning
tuner = Tuner(
    model_type=model_type,
    framework=framework,
)
best_params = tuner.tune(
    X_train=X_train,
    y_train=y_train,
    n_trials=n_trials,
    cv_splits=cv_splits,
)

# Model training
trainer = Trainer(
    model_type=model_type,
    params=best_params,
    framework=framework,
)
trainer.train(
    (X_train, y_train),
)

# Model evaluation
trainer.evaluate(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    rounding=4,
)

# Save results
trainer.save_results(output_dir / framework / model_type)
```
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import optuna
import torch

from .core.registry import requires_scaling
from .core.training import Trainer
from .core.tuning import Tuner
from .core.utils import set_all_seeds
from .dataprep.preprocessing import (
    FeatureSelector,
    create_sequences,
    load_data,
    set_up_dataloader,
)

# Suppress Optuna warnings (optional)
optuna.logging.set_verbosity(optuna.logging.WARNING)


if __name__ == "__main__":
    # --- Set random seed ---
    seed = 42
    set_all_seeds(seed)

    # --- Hyperparameters ---
    cv_splits = 3
    test_size = 0.2
    mi_threshold = 0.2
    corr_threshold = 0.9
    learning_rate = 0.001
    n_epochs = 150
    min_epochs = 5
    patience = 10
    forecast_horizon = 1
    gap = 0
    batch_size = 64
    n_trials = 100
    direction = "minimize"
    scoring = "neg_mean_squared_error"
    lookback_windows = [1]

    # --- Device configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Output directory ---
    dataset_name = "hk_unemp"
    output_dir = Path(f"./outputs/{dataset_name}/")

    # --- Load and split data ---
    X, y = load_data(
        data_path=f"./data/{dataset_name}.csv",
        target_col="Total unemployment rate",
    )

    split_index = int(len(X) * (1 - test_size))
    X_train_frame, X_test_frame = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index].to_numpy(), y.iloc[split_index:].to_numpy()
    selector = FeatureSelector(mi_threshold, corr_threshold)
    X_train = selector.fit_transform(X_train_frame, y_train)
    X_test = selector.transform(X_test_frame)

    for lookback in lookback_windows:
        print(f"Running experiments with lookback = {lookback}")
        # --- Create sequences and dataloaders ---
        X_train_seq, y_train_seq = create_sequences(
            X_train,
            y_train,
            lookback=lookback,
            forecast_horizon=forecast_horizon,
            gap=gap,
            return_type="np",
            flatten=True,
        )
        X_test_seq, y_test_seq = create_sequences(
            X_test,
            y_test,
            lookback=lookback,
            forecast_horizon=forecast_horizon,
            gap=gap,
            return_type="np",
            flatten=True,
        )

        # --- Sklearn models ---
        # for model_type in ["linear", "random_forest", "xgboost", "svm"]:
        #     framework = "sklearn"

        #     # --- Hyperparameter tuning ---
        #     print(f"Tuning {model_type}...")
        #     tuner = Tuner(
        #         model_type=model_type,
        #         framework=framework,
        #     )
        #     best_params = tuner.tune(
        #         X_train=X_train_seq,
        #         y_train=y_train_seq,
        #         n_trials=n_trials,
        #         seed=seed,
        #         direction=direction,
        #         scoring=scoring,
        #         cv_splits=cv_splits,
        #     )

        #     # --- Initialize Trainer ---
        #     trainer = Trainer(
        #         model_type=model_type,
        #         params=best_params,
        #         framework=framework,
        #     )

        #     trainer.train(
        #         (X_train_seq, y_train_seq),
        #     )
        #     trainer.evaluate(
        #         X_train=X_train_seq,
        #         y_train=y_train_seq,
        #         X_test=X_test_seq,
        #         y_test=y_test_seq,
        #         rounding=4,
        #     )
        #     trainer.save_results(
        #         output_dir / framework / model_type, suffix=f"lb{lookback}m"
        #     )
        #     print(f"[✅] Finished training {model_type}.")

        # --- Non-sequence deep learning models ---
        for model_type in ["mlp"]:
            framework = "torch"

            # --- Hyperparameter tuning ---
            print(f"Tuning {model_type}...")
            tuner = Tuner(
                model_type=model_type,
                framework=framework,
                device=device,
            )
            best_params = tuner.tune(
                X_train=X_train_seq,
                y_train=y_train_seq,
                n_trials=n_trials,
                cv_splits=cv_splits,
            )

            n_layers = best_params.pop("n_layers", 1)
            hidden_size = best_params.pop("hidden_size", 64)
            best_params["hidden_sizes"] = [hidden_size] * n_layers

            # --- Initialize Trainer ---
            trainer = Trainer(
                model_type=model_type,
                params=best_params,
                framework=framework,
            )

            if requires_scaling(model_type):
                scaler = StandardScaler()
                X_train_seq = scaler.fit_transform(X_train_seq)
                X_test_seq = scaler.transform(X_test_seq)

            train_loader, test_loader = set_up_dataloader(
                torch.tensor(X_train_seq, dtype=torch.float32),
                torch.tensor(y_train_seq, dtype=torch.float32),
                torch.tensor(X_test_seq, dtype=torch.float32),
                torch.tensor(y_test_seq, dtype=torch.float32),
                batch_size=batch_size,
                device=device,
            )

            trainer.train(
                train_loader,
            )
            trainer.evaluate(
                X_train=X_train_seq,
                y_train=y_train_seq,
                X_test=X_test_seq,
                y_test=y_test_seq,
                rounding=4,
            )
            trainer.save_results(
                output_dir / framework / model_type, suffix=f"lb{lookback}m"
            )
            print(f"[✅] Finished training {model_type}.")

    # --- Sequence deep learning models ---
    # for lookback in lookback_windows:
    #     print(f"Running experiments with lookback = {lookback}")
    #     # --- Create sequences and dataloaders ---
    #     X_train_seq, y_train_seq = create_sequences(
    #         X_train,
    #         y_train,
    #         lookback=lookback,
    #         forecast_horizon=forecast_horizon,
    #         gap=gap,
    #         return_type="np",
    #         flatten=False,
    #     )
    #     X_test_seq, y_test_seq = create_sequences(
    #         X_test,
    #         y_test,
    #         lookback=lookback,
    #         forecast_horizon=forecast_horizon,
    #         gap=gap,
    #         return_type="np",
    #         flatten=False,
    #     )

    #     train_loader, test_loader = set_up_dataloader(
    #         torch.tensor(X_train_seq, dtype=torch.float32),
    #         torch.tensor(y_train_seq, dtype=torch.float32),
    #         torch.tensor(X_test_seq, dtype=torch.float32),
    #         torch.tensor(y_test_seq, dtype=torch.float32),
    #         batch_size=batch_size,
    #         device=device,
    #     )

    #     framework = "torch"
    #     for model_type in ["lstm"]:
    #         # --- Hyperparameter tuning ---
    #         print(f"Tuning {model_type}...")
    #         tuner = Tuner(
    #             model_type=model_type,
    #             framework=framework,
    #             device=device,
    #         )
    #         best_params = tuner.tune(
    #             X_train=X_train_seq,
    #             y_train=y_train_seq,
    #             n_trials=n_trials,
    #             cv_splits=cv_splits,
    #         )

    #         # --- Initialize Trainer ---
    #         trainer = Trainer(
    #             model_type=model_type,
    #             params=best_params,
    #             framework=framework,
    #         )

    #         trainer.train(
    #             train_loader,
    #         )
    #         trainer.evaluate(
    #             X_train=X_train_seq,
    #             y_train=y_train_seq,
    #             X_test=X_test_seq,
    #             y_test=y_test_seq,
    #             rounding=4,
    #         )
    #         trainer.save_results(
    #             output_dir / framework / model_type, suffix=f"lb{lookback}m"
    #         )
    #         print(f"[✅] Finished training {model_type}.")

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger("src.model_training.train_model")


def load_data() -> pd.DataFrame:
    # Usa as janelas criadas em feature_engineering
    train_path = "data/processed/train_processed.csv"
    logger.info(f"Loading feature data from {train_path}")
    train_data = pd.read_csv(train_path)
    return train_data


def load_params() -> dict[str, float | int]:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["train"]


def prepare_data(
    train_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Espera um CSV com várias colunas de features (janela de tempo)
    e uma coluna 'target' com o Volume.

    Retorna:
    - X_train: (amostras, timesteps, 1)
    - y_train_scaled: (amostras, 1), Volume escalado
    - y_scaler: MinMaxScaler usado no target
    """

    # X = todas as colunas menos 'target'
    X_train = train_data.drop("target", axis=1).values.astype("float32")
    y_train = train_data["target"].values.reshape(-1, 1).astype("float32")

    # Escala o target (Volume) — ESSA É A PARTE CRÍTICA
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    # LSTM espera (amostras, timesteps, features)
    # timesteps = número de colunas (tamanho da janela)
    # features = 1 (apenas 1 série: Open ao longo do tempo)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train_scaled, y_scaler


def create_model(input_shape, params):
    """
    input_shape: (timesteps, n_features)
    """
    model = Sequential(
        [
            LSTM(
                params["lstm_1_units"],
                return_sequences=True,
                input_shape=input_shape,
            ),
            Dropout(params["dropout_rate"]),

            LSTM(
                params["lstm_2_units"],
                return_sequences=True,
            ),
            Dropout(params["dropout_rate"]),

            LSTM(
                params["lstm_3_units"],
                return_sequences=True,
            ),
            Dropout(params["dropout_rate"]),

            LSTM(
                params["lstm_4_units"],
                return_sequences=False,
            ),
            Dropout(params["dropout_rate"]),

            Dense(1, activation="linear"),  # regressão
        ]
    )

    optimizer = Adam(learning_rate=params["learning_rate"])

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    return model


def save_training_artifacts(
    model: tf.keras.Model,
    y_scaler: MinMaxScaler,
) -> None:
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.keras")
    scaler_path = os.path.join(models_dir, "y_scaler.pkl")

    logger.info(f"Saving model to {model_path}")
    model.save(model_path)

    logger.info(f"Saving target scaler to {scaler_path}")
    joblib.dump(y_scaler, scaler_path)


def train_model(train_data: pd.DataFrame, params: dict[str, int | float]) -> None:
    # não precisa usar pop, só ler
    tf.keras.utils.set_random_seed(params["random_seed"])

    X_train, y_train, y_scaler = prepare_data(train_data)

    model = create_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        params=params,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    logger.info("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=[early_stopping],
        verbose=1,
    )

    save_training_artifacts(model, y_scaler)

    metrics = {
        metric: float(history.history[metric][-1])
        for metric in history.history
    }

    os.makedirs("metrics", exist_ok=True)
    metrics_path = "metrics/training.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()
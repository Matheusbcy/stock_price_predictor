import logging
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger("src.model_evaluation.evaluate_model")


def load_model() -> tf.keras.Model:
    """Carrega o modelo Keras treinado do disco."""
    model_path = "models/model.keras"
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def load_y_scaler():
    """Carrega o scaler do target (Volume) usado no treinamento."""
    scaler_path = "models/y_scaler.pkl"
    logger.info(f"Loading target scaler from {scaler_path}")
    y_scaler = joblib.load(scaler_path)
    return y_scaler


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega os dados de teste já processados (janelas) do disco.

    Retorna:
        X_test: np.ndarray com shape (amostras, timesteps, 1)
        y_true: np.ndarray com shape (amostras, 1) em escala REAL de Volume
    """
    data_path = "data/processed/test_processed.csv"
    logger.info(f"Loading test data from {data_path}")
    data = pd.read_csv(data_path)

    # X = todas as colunas exceto 'target'
    X = data.drop("target", axis=1).values.astype("float32")
    y_true = data["target"].values.reshape(-1, 1).astype("float32")

    # LSTM espera (amostras, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y_true


def evaluate_model(
    model: tf.keras.Model,
    y_scaler,
    X_test: np.ndarray,
    y_true_real: np.ndarray,
) -> None:
    logger.info("Generating predictions on test set...")

    # Predição em escala NORMALIZADA (porque o modelo foi treinado assim)
    y_pred_scaled = model.predict(X_test)

    # Volta as previsões para a escala REAL de Volume
    y_pred_real = y_scaler.inverse_transform(y_pred_scaled)

    # Achata para 1D para usar nas métricas
    y_true_flat = y_true_real.flatten()
    y_pred_flat = y_pred_real.flatten()

    # Métricas de regressão
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    # MAPE pode explodir se tiver valor muito próximo de zero; use com cautela
    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    r2 = r2_score(y_true_flat, y_pred_flat)

    evaluation = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "r2": float(r2),
    }

    logger.info(
        "Regression metrics on test set:\n"
        f"  MAE  (Volume real): {mae:.2f}\n"
        f"  MSE  (Volume real): {mse:.2f}\n"
        f"  RMSE (Volume real): {rmse:.2f}\n"
        f"  MAPE (%%)         : {mape:.2f}\n"
        f"  R²                : {r2:.4f}"
    )

    os.makedirs("metrics", exist_ok=True)
    evaluation_path = "metrics/evaluation.json"
    with open(evaluation_path, "w") as f:
        json.dump(evaluation, f, indent=2)


def main() -> None:
    logger.info("Starting model evaluation...")

    model = load_model()
    y_scaler = load_y_scaler()
    X_test, y_true = load_test_data()
    evaluate_model(model, y_scaler, X_test, y_true)

    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()
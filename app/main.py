import io
import logging
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from scripts.prepare_api_input import prepare_api_input

logger = logging.getLogger("app.main")
logging.basicConfig(level=logging.INFO)


class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        logger.info("Loading artifacts from local project folder")

        models_dir = "models"
        data_dir = "data/processed"
        model_path = os.path.join(models_dir, "model.keras")
        y_scaler_path = os.path.join(models_dir, "y_scaler.pkl")
        train_processed_path = os.path.join(data_dir, "train_processed.csv")

        self.model = load_model(model_path)

        self.y_scaler = joblib.load(y_scaler_path)

        train_df = pd.read_csv(train_processed_path)
        self.feature_columns = [col for col in train_df.columns if col != "target"]
        self.timesteps = len(self.feature_columns)

        logger.info("Successfully loaded model, target scaler and feature schema")
        logger.info(f"Feature columns ({self.timesteps} timesteps): {self.feature_columns}")

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:

        if "target" in features.columns:
            features = features.drop(columns=["target"])

        missing_cols = [c for c in self.feature_columns if c not in features.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        X = features[self.feature_columns].values.astype("float32")

        X = X.reshape((X.shape[0], self.timesteps, 1))

        y_pred_scaled = self.model.predict(X)

        y_pred_real = self.y_scaler.inverse_transform(y_pred_scaled).ravel()

        return pd.DataFrame({"Prediction_Volume": y_pred_real}, index=features.index)


def create_routes(app: Flask) -> None:

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload() -> str:
        file = request.files.get("file")
        if file is None or file.filename == "":
            return render_template("index.html", error="Envie um arquivo CSV.")

        if not file.filename.endswith(".csv"):
            return render_template("index.html", error="Por favor, envie um arquivo CSV.")

        try:
            content = file.read().decode("utf-8")
            features = pd.read_csv(io.StringIO(content))

            predictions = app.model_service.predict(features)

            result = predictions.to_string()

            return render_template("index.html", predictions=result)

        except Exception as e:
            logger.error("Error processing file", exc_info=True)
            return render_template(
                "index.html",
                error=f"Error processing file: {str(e)}",
            )

app = Flask(__name__)
app.model_service = ModelService()
create_routes(app)
logger.info("Application initialized with model service and routes")


def main() -> None:
    prepare_api_input()
    
    app.run(host="0.0.0.0", port=5001, debug=True)


if __name__ == "__main__":
    main()
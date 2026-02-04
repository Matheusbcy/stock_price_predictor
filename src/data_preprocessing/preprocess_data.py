import logging
import os
import yaml

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger("src.data_preprocessing.preprocess_data")


def load_data() -> pd.DataFrame:
    input_path = "data/raw/raw.csv"
    logger.info(f"Loading raw data from {input_path}")
    data = pd.read_csv(input_path)
    return data


def load_params() -> dict[str, float | int]:
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["preprocess_data"]


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = load_params()
    logger.info("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(
        data, test_size=params["test_size"], random_state=params["random_seed"]
    )
    return train_data, test_data


def preprocess_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    logger.info("Preprocessing data...")
    
    train_target = train_data['Volume'].values
    test_target = test_data['Volume'].values
    train_features = train_data[["Open"]].values
    test_features = test_data[["Open"]].values
    
    normalizer = MinMaxScaler(feature_range = (0, 1))
    train_features = normalizer.fit_transform(train_features)
    test_features = normalizer.transform(test_features)

    train_processed = pd.DataFrame(train_features, columns=["Open"])
    train_processed["target"] = train_target.tolist()
    test_processed = pd.DataFrame(test_features, columns=["Open"])
    test_processed["target"] = test_target.tolist()
    
    return train_processed, test_processed, normalizer

    
def save_artifacts(
    train_data: pd.DataFrame, test_data: pd.DataFrame, normalizer: MinMaxScaler
) -> None:
    data_dir = "data/preprocessed"
    artifacts_dir = "artifacts"
    logger.info(f"Saving processed data to {data_dir}")

    train_path = os.path.join(data_dir, "train_preprocessed.csv")
    test_path = os.path.join(data_dir, "test_preprocessed.csv")
    scaler_path = os.path.join(artifacts_dir, "open_scaler.pkl")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    joblib.dump(normalizer, scaler_path)
def main() -> None:
    raw_data = load_data()
    train_data, test_data = split_data(raw_data)
    train_processed, test_processed, normalizer = preprocess_data(train_data, test_data)
    save_artifacts(train_processed, test_processed, normalizer)
    logger.info("Data preprocessing completed")

if __name__ == "__main__":
    main()

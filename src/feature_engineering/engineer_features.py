import logging
import os

import pandas as pd
import numpy as np

logger = logging.getLogger("src.feature_engineering.engineer_features")


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = "data/preprocessed/train_preprocessed.csv"
    test_path = "data/preprocessed/test_preprocessed.csv"
    logger.info(f"Loading preprocessed data from {train_path} and {test_path}")
    train_preprocessed = pd.read_csv(train_path)
    test_preprocessed = pd.read_csv(test_path)
    return train_preprocessed, test_preprocessed


def engineer_features(
    train_preprocessed: pd.DataFrame, 
    test_preprocessed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Engineering features...")

    window = 90

    feature_col = "Open"
    target_col = "target"

    train_X = []
    train_y = []

    for i in range(window, len(train_preprocessed)):
        window_values = train_preprocessed.loc[i - window:i - 1, feature_col].to_numpy()
        train_X.append(window_values)
        train_y.append(train_preprocessed.loc[i, target_col])

    test_X = []
    test_y = []

    for i in range(window, len(test_preprocessed)):
        window_values = test_preprocessed.loc[i - window:i - 1, feature_col].to_numpy()
        test_X.append(window_values)
        test_y.append(test_preprocessed.loc[i, target_col])

    train_X = np.array(train_X)
    test_X = np.array(test_X)

    train_processed = pd.DataFrame(train_X)
    train_processed[target_col] = train_y

    test_processed = pd.DataFrame(test_X)
    test_processed[target_col] = test_y

    return train_processed, test_processed


def save_artifacts(
    train_processed: pd.DataFrame, 
    test_processed: pd.DataFrame
) -> None:
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving engineered features to {output_dir}")

    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)


def main() -> None:
    train_preprocessed, test_preprocessed = load_preprocessed_data()
    train_processed, test_processed = engineer_features(train_preprocessed, test_preprocessed)
    save_artifacts(train_processed, test_processed)
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
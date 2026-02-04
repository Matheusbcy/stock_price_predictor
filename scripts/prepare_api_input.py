import pandas as pd
import logging

logger = logging.getLogger("scripts.prepare_api_input")


def prepare_api_input(
    source_path="data/processed/test_processed.csv",
    output_path="data/processed/test_for_api.csv",
) -> None:
    logger.info("Preparing API input CSV...")
    df = pd.read_csv(source_path)

    if "target" in df.columns:
        df = df.drop(columns=["target"])

    df.to_csv(output_path, index=False)
    logger.info(f"API input file saved to {output_path}")


def main():
    prepare_api_input()


if __name__ == "__main__":
    main()
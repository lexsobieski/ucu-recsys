import pandas as pd
from recommenders.datasets.python_splitters import python_chrono_split


def split_temporal(
    dataframe: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    train_validation_ratio = train_ratio + validation_ratio
    relative_validation_ratio = validation_ratio / train_validation_ratio

    train_validation, test = python_chrono_split(
        dataframe,
        ratio=train_validation_ratio,
        col_user="user_id",
        col_timestamp="timestamp"
    )

    train, validation = python_chrono_split(
        train_validation,
        ratio=(1 - relative_validation_ratio),
        col_user="user_id",
        col_timestamp="timestamp"
    )

    return train, validation, test

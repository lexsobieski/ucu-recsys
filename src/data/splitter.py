import pandas as pd
from recommenders.datasets.python_splitters import python_chrono_split

COLUMN_USER = "user_id"
COLUMN_TIMESTAMP = "timestamp"

EXPECTED_RATIO_SUM = 1.0


def split_temporal(
    dataframe: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + validation_ratio + test_ratio

    if total != EXPECTED_RATIO_SUM:
        raise ValueError(f"Ratios must sum to {EXPECTED_RATIO_SUM}, got {total}")

    train_validation_ratio = train_ratio + validation_ratio
    relative_validation_ratio = validation_ratio / train_validation_ratio
    relative_train_ratio = EXPECTED_RATIO_SUM - relative_validation_ratio

    train_validation, test = python_chrono_split(
        data=dataframe,
        ratio=train_validation_ratio,
        col_user=COLUMN_USER,
        col_timestamp=COLUMN_TIMESTAMP
    )

    train, validation = python_chrono_split(
        data=train_validation,
        ratio=relative_train_ratio,
        col_user=COLUMN_USER,
        col_timestamp=COLUMN_TIMESTAMP
    )

    return train, validation, test

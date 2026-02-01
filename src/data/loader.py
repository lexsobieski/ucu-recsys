from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"
FILE_SEPARATOR = "::"
FILE_ENCODING = "latin-1"
CSV_ENGINE = "python"

RATINGS_FILENAME = "ratings.dat"
MOVIES_FILENAME = "movies.dat"
USERS_FILENAME = "users.dat"

RATINGS_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
MOVIES_COLUMNS = ["item_id", "title", "genres"]
USERS_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]


def load_ratings(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DEFAULT_DATA_DIR
    ratings_path = data_dir / RATINGS_FILENAME

    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found at {ratings_path}")

    dataframe = pd.read_csv(
        filepath_or_buffer=ratings_path,
        sep=FILE_SEPARATOR,
        engine=CSV_ENGINE,
        names=RATINGS_COLUMNS,
        encoding=FILE_ENCODING
    )

    print(f"Loaded {len(dataframe)} ratings from {ratings_path}")

    return dataframe


def load_movies(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DEFAULT_DATA_DIR
    movies_path = data_dir / MOVIES_FILENAME

    if not movies_path.exists():
        raise FileNotFoundError(f"Movies file not found at {movies_path}")

    dataframe = pd.read_csv(
        filepath_or_buffer=movies_path,
        sep=FILE_SEPARATOR,
        engine=CSV_ENGINE,
        names=MOVIES_COLUMNS,
        encoding=FILE_ENCODING
    )

    print(f"Loaded {len(dataframe)} movies from {movies_path}")

    return dataframe


def load_users(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DEFAULT_DATA_DIR
    users_path = data_dir / USERS_FILENAME

    if not users_path.exists():
        raise FileNotFoundError(f"Users file not found at {users_path}")

    dataframe = pd.read_csv(
        filepath_or_buffer=users_path,
        sep=FILE_SEPARATOR,
        engine=CSV_ENGINE,
        names=USERS_COLUMNS,
        encoding=FILE_ENCODING
    )

    print(f"Loaded {len(dataframe)} users from {users_path}")

    return dataframe

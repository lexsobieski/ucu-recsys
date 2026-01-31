import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ml-1m"
SEPARATOR = "::"
ENCODING = "Latin-1"


def load_ratings(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DEFAULT_DATA_DIR
    ratings_path = data_dir / "ratings.dat"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found at {ratings_path}")

    dataframe = pd.read_csv(
        ratings_path,
        sep=SEPARATOR,
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding=ENCODING
    )

    logger.info(f"Loaded {len(dataframe)} ratings from {ratings_path}")
    return dataframe


def load_movies(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DEFAULT_DATA_DIR
    movies_path = data_dir / "movies.dat"

    if not movies_path.exists():
        raise FileNotFoundError(f"Movies file not found at {movies_path}")

    dataframe = pd.read_csv(
        movies_path,
        sep=SEPARATOR,
        engine="python",
        names=["item_id", "title", "genres"],
        encoding=ENCODING
    )

    logger.info(f"Loaded {len(dataframe)} movies from {movies_path}")
    return dataframe


def load_users(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DEFAULT_DATA_DIR
    users_path = data_dir / "users.dat"

    if not users_path.exists():
        raise FileNotFoundError(f"Users file not found at {users_path}")

    dataframe = pd.read_csv(
        users_path,
        sep=SEPARATOR,
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding=ENCODING
    )

    logger.info(f"Loaded {len(dataframe)} users from {users_path}")
    return dataframe

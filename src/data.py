import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")


def load_ratings(path: Path = DATA_RAW / "ratings.dat") -> pd.DataFrame:
    """Load ratings.dat -> [user_id, movie_id, rating, timestamp]"""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )


def load_movies(path: Path = DATA_RAW / "movies.dat") -> pd.DataFrame:
    """Load movies.dat -> [movie_id, title, genres]"""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )


def load_users(path: Path = DATA_RAW / "users.dat") -> pd.DataFrame:
    """Load users.dat -> [user_id, gender, age, occupation, zip_code]"""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )


def temporal_split(
    ratings: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split per user: earliest (train_ratio)% -> train, rest -> test.
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])

    def split_user(group):
        n_train = int(len(group) * train_ratio)
        group = group.copy()
        group["split"] = ["train"] * n_train + ["test"] * (len(group) - n_train)
        return group

    labeled = ratings.groupby("user_id", group_keys=False).apply(split_user)
    train = labeled[labeled["split"] == "train"].drop(columns=["split"])
    test = labeled[labeled["split"] == "test"].drop(columns=["split"])

    return train.reset_index(drop=True), test.reset_index(drop=True)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.base import InteractionData


class DataAdapter:
    """Converts DataFrames to InteractionData for model consumption."""

    def __init__(
        self,
        col_user: str = "user_id",
        col_item: str = "item_id",
        col_rating: str = "rating",
    ):
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating

    def to_interactions(self, df: pd.DataFrame) -> InteractionData:
        """
        Convert a DataFrame to InteractionData.

        :param df: DataFrame with user, item, rating columns (no duplicates expected)
        :return: InteractionData ready for model.fit()
        """
        # Sort IDs for reproducible index mappings
        users = np.array(sorted(df[self.col_user].unique(), key=str), dtype=object)
        items = np.array(sorted(df[self.col_item].unique(), key=str), dtype=object)

        user_to_idx = {u: i for i, u in enumerate(users)}
        item_to_idx = {it: i for i, it in enumerate(items)}

        user_indices = df[self.col_user].map(user_to_idx).values
        item_indices = df[self.col_item].map(item_to_idx).values
        ratings = df[self.col_rating].values.astype(np.float32)

        X_ui = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(len(users), len(items)),
        )

        global_mean = float(df[self.col_rating].mean())

        user_items_set = (
            df.groupby(self.col_user)[self.col_item]
            .apply(lambda s: set(s.values))
            .to_dict()
        )

        return InteractionData(
            X_ui=X_ui,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            idx_to_user=users,
            idx_to_item=items,
            global_mean=global_mean,
            user_items_set=user_items_set,
        )

    def to_implicit(self, df: pd.DataFrame, threshold: float = 4.0) -> InteractionData:
        """
        Convert explicit ratings to implicit feedback for ALS.

        Only keeps positive interactions (rating >= threshold) for training.
        Values are set to 1.0 (binary) for implicit feedback.
        """
        # Build mappings from ALL data (full item/user universe)
        users = np.array(sorted(df[self.col_user].unique(), key=str), dtype=object)
        items = np.array(sorted(df[self.col_item].unique(), key=str), dtype=object)

        user_to_idx = {u: i for i, u in enumerate(users)}
        item_to_idx = {it: i for i, it in enumerate(items)}

        # Filter to positive interactions for the training matrix
        positive_df = df[df[self.col_rating] >= threshold]

        if len(positive_df) == 0:
            raise ValueError(f"No ratings >= {threshold} found in data")

        user_indices = positive_df[self.col_user].map(user_to_idx).values
        item_indices = positive_df[self.col_item].map(item_to_idx).values
        ratings = np.ones(len(positive_df), dtype=np.float32)  # Binary

        X_ui = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(len(users), len(items)),
        )

        # Build user_items_set from ALL ratings (not just positives)
        all_user_items_set = (
            df.groupby(self.col_user)[self.col_item]
            .apply(lambda s: set(s.values))
            .to_dict()
        )

        return InteractionData(
            X_ui=X_ui,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            idx_to_user=users,
            idx_to_item=items,
            global_mean=0.0,
            user_items_set=all_user_items_set,
        )

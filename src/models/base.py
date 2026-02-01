from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Iterable, Hashable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class InteractionData:
    """Canonical data structure for user-item interactions."""
    X_ui: csr_matrix  # users × items, values = ratings/confidence
    user_to_idx: dict  # user_id -> row index
    item_to_idx: dict  # item_id -> col index
    idx_to_user: np.ndarray  # row index -> user_id (array for vectorized access)
    idx_to_item: np.ndarray  # col index -> item_id (array for vectorized access)
    global_mean: float = 0.0
    user_items_set: dict = field(default_factory=dict)  # user_id -> set of item_ids


class BaseRecommender(ABC):
    """Abstract base class for recommender models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, data: InteractionData) -> None:
        """Train the model on interaction data."""
        pass

    @abstractmethod
    def recommend(self, user_id: Hashable, k: int = 10) -> List[Hashable]:
        """Generate top-K recommendations for a user."""
        pass

    def score(self, user_id: Hashable, item_id: Hashable) -> float:
        """Predict preference/utility for a user-item pair."""
        raise NotImplementedError

    def recommend_batch(self, user_ids: Iterable, k: int = 10) -> dict:
        """Generate recommendations for multiple users."""
        return {u: self.recommend(u, k) for u in user_ids}

    def similar_items(self, item_id: Hashable, k: int = 10) -> List[Hashable]:
        """Find similar items (only for models that support it)."""
        raise NotImplementedError

    def predict(
        self,
        df: pd.DataFrame,
        col_user: str = "user_id",
        col_item: str = "item_id",
    ) -> pd.DataFrame:
        """
        Generate predictions for user-item pairs in a DataFrame.

        Returns DataFrame with [user_id, item_id, prediction] for use
        with RMSE/MAE evaluation.
        """
        predictions = []
        for _, row in df.iterrows():
            pred = self.score(row[col_user], row[col_item])
            predictions.append(pred)

        return pd.DataFrame({
            col_user: df[col_user].values,
            col_item: df[col_item].values,
            "prediction": predictions,
        })

    def recommend_for_eval(
        self,
        user_ids,
        k: int = 10,
        col_user: str = "user_id",
        col_item: str = "item_id",
    ) -> pd.DataFrame:
        """
        Generate top-K recommendations for users.

        Returns DataFrame with [user_id, item_id, prediction] for use
        with ranking metrics (NDCG, Precision, Recall).
        """
        rows = []
        for user_id in user_ids:
            recs = self.recommend(user_id, k=k)
            for rank, item_id in enumerate(recs):
                rows.append({
                    col_user: user_id,
                    col_item: item_id,
                    "prediction": k - rank,  # Higher rank = higher score
                })
        return pd.DataFrame(rows)

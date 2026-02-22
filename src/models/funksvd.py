import numpy as np
import pandas as pd
from typing import List, Hashable

from lenskit.algorithms.funksvd import FunkSVD as LenskitFunkSVD

from src.models.base import BaseRecommender, InteractionData


class FunkSVD(BaseRecommender):
    """
    FunkSVD matrix factorization using LensKit.

    Predicts: r̂_ui = μ + b_u + b_i + p_u · q_i
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 100,
        lr_all: float = 0.001,
        reg_all: float = 0.015,
        damping: float = 5.0,
        rating_range: tuple = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__(name=f"FunkSVD-k{n_factors}-lr{lr_all}")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr_all
        self.reg = reg_all
        self.damping = damping
        self.rating_range = rating_range
        self.random_state = random_state
        self.verbose = verbose  # Note: LensKit FunkSVD doesn't use this, kept for API compat

        self.model: LenskitFunkSVD | None = None
        self.data: InteractionData | None = None
        self._all_items: pd.Index | None = None

    def fit(self, data: InteractionData) -> None:
        """Train via LensKit's FunkSVD."""
        self.data = data

        # Vectorized sparse->DataFrame conversion using array indexing
        coo = data.X_ui.tocoo()
        train_df = pd.DataFrame({
            'user': data.idx_to_user[coo.row],
            'item': data.idx_to_item[coo.col],
            'rating': coo.data,
        })

        self.model = LenskitFunkSVD(
            features=self.n_factors,
            iterations=self.n_epochs,
            lrate=self.lr,
            reg=self.reg,
            damping=self.damping,
            range=self.rating_range,
            random_state=self.random_state,
        )

        self.model.fit(train_df)

        # Cache all items as Index for fast set difference
        self._all_items = pd.Index(data.item_to_idx.keys())

    def score(self, user_id: Hashable, item_id: Hashable) -> float:
        """Predict rating for a user-item pair."""
        if self.model is None or self.data is None:
            return 0.0

        preds = self.model.predict_for_user(user_id, [item_id])
        if preds is None or len(preds) == 0 or pd.isna(preds.iloc[0]):
            return self.data.global_mean

        return float(preds.iloc[0])

    def recommend(self, user_id: Hashable, k: int = 10) -> List[Hashable]:
        """Generate top-K recommendations for a user."""
        if self.model is None or self.data is None:
            return []

        # Get candidates (all items not rated by user) via fast set difference
        rated_items = self.data.user_items_set.get(user_id, set())
        candidates = self._all_items.difference(rated_items)

        if len(candidates) == 0:
            return []

        # Score candidates and return top-k
        preds = self.model.predict_for_user(user_id, candidates)
        if preds is None or len(preds) == 0:
            return []

        preds = preds.dropna().sort_values(ascending=False)
        return list(preds.head(k).index)

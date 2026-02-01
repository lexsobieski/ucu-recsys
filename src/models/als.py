import numpy as np
import pandas as pd
from typing import List, Hashable

from lenskit.algorithms.als import BiasedMF, ImplicitMF

from src.models.base import BaseRecommender, InteractionData


class ALS(BaseRecommender):
    """
    ALS matrix factorization using LensKit.
    """

    def __init__(
        self,
        factors: int = 50,
        iterations: int = 20,
        regularization: float = 0.1,
        damping: float = 5.0,
        implicit: bool = True,
        weight: float = 40.0,
        random_state: int = 42,
    ):
        super().__init__(name=f"ALS-k{factors}-reg{regularization}")
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.damping = damping
        self.implicit = implicit
        self.weight = weight
        self.random_state = random_state

        self.model: BiasedMF | ImplicitMF | None = None
        self.data: InteractionData | None = None
        self._all_items: pd.Index | None = None

    def fit(self, data: InteractionData) -> None:
        """Train ALS model."""
        self.data = data

        # Vectorized sparse->DataFrame conversion using array indexing
        coo = data.X_ui.tocoo()
        train_df = pd.DataFrame({
            'user': data.idx_to_user[coo.row],
            'item': data.idx_to_item[coo.col],
            'rating': coo.data,
        })

        if self.implicit:
            self.model = ImplicitMF(
                features=self.factors,
                iterations=self.iterations,
                reg=self.regularization,
                weight=self.weight,
                rng_spec=self.random_state,
            )
        else:
            self.model = BiasedMF(
                features=self.factors,
                iterations=self.iterations,
                reg=self.regularization,
                damping=self.damping,
                rng_spec=self.random_state,
            )

        self.model.fit(train_df)

        # Cache all items as Index for fast set difference
        self._all_items = pd.Index(data.item_to_idx.keys())

    def score(self, user_id: Hashable, item_id: Hashable) -> float:
        """Predict score for a user-item pair."""
        if self.model is None or self.data is None:
            return 0.0

        preds = self.model.predict_for_user(user_id, [item_id])
        if preds is None or len(preds) == 0 or pd.isna(preds.iloc[0]):
            # Implicit: 0.0 (unknown preference), Explicit: global_mean
            return 0.0 if self.implicit else self.data.global_mean

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

    def similar_items(self, item_id: Hashable, k: int = 10) -> List[Hashable]:
        """Find similar items based on item factors."""
        if self.model is None or self.data is None:
            return []

        # Use LensKit's item index, not our data mappings
        item_index = self.model.item_index_
        if item_id not in item_index:
            return []

        item_factors = self.model.item_features_
        if item_factors is None:
            return []

        # Get row index in LensKit's factor matrix
        i_idx = item_index.get_loc(item_id)
        target_vec = item_factors[i_idx]

        # Compute cosine similarity with all items
        norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        normalized = item_factors / norms
        target_norm = target_vec / (np.linalg.norm(target_vec) or 1)

        similarities = normalized @ target_norm

        # Get top k+1 (excluding self), map back via LensKit's index
        top_indices = np.argsort(similarities)[::-1]
        result = []
        for j in top_indices:
            if j == i_idx:
                continue
            result.append(item_index[j])
            if len(result) == k:
                break

        return result

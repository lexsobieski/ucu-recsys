import numpy as np

from src.models.base import BaseRecommender, InteractionData


class PopularityRecommender(BaseRecommender):

    def __init__(self):
        super().__init__(name="Popularity")
        self.data = None
        self._item_popularity = None
        self._popular_item_indices = None

    def fit(self, data):
        self.data = data

        self._item_popularity = np.array(
            data.X_ui.getnnz(axis=0),
            dtype=np.float64,
        )

        self._popular_item_indices = np.argsort(self._item_popularity)[::-1]

    def score(self, user_id, item_id):
        if self.data is None or self._item_popularity is None:
            return 0.0

        item_index = self.data.item_to_idx.get(item_id)
        if item_index is None:
            return 0.0

        return float(self._item_popularity[item_index])

    def recommend(self, user_id, k):
        if self.data is None or self._popular_item_indices is None:
            return []

        seen_items = self.data.user_items_set.get(user_id, set())

        result = []
        for item_index in self._popular_item_indices:
            item_id = self.data.idx_to_item[item_index]
            if item_id not in seen_items:
                result.append(item_id)
                if len(result) == k:
                    break

        return result

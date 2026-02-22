import numpy as np

from src.models.base import BaseRecommender, InteractionData


class ItemGraphRecommender(BaseRecommender):

    def __init__(self):
        super().__init__(name="ItemGraph")
        self.data = None
        self.cooccurrence = None

    def fit(self, data):
        self.data = data

        self.cooccurrence = data.X_ui.T.dot(data.X_ui)
        self.cooccurrence.setdiag(0)
        self.cooccurrence.eliminate_zeros()

    def score(self, user_id, item_id):
        if self.data is None:
            return 0.0

        user_index = self.data.user_to_idx.get(user_id)
        item_index = self.data.item_to_idx.get(item_id)
        if user_index is None or item_index is None:
            return 0.0

        user_vector = np.asarray(self.data.X_ui[user_index].todense()).flatten()
        return float(self.cooccurrence[item_index].dot(user_vector.T).item())

    def recommend(self, user_id, k):
        if self.data is None:
            return []

        user_index = self.data.user_to_idx.get(user_id)
        if user_index is None:
            return []

        user_vector = np.asarray(self.data.X_ui[user_index].todense()).flatten()
        scores = self.cooccurrence.dot(user_vector)

        seen_items = self.data.user_items_set.get(user_id, set())
        for item_id in seen_items:
            item_index = self.data.item_to_idx.get(item_id)
            if item_index is not None:
                scores[item_index] = -np.inf

        top_k_indices = np.argsort(scores)[::-1][:k]
        return [self.data.idx_to_item[i] for i in top_k_indices]

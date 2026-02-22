import numpy as np

from src.models.base import BaseRecommender, InteractionData


class PersonalizedPageRankRecommender(BaseRecommender):

    def __init__(self, damping, iterations):
        super().__init__(name="PersonalizedPageRank")
        self.damping = damping
        self.iterations = iterations
        self.data = None
        self.propagate_to_items = None
        self.propagate_to_users = None

    def fit(self, data):
        self.data = data

        user_degrees = np.asarray(data.X_ui.sum(axis=1)).flatten()
        user_degrees[user_degrees == 0] = 1.0
        normalized_user_to_item = data.X_ui.multiply(
            1.0 / user_degrees[:, np.newaxis],
        )
        self.propagate_to_items = normalized_user_to_item.T.tocsr()

        item_degrees = np.asarray(data.X_ui.sum(axis=0)).flatten()
        item_degrees[item_degrees == 0] = 1.0
        normalized_item_to_user = data.X_ui.T.multiply(
            1.0 / item_degrees[:, np.newaxis],
        )
        self.propagate_to_users = normalized_item_to_user.T.tocsr()

    def _compute_item_scores(self, user_index):
        n_users = self.data.X_ui.shape[0]
        restart = np.zeros(n_users)
        restart[user_index] = 1.0

        user_scores = restart.copy()

        for _ in range(self.iterations):
            item_scores = self.propagate_to_items.dot(user_scores)
            user_scores = (
                self.damping * restart
                + (1.0 - self.damping) * self.propagate_to_users.dot(item_scores)
            )

        return self.propagate_to_items.dot(user_scores)

    def score(self, user_id, item_id):
        if self.data is None:
            return 0.0

        user_index = self.data.user_to_idx.get(user_id)
        item_index = self.data.item_to_idx.get(item_id)
        if user_index is None or item_index is None:
            return 0.0

        return float(self._compute_item_scores(user_index)[item_index])

    def recommend(self, user_id, k):
        if self.data is None:
            return []

        user_index = self.data.user_to_idx.get(user_id)
        if user_index is None:
            return []

        scores = self._compute_item_scores(user_index)

        seen_items = self.data.user_items_set.get(user_id, set())
        for item_id in seen_items:
            item_index = self.data.item_to_idx.get(item_id)
            if item_index is not None:
                scores[item_index] = -np.inf

        top_k_indices = np.argsort(scores)[::-1][:k]
        return [self.data.idx_to_item[i] for i in top_k_indices]

import numpy as np
from tqdm import tqdm

from src.models.base import BaseRecommender, InteractionData

INIT_STD = 0.01
SIGMOID_CLIP = 500.0
LOG_EPSILON = 1e-10


class BPR(BaseRecommender):

    def __init__(
        self,
        n_factors,
        learning_rate,
        reg,
        n_epochs,
        n_neg_samples,
        batch_size,
        seed,
        verbose,
    ):
        super().__init__(
            name=f"BPR-f{n_factors}-lr{learning_rate}-reg{reg}",
        )
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_neg_samples = n_neg_samples
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        self.data = None
        self.user_factors = None
        self.item_factors = None
        self.training_loss_history = []
        self.validation_metrics_history = []

    def _sample_negatives(self, rng, user_indices, user_positive_items, n_items):
        negative_indices = rng.randint(
            0,
            n_items,
            size=len(user_indices),
        )

        for position in range(len(user_indices)):
            while negative_indices[position] in user_positive_items[user_indices[position]]:
                negative_indices[position] = rng.randint(0, n_items)

        return negative_indices

    def fit(self, data, validation_callback=None):
        self.data = data
        n_users, n_items = data.X_ui.shape

        rng = np.random.RandomState(self.seed)

        self.user_factors = rng.normal(
            0,
            INIT_STD,
            size=(n_users, self.n_factors),
        )
        self.item_factors = rng.normal(
            0,
            INIT_STD,
            size=(n_items, self.n_factors),
        )

        coo = data.X_ui.tocoo()
        positive_user_indices = coo.row.astype(np.int32)
        positive_item_indices = coo.col.astype(np.int32)
        n_positives = len(positive_user_indices)

        user_positive_items = {}
        for user_index, item_index in zip(positive_user_indices, positive_item_indices):
            if user_index not in user_positive_items:
                user_positive_items[user_index] = set()
            user_positive_items[user_index].add(item_index)

        self.training_loss_history = []
        self.validation_metrics_history = []

        epoch_iterator = tqdm(
            range(self.n_epochs),
            desc="BPR Training",
            disable=not self.verbose,
        )

        for epoch in epoch_iterator:
            order = rng.permutation(n_positives)
            shuffled_users = positive_user_indices[order]
            shuffled_positive_items = positive_item_indices[order]

            epoch_loss = 0.0
            n_updates = 0

            for start_idx in range(0, n_positives, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_positives)
                batch_users = shuffled_users[start_idx:end_idx]
                batch_positive_items = shuffled_positive_items[start_idx:end_idx]

                for _ in range(self.n_neg_samples):
                    batch_negative_items = self._sample_negatives(
                        rng,
                        batch_users,
                        user_positive_items,
                        n_items,
                    )

                    user_vectors = self.user_factors[batch_users]
                    positive_vectors = self.item_factors[batch_positive_items]
                    negative_vectors = self.item_factors[batch_negative_items]

                    score_differences = np.sum(
                        user_vectors * (positive_vectors - negative_vectors),
                        axis=1,
                    )

                    clipped = np.clip(score_differences, -SIGMOID_CLIP, SIGMOID_CLIP)
                    sigmoid_negative = 1.0 / (1.0 + np.exp(clipped))

                    epoch_loss += np.sum(
                        np.logaddexp(0, -clipped),
                    )

                    gradient_multiplier = sigmoid_negative[:, np.newaxis]

                    user_gradient = gradient_multiplier * (positive_vectors - negative_vectors) - self.reg * user_vectors
                    positive_gradient = gradient_multiplier * user_vectors - self.reg * positive_vectors
                    negative_gradient = gradient_multiplier * (-user_vectors) - self.reg * negative_vectors

                    np.add.at(
                        self.user_factors,
                        batch_users,
                        self.learning_rate * user_gradient,
                    )
                    np.add.at(
                        self.item_factors,
                        batch_positive_items,
                        self.learning_rate * positive_gradient,
                    )
                    np.add.at(
                        self.item_factors,
                        batch_negative_items,
                        self.learning_rate * negative_gradient,
                    )

                    n_updates += len(batch_users)

            average_loss = epoch_loss / max(n_updates, 1)
            self.training_loss_history.append(average_loss)

            if validation_callback is not None:
                validation_result = validation_callback(self, epoch)
                self.validation_metrics_history.append(validation_result)

            if self.verbose:
                postfix = {"loss": f"{average_loss:.6f}"}
                if self.validation_metrics_history and "ndcg" in self.validation_metrics_history[-1]:
                    postfix["val_ndcg"] = f"{self.validation_metrics_history[-1]['ndcg']:.4f}"
                epoch_iterator.set_postfix(postfix)

    def score(self, user_id, item_id):
        if self.data is None or self.user_factors is None:
            return 0.0

        user_index = self.data.user_to_idx.get(user_id)
        item_index = self.data.item_to_idx.get(item_id)
        if user_index is None or item_index is None:
            return 0.0

        return float(np.dot(
            self.user_factors[user_index],
            self.item_factors[item_index],
        ))

    def recommend(self, user_id, k):
        if self.data is None or self.user_factors is None:
            return []

        user_index = self.data.user_to_idx.get(user_id)
        if user_index is None:
            return []

        scores = self.item_factors @ self.user_factors[user_index]

        seen_items = self.data.user_items_set.get(user_id, set())
        for item_id in seen_items:
            item_index = self.data.item_to_idx.get(item_id)
            if item_index is not None:
                scores[item_index] = -np.inf

        top_k_indices = np.argsort(scores)[::-1][:k]
        return [self.data.idx_to_item[i] for i in top_k_indices]

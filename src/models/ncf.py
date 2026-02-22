import tempfile
import os
import numpy as np
import pandas as pd
from typing import List, Hashable

from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset

from src.models.base import BaseRecommender, InteractionData


class NeuralCF(BaseRecommender):
    """
    Neural Collaborative Filtering (NCF) wrapper using the recommenders library.

    Supports GMF, MLP, and NeuMF model types. Operates on implicit feedback
    (binary: interacted or not). Predictions are sigmoid scores in [0, 1].
    """

    def __init__(
        self,
        model_type: str = "NeuMF",
        n_factors: int = 8,
        layer_sizes: list = None,
        n_epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 5e-3,
        n_neg: int = 4,
        seed: int = 42,
    ):
        super().__init__(name=f"NCF-{model_type}-k{n_factors}")
        self.model_type = model_type
        self.n_factors = n_factors
        self.layer_sizes = layer_sizes or [16, 8, 4]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_neg = n_neg
        self.seed = seed

        self.ncf_model = None
        self.data: InteractionData | None = None
        self._temp_dir = None

    def fit(self, data: InteractionData) -> None:
        """Train NCF model on interaction data."""
        self.data = data

        # Convert sparse matrix → DataFrame with integer indices, sorted by user
        coo = data.X_ui.tocoo()
        train_df = pd.DataFrame({
            "userID": coo.row.astype(int),
            "itemID": coo.col.astype(int),
            "rating": coo.data.astype(float),
        })
        train_df = train_df.sort_values("userID").reset_index(drop=True)

        # Write to temp CSV (NCFDataset reads from file)
        self._temp_dir = tempfile.mkdtemp()
        train_csv = os.path.join(self._temp_dir, "train.csv")
        train_df.to_csv(train_csv, index=False)

        # Create NCF dataset
        dataset = NCFDataset(
            train_file=train_csv,
            col_user="userID",
            col_item="itemID",
            col_rating="rating",
            n_neg=self.n_neg,
            seed=self.seed,
            binary=True,
        )

        # Create and train NCF model
        self.ncf_model = NCF(
            n_users=dataset.n_users,
            n_items=dataset.n_items,
            model_type=self.model_type,
            n_factors=self.n_factors,
            layer_sizes=self.layer_sizes,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=1,
            seed=self.seed,
        )
        self.ncf_model.fit(dataset)

    def score(self, user_id: Hashable, item_id: Hashable) -> float:
        """Predict score for a user-item pair (sigmoid output, 0-1)."""
        if self.ncf_model is None or self.data is None:
            return 0.0

        # Map our IDs to internal integer indices
        u_idx = self.data.user_to_idx.get(user_id)
        i_idx = self.data.item_to_idx.get(item_id)
        if u_idx is None or i_idx is None:
            return 0.0

        # Check that both exist in NCF's internal mappings
        # (items/users with no positive interactions won't be in the NCF model)
        if u_idx not in self.ncf_model.user2id or i_idx not in self.ncf_model.item2id:
            return 0.0

        return self.ncf_model.predict(u_idx, i_idx, is_list=False)

    def recommend(self, user_id: Hashable, k: int = 10) -> List[Hashable]:
        """Generate top-K recommendations for a user."""
        if self.ncf_model is None or self.data is None:
            return []

        u_idx = self.data.user_to_idx.get(user_id)
        if u_idx is None or u_idx not in self.ncf_model.user2id:
            return []

        # Get candidate items (all items minus user's rated items),
        # filtered to items NCF actually knows about
        rated_items = self.data.user_items_set.get(user_id, set())
        ncf_known_items = self.ncf_model.item2id
        candidate_ids = []
        candidate_indices = []
        for item_id, i_idx in self.data.item_to_idx.items():
            if item_id not in rated_items and i_idx in ncf_known_items:
                candidate_ids.append(item_id)
                candidate_indices.append(i_idx)

        if not candidate_ids:
            return []

        user_indices = [u_idx] * len(candidate_indices)

        # Batch predict
        scores = self.ncf_model.predict(
            np.array(user_indices),
            np.array(candidate_indices),
            is_list=True,
        )

        # Get top-k
        top_k_pos = np.argsort(scores)[::-1][:k]
        return [candidate_ids[i] for i in top_k_pos]

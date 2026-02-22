import numpy as np
import pandas as pd
from typing import List, Hashable

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from src.models.base import BaseRecommender, InteractionData


class MultVAE(BaseRecommender):
    """
    Multinomial Variational Autoencoder (Multi-VAE) for Collaborative Filtering.

    Implements the model from Liang et al. "Variational Autoencoders for
    Collaborative Filtering" (WWW 2018). Operates on implicit feedback
    (binary click matrix). Encodes full user rows through a VAE and
    reconstructs item scores for ranking.
    """

    def __init__(
        self,
        intermediate_dim: int = 200,
        latent_dim: int = 70,
        n_epochs: int = 100,
        batch_size: int = 100,
        beta: float = 1.0,
        drop_encoder: float = 0.5,
        drop_decoder: float = 0.5,
        learning_rate: float = 0.001,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(name=f"MultVAE-d{latent_dim}-e{n_epochs}")
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta = beta
        self.drop_encoder = drop_encoder
        self.drop_decoder = drop_decoder
        self.learning_rate = learning_rate
        self.val_fraction = val_fraction
        self.seed = seed

        self._model = None
        self.data: InteractionData | None = None
        self._train_matrix: np.ndarray | None = None

    def _build_model(self, original_dim: int):
        """Build the VAE Keras model.

        Uses add_loss() for the KL term so intermediate symbolic tensors
        (z_mean, z_log_var) stay within the graph — compatible with tf_keras 2.20.
        """
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Encoder
        x_input = Input(shape=(original_dim,))
        x_norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(x_input)
        h_enc = Dropout(self.drop_encoder)(x_norm)
        h_enc = Dense(self.intermediate_dim, activation="tanh")(h_enc)
        z_mean = Dense(self.latent_dim)(h_enc)
        z_log_var = Dense(self.latent_dim)(h_enc)

        # Reparameterization trick
        def sampling(args):
            _mean, _log_var = args
            epsilon = K.random_normal(
                shape=(K.shape(_mean)[0], self.latent_dim),
                mean=0.0, stddev=1.0, seed=self.seed,
            )
            return _mean + K.exp(_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        # Decoder
        h_dec = Dense(self.intermediate_dim, activation="tanh")(z)
        h_dec = Dropout(self.drop_decoder)(h_dec)
        x_decoded = Dense(original_dim)(h_dec)

        model = Model(x_input, x_decoded)

        # KL divergence added via add_loss (keeps symbolic tensors in-graph)
        kl_loss = -0.5 * K.mean(
            K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        )
        model.add_loss(self.beta * kl_loss)

        # Reconstruction loss: multinomial NLL (compiled loss only sees x_true, x_pred)
        def nll_loss(x_true, x_pred):
            log_softmax = tf.nn.log_softmax(x_pred)
            return -tf.reduce_mean(
                tf.reduce_sum(log_softmax * x_true, axis=-1)
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=nll_loss,
        )
        return model

    def fit(self, data: InteractionData) -> None:
        """Train MultVAE on interaction data."""
        self.data = data
        n_users, n_items = data.X_ui.shape

        # Convert to dense binary matrix
        full_matrix = np.array(data.X_ui.todense(), dtype=np.float32)
        full_matrix[full_matrix > 0] = 1.0

        # Split users into train/validation sets
        rng = np.random.RandomState(self.seed)
        n_val_users = max(1, int(n_users * self.val_fraction))
        val_user_indices = rng.choice(n_users, size=n_val_users, replace=False)
        train_user_mask = np.ones(n_users, dtype=bool)
        train_user_mask[val_user_indices] = False

        x_train = full_matrix[train_user_mask]
        x_valid = full_matrix[val_user_indices]

        # Store the full training matrix (all users) for predictions later
        self._train_matrix = full_matrix

        # Build and train
        self._model = self._build_model(n_items)

        self._model.fit(
            x_train, x_train,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            validation_data=(x_valid, x_valid),
            shuffle=True,
            verbose=1,
        )

    def score(self, user_id: Hashable, item_id: Hashable) -> float:
        """Predict score for a user-item pair."""
        if self._model is None or self.data is None:
            return 0.0

        u_idx = self.data.user_to_idx.get(user_id)
        i_idx = self.data.item_to_idx.get(item_id)
        if u_idx is None or i_idx is None:
            return 0.0

        user_row = self._train_matrix[u_idx:u_idx + 1]
        scores = self._model.predict(user_row, verbose=0)
        return float(scores[0, i_idx])

    def recommend(self, user_id: Hashable, k: int = 10) -> List[Hashable]:
        """Generate top-K recommendations for a user."""
        if self._model is None or self.data is None:
            return []

        u_idx = self.data.user_to_idx.get(user_id)
        if u_idx is None:
            return []

        user_row = self._train_matrix[u_idx:u_idx + 1]
        scores = self._model.predict(user_row, verbose=0)[0]

        # Mask already-rated items
        rated_items = self.data.user_items_set.get(user_id, set())
        for it in rated_items:
            i_idx = self.data.item_to_idx.get(it)
            if i_idx is not None:
                scores[i_idx] = -np.inf

        # Return top-k item IDs
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [self.data.idx_to_item[i] for i in top_k_indices]

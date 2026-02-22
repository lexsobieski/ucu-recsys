import pandas as pd
from typing import Optional

from src.models.base import BaseRecommender
from src.models.content_based import ContentBasedRecommender


class CascadeHybridRecommender:
    """
    Candidate Generation + Reranking Hybrid Recommender.

    Architecture:
    1. Collaborative Filtering retrieves top N candidates (High Recall)
    2. Reranker (Content-Based) scores these N candidates (High Precision)
    3. Final recommendations are the top K items sorted by the Reranker's score
    """

    def __init__(
        self,
        generator_model: BaseRecommender,
        reranker_model: ContentBasedRecommender,
        num_candidates: int = 100,
    ) -> None:
        self.generator_model = generator_model
        self.reranker_model = reranker_model
        self.num_candidates = num_candidates
        self.train_pivot: Optional[pd.DataFrame] = None

    def fit(self, train_pivot: pd.DataFrame) -> None:
        """
        Store the train_pivot matrix needed for the content-based reranker
        Assume the base models are already fitted before being passed into this hybrid
        """
        self.train_pivot = train_pivot

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        candidates_df = self.generator_model.recommend(user_id, top_k=self.num_candidates)

        if candidates_df is None or candidates_df.empty:
            return pd.DataFrame()

        candidate_item_ids = set(candidates_df["item_id"].values)
        total_items = len(self.reranker_model.sim_df.columns)

        cb_recs_df = self.reranker_model.recommend(user_id, self.train_pivot, top_k=total_items)

        if cb_recs_df is None or cb_recs_df.empty:
            return candidates_df.head(top_k)

        reranked_df = cb_recs_df[cb_recs_df["item_id"].isin(candidate_item_ids)].copy()
        final_recs = reranked_df.head(top_k)

        if len(final_recs) < top_k:
            missing_count = top_k - len(final_recs)
            used_items = set(final_recs["item_id"].values)
            backfill_candidates = candidates_df[~candidates_df["item_id"].isin(used_items)]
            final_recs = pd.concat([final_recs, backfill_candidates.head(missing_count)])

        return final_recs.head(top_k)

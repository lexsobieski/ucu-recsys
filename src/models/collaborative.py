import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemRecommender:
    
    def __init__(self, method='cosine'):
        self.method = method
        self.similarity_matrix = None
        self.item_ids = None
        self.train_matrix = None
        
    def fit(self, train_df: pd.DataFrame, user_col='user_id', item_col='item_id', rating_col='rating'):
        self.train_matrix = train_df.pivot(index=user_col, columns=item_col, values=rating_col).fillna(0)
        self.item_ids = self.train_matrix.columns
        
        if self.method == 'cosine':
            self.similarity_matrix = cosine_similarity(self.train_matrix.T)
            
        elif self.method == 'pearson':
            mask = self.train_matrix > 0
            
            item_means = self.train_matrix.replace(0, np.nan).mean(axis=0)
            
            train_centered = self.train_matrix.sub(item_means, axis=1)
            train_centered = train_centered.where(mask, 0)
            
            self.similarity_matrix = cosine_similarity(train_centered.T)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        self.sim_df = pd.DataFrame(self.similarity_matrix, index=self.item_ids, columns=self.item_ids)
        
    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        if user_id not in self.train_matrix.index:
            return pd.DataFrame()
            
        user_ratings = self.train_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        valid_rated = rated_items.intersection(self.sim_df.index)
        
        if len(valid_rated) == 0:
            return pd.DataFrame()
            
        u_vec = user_ratings[valid_rated].values.reshape(1, -1)
        sim_subset = self.sim_df.loc[valid_rated].values
        
        scores = u_vec.dot(sim_subset).flatten()
        
        scores_series = pd.Series(scores, index=self.sim_df.columns)
        scores_series = scores_series.drop(index=rated_items, errors='ignore')
        
        top_items = scores_series.nlargest(top_k)
        
        return pd.DataFrame({
            'user_id': user_id,
            'item_id': top_items.index,
            'prediction': top_items.values
        })

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class ContentBasedRecommender:
    
    def __init__(self, item_col='item_id', similarity_method='jaccard'):
        self.item_col = item_col
        self.similarity_method = similarity_method
        self.similarity_matrix = None
        self.item_ids = None
        
    def _compute_jaccard_similarity(self, df: pd.DataFrame):
        df['genres_list'] = df['genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(df['genres_list'])
        
        jaccard_dist = pairwise_distances(genres_encoded.astype(bool), metric='jaccard')
        self.similarity_matrix = 1 - jaccard_dist
    
    def _compute_tfidf_similarity(self, df: pd.DataFrame):
        df['title_clean'] = df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        df['genres_str'] = df['genres'].str.replace('|', ' ')
        df['metadata_soup'] = df['title_clean'] + " " + df['genres_str']
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['metadata_soup'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
    
    def _compute_cosine_similarity(self, df: pd.DataFrame):
        if 'year' not in df.columns:
             df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
        df['year'] = df['year'].fillna(df['year'].median())
        scaler = MinMaxScaler()
        year_scaled = scaler.fit_transform(df[['year']])
        
        df['genres_list'] = df['genres'].str.split('|')
        mlb = MultiLabelBinarizer()
        genres_encoded = mlb.fit_transform(df['genres_list'])
        
        features = np.hstack([genres_encoded, year_scaled])
        self.similarity_matrix = cosine_similarity(features)
        
    def fit(self, items_df: pd.DataFrame):
        df = items_df.copy()
        self.item_ids = df[self.item_col].values
        
        if self.similarity_method == 'jaccard':
            self._compute_jaccard_similarity(df)
        elif self.similarity_method == 'tfidf':
            self._compute_tfidf_similarity(df)
        elif self.similarity_method == 'cosine':
            self._compute_cosine_similarity(df)
        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")
            
        self.sim_df = pd.DataFrame(self.similarity_matrix, index=self.item_ids, columns=self.item_ids)
        
    def recommend(self, user_id: int, user_history_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        if user_id not in user_history_df.index:
            return pd.DataFrame()
            
        user_ratings = user_history_df.loc[user_id]
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

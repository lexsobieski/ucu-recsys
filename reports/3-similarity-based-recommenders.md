# Similarity-Based Recommenders Analysis

This report documents the implementation and evaluation of similarity-based recommendation models for the MovieLens 1M dataset. The goal was to compare content-based approaches against collaborative filtering and explore if a hybrid combination could outperform individual models.

## 1. Models implemented

We implemented modular recommender classes in `src/models/`, focusing on item-item similarity.

### A. Content-Based filtering (`src/models/content_based.py`)
Uses item metadata to find similar movies. We tested two feature extraction strategies:
1. Jaccard Similarity (Baseline): Treating genres as binary sets. This is simple but robust for categorical data
2. TF-IDF: Combining title and genres into a text string. This helps identify sequels (e.g., "Toy Story" vs "Toy Story 2") and down-weights overly common genres like "Drama"

### B. Collaborative filtering (`src/models/collaborative.py`)
Uses user interaction history. We chose item-item CF because the number of movies (3.8k) is much smaller than the number of users (6k), making the similarity matrix denser and more stable.
1. Cosine Similarity: Standard vector similarity on the user-item matrix.
2. Pearson Correlation: Centers the data by subtracting item mean ratings before computing cosine. This theoretically handles user bias better but can be sensitive to sparsity.

### C. Hybrid model
A simple linear weighted ensemble:
$$ Score = \alpha \cdot Score_{CF} + (1 - \alpha) \cdot Score_{CB} $$
We tested $\alpha=0.8$, giving primary weight to the stronger CF model while using CB to fill in gaps.

---

## 2. Experiment results

We evaluated models using a temporal split (Train 80% / Test 10%) and ranking metrics at K=10.

| Model | NDCG@10 | Precision@10 | Recall@10 |
| :--- | :--- | :--- | :--- |
| **CF Item-Item (Cosine)** | **0.1488** | **0.1380** | **0.0865** |
| CF Item-Item (Pearson) | 0.1291 | 0.1160 | 0.0609 |
| Hybrid (CF + TF-IDF) | 0.1410 | 0.1260 | 0.0826 |
| CB (Jaccard) | 0.0331 | 0.0320 | 0.0131 |
| CB (TF-IDF) | 0.0306 | 0.0290 | 0.0170 |

---

## 3. Key findings & interpretation

### Why collaborative filtering won
Collaborative filtering with cosine similarity was the clear winner, outperforming the best content-based model by a factor of approximately 4x in NDCG. MovieLens 1M is a dense dataset. User behavior patterns (co-viewing) contain much richer signals about quality and taste than static metadata. Pearson performed worse than standard cosine. In sparse item-item matrices, mean-centering can sometimes introduce noise or dampen the signal of co-rated items, making the simpler cosine metric more robust here.

### Content-Based: Precision vs. Recall
Jaccard (genres) had better top-ranking precision (NDCG) than TF-IDF. Genres are a strong, explicit signal for grouping movies. TF-IDF (titles) slightly improved recall (0.017 vs 0.013). By matching keywords in titles, the model could find specific relevant items (like sequels) that Jaccard missed, even if it ranked them lower on average

### The hybrid failure
Mixing the models ($\alpha=0.8$) did not improve performance compared to pure CF. For users with sufficient history (which is most users in MovieLens 1M), the content-based signal is significantly weaker and acts as noise when added to the high-quality CF signal. Instead of a weighted ensemble for everyone, a switching hybrid would be better: use CF for active users, and fall back to content-based only for new users/items (cold start)

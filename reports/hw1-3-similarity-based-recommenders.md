# Similarity-Based Recommenders Analysis

This report documents the implementation and evaluation of similarity-based recommendation models for the MovieLens 1M dataset. The goal was to compare content-based approaches against collaborative filtering and explore if a hybrid combination could outperform individual models

## 1. Models implemented

We implemented modular recommender classes in `src/models/`, focusing on item-item similarity

### A. Content-Based filtering (`src/models/content_based.py`)
Uses item metadata to find similar movies. We tested two feature extraction strategies:
1. Jaccard Similarity (Baseline): Treating genres as binary sets. This is simple but robust for categorical data
2. TF-IDF: Combining title and genres into a text string. This helps identify sequels (e.g., "Toy Story" vs "Toy Story 2") and down-weights overly common genres like "Drama"

### B. Collaborative filtering (`src/models/collaborative.py`)
Uses user interaction history. We chose item-item CF because the number of movies (3.8k) is much smaller than the number of users (6k), making the similarity matrix denser and more stable
1. Cosine Similarity: Standard vector similarity on the user-item matrix
2. Pearson Correlation: Centers the data by subtracting item mean ratings before computing cosine. This theoretically handles user bias better but can be sensitive to sparsity

### C. Hybrid model
A simple linear weighted ensemble:
$$ Score = \alpha \cdot Score_{CF} + (1 - \alpha) \cdot Score_{CB} $$
We tested $\alpha=0.8$, giving primary weight to the stronger CF model while using CB to fill in gaps

---

## 2. Experiment results

We evaluated models using a temporal split (Train 80% / Test 10%) and ranking metrics at K=10, 20, and 50

| Model | NDCG@10 | Precision@10 | Recall@10 |
| :--- | :--- | :--- | :--- |
| **CF Item-Item (Cosine)** | **0.1620** | **0.1490** | **0.1054** |
| Hybrid (alpha=0.8) | 0.1598 | 0.1450 | 0.1037 |
| CF Item-Item (Pearson) | 0.1177 | 0.1060 | 0.0664 |
| CB (TF-IDF) | 0.0301 | 0.0200 | 0.0201 |
| CB (Jaccard) | 0.0116 | 0.0110 | 0.0052 |

---

## 3. Key findings & interpretation

### Why collaborative filtering won
Collaborative filtering with cosine similarity was the clear winner (NDCG@10 = 0.1620), massively outperforming the best content-based model. MovieLens 1M is a dense dataset. User behavior patterns (co-viewing) contain much richer signals about quality and taste than static metadata. Pearson performed worse than standard cosine. In sparse item-item matrices, mean-centering can sometimes introduce noise or dampen the signal of co-rated items, making the simpler cosine metric more robust here

### Content-Based: Precision vs. Recall
Unlike initial tests, TF-IDF (titles + genres) clearly outperformed pure Jaccard (genres) across all metrics (NDCG 0.030 vs 0.011 at K=10). By matching keywords in titles, the model could find specific relevant items (like direct sequels or franchises) that pure genre-matching missed, leading to better precision and recall deeper in the list (Recall@50: 0.064 vs 0.033)

### Hybrid performance analysis
The weighted hybrid model ($\alpha=0.8$) slightly underperformed pure CF at K=10 (NDCG 0.1598 vs 0.1620), confirming that naive blending injects noise into strong collaborative signals. However, as the list size grew to K=20 and K=50, the hybrid model matched or marginally improved certain metrics (e.g., NDCG@20: 0.1663 vs 0.1649 for pure CF). This suggests that content metadata can occasionally help surface relevant "tail" items deeper in the recommendation list, but for the very top recommendations, trusting pure crowd behavior is safer

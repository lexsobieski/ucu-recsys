# UCU Recommender Systems Capstone

Classical recommender systems on MovieLens 1M dataset.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Structure

```
├── data/
│   ├── raw/               # movies.dat, ratings.dat, users.dat
│   └── processed/         # train.parquet, test.parquet
├── artifacts/             # Trained model artifacts
├── experiments/           # Dated experiment folders
│   └── YYYYMMDD_descriptive_name/
│       ├── results/
│       ├── description.md
│       └── experiment.ipynb
├── scripts/               # Utility scripts
├── src/
│   ├── models/
│   │   ├── base.py        # BaseRecommender ABC
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   └── matrix_fact.py
│   ├── constants.py       # DataSchema
│   ├── metrics.py         # RMSE, Precision@K, Recall@K
│   └── utils.py
└── requirements.txt
```

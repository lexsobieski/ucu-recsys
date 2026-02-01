# Recommender Systems
## Capstone Project

**Macrodata Refinement** Team

- Mykyta Berehulia
- Oleksandr Sobetskyi
- Mark Matviiv

## Project Setup

Python: **3.12.10**

```shell
python -m venv .venv
source .venv/bin/activate

# macOS
pip install -r requirements_mac.txt

# Windows
pip install -r requirements_win.txt
```

## Dataset Download

```shell
python scripts/download_movielens.py
```

## Project Structure

```
├── artifacts/
├── data/
│   └── ml-1m/
│       ├── movies.dat
│       ├── ratings.dat
│       └── users.dat
├── experiments/
│   ├── als-tuning.ipynb
│   ├── exploratory-data-analysis.ipynb
│   ├── funksvd-tuning.ipynb
│   ├── matrix-factorization.ipynb
│   ├── metrics-example.ipynb
│   └── similarity-based-recommenders.ipynb
├── reports/
│   ├── 1-exploratory-data-analysis.md
│   ├── 2-offline-evaluation-strategy.md
│   ├── 3-similarity-based-recommenders.md
│   ├── 4-matrix-factorization.md
│   └── 5-final-summary.md
├── scripts/
│   └── download_movielens.py
├── src/
│   ├── data/
│   │   ├── adapter.py
│   │   ├── loader.py
│   │   └── splitter.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── models/
│       ├── als.py
│       ├── base.py
│       ├── collaborative.py
│       └── content_based.py
├── requirements_mac.txt
└── requirements_win.txt
```

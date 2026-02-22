# Recommender Systems
## Capstone Project

**Macrodata Refinement** Team

- Mykyta Berehulia
- Oleksandr Sobetskyi
- Mark Matviiv

## Project Setup

Python: **3.11.9**

```shell
pyenv install 3.11.9
pyenv local 3.11.9

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
│   ├── multvae-tuning.ipynb
│   ├── ncf-tuning.ipynb
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
│       ├── content_based.py
│       ├── funksvd.py
│       ├── multvae.py
│       └── ncf.py
└── requirements.txt
```

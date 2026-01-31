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
│   ├── exploratory-data-analysis.ipynb
│   ├── matrix-factorization.ipynb
│   └── similarity-based-recommenders.ipynb
├── scripts/
│   └── download_movielens.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── splitter.py
│   └── evaluation/
│       └── metrics.py
├── requirements.txt
```

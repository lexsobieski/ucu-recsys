from typing import Any

import pandas as pd
from recommenders.evaluation import python_evaluation

COLUMN_USER = "user_id"
COLUMN_ITEM = "item_id"
COLUMN_RATING = "rating"
COLUMN_PREDICTION = "prediction"


def compute_ndcg(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int
) -> float:
    return python_evaluation.ndcg_at_k(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION,
        k=top_k
    )


def compute_map(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int
) -> float:
    return python_evaluation.map_at_k(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION,
        k=top_k
    )


def compute_precision(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int
) -> float:
    return python_evaluation.precision_at_k(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION,
        k=top_k
    )


def compute_recall(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int
) -> float:
    return python_evaluation.recall_at_k(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION,
        k=top_k
    )


def compute_rmse(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    return python_evaluation.rmse(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION
    )


def compute_mae(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    return python_evaluation.mae(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION
    )


def compute_r_squared(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    return python_evaluation.rsquared(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION
    )


def compute_coverage(predictions: pd.DataFrame, catalog: list[Any]) -> float:
    return python_evaluation.catalog_coverage(
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        catalog=catalog
    )


def compute_novelty(train_dataframe: pd.DataFrame, predictions: pd.DataFrame) -> float:
    return python_evaluation.novelty(
        train_df=train_dataframe,
        reco_df=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM
    )


def compute_diversity(train_dataframe: pd.DataFrame, predictions: pd.DataFrame) -> float:
    return python_evaluation.diversity(
        train_df=train_dataframe,
        reco_df=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM
    )


def compute_ranking_metrics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int
) -> dict[str, float]:
    return {
        "ndcg": compute_ndcg(ground_truth, predictions, top_k),
        "map": compute_map(ground_truth, predictions, top_k),
        "precision": compute_precision(ground_truth, predictions, top_k),
        "recall": compute_recall(ground_truth, predictions, top_k)
    }


def compute_rating_metrics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame
) -> dict[str, float]:
    return {
        "rmse": compute_rmse(ground_truth, predictions),
        "mae": compute_mae(ground_truth, predictions),
        "r_squared": compute_r_squared(ground_truth, predictions)
    }


def compute_beyond_accuracy_metrics(
    train_dataframe: pd.DataFrame,
    predictions: pd.DataFrame,
    catalog: list[Any] | None = None
) -> dict[str, float]:
    if catalog is None:
        catalog = train_dataframe[COLUMN_ITEM].unique().tolist()

    return {
        "coverage": compute_coverage(predictions, catalog),
        "novelty": compute_novelty(train_dataframe, predictions),
        "diversity": compute_diversity(train_dataframe, predictions)
    }


def compute_all_metrics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    top_k: int,
    train_dataframe: pd.DataFrame,
    catalog: list[Any] | None = None
) -> dict[str, float]:
    if catalog is None:
        catalog = train_dataframe[COLUMN_ITEM].unique().tolist()

    return {
        "ndcg": compute_ndcg(ground_truth, predictions, top_k),
        "map": compute_map(ground_truth, predictions, top_k),
        "precision": compute_precision(ground_truth, predictions, top_k),
        "recall": compute_recall(ground_truth, predictions, top_k),
        "rmse": compute_rmse(ground_truth, predictions),
        "mae": compute_mae(ground_truth, predictions),
        "r_squared": compute_r_squared(ground_truth, predictions),
        "coverage": compute_coverage(predictions, catalog),
        "novelty": compute_novelty(train_dataframe, predictions),
        "diversity": compute_diversity(train_dataframe, predictions)
    }

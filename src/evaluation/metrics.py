import pandas as pd
from sklearn.metrics import explained_variance_score
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


def compute_rmse(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame
) -> float:
    return python_evaluation.rmse(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION
    )


def compute_mae(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame
) -> float:
    return python_evaluation.mae(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION
    )


def compute_r_squared(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame
) -> float:
    return python_evaluation.rsquared(
        rating_true=ground_truth,
        rating_pred=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM,
        col_rating=COLUMN_RATING,
        col_prediction=COLUMN_PREDICTION
    )


def compute_explained_variance(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame
) -> float:
    merged = ground_truth.merge(
        predictions,
        on=[COLUMN_USER, COLUMN_ITEM]
    )
    return explained_variance_score(
        y_true=merged[COLUMN_RATING],
        y_pred=merged[COLUMN_PREDICTION]
    )


def compute_coverage(
    train_dataframe: pd.DataFrame,
    predictions: pd.DataFrame
) -> float:
    return python_evaluation.catalog_coverage(
        train_df=train_dataframe,
        reco_df=predictions,
        col_user=COLUMN_USER,
        col_item=COLUMN_ITEM
    )


def compute_novelty(
    train_dataframe: pd.DataFrame,
    predictions: pd.DataFrame
) -> float:
    return python_evaluation.novelty(
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
        "ndcg": compute_ndcg(
            ground_truth=ground_truth,
            predictions=predictions,
            top_k=top_k
        ),
        "map": compute_map(
            ground_truth=ground_truth,
            predictions=predictions,
            top_k=top_k
        ),
        "precision": compute_precision(
            ground_truth=ground_truth,
            predictions=predictions,
            top_k=top_k
        ),
        "recall": compute_recall(
            ground_truth=ground_truth,
            predictions=predictions,
            top_k=top_k
        )
    }


def compute_rating_metrics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame
) -> dict[str, float]:
    return {
        "rmse": compute_rmse(
            ground_truth=ground_truth,
            predictions=predictions
        ),
        "mae": compute_mae(
            ground_truth=ground_truth,
            predictions=predictions
        ),
        "r_squared": compute_r_squared(
            ground_truth=ground_truth,
            predictions=predictions
        ),
        "explained_variance": compute_explained_variance(
            ground_truth=ground_truth,
            predictions=predictions
        )
    }


def compute_beyond_accuracy_metrics(
    train_dataframe: pd.DataFrame,
    predictions: pd.DataFrame
) -> dict[str, float]:
    return {
        "coverage": compute_coverage(
            train_dataframe=train_dataframe,
            predictions=predictions
        ),
        "novelty": compute_novelty(
            train_dataframe=train_dataframe,
            predictions=predictions
        )
    }



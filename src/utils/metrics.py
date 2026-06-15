"""평가지표 — TASK별로 달라지는 채점 기준을 한곳에 모은다.

값이 클수록 좋게 통일한다(회귀는 -RMSE 반환). 이렇게 하면 Optuna는 항상
maximize, 앙상블 가중치 탐색도 항상 maximize로 통일돼서 코드가 단순해진다.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    mean_squared_error,
)


def get_metric(task: str, y_true, y_pred):
    """task에 맞는 점수를 반환 (클수록 좋음).

    Parameters
    ----------
    task : "binary" | "multiclass" | "regression"
    y_pred :
        - binary: 양성 확률 (1차원)
        - multiclass: 클래스 확률 (N x C)
        - regression: 예측값 (1차원)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task == "binary":
        return roc_auc_score(y_true, y_pred)
    elif task == "multiclass":
        return f1_score(y_true, y_pred.argmax(axis=1), average="macro")
    elif task == "regression":
        return -np.sqrt(mean_squared_error(y_true, y_pred))
    raise ValueError(f"unknown task: {task}")


def metric_name(task: str) -> str:
    return {"binary": "AUC", "multiclass": "MacroF1", "regression": "-RMSE"}[task]


def to_submission(task: str, pred):
    """최종 예측을 제출 형태로 변환."""
    pred = np.asarray(pred)
    if task == "multiclass":
        return pred.argmax(axis=1)
    return pred  # binary는 확률, regression은 값 그대로 (대회 양식에 맞게 조정)


# ── Task 2: 선박 ID 검색 ────────────────────────────────────────────────────

def retrieval_score(y_true_ids, top5_pred_ids):
    """Task 2 평가: Recall@1×0.5 + Recall@3×0.3 + Recall@5×0.2

    Parameters
    ----------
    y_true_ids    : list[int]        정답 ship_id (쿼리 수 N)
    top5_pred_ids : list[list[int]]  예측 Top-5 ship_id 리스트 (N × 5)

    Returns
    -------
    score : float  (0~1, 클수록 좋음)
    """
    n = len(y_true_ids)
    r1 = sum(t == p[0]    for t, p in zip(y_true_ids, top5_pred_ids)) / n
    r3 = sum(t in p[:3]   for t, p in zip(y_true_ids, top5_pred_ids)) / n
    r5 = sum(t in p[:5]   for t, p in zip(y_true_ids, top5_pred_ids)) / n
    return r1 * 0.5 + r3 * 0.3 + r5 * 0.2


def recall_at_k(y_true_ids, top5_pred_ids):
    """Recall@1, @3, @5를 각각 반환 (로깅용)."""
    n = len(y_true_ids)
    r1 = sum(t == p[0]  for t, p in zip(y_true_ids, top5_pred_ids)) / n
    r3 = sum(t in p[:3] for t, p in zip(y_true_ids, top5_pred_ids)) / n
    r5 = sum(t in p[:5] for t, p in zip(y_true_ids, top5_pred_ids)) / n
    return {"R@1": r1, "R@3": r3, "R@5": r5}

"""교차검증 전략 — task와 상황에 맞는 fold 분할기를 만든다.

핵심: 분류는 StratifiedKFold(클래스 비율 유지), 회귀는 KFold,
녹음/그룹 단위 누수가 있으면 GroupKFold를 써야 한다.
"""
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    GroupKFold,
    StratifiedGroupKFold,
)


def get_folder(task: str, n_splits: int = 5, seed: int = 42, group: bool = False):
    """fold 분할기를 반환.

    Parameters
    ----------
    task : "binary" | "multiclass" | "regression"
    group : True면 그룹 단위 누수 방지 (녹음 ID, 환자 ID 등)
    """
    is_clf = task in ("binary", "multiclass")

    if group:
        if is_clf:
            # 클래스 비율도 맞추고 그룹 누수도 막음 (가능하면 이게 최선)
            return StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
        return GroupKFold(n_splits=n_splits)

    if is_clf:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def split(folder, X, y, groups=None):
    """folder.split을 group 유무에 따라 호출하는 래퍼."""
    if groups is not None:
        return folder.split(X, y, groups=groups)
    return folder.split(X, y)

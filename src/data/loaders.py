"""데이터 로더 — task별 데이터를 읽어 (X, y, groups, test) 형태로 통일 반환.

실제 대회에선 이 파일의 경로/컬럼명만 수정하면 된다.
데모 모드(demo=True)는 합성 데이터를 생성해 파이프라인 전체를 검증할 수 있다.
"""
import numpy as np
import pandas as pd


def load_tabular(cfg, demo: bool = True):
    """정형 데이터 로드. (X, y, groups, X_test, test_ids) 반환."""
    if demo:
        from sklearn.datasets import make_classification, make_regression
        task = cfg["task"]
        if task == "binary":
            X, y = make_classification(n_samples=2000, n_features=15,
                                       n_informative=8, n_classes=2, random_state=cfg["seed"])
        elif task == "multiclass":
            X, y = make_classification(n_samples=2000, n_features=15, n_informative=10,
                                       n_classes=4, n_clusters_per_class=1, random_state=cfg["seed"])
        else:
            X, y = make_regression(n_samples=2000, n_features=15, n_informative=8,
                                   noise=15.0, random_state=cfg["seed"])
        cols = [f"f{i}" for i in range(X.shape[1])]
        Xdf = pd.DataFrame(X, columns=cols)
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(
            pd.concat([Xdf, pd.Series(y, name="target")], axis=1),
            test_size=0.25, random_state=cfg["seed"],
        )
        X_tr = tr[cols].reset_index(drop=True)
        y_tr = tr["target"].values
        X_te = te[cols].reset_index(drop=True)
        return X_tr, y_tr, None, X_te, np.arange(len(X_te))

    # ===== 실제 대회: 아래를 본인 데이터에 맞게 =====
    train = pd.read_csv(cfg["paths"]["train"])
    test = pd.read_csv(cfg["paths"]["test"])
    target = cfg["target_col"]
    id_col = cfg.get("id_col")
    feature_cols = [c for c in train.columns if c not in [target, id_col]]
    groups = train[cfg["group_col"]].values if cfg.get("group_col") else None
    test_ids = test[id_col].values if id_col else np.arange(len(test))
    return train[feature_cols], train[target].values, groups, test[feature_cols], test_ids


def load_vision(cfg, demo: bool = True):
    """비전: 이미지 경로+라벨 DataFrame 반환. 실제 학습은 Dataset에서 처리."""
    if demo:
        # 데모는 train.py의 vision 분기에서 합성 텐서로 대체
        return pd.DataFrame({"filepath": [], "label": []})
    df = pd.read_csv(cfg["paths"]["train"])
    return df

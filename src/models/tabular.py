"""부스팅 3종 통일 인터페이스.

LightGBM / XGBoost / CatBoost를 같은 방식으로 다룰 수 있게 래핑한다.
train.py는 모델 종류를 신경 쓰지 않고 fit/predict만 호출하면 된다.

각 모델은 (task, params)로 생성하고:
  - fit(X_tr, y_tr, X_va, y_va, cat_features)
  - predict(X)  → binary=확률, multiclass=확률(N,C), regression=값
"""
import numpy as np


class TabularModel:
    """부스팅 3종 공통 래퍼."""

    def __init__(self, kind: str, task: str, params: dict = None, n_class: int = None):
        self.kind = kind            # "lgb" | "xgb" | "cat"
        self.task = task
        self.params = params or {}
        self.n_class = n_class
        self.model = None

    # ---------- 학습 ----------
    def fit(self, X_tr, y_tr, X_va, y_va, cat_features=None):
        if self.kind == "lgb":
            self._fit_lgb(X_tr, y_tr, X_va, y_va, cat_features)
        elif self.kind == "xgb":
            self._fit_xgb(X_tr, y_tr, X_va, y_va)
        elif self.kind == "cat":
            self._fit_cat(X_tr, y_tr, X_va, y_va, cat_features)
        else:
            raise ValueError(f"unknown kind: {self.kind}")
        return self

    def _fit_lgb(self, X_tr, y_tr, X_va, y_va, cat_features):
        import lightgbm as lgb
        common = dict(n_estimators=2000, random_state=42, verbose=-1)
        common.update(self.params)
        if self.task == "regression":
            self.model = lgb.LGBMRegressor(objective="regression", **common)
        elif self.task == "binary":
            self.model = lgb.LGBMClassifier(objective="binary", **common)
        else:
            self.model = lgb.LGBMClassifier(objective="multiclass",
                                            num_class=self.n_class, **common)
        self.model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)],
            categorical_feature=cat_features or "auto",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

    def _fit_xgb(self, X_tr, y_tr, X_va, y_va):
        import xgboost as xgb
        common = dict(n_estimators=2000, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, random_state=42,
                      early_stopping_rounds=50, verbosity=0)
        common.update(self.params)   # config 파라미터가 기본값을 덮어씀 (충돌 방지)
        if self.task == "regression":
            self.model = xgb.XGBRegressor(**common)
        elif self.task == "binary":
            self.model = xgb.XGBClassifier(eval_metric="auc", **common)
        else:
            self.model = xgb.XGBClassifier(objective="multi:softprob",
                                           num_class=self.n_class, **common)
        self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    def _fit_cat(self, X_tr, y_tr, X_va, y_va, cat_features):
        from catboost import CatBoostClassifier, CatBoostRegressor
        common = dict(iterations=2000, learning_rate=0.05, depth=6,
                      random_seed=42, verbose=False, early_stopping_rounds=50,
                      cat_features=cat_features)
        common.update(self.params)
        if self.task == "regression":
            self.model = CatBoostRegressor(**common)
        else:
            self.model = CatBoostClassifier(**common)
        self.model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    # ---------- 예측 ----------
    def predict(self, X):
        if self.task == "regression":
            return self.model.predict(X)
        if self.task == "binary":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)   # multiclass: (N, C)


def make_models(task: str, n_class: int = None, params_by_kind: dict = None):
    """설정에 따라 부스팅 3종 인스턴스를 만들어 리스트로 반환."""
    params_by_kind = params_by_kind or {}
    kinds = ["lgb", "xgb", "cat"]
    return [
        TabularModel(k, task, params_by_kind.get(k, {}), n_class) for k in kinds
    ]

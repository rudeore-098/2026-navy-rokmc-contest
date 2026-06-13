"""시계열 모델.

실전 팁: 시계열은 feature engineering 후 LightGBM/XGBoost로 푸는 게
딥러닝보다 자주 이긴다. 그래서 1순위는 정형 접근(src/models/tabular.py 재사용),
딥러닝이 필요할 때만 아래 골격을 쓴다.
"""


def create_timeseries_model(kind: str = "lgbm", **kwargs):
    """시계열 모델 생성.

    kind:
      - "lgbm": feature를 만든 뒤 tabular.py의 부스팅을 그대로 사용 (권장 1순위)
      - "lstm": 순환 신경망 (torch 필요)
      - "tcn" : Temporal Convolutional Network (torch 필요)
    """
    if kind == "lgbm":
        from .tabular import make_models
        return make_models(kwargs.get("task", "regression"))

    if kind in ("lstm", "tcn"):
        # torch 기반 골격. 실제 구현은 데이터 형태에 맞춰 채운다.
        raise NotImplementedError(
            "LSTM/TCN은 데이터 형태(시퀀스 길이, 변수 수)에 맞춰 구현하세요. "
            "대부분의 표형 시계열은 feature engineering + LightGBM이 더 강합니다."
        )
    raise ValueError(f"unknown kind: {kind}")


def make_lag_features(df, target_col, lags=(1, 2, 3, 7), rolling=(3, 7)):
    """시계열 → 정형 변환용 기본 feature (lag, rolling 통계).

    이렇게 만든 표를 tabular 파이프라인에 넣는 게 실전에서 강력하다.
    """
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    for win in rolling:
        out[f"{target_col}_rollmean{win}"] = out[target_col].shift(1).rolling(win).mean()
        out[f"{target_col}_rollstd{win}"] = out[target_col].shift(1).rolling(win).std()
    return out

"""공통 학습 루프 — config를 받아 task별로 분기.

핵심 산출물: OOF 예측(oof.npy)과 test 예측. 이게 앙상블의 입력이 된다.

사용법:
    python -m src.train --config configs/tabular.yaml --exp exp_001
    python -m src.train --config configs/tabular.yaml --exp exp_001 --demo
"""
import os
import argparse
import numpy as np

from src.utils.seed import seed_everything
from src.utils.metrics import get_metric, metric_name
from src.utils.logger import get_logger, log_experiment
from src.data.cv import get_folder, split as cv_split
from src.data import loaders


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_tabular(X_tr, X_te):
    """범주형(object/str)을 정수 라벨로 인코딩. 결측은 부스팅이 자체 처리."""
    import pandas as pd
    cat_cols = [c for c in X_tr.columns
                if X_tr[c].dtype == "object" or X_tr[c].dtype.name in ("str", "string")
                or isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in cat_cols:
        cats = X_tr[c].astype("category").cat.categories
        m = {v: i for i, v in enumerate(cats)}
        X_tr[c] = X_tr[c].map(m).fillna(-1).astype(int)
        X_te[c] = X_te[c].map(m).fillna(-1).astype(int)
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
    return X_tr, X_te, cat_idx


def train_tabular(cfg, exp_dir, demo, logger):
    """부스팅 3종 학습 → 각 모델의 OOF/test 예측을 저장."""
    from src.models.tabular import make_models

    task = cfg["task"]
    X, y, groups, X_test, test_ids = loaders.load_tabular(cfg, demo=demo)
    X, X_test, cat_idx = preprocess_tabular(X, X_test)
    n_class = len(np.unique(y)) if task == "multiclass" else None

    folder = get_folder(task, cfg["n_folds"], cfg["seed"],
                        group=bool(cfg.get("group_col")))
    models = make_models(task, n_class, cfg.get("params_by_kind"))

    results = {}
    for m in models:
        if task == "multiclass":
            oof = np.zeros((len(X), n_class)); test_pred = np.zeros((len(X_test), n_class))
        else:
            oof = np.zeros(len(X)); test_pred = np.zeros(len(X_test))

        for tr_idx, va_idx in cv_split(folder, X, y, groups):
            m_fold = make_models(task, n_class, cfg.get("params_by_kind"))
            # 같은 kind만 다시 만들어 사용
            mk = next(x for x in m_fold if x.kind == m.kind)
            mk.fit(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx],
                   cat_features=cat_idx)
            oof[va_idx] = mk.predict(X.iloc[va_idx])
            test_pred += mk.predict(X_test) / cfg["n_folds"]

        score = get_metric(task, y, oof)
        logger.info(f"[{m.kind}] OOF {metric_name(task)}: {score:.5f}")
        np.save(os.path.join(exp_dir, f"oof_{m.kind}.npy"), oof)
        np.save(os.path.join(exp_dir, f"test_{m.kind}.npy"), test_pred)
        results[m.kind] = score

    # 단순 평균 OOF도 저장 (앙상블 입력의 기본)
    oof_files = [np.load(os.path.join(exp_dir, f"oof_{k}.npy")) for k in results]
    np.save(os.path.join(exp_dir, "oof.npy"), np.mean(oof_files, axis=0))
    np.save(os.path.join(exp_dir, "y_true.npy"), y)
    np.save(os.path.join(exp_dir, "test_ids.npy"), test_ids)
    return results


def train_vision(cfg, exp_dir, demo, logger):
    """비전 학습 골격. GPU(torch/timm) 환경에서 동작.

    데모 환경엔 torch가 없을 수 있어, 구조만 제공하고 안내한다.
    실제 구현은 notebooks/01_quick_baseline 또는 음향 파이프라인 노트북 참고.
    """
    logger.info("vision 학습은 GPU(torch/timm) 환경에서 실행하세요.")
    logger.info("OOF 골격은 tabular과 동일: K-Fold로 fold별 검증 예측을 모아 oof.npy 저장.")
    raise NotImplementedError(
        "vision 학습 본체는 GPU 환경에서 채우세요. "
        "구조는 train_tabular와 동일(K-Fold→OOF), 모델만 timm 백본으로 교체."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="exp_001")
    ap.add_argument("--demo", action="store_true", help="합성 데이터로 파이프라인 검증")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    logger = get_logger()

    exp_dir = os.path.join("experiments", args.exp)
    os.makedirs(exp_dir, exist_ok=True)

    task_type = cfg.get("type", "tabular")
    logger.info(f"=== train | type={task_type} | task={cfg['task']} | exp={args.exp} ===")

    if task_type == "tabular":
        results = train_tabular(cfg, exp_dir, args.demo, logger)
    elif task_type == "vision":
        results = train_vision(cfg, exp_dir, args.demo, logger)
    else:
        raise ValueError(f"unknown type: {task_type}")

    # 실험 기록 (best 모델 점수로)
    best_kind = max(results, key=results.get)
    log_experiment(
        exp_id=args.exp, model=best_kind, params=cfg.get("params_by_kind", {}),
        oof_score=results[best_kind], cv=f"{cfg['n_folds']}fold",
        seed=cfg["seed"], notes=f"types={task_type}, all={results}",
    )
    logger.info(f"완료. 산출물: {exp_dir}/ (oof.npy, test_*.npy)")


if __name__ == "__main__":
    main()

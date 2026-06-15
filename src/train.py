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


def train_audio(cfg, exp_dir, demo, logger):
    """음향 CNN + AIS 융합 학습 → OOF/test 예측 저장 (tabular 과 동일한 OOF 컨벤션).

    - ship_id GroupKFold (배 누수 차단)
    - class-weighted CrossEntropy (Macro F1 / 소수클래스 C_Passenger 대응)
    - mel npy 우선 로드(scripts/precompute_mel.py), 없으면 WAV fallback
    - fold 내 best Macro F1 에폭의 가중치를 복원해 OOF·test 예측을 동일 체크포인트로 통일
    task 는 'multiclass' (= Macro F1) 를 사용. GPU(torch/timm/torchaudio) 환경 필요.
    """
    import copy
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.data.loaders import SHIP_TYPE_TO_IDX, IDX_TO_SHIP_TYPE
    from src.models.audio import (
        build_ais_features, make_dataset, create_audio_model, class_weights)

    if demo:
        logger.info("audio 데모는 생략 — 실제 데이터(mel npy/WAV)로 실행하세요.")
        raise NotImplementedError("audio 학습은 실제 데이터 + GPU 환경에서 실행")

    task = cfg["task"]            # 'multiclass' → Macro F1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cls2idx = SHIP_TYPE_TO_IDX
    n_class = len(cls2idx)
    nw = cfg.get("num_workers", 4)

    train = pd.read_csv(cfg["paths"]["train"])
    test  = pd.read_csv(cfg["paths"]["task1_test"])
    y = train[cfg["target_col"]].map(cls2idx).values
    groups = train[cfg["group_col"]].values
    tr_ais, te_ais = build_ais_features(train), build_ais_features(test)

    cfg_tr = {**cfg, "_audio_dir": cfg["paths"]["train_audio"]}
    cfg_te = {**cfg, "_audio_dir": cfg["paths"]["task1_audio"]}

    # use_npy=true 인데 사전생성이 안 됐으면, 워커가 죽는 대신 여기서 명확히 안내
    if cfg.get("use_npy", False):
        md = cfg.get("mel_dir", "data/mel_npy")
        probe = os.path.join(md, str(train["filename"].iloc[0]).replace(".wav", ".npy"))
        if not os.path.exists(probe):
            raise FileNotFoundError(
                f"use_npy=true 인데 mel npy가 없습니다 ({probe}). "
                f"먼저 `python scripts/precompute_mel.py --config <config> --fp16` 를 실행하세요. "
                f"(또는 config에서 use_npy: false 로 바꾸면 WAV에서 즉석 변환)")

    # 정답이 있는 홀드아웃(task1_val) — 있으면 OOF 와 별개로 실제 Macro F1 도 측정
    val_path = cfg["paths"].get("task1_val")
    has_val = bool(val_path) and os.path.exists(val_path)
    if has_val:
        val = pd.read_csv(val_path)
        y_val = val[cfg["target_col"]].map(cls2idx).values
        va_ais = build_ais_features(val)
        # val 입력(npy/WAV)이 실제로 존재하는지 확인 → 없으면 홀드아웃만 생략
        md = cfg.get("mel_dir", "data/mel_npy")
        fn0 = str(val["filename"].iloc[0])
        vprobe = (os.path.join(md, fn0.replace(".wav", ".npy")) if cfg.get("use_npy", False)
                  else os.path.join(cfg["paths"]["task1_audio"], fn0))
        if not os.path.exists(vprobe):
            logger.info(f"[audio] val 입력 없음({vprobe}) → 홀드아웃 평가 생략")
            has_val = False

    def predict(model, loader):
        """(mel, ais[, y]) 배치를 받아 softmax 확률 (N, C) 반환. 라벨 유무 무관."""
        model.eval(); out = []
        with torch.no_grad():
            for batch in loader:
                mel, ais = batch[0], batch[1]
                with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                    out.append(model(mel.to(dev), ais.to(dev)).float().softmax(1).cpu().numpy())
        return np.concatenate(out)

    folder = get_folder(task, cfg["n_folds"], cfg["seed"], group=True)
    oof = np.zeros((len(train), n_class))
    test_pred = np.zeros((len(test), n_class))
    val_pred = np.zeros((len(val), n_class)) if has_val else None

    for fold, (tri, vai) in enumerate(cv_split(folder, train, y, groups)):
        tl = DataLoader(make_dataset(train.iloc[tri], tr_ais[tri], cfg_tr, y[tri], train=True),
                        batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=nw, pin_memory=True, drop_last=True)
        vl = DataLoader(make_dataset(train.iloc[vai], tr_ais[vai], cfg_tr, y[vai], train=False),
                        batch_size=cfg["batch_size"], shuffle=False, num_workers=nw)

        model = create_audio_model(cfg["model_name"], n_class, cfg.get("pretrained", True)).to(dev)
        crit = nn.CrossEntropyLoss(weight=class_weights(y[tri], n_class).to(dev))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-5))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
        scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

        best_f1, best_oof, best_state = -1, None, None
        for ep in range(cfg["epochs"]):
            model.train()
            for mel, ais, yy in tl:
                mel, ais, yy = mel.to(dev), ais.to(dev), yy.to(dev)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                    loss = crit(model(mel, ais), yy)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
            probs = predict(model, vl)                      # fold val 예측
            f1 = get_metric(task, y[vai], probs)
            logger.info(f"[fold{fold}] ep{ep+1:02d} {metric_name(task)}={f1:.4f}")
            if f1 > best_f1:
                best_f1, best_oof = f1, probs
                best_state = copy.deepcopy(model.state_dict())   # best 에폭 가중치 보관
        oof[vai] = best_oof

        # OOF 와 동일한 best 체크포인트로 test/val 예측 (last 에폭 아님)
        if best_state is not None:
            model.load_state_dict(best_state)
        tel = DataLoader(make_dataset(test, te_ais, cfg_te, None, train=False),
                         batch_size=cfg["batch_size"], shuffle=False, num_workers=nw)
        test_pred += predict(model, tel) / cfg["n_folds"]
        if has_val:
            vel = DataLoader(make_dataset(val, va_ais, cfg_te, None, train=False),
                             batch_size=cfg["batch_size"], shuffle=False, num_workers=nw)
            val_pred += predict(model, vel) / cfg["n_folds"]
        torch.save(model.state_dict(), os.path.join(exp_dir, f"audio_fold{fold}.pt"))

    score = get_metric(task, y, oof)
    logger.info(f"[audio] OOF {metric_name(task)}: {score:.5f}")
    if has_val:
        val_score = get_metric(task, y_val, val_pred)
        logger.info(f"[audio] VAL(holdout) {metric_name(task)}: {val_score:.5f}")
        np.save(os.path.join(exp_dir, "val_audio.npy"), val_pred)

    np.save(os.path.join(exp_dir, "oof_audio.npy"), oof)
    np.save(os.path.join(exp_dir, "oof.npy"), oof)
    np.save(os.path.join(exp_dir, "test_audio.npy"), test_pred)
    np.save(os.path.join(exp_dir, "y_true.npy"), y)

    # 제출 파일 (filename, ship_type) — 5-fold 평균 확률의 argmax → 라벨 문자열
    target_col = cfg.get("target_col", "ship_type")
    sub = pd.DataFrame({
        "filename": test["filename"].values,
        target_col: [IDX_TO_SHIP_TYPE[i] for i in test_pred.argmax(axis=1)],
    })
    sub_path = os.path.join(exp_dir, "submission_task1.csv")
    sub.to_csv(sub_path, index=False)
    logger.info(f"[audio] 제출 파일 저장: {sub_path} ({len(sub)}행) | 분포={sub[target_col].value_counts().to_dict()}")
    return {"audio": score}


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
    elif task_type == "audio":
        results = train_audio(cfg, exp_dir, args.demo, logger)
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

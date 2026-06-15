"""Task 1 학습 루프 — AST audio + AIS 융합 (4-class Macro F1).

핵심 산출물: OOF 예측(oof.npy)과 test 예측. 이게 앙상블의 입력이 된다.

사용법:
    python -m src.train --config configs/audio_task1_ast.yaml --exp exp_t1_ast

[변경 사항 - 2026-06-15]
- tabular / vision 분기 제거 (음향 대회 전용)
- audio 학습 로직만 유지
- StratifiedGroupKFold 지원 (cv_type 키)
- DataLoader persistent_workers / prefetch_factor 보강
- fold 단위 중간 저장 (Ctrl+C 안전망)
"""
import os
import argparse
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.metrics import get_metric, metric_name
from src.utils.logger import get_logger, log_experiment
from src.data.cv import get_folder, split as cv_split


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_audio(cfg, exp_dir, logger):
    """AST audio + AIS 융합 학습 → OOF/test 예측 저장.

    - ship_id StratifiedGroupKFold (배 누수 차단 + 클래스 균등)
    - effective number weighted CrossEntropy (Macro F1 / 소수클래스 대응)
    - AST fbank npy 우선 로드(scripts/precompute_ast.py), 없으면 WAV fallback
    - fold 내 best Macro F1 에폭의 가중치를 복원해 OOF·test 예측을 동일 체크포인트로 통일
    - fold 단위로 submission.csv 중간 저장 (Ctrl+C 시 마지막 fold까지의 결과 보존)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.data.loaders import SHIP_TYPE_TO_IDX, IDX_TO_SHIP_TYPE
    from src.models.audio import (
        build_ais_features, make_dataset, create_audio_model, class_weights)

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

    # 정답이 있는 홀드아웃(task1_val) — OOF 와 별개로 실제 Macro F1 측정
    val_path = cfg["paths"].get("task1_val")
    has_val = bool(val_path) and os.path.exists(val_path)
    if has_val:
        val = pd.read_csv(val_path)
        y_val = val[cfg["target_col"]].map(cls2idx).values
        va_ais = build_ais_features(val)
        # val 입력(WAV/npy)이 실제로 존재하는지 확인
        fn0 = str(val["filename"].iloc[0])
        vprobe = os.path.join("data/ast_npy", fn0.replace(".wav", ".npy"))
        if not os.path.exists(vprobe):
            vprobe = os.path.join(cfg["paths"]["task1_audio"], fn0)
        if not os.path.exists(vprobe):
            logger.info(f"[audio] val 입력 없음({vprobe}) → 홀드아웃 평가 생략")
            has_val = False

    def predict(model, loader):
        """(mel, ais[, y]) 배치를 받아 softmax 확률 (N, C) 반환."""
        model.eval()
        out = []
        with torch.no_grad():
            for batch in loader:
                mel, ais = batch[0], batch[1]
                with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                    out.append(model(mel.to(dev), ais.to(dev)).float().softmax(1).cpu().numpy())
        return np.concatenate(out)

    # === CV splitter 선택 (config의 cv_type 키로 분기) ===
    cv_type = cfg.get("cv_type", "stratified_group")
    if cv_type == "stratified_group":
        from sklearn.model_selection import StratifiedGroupKFold
        folder = StratifiedGroupKFold(
            n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
        splits = list(folder.split(train, y, groups))
        logger.info(f"[cv] StratifiedGroupKFold (n={cfg['n_folds']}, group=ship_id)")
    elif cv_type == "stratified":
        from sklearn.model_selection import StratifiedKFold
        folder = StratifiedKFold(
            n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
        splits = list(folder.split(train, y))
        logger.info(f"[cv] StratifiedKFold (n={cfg['n_folds']}, ⚠ ship_id leak 가능)")
    else:
        folder = get_folder(task, cfg["n_folds"], cfg["seed"], group=True)
        splits = list(cv_split(folder, train, y, groups))
        logger.info(f"[cv] 기존 get_folder (GroupKFold)")

    # fold별 sanity check — 클래스 분포 + ship_id 누수 확인
    for fi, (tri, vai) in enumerate(splits):
        val_dist = np.bincount(y[vai], minlength=n_class)
        train_groups = set(groups[tri])
        val_groups = set(groups[vai])
        leak = len(train_groups & val_groups)
        logger.info(f"[cv] fold{fi} val_dist={val_dist.tolist()} "
                    f"groups(train={len(train_groups)}, val={len(val_groups)}, leak={leak})")

    oof = np.zeros((len(train), n_class))
    test_pred = np.zeros((len(test), n_class))
    val_pred = np.zeros((len(val), n_class)) if has_val else None

    for fold, (tri, vai) in enumerate(splits):
        tl = DataLoader(make_dataset(train.iloc[tri], tr_ais[tri], cfg_tr, y[tri], train=True),
                        batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=nw, pin_memory=True, drop_last=True,
                        persistent_workers=(nw > 0),
                        prefetch_factor=2 if nw > 0 else None)
        vl = DataLoader(make_dataset(train.iloc[vai], tr_ais[vai], cfg_tr, y[vai], train=False),
                        batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=nw, pin_memory=True,
                        persistent_workers=(nw > 0),
                        prefetch_factor=2 if nw > 0 else None)

        model = create_audio_model(cfg["model_name"], n_class, cfg.get("pretrained", True)).to(dev)
        # class_weights: effective number (β=0.9999 기본, config에서 override 가능)
        beta = cfg.get("class_weight_beta", 0.9999)
        crit = nn.CrossEntropyLoss(weight=class_weights(y[tri], n_class, beta=beta).to(dev))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 1e-5))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
        scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))

        best_f1, best_oof, best_state = -1, None, None
        for ep in range(cfg["epochs"]):
            model.train()
            for mel, ais, yy in tqdm(tl, desc=f"fold{fold} ep{ep+1}", leave=False):
                mel, ais, yy = mel.to(dev), ais.to(dev), yy.to(dev)
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                    loss = crit(model(mel, ais), yy)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()
            probs = predict(model, vl)                      # fold val 예측
            f1 = get_metric(task, y[vai], probs)
            logger.info(f"[fold{fold}] ep{ep+1:02d} {metric_name(task)}={f1:.4f}")
            if f1 > best_f1:
                best_f1, best_oof = f1, probs
                best_state = copy.deepcopy(model.state_dict())
        oof[vai] = best_oof

        # OOF 와 동일한 best 체크포인트로 test/val 예측
        if best_state is not None:
            model.load_state_dict(best_state)
        tel = DataLoader(make_dataset(test, te_ais, cfg_te, None, train=False),
                         batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=nw, pin_memory=True,
                         prefetch_factor=2 if nw > 0 else None)
        test_pred += predict(model, tel) / cfg["n_folds"]
        if has_val:
            vel = DataLoader(make_dataset(val, va_ais, cfg_te, None, train=False),
                             batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=nw, pin_memory=True,
                             prefetch_factor=2 if nw > 0 else None)
            val_pred += predict(model, vel) / cfg["n_folds"]
        torch.save(model.state_dict(), os.path.join(exp_dir, f"audio_fold{fold}.pt"))

        # === fold 단위 중간 저장 (Ctrl+C 안전망) ===
        n_done = fold + 1
        partial_test = test_pred * (cfg["n_folds"] / n_done)   # 누적합 → 평균 복원
        np.save(os.path.join(exp_dir, "oof.npy"), oof)
        np.save(os.path.join(exp_dir, "oof_audio.npy"), oof)
        np.save(os.path.join(exp_dir, "test_audio.npy"), partial_test)
        np.save(os.path.join(exp_dir, "y_true.npy"), y)
        if has_val:
            partial_val = val_pred * (cfg["n_folds"] / n_done)
            np.save(os.path.join(exp_dir, "val_audio.npy"), partial_val)

        # 누적 평균 기반 submission
        sub = pd.DataFrame({
            "filename": test["filename"].values,
            "predicted_class": [IDX_TO_SHIP_TYPE[i] for i in partial_test.argmax(axis=1)],
        })
        sub.to_csv(os.path.join(exp_dir, "submission_task1.csv"), index=False)

        # 진행 상황 로그 — 지금까지 완료한 fold의 val만 모아서 partial OOF 계산
        done_idx = np.concatenate([v for _, v in splits[:n_done]])
        partial_oof_score = get_metric(task, y[done_idx], oof[done_idx])
        logger.info(f"[checkpoint] fold {n_done}/{cfg['n_folds']} 완료 | "
                    f"partial OOF {metric_name(task)}={partial_oof_score:.5f} | "
                    f"submission 저장 ({len(sub)}행)")

    # === 5 fold 완주 시 최종 정리 (중간 저장본을 덮어쓰기) ===
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

    # 제출 파일 (filename, predicted_class) — n-fold 평균 확률의 argmax → 라벨 문자열
    sub = pd.DataFrame({
        "filename": test["filename"].values,
        "predicted_class": [IDX_TO_SHIP_TYPE[i] for i in test_pred.argmax(axis=1)],
    })
    sub_path = os.path.join(exp_dir, "submission_task1.csv")
    sub.to_csv(sub_path, index=False)
    logger.info(f"[audio] 제출 파일 저장: {sub_path} ({len(sub)}행) | "
                f"분포={sub['predicted_class'].value_counts().to_dict()}")
    return {"audio": score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="exp_001")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    logger = get_logger()

    exp_dir = os.path.join("experiments", args.exp)
    os.makedirs(exp_dir, exist_ok=True)

    task_type = cfg.get("type", "audio")
    logger.info(f"=== train | type={task_type} | task={cfg['task']} | exp={args.exp} ===")

    if task_type != "audio":
        raise ValueError(
            f"이 train.py 는 audio 전용입니다 (type={task_type} 미지원). "
            f"tabular/vision 등 다른 type 은 별도 학습 스크립트가 필요합니다."
        )

    results = train_audio(cfg, exp_dir, logger)

    log_experiment(
        exp_id=args.exp, model="audio",
        params={"model_name": cfg.get("model_name"),
                "class_weight_beta": cfg.get("class_weight_beta", 0.9999),
                "n_folds": cfg["n_folds"]},
        oof_score=results["audio"], cv=f"{cfg['n_folds']}fold",
        seed=cfg["seed"], notes=f"AIS 14-d + effective_num_weight + StratifiedGroupKFold",
    )
    logger.info(f"완료. 산출물: {exp_dir}/ (oof.npy, test_*.npy, val_audio.npy, submission_task1.csv)")


if __name__ == "__main__":
    main()
"""공통 학습 루프 — config를 받아 task별로 분기.

핵심 산출물: OOF 예측(oof.npy)과 test 예측. 이게 앙상블의 입력이 된다.

사용법:
    python -m src.train --config configs/audio_task1.yaml --exp exp_t1_001
    python -m src.train --config configs/audio_task2.yaml --exp exp_t2_001
    python -m src.train --config configs/tabular.yaml    --exp exp_tab_001
    python -m src.train --config configs/audio_task1.yaml --exp smoke --demo
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


# ── 정형 ─────────────────────────────────────────────────────────────────────

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

    oof_files = [np.load(os.path.join(exp_dir, f"oof_{k}.npy")) for k in results]
    np.save(os.path.join(exp_dir, "oof.npy"), np.mean(oof_files, axis=0))
    np.save(os.path.join(exp_dir, "y_true.npy"), y)
    np.save(os.path.join(exp_dir, "test_ids.npy"), test_ids)
    return results


# ── 오디오 공통 유틸 ──────────────────────────────────────────────────────────

def spec_augment(spec, T: int = 40, F: int = 20, num_T: int = 2, num_F: int = 2):
    """SpecAugment in-place on a batch tensor (B, 1, n_mels, time)."""
    import torch
    B, _, n_mels, time = spec.shape
    spec = spec.clone()
    for _ in range(num_T):
        t0 = torch.randint(0, max(1, time - T), (B,))
        tl = torch.randint(1, T + 1, (B,))
        for b in range(B):
            spec[b, :, :, t0[b]: t0[b] + tl[b]] = 0.0
    for _ in range(num_F):
        f0 = torch.randint(0, max(1, n_mels - F), (B,))
        fl = torch.randint(1, F + 1, (B,))
        for b in range(B):
            spec[b, :, f0[b]: f0[b] + fl[b], :] = 0.0
    return spec


def _train_epoch(model, loader, criterion, optimizer, scaler, device, augment=True):
    import torch
    model.train()
    total_loss = 0.0
    for spec, ais, label in loader:
        spec, ais, label = spec.to(device), ais.to(device), label.to(device)
        if augment:
            spec = spec_augment(spec)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            loss = criterion(model(spec, ais), label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def _eval_epoch(model, loader, device):
    """Returns (preds: ndarray N×C, labels: ndarray N)."""
    import torch
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            spec, ais, label = batch
            spec, ais = spec.to(device), ais.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(spec, ais)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def _predict(model, loader, device):
    """Returns softmax probs ndarray N×C (no labels)."""
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            spec, ais = batch[0].to(device), batch[1].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(spec, ais)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def _make_loader(df, audio_dir, cfg, demo, is_test=False,
                 label_col=None, label_map=None, shuffle=False):
    from torch.utils.data import DataLoader
    from src.data.loaders import AudioDataset
    ds = AudioDataset(df, audio_dir, cfg, is_test=is_test, demo=demo,
                      label_col=label_col, label_map=label_map)
    bs = cfg["batch_size"] if not is_test else cfg["batch_size"] * 2
    nw = cfg.get("num_workers", 0)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=nw, pin_memory=True)


# ── Task 1: 선종 분류 (GroupKFold + Focal Loss + OOF) ────────────────────────

def train_audio_task1(cfg, exp_dir, demo, logger):
    import torch
    from src.models.audio import AudioModel, FocalLoss
    from src.data.loaders import load_task1, SHIP_TYPE_TO_IDX

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    train_df, audio_dir = load_task1(cfg, "train", demo=demo)
    labels  = train_df["ship_type"].map(SHIP_TYPE_TO_IDX).values
    groups  = train_df["ship_id"].values
    n_cls   = len(SHIP_TYPE_TO_IDX)  # 4
    n_folds = cfg.get("n_folds", 5)

    folder = get_folder("multiclass", n_folds, cfg["seed"], group=True)
    oof    = np.zeros((len(train_df), n_cls), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(cv_split(folder, train_df, labels, groups)):
        logger.info(f"=== Fold {fold + 1}/{n_folds} ===")
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        tr_loader = _make_loader(tr_df, audio_dir, cfg, demo, shuffle=True)
        va_loader = _make_loader(va_df, audio_dir, cfg, demo)

        # class weights from training fold
        cnts = np.bincount(labels[tr_idx], minlength=n_cls).astype(float)
        w    = torch.tensor((cnts.sum() / (n_cls * cnts)).clip(0.1, 10),
                            dtype=torch.float32, device=device)
        criterion = FocalLoss(gamma=2.0, weight=w)

        model = AudioModel(
            model_name=cfg["model_name"],
            num_classes=n_cls,
            in_chans=cfg.get("in_chans", 1),
            pretrained=cfg.get("pretrained", True),
            ais_dim=cfg.get("ais_dim", 5),
            embed_dim=cfg.get("embed_dim", 512),
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg["lr"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best_score, best_oof_fold, patience_cnt = -np.inf, None, 0
        patience = cfg.get("patience", 5)

        for epoch in range(cfg["epochs"]):
            loss = _train_epoch(model, tr_loader, criterion, optimizer, scaler, device)
            va_preds, va_labels = _eval_epoch(model, va_loader, device)
            score = get_metric("multiclass", va_labels, va_preds)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            logger.info(f"  ep {epoch+1:3d}/{cfg['epochs']} | loss={loss:.4f} "
                        f"| MacroF1={score:.4f} | lr={lr_now:.2e}")

            if score > best_score:
                best_score, best_oof_fold, patience_cnt = score, va_preds.copy(), 0
                torch.save(model.state_dict(),
                           os.path.join(exp_dir, f"model_fold{fold}.pt"))
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    logger.info(f"  early stop at epoch {epoch + 1}")
                    break

        oof[va_idx] = best_oof_fold
        logger.info(f"  Fold {fold + 1} best MacroF1: {best_score:.5f}")

    oof_score = get_metric("multiclass", labels, oof)
    logger.info(f"OOF MacroF1: {oof_score:.5f}")
    np.save(os.path.join(exp_dir, "oof.npy"), oof)
    np.save(os.path.join(exp_dir, "y_true.npy"), labels)

    # ── t1_val / t1_test 앙상블 예측 ──────────────────────────────────────────
    val_df,  val_audio  = load_task1(cfg, "val",  demo=demo)
    test_df, test_audio = load_task1(cfg, "test", demo=demo)

    val_preds  = np.zeros((len(val_df),  n_cls), dtype=np.float32)
    test_preds = np.zeros((len(test_df), n_cls), dtype=np.float32)
    loaded = 0

    for fold in range(n_folds):
        ckpt = os.path.join(exp_dir, f"model_fold{fold}.pt")
        if not os.path.exists(ckpt):
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device))
        val_preds  += _predict(model, _make_loader(val_df,  val_audio,  cfg, demo, is_test=True), device)
        test_preds += _predict(model, _make_loader(test_df, test_audio, cfg, demo, is_test=True), device)
        loaded += 1

    if loaded:
        val_preds  /= loaded
        test_preds /= loaded

    np.save(os.path.join(exp_dir, "val_pred.npy"),  val_preds)
    np.save(os.path.join(exp_dir, "test_pred.npy"), test_preds)

    val_labels = val_df["ship_type"].map(SHIP_TYPE_TO_IDX).values
    val_score  = get_metric("multiclass", val_labels, val_preds)
    logger.info(f"t1_val MacroF1 (ensemble): {val_score:.5f}")

    return {"oof_macro_f1": oof_score, "val_macro_f1": val_score}


# ── Task 2: 선박 ID 검색 (362-class 분류 → 임베딩 추출용) ─────────────────────

def train_audio_task2(cfg, exp_dir, demo, logger):
    """362-class softmax로 audio-only 임베딩 모델 학습.

    검색(retrieval)은 src/infer_retrieval.py 에서 수행.
    """
    import torch
    import pandas as pd
    from src.models.audio import AudioModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # ── 학습 데이터: train/train.csv (362척) ──────────────────────────────────
    train_df = pd.read_csv(cfg["paths"]["train"])
    audio_dir = cfg["paths"]["train_audio"]

    # ship_id → 연속 정수 매핑
    ship_ids = sorted(train_df["ship_id"].unique())
    id_map   = {sid: i for i, sid in enumerate(ship_ids)}
    n_cls    = len(ship_ids)  # 362
    np.save(os.path.join(exp_dir, "ship_id_map.npy"), id_map)
    logger.info(f"Task2: {n_cls} ships, {len(train_df)} clips, audio-only (ais_dim=0)")

    # 80/20 holdout on ship level
    rng = np.random.default_rng(cfg["seed"])
    val_ships  = set(rng.choice(ship_ids, size=int(n_cls * 0.2), replace=False).tolist())
    tr_df  = train_df[~train_df["ship_id"].isin(val_ships)].reset_index(drop=True)
    va_df  = train_df[ train_df["ship_id"].isin(val_ships)].reset_index(drop=True)

    tr_loader = _make_loader(tr_df, audio_dir, cfg, demo, shuffle=True,
                             label_col="ship_id", label_map=id_map)
    va_loader = _make_loader(va_df, audio_dir, cfg, demo,
                             label_col="ship_id", label_map=id_map)

    model = AudioModel(
        model_name=cfg["model_name"],
        num_classes=n_cls,
        in_chans=cfg.get("in_chans", 1),
        pretrained=cfg.get("pretrained", True),
        ais_dim=0,                           # AIS 없음: test에도 없으니까
        embed_dim=cfg.get("embed_dim", 512),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc, patience_cnt = -np.inf, 0
    patience = cfg.get("patience", 5)

    for epoch in range(cfg["epochs"]):
        loss = _train_epoch(model, tr_loader, criterion, optimizer, scaler, device)
        va_preds, va_labels = _eval_epoch(model, va_loader, device)
        acc = (va_preds.argmax(1) == va_labels).mean()
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        logger.info(f"  ep {epoch+1:3d}/{cfg['epochs']} | loss={loss:.4f} "
                    f"| val_acc={acc:.4f} | lr={lr_now:.2e}")

        if acc > best_acc:
            best_acc, patience_cnt = acc, 0
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, "model_task2.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"  early stop at epoch {epoch + 1}")
                break

    logger.info(f"Task2 best val acc: {best_acc:.5f}")
    logger.info("Saved model_task2.pt  — run src/infer_retrieval.py for Recall@K")
    return {"val_acc": best_acc}


# ── 엔트리포인트 ───────────────────────────────────────────────────────────────

def train_audio(cfg, exp_dir, demo, logger):
    task = cfg.get("task", "multiclass")
    if task == "multiclass":
        return train_audio_task1(cfg, exp_dir, demo, logger)
    elif task == "retrieval":
        return train_audio_task2(cfg, exp_dir, demo, logger)
    else:
        raise ValueError(f"unknown audio task: {task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp",    default="exp_001")
    ap.add_argument("--demo",   action="store_true",
                    help="합성 데이터로 파이프라인 검증")
    args = ap.parse_args()

    cfg  = load_config(args.config)
    seed_everything(cfg["seed"])
    logger = get_logger()

    exp_dir = os.path.join("experiments", args.exp)
    os.makedirs(exp_dir, exist_ok=True)

    task_type = cfg.get("type", "tabular")
    logger.info(f"=== train | type={task_type} | task={cfg['task']} | exp={args.exp} ===")

    if task_type == "tabular":
        results = train_tabular(cfg, exp_dir, args.demo, logger)
        best_kind = max(results, key=results.get)
        log_experiment(
            exp_id=args.exp, model=best_kind, params=cfg.get("params_by_kind", {}),
            oof_score=results[best_kind], cv=f"{cfg['n_folds']}fold",
            seed=cfg["seed"], notes=f"type={task_type}, all={results}",
        )
    elif task_type == "audio":
        results = train_audio(cfg, exp_dir, args.demo, logger)
        primary = list(results.values())[0]
        log_experiment(
            exp_id=args.exp, model=cfg.get("model_name", "audio"),
            params={k: cfg.get(k) for k in ("lr", "epochs", "batch_size", "ais_dim")
                    if cfg.get(k) is not None},
            oof_score=primary, cv=f"{cfg.get('n_folds', 1)}fold",
            seed=cfg["seed"], notes=str(results),
        )
    else:
        raise ValueError(f"unknown type: {task_type}")

    logger.info(f"완료. 산출물: {exp_dir}/")


if __name__ == "__main__":
    main()

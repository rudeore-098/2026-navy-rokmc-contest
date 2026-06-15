"""오디오 학습 루프 — config를 받아 Task별로 분기.

핵심 산출물: OOF 예측(oof.npy)과 test 예측. 앙상블의 입력이 된다.

사용법:
    python -m src.train --config configs/audio_task1.yaml --exp exp_t1_001
    python -m src.train --config configs/audio_task1.yaml --exp smoke --demo
"""
import os
import argparse
import numpy as np
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.metrics import get_metric
from src.utils.logger import get_logger, log_experiment
from src.data.cv import get_folder, split as cv_split


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 2단계 파인튜닝 헬퍼 ──────────────────────────────────────────────────────────

def _get_backbone(model):
    """AudioModel → backbone, ASTWrapper → ast, 그 외 → None."""
    if hasattr(model, "backbone"):
        return model.backbone
    if hasattr(model, "ast"):
        return model.ast
    return None


def _freeze_backbone(model):
    bb = _get_backbone(model)
    if bb:
        for p in bb.parameters():
            p.requires_grad = False


def _unfreeze_backbone(model):
    bb = _get_backbone(model)
    if bb:
        for p in bb.parameters():
            p.requires_grad = True


def _make_optimizer(model, lr, backbone_lr_scale=1.0, weight_decay=1e-4):
    """백본/헤드 LR을 분리한 AdamW 옵티마이저.

    backbone_lr_scale < 1.0 이면 백본에 낮은 LR 적용 (레이어별 LR 감쇠).
    """
    import torch
    bb = _get_backbone(model)
    bb_ids = {id(p) for p in bb.parameters()} if bb else set()

    head_params = [p for p in model.parameters()
                   if id(p) not in bb_ids and p.requires_grad]
    bb_params   = [p for p in bb.parameters() if p.requires_grad] if bb else []

    groups = [{"params": head_params, "lr": lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": lr * backbone_lr_scale})

    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ── SpecAugment ───────────────────────────────────────────────────────────────

def spec_augment(spec, T: int = 40, F: int = 20, num_T: int = 2, num_F: int = 2):
    """Batch-level SpecAugment (벡터화 — Python 루프 없음). spec: (B, C, n_mels, time)"""
    import torch
    B, _, n_mels, time = spec.shape
    dev = spec.device
    spec = spec.clone()

    t_idx = torch.arange(time,   device=dev).view(1, 1, 1, time)
    f_idx = torch.arange(n_mels, device=dev).view(1, 1, n_mels, 1)

    for _ in range(num_T):
        t0 = torch.randint(0, max(1, time   - T), (B, 1, 1, 1), device=dev)
        tl = torch.randint(1, T + 1,              (B, 1, 1, 1), device=dev)
        spec.masked_fill_((t_idx >= t0) & (t_idx < t0 + tl), 0.0)

    for _ in range(num_F):
        f0 = torch.randint(0, max(1, n_mels - F), (B, 1, 1, 1), device=dev)
        fl = torch.randint(1, F + 1,              (B, 1, 1, 1), device=dev)
        spec.masked_fill_((f_idx >= f0) & (f_idx < f0 + fl), 0.0)

    return spec


def mixup_data(spec, ais, alpha=0.4):
    """배치 내 두 샘플을 λ:(1-λ)로 보간. λ-트릭으로 정수 라벨 FocalLoss와 호환."""
    import torch
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(spec.size(0), device=spec.device)
    mixed_spec = lam * spec + (1 - lam) * spec[idx]
    mixed_ais  = lam * ais  + (1 - lam) * ais[idx]
    return mixed_spec, mixed_ais, idx, lam


# ── 공통 루프 헬퍼 ────────────────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, scaler, device, augment=True, desc="train",
                 mixup_p=0.0, mixup_alpha=0.4):
    import torch
    model.train()
    total_loss = 0.0
    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for spec, ais, label in bar:
        spec, ais, label = spec.to(device), ais.to(device), label.to(device)
        if augment:
            spec = spec_augment(spec)
        use_mix = augment and mixup_p > 0 and float(np.random.rand()) < mixup_p
        mix_idx, lam = None, 1.0
        if use_mix:
            spec, ais, mix_idx, lam = mixup_data(spec, ais, alpha=mixup_alpha)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(spec, ais)
            if use_mix:
                loss = lam * criterion(out, label) + (1 - lam) * criterion(out, label[mix_idx])
            else:
                loss = criterion(out, label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(loader), 1)


def _eval_epoch(model, loader, device, desc="val"):
    """Returns (preds: ndarray N×C, labels: ndarray N)."""
    import torch
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for spec, ais, label in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            spec, ais = spec.to(device), ais.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(spec, ais)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(label.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def _predict(model, loader, device, desc="predict"):
    """Returns softmax probs ndarray N×C (no labels)."""
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            spec, ais = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(spec, ais)
            preds.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def _predict_tta(model, loader, device, n_tta: int = 3, desc="predict"):
    """TTA: time-shift ±step 프레임 n_tta 패스 평균. n_tta=1이면 일반 predict."""
    import torch
    if n_tta <= 1:
        return _predict(model, loader, device, desc)
    step = 15  # 프레임 ≈ 0.24초 (hop_length=512, sr=32000)
    half = n_tta // 2
    shifts = [k * step for k in range(-half, half + 1)][:n_tta]
    model.eval()
    all_preds = []
    for shift in shifts:
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{desc}[s={shift:+d}]", leave=False, dynamic_ncols=True):
                spec, ais = batch[0].to(device), batch[1].to(device)
                if shift:
                    spec = torch.roll(spec, shift, dims=-1)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(spec, ais)
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_preds.append(np.concatenate(preds))
    return np.mean(all_preds, axis=0)


def _make_loader(df, audio_dir, cfg, demo, is_test=False,
                 label_col=None, label_map=None, shuffle=False):
    from torch.utils.data import DataLoader
    from src.data.loaders import AudioDataset

    cache_dir = cfg.get("cache_dir", "data/spec_cache")
    ds = AudioDataset(df, audio_dir, cfg, is_test=is_test, demo=demo,
                      label_col=label_col, label_map=label_map,
                      cache_dir=cache_dir)
    bs = cfg["batch_size"] if not is_test else cfg["batch_size"] * 2
    nw = cfg.get("num_workers", 4)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=nw, pin_memory=True)


# ── Task 1: 선종 분류 (GroupKFold + Focal Loss + OOF) ────────────────────────

def train_audio_task1(cfg, exp_dir, demo, logger):
    import torch
    from src.models.audio import FocalLoss, create_audio_model
    from src.data.loaders import load_task1, SHIP_TYPE_TO_IDX

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    train_df, audio_dir = load_task1(cfg, "train", demo=demo)
    labels  = train_df["ship_type"].map(SHIP_TYPE_TO_IDX).values
    groups  = train_df["ship_id"].values
    n_cls   = len(SHIP_TYPE_TO_IDX)  # 4
    n_folds = cfg.get("n_folds", 5)

    # ── 단일 학습 (n_folds=1): 전체 train.csv → task1_val.csv 검증 ─────────────
    if n_folds == 1:
        logger.info("=== Single Fold (no CV): train 전체 → val.csv 검증 ===")
        val_df_ext, val_audio_ext = load_task1(cfg, "val", demo=demo)
        va_labels_ext = val_df_ext["ship_type"].map(SHIP_TYPE_TO_IDX).values

        tr_loader = _make_loader(train_df, audio_dir, cfg, demo, shuffle=True)
        va_loader = _make_loader(val_df_ext, val_audio_ext, cfg, demo)

        cnts = np.bincount(labels, minlength=n_cls).astype(float)
        w    = torch.tensor((cnts.sum() / (n_cls * cnts)).clip(0.1, 10),
                            dtype=torch.float32, device=device)
        criterion = FocalLoss(gamma=2.0, weight=w,
                              label_smoothing=cfg.get("label_smoothing", 0.0))
        model = create_audio_model(cfg, n_cls).to(device)

        freeze_epochs     = cfg.get("freeze_epochs", 0)
        unfreeze_lr_scale = cfg.get("unfreeze_lr_scale", 0.1)
        total_epochs      = cfg["epochs"]
        base_lr           = cfg["lr"]
        patience          = cfg.get("patience", 5)

        if freeze_epochs > 0:
            _freeze_backbone(model)
            logger.info(f"  [Phase 1] 백본 freeze — {freeze_epochs} epochs")
            optimizer = _make_optimizer(model, base_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=freeze_epochs, eta_min=base_lr * 0.01)
        else:
            optimizer = _make_optimizer(model, base_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=base_lr * 0.01)

        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        best_score, patience_cnt = -np.inf, 0

        epoch_bar = tqdm(range(total_epochs), desc="Single", unit="ep", dynamic_ncols=True)
        for epoch in epoch_bar:
            if freeze_epochs > 0 and epoch == freeze_epochs:
                _unfreeze_backbone(model)
                optimizer = _make_optimizer(model, base_lr,
                                            backbone_lr_scale=unfreeze_lr_scale)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_epochs - freeze_epochs, eta_min=base_lr * 0.01)
                best_score, patience_cnt = -np.inf, 0
                logger.info(f"  [Phase 2] 백본 unfreeze")

            loss = _train_epoch(model, tr_loader, criterion, optimizer, scaler, device,
                                mixup_p=cfg.get("mixup_p", 0.0),
                                mixup_alpha=cfg.get("mixup_alpha", 0.4),
                                desc="  train")
            va_preds, _ = _eval_epoch(model, va_loader, device, desc="  val")
            score = get_metric("multiclass", va_labels_ext, va_preds)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            epoch_bar.set_postfix(loss=f"{loss:.4f}", F1=f"{score:.4f}",
                                  best=f"{best_score:.4f}" if best_score > -np.inf else "-",
                                  lr=f"{lr_now:.2e}")
            logger.info(f"  ep {epoch+1:3d}/{total_epochs} | loss={loss:.4f} "
                        f"| MacroF1={score:.4f} | lr={lr_now:.2e}")

            if score > best_score:
                best_score, patience_cnt = score, 0
                torch.save(model.state_dict(), os.path.join(exp_dir, "model_fold0.pt"))
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    logger.info(f"  early stop at epoch {epoch + 1}")
                    epoch_bar.close()
                    break

        logger.info(f"best val MacroF1: {best_score:.5f}")

        # test 예측
        test_df, test_audio = load_task1(cfg, "test", demo=demo)
        tta_n = cfg.get("tta_n", 1)
        model.load_state_dict(torch.load(os.path.join(exp_dir, "model_fold0.pt"),
                                         map_location=device, weights_only=True))
        val_preds  = _predict_tta(model, _make_loader(val_df_ext, val_audio_ext,
                                                       cfg, demo, is_test=True),
                                  device, n_tta=tta_n, desc="  val")
        test_preds = _predict_tta(model, _make_loader(test_df, test_audio,
                                                       cfg, demo, is_test=True),
                                  device, n_tta=tta_n, desc="  test")

        # oof.npy는 앙상블 호환을 위해 val 예측으로 대체 (단일 fold이므로)
        np.save(os.path.join(exp_dir, "oof.npy"),      va_preds)
        np.save(os.path.join(exp_dir, "y_true.npy"),   va_labels_ext)
        np.save(os.path.join(exp_dir, "val_pred.npy"), val_preds)
        np.save(os.path.join(exp_dir, "test_pred.npy"), test_preds)
        return {"oof_macro_f1": best_score, "val_macro_f1": best_score}

    # ── KFold 학습 ──────────────────────────────────────────────────────────────
    folder = get_folder("multiclass", n_folds, cfg["seed"], group=True)
    oof    = np.zeros((len(train_df), n_cls), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(cv_split(folder, train_df, labels, groups)):
        logger.info(f"=== Fold {fold + 1}/{n_folds} ===")
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        tr_loader = _make_loader(tr_df, audio_dir, cfg, demo, shuffle=True)
        va_loader = _make_loader(va_df, audio_dir, cfg, demo)

        cnts = np.bincount(labels[tr_idx], minlength=n_cls).astype(float)
        w    = torch.tensor((cnts.sum() / (n_cls * cnts)).clip(0.1, 10),
                            dtype=torch.float32, device=device)
        criterion = FocalLoss(gamma=2.0, weight=w,
                              label_smoothing=cfg.get("label_smoothing", 0.0))

        model = create_audio_model(cfg, n_cls).to(device)

        freeze_epochs      = cfg.get("freeze_epochs", 0)
        unfreeze_lr_scale  = cfg.get("unfreeze_lr_scale", 0.1)
        total_epochs       = cfg["epochs"]
        base_lr            = cfg["lr"]
        patience           = cfg.get("patience", 5)

        # Phase 1: 백본 freeze, 헤드만 학습
        if freeze_epochs > 0:
            _freeze_backbone(model)
            logger.info(f"  [Phase 1] 백본 freeze — {freeze_epochs} epochs (헤드만 학습)")
            optimizer = _make_optimizer(model, base_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=freeze_epochs, eta_min=base_lr * 0.01)
        else:
            optimizer = _make_optimizer(model, base_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=base_lr * 0.01)

        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        best_score, best_oof_fold, patience_cnt = -np.inf, None, 0

        epoch_bar = tqdm(range(total_epochs), desc=f"Fold {fold+1}/{n_folds}",
                         unit="ep", dynamic_ncols=True)
        for epoch in epoch_bar:
            # Phase 1 → Phase 2 전환
            if freeze_epochs > 0 and epoch == freeze_epochs:
                _unfreeze_backbone(model)
                optimizer = _make_optimizer(model, base_lr,
                                            backbone_lr_scale=unfreeze_lr_scale)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_epochs - freeze_epochs, eta_min=base_lr * 0.01)
                best_score, patience_cnt = -np.inf, 0   # patience 리셋
                logger.info(f"  [Phase 2] 백본 unfreeze "
                            f"(backbone lr={base_lr * unfreeze_lr_scale:.2e}, "
                            f"head lr={base_lr:.2e})")

            loss = _train_epoch(model, tr_loader, criterion, optimizer, scaler, device,
                                mixup_p=cfg.get("mixup_p", 0.0),
                                mixup_alpha=cfg.get("mixup_alpha", 0.4),
                                desc="  train")
            va_preds, va_labels = _eval_epoch(model, va_loader, device, desc="  val")
            score = get_metric("multiclass", va_labels, va_preds)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            phase = 1 if epoch < freeze_epochs else 2
            epoch_bar.set_postfix(
                loss=f"{loss:.4f}", F1=f"{score:.4f}",
                best=f"{best_score:.4f}" if best_score > -np.inf else "-",
                lr=f"{lr_now:.2e}", ph=phase,
            )
            logger.info(f"  ep {epoch+1:3d}/{total_epochs} [ph{phase}] | loss={loss:.4f} "
                        f"| MacroF1={score:.4f} | lr={lr_now:.2e}")

            if score > best_score:
                best_score, best_oof_fold, patience_cnt = score, va_preds.copy(), 0
                torch.save(model.state_dict(),
                           os.path.join(exp_dir, f"model_fold{fold}.pt"))
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    logger.info(f"  early stop at epoch {epoch + 1} [phase {phase}]")
                    epoch_bar.close()
                    break

        oof[va_idx] = best_oof_fold
        logger.info(f"  Fold {fold + 1} best MacroF1: {best_score:.5f}")

    oof_score = get_metric("multiclass", labels, oof)
    logger.info(f"OOF MacroF1: {oof_score:.5f}")
    np.save(os.path.join(exp_dir, "oof.npy"), oof)
    np.save(os.path.join(exp_dir, "y_true.npy"), labels)

    # t1_val / t1_test 앙상블 예측
    val_df,  val_audio  = load_task1(cfg, "val",  demo=demo)
    test_df, test_audio = load_task1(cfg, "test", demo=demo)
    val_preds  = np.zeros((len(val_df),  n_cls), dtype=np.float32)
    test_preds = np.zeros((len(test_df), n_cls), dtype=np.float32)
    loaded = 0

    model = create_audio_model(cfg, n_cls).to(device)
    tta_n = cfg.get("tta_n", 1)
    for fold in tqdm(range(n_folds), desc="앙상블 예측", unit="fold", dynamic_ncols=True):
        ckpt = os.path.join(exp_dir, f"model_fold{fold}.pt")
        if not os.path.exists(ckpt):
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        val_preds  += _predict_tta(model, _make_loader(val_df,  val_audio,  cfg, demo, is_test=True),
                                   device, n_tta=tta_n, desc=f"  fold{fold} val")
        test_preds += _predict_tta(model, _make_loader(test_df, test_audio, cfg, demo, is_test=True),
                                   device, n_tta=tta_n, desc=f"  fold{fold} test")
        loaded += 1

    if loaded:
        val_preds /= loaded; test_preds /= loaded

    np.save(os.path.join(exp_dir, "val_pred.npy"),  val_preds)
    np.save(os.path.join(exp_dir, "test_pred.npy"), test_preds)

    val_score = get_metric("multiclass",
                           val_df["ship_type"].map(SHIP_TYPE_TO_IDX).values, val_preds)
    logger.info(f"t1_val MacroF1 (ensemble): {val_score:.5f}")
    return {"oof_macro_f1": oof_score, "val_macro_f1": val_score}


# ── 체크포인트 → npy 생성 (학습 중단 후 예측만 뽑을 때) ──────────────────────

def predict_only(cfg, exp_dir, demo, logger):
    """experiments/{exp}/model_fold*.pt 로 val_pred.npy, test_pred.npy 생성."""
    import torch
    import glob
    from src.models.audio import create_audio_model
    from src.data.loaders import load_task1, SHIP_TYPE_TO_IDX

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_cls  = len(SHIP_TYPE_TO_IDX)
    tta_n  = cfg.get("tta_n", 1)

    ckpts = sorted(glob.glob(os.path.join(exp_dir, "model_fold*.pt")))
    if not ckpts:
        logger.error(f"체크포인트 없음: {exp_dir}/model_fold*.pt")
        return

    logger.info(f"발견된 체크포인트 {len(ckpts)}개: {[os.path.basename(c) for c in ckpts]}")

    val_df,  val_audio  = load_task1(cfg, "val",  demo=demo)
    test_df, test_audio = load_task1(cfg, "test", demo=demo)
    val_preds  = np.zeros((len(val_df),  n_cls), dtype=np.float32)
    test_preds = np.zeros((len(test_df), n_cls), dtype=np.float32)

    model = create_audio_model(cfg, n_cls).to(device)
    for ckpt in tqdm(ckpts, desc="predict", unit="fold", dynamic_ncols=True):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        val_preds  += _predict_tta(model, _make_loader(val_df,  val_audio,  cfg, demo, is_test=True),
                                   device, n_tta=tta_n, desc=f"  {os.path.basename(ckpt)} val")
        test_preds += _predict_tta(model, _make_loader(test_df, test_audio, cfg, demo, is_test=True),
                                   device, n_tta=tta_n, desc=f"  {os.path.basename(ckpt)} test")

    val_preds  /= len(ckpts)
    test_preds /= len(ckpts)

    np.save(os.path.join(exp_dir, "val_pred.npy"),  val_preds)
    np.save(os.path.join(exp_dir, "test_pred.npy"), test_preds)

    val_score = get_metric("multiclass",
                           val_df["ship_type"].map(SHIP_TYPE_TO_IDX).values, val_preds)
    logger.info(f"val MacroF1: {val_score:.5f}  ({len(ckpts)} ckpt 평균)")
    logger.info(f"저장: {exp_dir}/val_pred.npy, test_pred.npy")


# ── 엔트리포인트 ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       required=True)
    ap.add_argument("--exp",          default="exp_001")
    ap.add_argument("--demo",         action="store_true")
    ap.add_argument("--predict-only", action="store_true",
                    help="학습 없이 기존 체크포인트로 val_pred/test_pred.npy만 생성")
    args = ap.parse_args()

    cfg    = load_config(args.config)
    seed_everything(cfg["seed"])
    logger = get_logger()

    exp_dir = os.path.join("experiments", args.exp)
    os.makedirs(exp_dir, exist_ok=True)

    if args.predict_only:
        logger.info(f"=== predict-only | exp={args.exp} ===")
        predict_only(cfg, exp_dir, args.demo, logger)
        return

    task     = cfg.get("task", "multiclass")
    task_type = cfg.get("type", "audio")
    logger.info(f"=== train | task={task} | exp={args.exp} ===")

    if task == "multiclass":
        results = train_audio_task1(cfg, exp_dir, args.demo, logger)
    else:
        raise ValueError(f"unknown task: {task}")

    primary = list(results.values())[0]
    log_experiment(
        exp_id=args.exp,
        model=cfg.get("model_name", "audio"),
        params={k: cfg.get(k) for k in ("lr", "epochs", "batch_size", "ais_dim", "n_folds")
                if cfg.get(k) is not None},
        oof_score=primary,
        cv=f"{cfg.get('n_folds', 1)}fold",
        seed=cfg["seed"],
        notes=str(results),
    )
    logger.info(f"완료. 산출물: {exp_dir}/")


if __name__ == "__main__":
    main()

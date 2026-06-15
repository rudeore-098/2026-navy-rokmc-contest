"""LightGBM AIS-only 학습 — Task 1 선종 분류.

음향 없이 AIS 7-dim 피처(Model B)만으로 학습.
산출물 포맷은 train.py와 동일 → ensemble.py에 바로 편입 가능.

사용법:
    python -m src.train_lgbm --config configs/lgbm_ais.yaml --exp exp_lgbm_ais_001

앙상블:
    python -m src.ensemble --task multiclass --exps exp_t1_effb2 exp_lgbm_ais_001
"""
import os
import argparse
import numpy as np
import pandas as pd

from src.utils.seed import seed_everything
from src.utils.metrics import get_metric
from src.utils.logger import get_logger, log_experiment
from src.data.cv import get_folder, split as cv_split
from src.data.loaders import SHIP_TYPE_TO_IDX


def encode_ais(df, sog_mean, sog_std):
    """DataFrame → 7-dim AIS 피처 (Model B 방식).

    [sog_norm, cog_sin, cog_cos, hdg_sin, hdg_cos, cog_missing, hdg_missing]

    Parameters
    ----------
    sog_mean, sog_std : train 전체 기준 SOG log-clip 통계 (누수 방지)
    """
    sog = df["sog"].astype(float).values
    cog = df["cog"].astype(float).values
    hdg = df["true_heading"].astype(float).values

    sog_norm = (np.log1p(np.clip(sog, 0, 30)) - sog_mean) / sog_std

    cog_missing = ((cog == 360.0) | (sog == 0.0)).astype(np.float32)
    hdg_missing = ((hdg == 0.0) | (hdg > 360.0)).astype(np.float32)

    cog_rad = np.deg2rad(cog)
    hdg_rad = np.deg2rad(hdg)
    cog_sin = np.where(cog_missing, 0.0, np.sin(cog_rad))
    cog_cos = np.where(cog_missing, 0.0, np.cos(cog_rad))
    hdg_sin = np.where(hdg_missing, 0.0, np.sin(hdg_rad))
    hdg_cos = np.where(hdg_missing, 0.0, np.cos(hdg_rad))

    return np.stack([
        sog_norm, cog_sin, cog_cos, hdg_sin, hdg_cos,
        cog_missing, hdg_missing,
    ], axis=1).astype(np.float32)


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_lgbm_ais(cfg, exp_dir, logger):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    train_df = pd.read_csv(cfg["paths"]["train"])
    val_df   = pd.read_csv(cfg["paths"]["task1_val"])
    test_df  = pd.read_csv(cfg["paths"]["task1_test"])

    labels = train_df["ship_type"].map(SHIP_TYPE_TO_IDX).values
    groups = train_df["ship_id"].values
    n_cls   = len(SHIP_TYPE_TO_IDX)
    n_folds = cfg.get("n_folds", 5)

    # ── 데이터 로딩 로그 ──────────────────────────────────────────────────────
    logger.info(f"train : {len(train_df):,}행  |  ships: {train_df['ship_id'].nunique()}")
    logger.info(f"val   : {len(val_df):,}행")
    logger.info(f"test  : {len(test_df):,}행")

    logger.info("── 클래스 분포 (train) ──")
    for cls, idx in SHIP_TYPE_TO_IDX.items():
        cnt = int((labels == idx).sum())
        logger.info(f"  {cls:<20} {cnt:5d}  ({cnt/len(labels)*100:.1f}%)")

    sog = train_df["sog"].astype(float).values
    cog = train_df["cog"].astype(float).values
    hdg = train_df["true_heading"].astype(float).values
    logger.info("── 센티넬 현황 (train) ──")
    logger.info(f"  SOG==0    : {(sog==0).sum():5d}  ({(sog==0).mean()*100:.1f}%)  → cog_missing")
    logger.info(f"  COG==360  : {(cog==360).sum():5d}  ({(cog==360).mean()*100:.1f}%)  → cog_missing")
    logger.info(f"  HDG==0    : {(hdg==0).sum():5d}  ({(hdg==0).mean()*100:.1f}%)  → hdg_missing")
    logger.info(f"  HDG>360   : {(hdg>360).sum():5d}  ({(hdg>360).mean()*100:.1f}%)  → hdg_missing")

    # ── AIS 인코딩 ────────────────────────────────────────────────────────────
    sog_log  = np.log1p(np.clip(sog, 0, 30))
    sog_mean = float(sog_log.mean())
    sog_std  = float(sog_log.std() + 1e-6)
    logger.info(f"── SOG log-clip 정규화: mean={sog_mean:.4f}, std={sog_std:.4f}")

    X_train = encode_ais(train_df, sog_mean, sog_std)
    X_val   = encode_ais(val_df,   sog_mean, sog_std)
    X_test  = encode_ais(test_df,  sog_mean, sog_std)

    FEAT_NAMES = ["sog_norm","cog_sin","cog_cos","hdg_sin","hdg_cos","cog_missing","hdg_missing"]
    logger.info(f"── 피처 {X_train.shape[1]}개: {FEAT_NAMES}")
    logger.info(f"   train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    lgbm_params = {**cfg.get("lgbm", {}), "random_state": cfg.get("seed", 42)}
    es_rounds   = cfg.get("early_stopping_rounds", 50)

    folder = get_folder("multiclass", n_folds, cfg["seed"], group=True)
    oof        = np.zeros((len(train_df), n_cls), dtype=np.float32)
    val_preds  = np.zeros((len(val_df),  n_cls), dtype=np.float32)
    test_preds = np.zeros((len(test_df), n_cls), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(cv_split(folder, train_df, labels, groups)):
        logger.info(f"=== Fold {fold+1}/{n_folds} ===")
        clf = LGBMClassifier(**lgbm_params)
        clf.fit(
            X_train[tr_idx], labels[tr_idx],
            eval_set=[(X_train[va_idx], labels[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(es_rounds), log_evaluation(0)],
        )
        oof[va_idx] = clf.predict_proba(X_train[va_idx])
        val_preds  += clf.predict_proba(X_val)
        test_preds += clf.predict_proba(X_test)

        fold_score = get_metric("multiclass", labels[va_idx], oof[va_idx])
        logger.info(f"  Fold {fold+1} MacroF1: {fold_score:.5f} | best_iter: {clf.best_iteration_}")

    val_preds  /= n_folds
    test_preds /= n_folds

    oof_score = get_metric("multiclass", labels, oof)
    val_score = get_metric("multiclass",
                           val_df["ship_type"].map(SHIP_TYPE_TO_IDX).values, val_preds)
    logger.info(f"OOF MacroF1:     {oof_score:.5f}")
    logger.info(f"val MacroF1:     {val_score:.5f}")

    np.save(os.path.join(exp_dir, "oof.npy"),       oof)
    np.save(os.path.join(exp_dir, "y_true.npy"),    labels)
    np.save(os.path.join(exp_dir, "val_pred.npy"),  val_preds)
    np.save(os.path.join(exp_dir, "test_pred.npy"), test_preds)

    return {"oof_macro_f1": oof_score, "val_macro_f1": val_score}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp",    default="exp_lgbm_001")
    args = ap.parse_args()

    cfg    = load_config(args.config)
    seed_everything(cfg["seed"])
    logger = get_logger()

    exp_dir = os.path.join("experiments", args.exp)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"=== LightGBM AIS-only | exp={args.exp} ===")
    results = train_lgbm_ais(cfg, exp_dir, logger)

    log_experiment(
        exp_id=args.exp,
        model="lightgbm_ais_7dim",
        params={"n_folds": cfg.get("n_folds"), "seed": cfg.get("seed"),
                **{k: cfg["lgbm"].get(k) for k in ("learning_rate", "num_leaves")
                   if cfg.get("lgbm")}},
        oof_score=results["oof_macro_f1"],
        cv=f"{cfg.get('n_folds', 5)}fold",
        seed=cfg["seed"],
        notes=str(results),
    )
    logger.info(f"완료. 산출물: {exp_dir}/")


if __name__ == "__main__":
    main()

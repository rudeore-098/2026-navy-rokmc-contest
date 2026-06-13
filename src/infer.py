"""추론 — 저장된 test 예측을 모아 제출 파일 생성.

train.py가 각 모델의 test_{kind}.npy를 저장해뒀으므로,
여기선 그것들을 합쳐(단순 평균 또는 ensemble.py의 가중치) submission.csv를 만든다.

사용법:
    python -m src.infer --config configs/tabular.yaml --exp exp_001
"""
import os
import argparse
import numpy as np
import pandas as pd

from src.utils.metrics import to_submission


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="exp_001")
    ap.add_argument("--weights", default=None,
                    help="콤마구분 가중치 'lgb,xgb,cat' 예: 0.5,0.2,0.3 (없으면 단순평균)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp_dir = os.path.join("experiments", args.exp)
    task = cfg["task"]

    kinds = ["lgb", "xgb", "cat"]
    test_preds, used = [], []
    for k in kinds:
        p = os.path.join(exp_dir, f"test_{k}.npy")
        if os.path.exists(p):
            test_preds.append(np.load(p)); used.append(k)
    if not test_preds:
        raise FileNotFoundError(f"{exp_dir}에 test_*.npy가 없습니다. 먼저 train을 실행하세요.")

    if args.weights:
        w = np.array([float(x) for x in args.weights.split(",")])
        w = w / w.sum()
        final = sum(wi * p for wi, p in zip(w, test_preds))
    else:
        final = np.mean(test_preds, axis=0)

    test_ids = np.load(os.path.join(exp_dir, "test_ids.npy"))
    sub = pd.DataFrame({
        cfg.get("id_col", "id"): test_ids,
        cfg.get("target_col", "target"): to_submission(task, final),
    })
    out = os.path.join(exp_dir, "submission.csv")
    sub.to_csv(out, index=False)
    print(f"제출 파일 저장: {out} (모델: {used})")
    print(sub.head())


if __name__ == "__main__":
    main()

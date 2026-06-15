"""Task 1 제출 파일 생성 — train.py가 저장한 test_pred.npy → submission.csv.

사용법:
    python -m src.infer --config configs/audio_task1.yaml --exp exp_t1_001
    python -m src.infer --config configs/audio_task1.yaml --exp exp_t1_001 --split val
"""
import os
import argparse
import numpy as np
import pandas as pd

from src.data.loaders import IDX_TO_SHIP_TYPE, SHIP_TYPE_TO_IDX
from src.utils.metrics import get_metric


def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp",    default="exp_001")
    ap.add_argument("--split",  default="test", choices=["test", "val"])
    args = ap.parse_args()

    cfg     = load_config(args.config)
    exp_dir = os.path.join("experiments", args.exp)
    paths   = cfg["paths"]

    preds = np.load(os.path.join(exp_dir, f"{args.split}_pred.npy"))  # (N, 4)

    if args.split == "test":
        df = pd.read_csv(paths["task1_test"])
    else:
        df = pd.read_csv(paths["task1_val"])

    sub = pd.DataFrame({
        "filename":  df["filename"],
        "ship_type": [IDX_TO_SHIP_TYPE[i] for i in preds.argmax(axis=1)],
    })

    out = os.path.join(exp_dir, f"submission_{args.split}.csv")
    sub.to_csv(out, index=False)
    print(f"Saved: {out}  ({len(sub)} rows)")
    print(sub["ship_type"].value_counts().to_string())

    if args.split == "val":
        true = df["ship_type"].map(SHIP_TYPE_TO_IDX).values
        score = get_metric("multiclass", true, preds)
        print(f"\nVal MacroF1: {score:.5f}")


if __name__ == "__main__":
    main()

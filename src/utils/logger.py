"""로깅 — 콘솔 출력 + 실험 결과 CSV 자동 기록.

노션 DB와 연결하는 흐름: 여기서 남긴 experiments/log.csv를 보고
노션에 옮기거나, exp_id를 노션 '실험ID'와 맞춰 추적한다.
"""
import os
import csv
import json
import logging
import datetime
from typing import Optional


def get_logger(name: str = "comp", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger


def log_experiment(
    exp_id: str,
    model: str,
    params: dict,
    oof_score: float,
    lb_score: Optional[float] = None,
    cv: str = "5fold",
    seed: int = 42,
    notes: str = "",
    log_dir: str = "experiments",
):
    """실험 한 건을 experiments/log.csv에 append + config를 JSON으로 저장.

    컬럼은 실행마다 합집합으로 관리되므로, 모델별 파라미터가 달라도 안전하다.
    """
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "log.csv")

    row = {
        "exp_id": exp_id,
        "datetime": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "cv": cv,
        "seed": seed,
        "oof_score": round(float(oof_score), 6),
        "lb_score": "" if lb_score is None else round(float(lb_score), 6),
        "notes": notes,
    }
    for k, v in params.items():
        row[f"p_{k}"] = v

    # config JSON 저장 (정확한 재현용)
    exp_dir = os.path.join(log_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"model": model, "params": params, "seed": seed, "cv": cv,
             "oof_score": oof_score, "lb_score": lb_score, "notes": notes},
            f, indent=2, ensure_ascii=False,
        )

    # CSV append (컬럼 합집합)
    rows, fields = [], []
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        rows = df.to_dict("records")
        fields = list(df.columns)
    for k in row:
        if k not in fields:
            fields.append(k)
    rows.append(row)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    return csv_path

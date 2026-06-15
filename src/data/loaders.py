"""데이터 로더 — Task 2 갤러리/쿼리 로드 + 선종 라벨 매핑.

Task 1 train/test 는 train.py 가 pd.read_csv 로 직접 읽으므로 별도 로더 불필요.

[변경 사항 - 2026-06-15]
- load_tabular / load_vision / wav_to_melspec / load_audio_df / load_task1 삭제
- AudioDataset 클래스 삭제 (CNN 경로용, AST 는 audio.py 의 _ASTDS 사용)
- Task 2 로더 + 선종 매핑만 유지
"""
import numpy as np
import pandas as pd


# ── 선종 4-class 라벨 매핑 (Task 1 / Task 2 공통) ────────────────────────────
SHIP_TYPE_TO_IDX = {
    "A_SmallWorking": 0,
    "B_MotorBoat":    1,
    "C_Passenger":    2,
    "D_LargeShip":    3,
}
IDX_TO_SHIP_TYPE = {v: k for k, v in SHIP_TYPE_TO_IDX.items()}


# ── Task 2 로더 ─────────────────────────────────────────────────────────────

def load_task2_gallery(cfg, demo: bool = False):
    """Task 2 갤러리 (100척 참조 클립) 로드.

    컬럼: filename, ship_id, ship_type, AIS

    Returns
    -------
    df        : DataFrame
    audio_dir : str
    """
    if demo:
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "filename": [f"gallery_{i:06d}.wav" for i in range(n)],
            "ship_id":  np.repeat(np.arange(10), 10),
            "ship_type": rng.choice(list(SHIP_TYPE_TO_IDX), n),
            "sog":       rng.uniform(0, 20, n).round(1),
            "cog":       rng.uniform(0, 360, n).round(1),
            "true_heading": rng.uniform(0, 360, n).round(1),
        })
        return df, None

    paths = cfg["paths"]
    df = pd.read_csv(paths["task2_gallery"])
    return df, paths["task2_audio"]


def load_task2_query(cfg, split: str = "val", demo: bool = False):
    """Task 2 쿼리 로드.

    Parameters
    ----------
    split : "val" | "test"
        - val  : filename, ship_id, ship_type  (라벨 있음, 자체 검증용)
        - test : filename only                 (라벨 없음, 제출 대상)

    Returns
    -------
    df        : DataFrame
    audio_dir : str | None
    """
    if demo:
        rng = np.random.default_rng(1)
        n = 50
        df = pd.DataFrame({
            "filename": [f"query_{i:06d}.wav" for i in range(n)],
            "ship_id":  rng.integers(0, 10, n),
        })
        return df, None

    paths = cfg["paths"]
    if split == "val":
        df = pd.read_csv(paths["task2_val"])
    elif split == "test":
        df = pd.read_csv(paths["task2_test"])
    else:
        raise ValueError(f"unknown split: {split}")
    return df, paths["task2_audio"]
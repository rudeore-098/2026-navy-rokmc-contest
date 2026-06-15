"""오디오 데이터 로더 — CSV 로딩, AudioDataset.

학습 전에 반드시 mel 변환을 먼저 실행:
    python -m src.data.mel --config configs/audio_task1.yaml
"""
import os
import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:
    _TorchDataset = object

# Task 1 선종 4-class 레이블 매핑
SHIP_TYPE_TO_IDX = {
    "A_SmallWorking": 0,
    "B_MotorBoat":    1,
    "C_Passenger":    2,
    "D_LargeShip":    3,
}
IDX_TO_SHIP_TYPE = {v: k for k, v in SHIP_TYPE_TO_IDX.items()}


# ── CSV 로더 ───────────────────────────────────────────────────────────────────

def load_audio_df(cfg, split: str = "train", demo: bool = False):
    """음향 DataFrame + 오디오 디렉토리 반환 (범용).

    split : "train" | "val" | "test" | "gallery"
    """
    if demo:
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "filename":     [f"demo_{i:06d}.wav" for i in range(n)],
            "ship_type":    rng.choice(list(SHIP_TYPE_TO_IDX), n),
            "ship_id":      rng.integers(0, 50, n),
            "sog":          rng.uniform(0, 20, n).round(1),
            "cog":          rng.uniform(0, 360, n).round(1),
            "true_heading": rng.uniform(0, 360, n).round(1),
        })
        return df, None

    df = pd.read_csv(cfg["paths"][split])
    audio_dir = cfg["paths"][f"{split}_audio"]
    return df, audio_dir


def load_task1(cfg, split: str = "train", demo: bool = False):
    """Task 1 선종분류 데이터 로드.

    split : "train" | "val" | "test"
      - train : filename, ship_id, ship_type, AIS  (라벨 있음)
      - val   : filename, ship_type, AIS           (라벨 있음)
      - test  : filename, AIS                      (라벨 없음)
    """
    if demo:
        return load_audio_df(cfg, demo=True)

    paths = cfg["paths"]
    if split == "train":
        return pd.read_csv(paths["train"]), paths["train_audio"]
    elif split == "val":
        return pd.read_csv(paths["task1_val"]), paths["task1_audio"]
    elif split == "test":
        return pd.read_csv(paths["task1_test"]), paths["task1_audio"]
    raise ValueError(f"unknown split: {split}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class AudioDataset(_TorchDataset):
    """WAV + AIS → (spec_tensor, ais_tensor, label) 반환하는 Dataset.

    Parameters
    ----------
    df        : load_task1() 등이 반환한 DataFrame
    audio_dir : WAV 파일 폴더 (demo 모드에선 None)
    cfg       : yaml config dict
    is_test   : True면 label 없이 (spec, ais) 만 반환
    demo      : True면 실제 파일 없이 랜덤 텐서로 동작
    cache_dir : 미리 변환된 .npy 폴더 (None이면 매번 librosa 호출)
    """

    _SOG_MEAN, _SOG_STD = 10.0, 8.0

    def __init__(self, df, audio_dir, cfg, is_test=False, demo=False,
                 label_col=None, label_map=None, cache_dir="data/spec_cache"):
        self.df        = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.is_test   = is_test
        self.demo      = demo
        self.cache_dir = cache_dir
        self.label_col = label_col or "ship_type"
        self.label_map = label_map if label_map is not None else SHIP_TYPE_TO_IDX

        self.sr         = cfg.get("sample_rate", 32000)
        self.duration   = cfg.get("duration",    5.0)
        self.n_mels     = cfg.get("n_mels",      128)
        self.n_fft      = cfg.get("n_fft",       2048)
        self.hop_length = cfg.get("hop_length",  512)
        self.f_min      = cfg.get("f_min",       50)
        self.f_max      = cfg.get("f_max",       16000)

        # packed 배열 로드 (있으면 파일 오픈 오버헤드 제거)
        self._packed = None
        self._packed_index = None
        if not demo:
            packed_path = os.path.join(cache_dir, "packed_all.npy")
            index_path  = os.path.join(cache_dir, "packed_all_index.json")
            if os.path.exists(packed_path) and os.path.exists(index_path):
                import json
                self._packed = np.load(packed_path, mmap_mode='r')
                self._packed_index = json.load(open(index_path))

    def __len__(self):
        return len(self.df)

    def _encode_ais(self, row):
        """AIS 메타데이터 → 5-dim float32 벡터 (sog + sin/cos×cog + sin/cos×heading)."""
        sog = (float(row.get("sog", 0.0)) - self._SOG_MEAN) / (self._SOG_STD + 1e-6)
        cog_rad = np.deg2rad(float(row.get("cog", 0.0)))
        hdg_rad = np.deg2rad(float(row.get("true_heading", 0.0)))
        return np.array([
            sog,
            np.sin(cog_rad), np.cos(cog_rad),
            np.sin(hdg_rad), np.cos(hdg_rad),
        ], dtype=np.float32)

    def _load_spec(self, filename):
        if self.demo:
            time_frames = int(self.sr * self.duration) // self.hop_length + 1
            return np.random.rand(self.n_mels, time_frames).astype(np.float32)

        # packed 배열 우선 (파일 오픈 없이 슬라이스)
        if self._packed is not None and filename in self._packed_index:
            return np.array(self._packed[self._packed_index[filename]])

        # fallback: 개별 .npy
        npy_path = os.path.join(self.cache_dir, filename.replace(".wav", ".npy"))
        try:
            return np.load(npy_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"캐시 파일이 없습니다: {npy_path}\n"
                "학습 전 mel 변환을 먼저 실행하세요:\n"
                "  python -m src.data.mel --config configs/audio_task1.yaml"
            ) from None

    def __getitem__(self, idx):
        import torch
        row = self.df.iloc[idx]

        spec_tensor = torch.from_numpy(self._load_spec(row["filename"])).unsqueeze(0)
        ais_tensor  = torch.from_numpy(self._encode_ais(row))

        if self.is_test:
            return spec_tensor, ais_tensor

        label = self.label_map[row[self.label_col]]
        return spec_tensor, ais_tensor, torch.tensor(label, dtype=torch.long)

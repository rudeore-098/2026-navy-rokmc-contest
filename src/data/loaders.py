"""오디오 데이터 로더 — WAV + AIS 데이터를 로드해 AudioDataset으로 반환."""
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


def wav_to_melspec(filepath, sr=32000, duration=5.0,
                   n_mels=128, n_fft=2048, hop_length=512,
                   f_min=50, f_max=16000, top_db=80.0):
    """WAV 파일 → 정규화된 Log-mel spectrogram.

    Returns
    -------
    log_mel : np.ndarray, shape (n_mels, time_frames), dtype float32
              값 범위 [0, 1]  (0 = 가장 조용, 1 = 최대)
    """
    import librosa
    y, _ = librosa.load(filepath, sr=sr, duration=duration, mono=True)

    # 5초에 맞게 zero-pad 또는 truncate
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=f_min, fmax=f_max,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)

    # [-top_db, 0] → [0, 1]
    log_mel = (log_mel + top_db) / top_db
    return log_mel.astype(np.float32)


def load_audio_df(cfg, split: str = "train", demo: bool = False):
    """음향 DataFrame + 오디오 디렉토리 반환 (범용).

    Parameters
    ----------
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


# ── 대회 데이터 구조 전용 로더 ───────────────────────────────────────────────
#
# 폴더 구조 (data/ 아래에 대회 데이터를 복사 또는 심볼릭링크):
#   data/train/train.csv  +  data/train/audio/
#   data/task1_test/val.csv | test.csv  +  data/task1_test/audio/
#   data/task2_test/gallery.csv | val.csv | test.csv  +  data/task2_test/audio/
#   data/ship_list.csv
#   data/task2_target_ships.csv

def load_task1(cfg, split: str = "train", demo: bool = False):
    """Task 1 선종분류 데이터 로드.

    Parameters
    ----------
    split : "train" | "val" | "test"
        - train : filename, ship_id, ship_type, AIS  (라벨 있음)
        - val   : filename, ship_type, AIS           (라벨 있음, 자체 검증용)
        - test  : filename, AIS                      (라벨 없음, 제출 대상)

    Returns
    -------
    df        : DataFrame
    audio_dir : str | None
    """
    if demo:
        return load_audio_df(cfg, demo=True)

    paths = cfg["paths"]
    if split == "train":
        df = pd.read_csv(paths["train"])
        audio_dir = paths["train_audio"]
    elif split == "val":
        df = pd.read_csv(paths["task1_val"])
        audio_dir = paths["task1_audio"]
    elif split == "test":
        df = pd.read_csv(paths["task1_test"])
        audio_dir = paths["task1_audio"]
    else:
        raise ValueError(f"unknown split: {split}")
    return df, audio_dir


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


class AudioDataset(_TorchDataset):
    """WAV + AIS → (spec_tensor, ais_tensor, label) 반환하는 Dataset.

    torch.utils.data.DataLoader에 바로 넣을 수 있다.

    Parameters
    ----------
    df        : load_audio_df() 반환 DataFrame
    audio_dir : WAV 파일 폴더 (demo 모드에선 None)
    cfg       : yaml config dict
    is_test   : True면 label 없이 (spec, ais) 만 반환
    demo      : True면 실제 파일 없이 랜덤 텐서로 동작
    """

    # AIS 정규화 상수 (sog, cog, true_heading 순)
    # cog/heading은 원형 → sin/cos 2차원으로 펼쳐 총 5-dim
    _SOG_MEAN, _SOG_STD = 10.0, 8.0

    def __init__(self, df, audio_dir, cfg, is_test=False, demo=False,
                 label_col=None, label_map=None):
        self.df        = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.is_test   = is_test
        self.demo      = demo
        self.label_col = label_col or "ship_type"
        self.label_map = label_map if label_map is not None else SHIP_TYPE_TO_IDX

        self.sr         = cfg.get("sample_rate", 32000)
        self.duration   = cfg.get("duration",    5.0)
        self.n_mels     = cfg.get("n_mels",      128)
        self.n_fft      = cfg.get("n_fft",       2048)
        self.hop_length = cfg.get("hop_length",  512)
        self.f_min      = cfg.get("f_min",       50)
        self.f_max      = cfg.get("f_max",       16000)

    def __len__(self):
        return len(self.df)

    def _encode_ais(self, row):
        """AIS 메타데이터 → 5-dim float32 벡터.

        sog 정규화 + cog/heading sin-cos 인코딩(원형 피처 처리).
        """
        sog = (float(row.get("sog", 0.0)) - self._SOG_MEAN) / (self._SOG_STD + 1e-6)

        cog_rad = np.deg2rad(float(row.get("cog", 0.0)))
        hdg_rad = np.deg2rad(float(row.get("true_heading", 0.0)))

        return np.array([
            sog,
            np.sin(cog_rad), np.cos(cog_rad),
            np.sin(hdg_rad), np.cos(hdg_rad),
        ], dtype=np.float32)

    def __getitem__(self, idx):
        import torch
        row = self.df.iloc[idx]

        # ── 스펙트로그램 ──────────────────────────────────────────────────────
        time_frames = int(self.sr * self.duration) // self.hop_length + 1
        if self.demo:
            spec = np.random.rand(self.n_mels, time_frames).astype(np.float32)
        else:
            path = os.path.join(self.audio_dir, row["filename"])
            try:
                spec = wav_to_melspec(
                    path, sr=self.sr, duration=self.duration,
                    n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length,
                    f_min=self.f_min, f_max=self.f_max,
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load {path}: {e}. Returning zeros.")
                spec = np.zeros((self.n_mels, time_frames), dtype=np.float32)
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, time)

        # ── AIS 피처 ─────────────────────────────────────────────────────────
        ais_tensor = torch.from_numpy(self._encode_ais(row))  # (5,)

        if self.is_test:
            return spec_tensor, ais_tensor

        label = self.label_map[row[self.label_col]]
        return spec_tensor, ais_tensor, torch.tensor(label, dtype=torch.long)

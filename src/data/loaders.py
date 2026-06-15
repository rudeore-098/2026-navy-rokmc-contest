"""데이터 로더 — task별 데이터를 읽어 (X, y, groups, test) 형태로 통일 반환.

실제 대회에선 이 파일의 경로/컬럼명만 수정하면 된다.
데모 모드(demo=True)는 합성 데이터를 생성해 파이프라인 전체를 검증할 수 있다.
"""
import os
import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:
    _TorchDataset = object


# ── 정형 ────────────────────────────────────────────────────────────────────

def load_tabular(cfg, demo: bool = False):
    """정형 데이터 로드. (X, y, groups, X_test, test_ids) 반환."""
    if demo:
        from sklearn.datasets import make_classification, make_regression
        task = cfg["task"]
        if task == "binary":
            X, y = make_classification(n_samples=2000, n_features=15,
                                       n_informative=8, n_classes=2, random_state=cfg["seed"])
        elif task == "multiclass":
            X, y = make_classification(n_samples=2000, n_features=15, n_informative=10,
                                       n_classes=4, n_clusters_per_class=1, random_state=cfg["seed"])
        else:
            X, y = make_regression(n_samples=2000, n_features=15, n_informative=8,
                                   noise=15.0, random_state=cfg["seed"])
        cols = [f"f{i}" for i in range(X.shape[1])]
        Xdf = pd.DataFrame(X, columns=cols)
        from sklearn.model_selection import train_test_split
        tr, te = train_test_split(
            pd.concat([Xdf, pd.Series(y, name="target")], axis=1),
            test_size=0.25, random_state=cfg["seed"],
        )
        X_tr = tr[cols].reset_index(drop=True)
        y_tr = tr["target"].values
        X_te = te[cols].reset_index(drop=True)
        return X_tr, y_tr, None, X_te, np.arange(len(X_te))

    # ===== 실제 대회: 아래를 본인 데이터에 맞게 =====
    train = pd.read_csv(cfg["paths"]["train"])
    test = pd.read_csv(cfg["paths"]["test"])
    target = cfg["target_col"]
    id_col = cfg.get("id_col")
    feature_cols = [c for c in train.columns if c not in [target, id_col]]
    groups = train[cfg["group_col"]].values if cfg.get("group_col") else None
    test_ids = test[id_col].values if id_col else np.arange(len(test))
    return train[feature_cols], train[target].values, groups, test[feature_cols], test_ids


def load_vision(cfg, demo: bool = True):
    """비전: 이미지 경로+라벨 DataFrame 반환. 실제 학습은 Dataset에서 처리."""
    if demo:
        return pd.DataFrame({"filepath": [], "label": []})
    df = pd.read_csv(cfg["paths"]["train"])
    return df


# ── 음향 ────────────────────────────────────────────────────────────────────

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

    def __init__(self, df, audio_dir, cfg, is_test=False, demo=False):
        self.df        = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.is_test   = is_test
        self.demo      = demo

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
        if self.demo:
            time_frames = int(self.sr * self.duration) // self.hop_length + 1
            spec = np.random.rand(self.n_mels, time_frames).astype(np.float32)
        else:
            path = os.path.join(self.audio_dir, row["filename"])
            spec = wav_to_melspec(
                path, sr=self.sr, duration=self.duration,
                n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length,
                f_min=self.f_min, f_max=self.f_max,
            )
        spec_tensor = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, time)

        # ── AIS 피처 ─────────────────────────────────────────────────────────
        ais_tensor = torch.from_numpy(self._encode_ais(row))  # (5,)

        if self.is_test:
            return spec_tensor, ais_tensor

        label = SHIP_TYPE_TO_IDX[row["ship_type"]]
        return spec_tensor, ais_tensor, torch.tensor(label, dtype=torch.long)

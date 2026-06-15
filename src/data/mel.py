"""WAV → Log-mel spectrogram 변환 및 .npy 캐시 빌드.

사전 변환 (학습 전 1회 실행):
    python -m src.data.mel --config configs/audio_task1.yaml
"""
import os
import numpy as np


def wav_to_melspec(filepath, sr=32000, duration=5.0,
                   n_mels=128, n_fft=2048, hop_length=512,
                   f_min=50, f_max=16000, top_db=80.0):
    """WAV 파일 → 정규화된 Log-mel spectrogram.

    Returns
    -------
    log_mel : np.ndarray, shape (n_mels, time_frames), dtype float32
              값 범위 [0, 1]
    """
    import librosa
    y, _ = librosa.load(filepath, sr=sr, duration=duration, mono=True)

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
    log_mel = (log_mel + top_db) / top_db
    return log_mel.astype(np.float32)


def build_spectrogram_cache(audio_dir, df, cfg, cache_dir, verbose=True):
    """audio_dir 안의 WAV를 cache_dir에 .npy로 저장. 이미 있으면 건너뜀."""
    os.makedirs(cache_dir, exist_ok=True)

    mel_kwargs = dict(
        sr=cfg.get("sample_rate", 32000),
        duration=cfg.get("duration", 5.0),
        n_mels=cfg.get("n_mels", 128),
        n_fft=cfg.get("n_fft", 2048),
        hop_length=cfg.get("hop_length", 512),
        f_min=cfg.get("f_min", 50),
        f_max=cfg.get("f_max", 16000),
    )

    filenames = df["filename"].tolist()
    missing = [
        f for f in filenames
        if not os.path.exists(os.path.join(cache_dir, f.replace(".wav", ".npy")))
    ]
    if not missing:
        if verbose:
            print(f"[cache] 이미 완료 ({len(filenames)}개) → {cache_dir}")
        return

    from tqdm import tqdm
    for fname in tqdm(missing, desc="mel cache", unit="file", dynamic_ncols=True):
        src = os.path.join(audio_dir, fname)
        dst = os.path.join(cache_dir, fname.replace(".wav", ".npy"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            np.save(dst, wav_to_melspec(src, **mel_kwargs))
        except Exception as e:
            import warnings
            tqdm.write(f"[경고] 변환 실패 {fname}: {e}")

    if verbose:
        print(f"[cache] 완료 → {cache_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_all(cfg):
    import pandas as pd
    cache_dir = cfg.get("cache_dir", "data/spec_cache")
    paths = cfg["paths"]

    splits = []
    # Task 1
    if "train" in paths and "train_audio" in paths:
        splits.append((paths["train_audio"], pd.read_csv(paths["train"])))
    if "task1_val" in paths and "task1_audio" in paths:
        splits.append((paths["task1_audio"], pd.read_csv(paths["task1_val"])))
    if "task1_test" in paths and "task1_audio" in paths:
        splits.append((paths["task1_audio"], pd.read_csv(paths["task1_test"])))
    for audio_dir, df in splits:
        build_spectrogram_cache(audio_dir, df, cfg, cache_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="WAV → mel .npy 사전 변환")
    ap.add_argument("--config", required=True, help="yaml config 경로")
    args = ap.parse_args()
    _build_all(_load_config(args.config))

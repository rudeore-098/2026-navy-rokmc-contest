"""WAV → Log-mel spectrogram 변환 및 캐시 빌드.

사전 변환 (학습 전 1회 실행):
    python -m src.data.mel --config configs/audio_task1.yaml

순서:
    1. 개별 .npy 변환 (build_spectrogram_cache)
    2. 하나의 packed_all.npy로 통합 (build_packed)
       → AudioDataset이 mmap으로 O(1) 슬라이스 접근
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


def wav_to_cqt(filepath, sr=32000, duration=5.0,
               n_bins=128, bins_per_octave=16, hop_length=512,
               f_min=50, top_db=80.0):
    """WAV 파일 → 정규화된 CQT spectrogram.

    n_bins=128, bins_per_octave=16 → fmax ≈ 50 × 2^8 = 12800 Hz
    hop_length=512 → mel과 동일한 시간 프레임 수 (313)

    Returns
    -------
    cqt : np.ndarray, shape (n_bins, time_frames), dtype float32, 값 범위 [0, 1]
    """
    import librosa
    y, _ = librosa.load(filepath, sr=sr, duration=duration, mono=True)

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop_length,
        fmin=f_min, n_bins=n_bins, bins_per_octave=bins_per_octave,
    ))
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max, top_db=top_db)
    log_cqt = (log_cqt + top_db) / top_db
    return log_cqt.astype(np.float32)


def build_cqt_cache(audio_dir, df, cfg, cache_dir, verbose=True):
    """CQT를 cache_dir에 {stem}_cqt.npy로 저장. 이미 있으면 건너뜀."""
    os.makedirs(cache_dir, exist_ok=True)

    cqt_kwargs = dict(
        sr=cfg.get("sample_rate", 32000),
        duration=cfg.get("duration", 5.0),
        n_bins=cfg.get("n_mels", 128),
        bins_per_octave=cfg.get("cqt_bins_per_octave", 16),
        hop_length=cfg.get("hop_length", 512),
        f_min=cfg.get("f_min", 50),
    )

    filenames = df["filename"].tolist()
    missing = [
        f for f in filenames
        if not os.path.exists(
            os.path.join(cache_dir, f.replace(".wav", "_cqt.npy"))
        )
    ]
    if not missing:
        if verbose:
            print(f"[cqt cache] 이미 완료 ({len(filenames)}개) → {cache_dir}")
        return

    from tqdm import tqdm
    for fname in tqdm(missing, desc="cqt cache", unit="file", dynamic_ncols=True):
        src = os.path.join(audio_dir, fname)
        dst = os.path.join(cache_dir, fname.replace(".wav", "_cqt.npy"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            np.save(dst, wav_to_cqt(src, **cqt_kwargs))
        except Exception as e:
            tqdm.write(f"[경고] CQT 변환 실패 {fname}: {e}")

    if verbose:
        print(f"[cqt cache] 완료 → {cache_dir}")


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
            tqdm.write(f"[경고] 변환 실패 {fname}: {e}")

    if verbose:
        print(f"[cache] 완료 → {cache_dir}")


def build_packed(cache_dir, all_filenames, verbose=True):
    """개별 .npy → packed_all.npy + packed_all_index.json.

    AudioDataset이 np.load(mmap_mode='r')로 파일 오픈 오버헤드 없이
    arr[idx] 슬라이스만으로 스펙트로그램을 가져올 수 있게 한다.

    Parameters
    ----------
    all_filenames : list[str]
        train + val + test 전체 파일명 (중복 자동 제거)
    """
    import json

    packed_path = os.path.join(cache_dir, "packed_all.npy")
    index_path  = os.path.join(cache_dir, "packed_all_index.json")

    if os.path.exists(packed_path) and os.path.exists(index_path):
        if verbose:
            print(f"[pack] 이미 완료 → {packed_path}")
        return

    seen = set()
    unique = [f for f in all_filenames if not (f in seen or seen.add(f))]

    from tqdm import tqdm
    specs, fname_to_idx = [], {}
    for fname in tqdm(unique, desc="packing", unit="file", dynamic_ncols=True):
        npy = os.path.join(cache_dir, fname.replace(".wav", ".npy"))
        if os.path.exists(npy):
            fname_to_idx[fname] = len(specs)
            specs.append(np.load(npy))
        else:
            tqdm.write(f"[경고] 없음(건너뜀): {npy}")

    np.save(packed_path, np.stack(specs))
    with open(index_path, "w") as f:
        json.dump(fname_to_idx, f)

    if verbose:
        size_gb = os.path.getsize(packed_path) / 1e9
        print(f"[pack] 완료: {len(specs)}개, {size_gb:.2f} GB → {packed_path}")


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
    if "train" in paths and "train_audio" in paths:
        splits.append((paths["train_audio"], pd.read_csv(paths["train"])))
    if "task1_val" in paths and "task1_audio" in paths:
        splits.append((paths["task1_audio"], pd.read_csv(paths["task1_val"])))
    if "task1_test" in paths and "task1_audio" in paths:
        splits.append((paths["task1_audio"], pd.read_csv(paths["task1_test"])))

    all_filenames = []
    for audio_dir, df in splits:
        build_spectrogram_cache(audio_dir, df, cfg, cache_dir)
        if cfg.get("use_cqt", False):
            build_cqt_cache(audio_dir, df, cfg, cache_dir)
        all_filenames.extend(df["filename"].tolist())

    build_packed(cache_dir, all_filenames)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="WAV → mel .npy 사전 변환 + 패킹")
    ap.add_argument("--config", required=True, help="yaml config 경로")
    args = ap.parse_args()
    _build_all(_load_config(args.config))

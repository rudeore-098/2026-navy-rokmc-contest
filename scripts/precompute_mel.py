"""mel npy 사전생성 + 팀 공유용 패키징.

src/models/audio.py 의 _DS._mel (use_npy=False 분기)와 '동일한' 전처리로
모든 클립의 log-mel 을 .npy 로 저장하고, 한 명이 만든 결과를 팀이 그대로
쓸 수 있도록 공유 패키지(tar + config + manifest + 체크섬)를 함께 만든다.

⚠️ config 의 mel 파라미터(n_mels/n_fft/hop_length/f_min/f_max/hp_cutoff/sample_rate)와
   audio.py 의 Dataset 이 정확히 일치해야 함 (둘 다 같은 config 를 읽는다).
저장 규격: (n_mels, T),  파일명 = <원본>.wav → <원본>.npy

사용:
    # 변환만
    python scripts/precompute_mel.py --config configs/audio_task1.yaml
    # fp16(용량 절반) + 끝나면 공유용 tar 까지 생성
    python scripts/precompute_mel.py --config configs/audio_task1.yaml --fp16 --pack
    # 변환 없이 기존 npy 폴더만 패키징
    python scripts/precompute_mel.py --config configs/audio_task1.yaml --pack_only

동봉물 (mel_dir 안):
    _mel_config.yaml   변환에 쓴 config 스냅샷 (학습 일관성 검증용)
    _manifest.json     파일 수/shape/dtype/파라미터/개별 해시 요약
공유 패키지 (mel_dir 옆):
    mel_npy.tar.gz     위 npy + 동봉물 전체
    mel_npy.tar.gz.md5 무결성 체크섬
"""
import os, sys, json, glob, time, hashlib, argparse, tarfile, shutil
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 학습 코드(audio.py)와 동일 전처리 검증을 위해 mel 파라미터 키 (audio_task1.yaml 규약)
MEL_KEYS = ["sample_rate", "duration", "hp_cutoff", "n_mels", "n_fft", "hop_length", "f_min", "f_max"]


def mel_transform(cfg):
    import torchaudio
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.get("sample_rate", 32000),
        n_fft=cfg.get("n_fft", 2048), hop_length=cfg.get("hop_length", 512),
        n_mels=cfg.get("n_mels", 128), f_min=cfg.get("f_min", 50), f_max=cfg.get("f_max", 16000))
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    return melspec, to_db


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _splits(cfg):
    """(csv_path, audio_dir) 목록. 존재하는 split 만 사용."""
    p = cfg["paths"]
    out = [(p["train"], p["train_audio"])]
    for csv_key, audio_key in [("task1_val", "task1_audio"), ("task1_test", "task1_audio")]:
        if csv_key in p:
            out.append((p[csv_key], p[audio_key]))
    return out


def convert_all(cfg, out_dir, fp16):
    import torchaudio
    import torch.nn.functional as F

    melspec, to_db = mel_transform(cfg)
    sr = cfg.get("sample_rate", 32000)
    target = int(sr * cfg.get("duration", 5.0))
    hp_cut = cfg.get("hp_cutoff", 20.0)
    dtype = np.float16 if fp16 else np.float32

    seen, n_ok, n_skip, n_err = set(), 0, 0, 0
    for csv_path, audio_dir in _splits(cfg):
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} 없음"); continue
        df = pd.read_csv(csv_path)
        for fn in df["filename"]:
            if fn in seen: continue
            seen.add(fn)
            out = os.path.join(out_dir, fn.replace(".wav", ".npy"))
            if os.path.exists(out): n_skip += 1; continue
            try:
                wav, _sr = torchaudio.load(os.path.join(audio_dir, fn))
                if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
                if _sr != sr:
                    wav = torchaudio.functional.resample(wav, _sr, sr)
                wav = torchaudio.functional.highpass_biquad(wav, sr, hp_cut)
                wav = F.pad(wav, (0, target - wav.shape[1])) if wav.shape[1] < target else wav[:, :target]
                mel = to_db(melspec(wav))
                mel = (mel - mel.mean()) / (mel.std() + 1e-6)
                np.save(out, mel.squeeze(0).numpy().astype(dtype))      # (n_mels, T)
                n_ok += 1
                if n_ok % 2000 == 0: print(f"  ... {n_ok} done")
            except Exception as e:
                n_err += 1
                if n_err <= 10: print(f"  [ERR] {fn}: {e}")
    return n_ok, n_skip, n_err


def write_embedded(cfg, config_path, out_dir):
    """config 스냅샷 + manifest 동봉. 학습 시 일관성 검증에 사용."""
    # 1) config 스냅샷 (변환에 쓴 그대로)
    snap = os.path.join(out_dir, "_mel_config.yaml")
    shutil.copyfile(config_path, snap)

    # 2) manifest: 파일 목록/shape/dtype/파라미터 + 폴더 단위 요약 해시
    npys = sorted(glob.glob(os.path.join(out_dir, "*.npy")))
    sample = np.load(npys[0]) if npys else None
    # 폴더 요약 해시: 파일명 정렬 후 (이름+크기) 누적 → 빠른 일치 확인용
    h = hashlib.md5()
    files_meta = []
    for p in npys:
        size = os.path.getsize(p)
        h.update(os.path.basename(p).encode()); h.update(str(size).encode())
        files_meta.append({"name": os.path.basename(p), "bytes": size})
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_files": len(npys),
        "mel_params": {k: cfg[k] for k in MEL_KEYS if k in cfg},
        "model_input": {  # 학습 측이 기대하는 형식
            "shape": list(sample.shape) if sample is not None else None,
            "dtype": str(sample.dtype) if sample is not None else None,
            "axis_order": "(n_mels, T)",
        },
        "folder_summary_md5": h.hexdigest(),
        "files": files_meta,
    }
    with open(os.path.join(out_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def pack(out_dir):
    """mel_dir 전체(npy + 동봉물)를 tar.gz + md5 로 묶어 팀 공유 파일 생성."""
    parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    base = os.path.basename(os.path.normpath(out_dir))
    tar_path = os.path.join(parent, base + ".tar.gz")
    print(f"\n[pack] 압축 중 → {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=base)
    digest = md5_of(tar_path)
    with open(tar_path + ".md5", "w") as f:
        f.write(f"{digest}  {os.path.basename(tar_path)}\n")
    size_gb = os.path.getsize(tar_path) / 1e9
    print(f"[pack] 완료: {tar_path}  ({size_gb:.2f} GB)")
    print(f"[pack] md5 : {digest}")
    print("\n팀원 받는 법:")
    print(f"  # (구글드라이브 공유 시) pip install gdown && gdown <링크> -O {base}.tar.gz")
    print(f"  md5sum -c {base}.tar.gz.md5          # 무결성 확인")
    print(f"  tar -xzf {base}.tar.gz -C data/      # → data/{base}/")
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/audio_task1.yaml")
    ap.add_argument("--fp16", action="store_true", help="mel 을 float16 로 저장(용량 절반)")
    ap.add_argument("--pack", action="store_true", help="변환 후 공유용 tar.gz + md5 생성")
    ap.add_argument("--pack_only", action="store_true", help="변환 생략, 기존 npy 폴더만 패키징")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = cfg.get("mel_dir", "data/mel_npy"); os.makedirs(out_dir, exist_ok=True)

    if not args.pack_only:
        print(f"변환 시작 → {out_dir}  (fp16={args.fp16})")
        n_ok, n_skip, n_err = convert_all(cfg, out_dir, args.fp16)
        print(f"\n변환 완료: ok={n_ok}, 이미존재={n_skip}, 오류={n_err}")

    # 동봉물 작성 (항상)
    manifest = write_embedded(cfg, args.config, out_dir)
    mp = manifest["model_input"]
    print(f"\n[manifest] 파일 {manifest['n_files']}개 | shape={mp['shape']} {mp['dtype']} "
          f"({mp['axis_order']}) | 요약md5={manifest['folder_summary_md5'][:12]}…")
    print(f"[manifest] mel_params={manifest['mel_params']}")
    print(f"동봉: {out_dir}/_mel_config.yaml , {out_dir}/_manifest.json")

    if args.pack or args.pack_only:
        pack(out_dir)


if __name__ == "__main__":
    main()

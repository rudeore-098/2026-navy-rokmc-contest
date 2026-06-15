"""AST fbank precompute — 한 번만 굽고, 학습은 npy만 읽음."""
import os, argparse, yaml
import numpy as np, pandas as pd, torch, torchaudio
from transformers import ASTFeatureExtractor
from tqdm import tqdm

SPLITS = {
    "train":      ("train",      "train_audio"),
    "task1_val":  ("task1_val",  "task1_audio"),
    "task1_test": ("task1_test", "task1_audio"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--out_dir", default="data/ast_npy")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ast_id = cfg.get("ast_model_id", "MIT/ast-finetuned-audioset-10-10-0.4593")
    fe = ASTFeatureExtractor.from_pretrained(ast_id)
    os.makedirs(args.out_dir, exist_ok=True)

    splits = list(SPLITS) if args.split == "all" else [args.split]
    rs = {}
    for split in splits:
        ck, ak = SPLITS[split]
        csv_path, audio_dir = cfg["paths"].get(ck), cfg["paths"].get(ak)
        if not csv_path or not os.path.exists(csv_path):
            print(f"[{split}] skip — {csv_path} 없음"); continue
        df = pd.read_csv(csv_path)
        print(f"[{split}] {len(df)} clips → {args.out_dir}")
        for fn in tqdm(df["filename"].tolist(), desc=split):
            out = os.path.join(args.out_dir, fn.replace(".wav", ".npy"))
            if os.path.exists(out): continue
            wav, sr = torchaudio.load(os.path.join(audio_dir, fn))
            wav = wav.mean(0)
            if sr != 16000:
                if sr not in rs: rs[sr] = torchaudio.transforms.Resample(sr, 16000)
                wav = rs[sr](wav)
            x = fe(wav.numpy(), sampling_rate=16000, return_tensors="np")["input_values"][0]
            np.save(out, x.astype(np.float16))

if __name__ == "__main__":
    main()

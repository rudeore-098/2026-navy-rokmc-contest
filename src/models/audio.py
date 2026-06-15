"""음향 모델 — log-mel(CNN 백본) + AIS(MLP) 융합 분류기.

vision.py 가 timm 백본 래퍼인 것과 같은 결로, 음향 전용 융합 모델을 제공한다.
스펙트로그램을 1채널 이미지로 보고 timm CNN 에 넣되, AIS 정형 피처를 함께 융합한다.

핵심 설계 (대회 EDA 반영):
- SOG=102.3 = AIS 결측 아티팩트 → sog_invalid 플래그 + 값 제거 (음향은 정상)
- heading==0 다수가 미보고 → head_missing 플래그 / heading>360 클리핑
- 80% 정지 클립 → is_stop 플래그로 'AIS 신뢰도 낮음' 을 모델에 명시
- mel npy 사전생성(scripts/precompute_mel.py) 우선 로드, 없으면 WAV fallback

config 키는 기존 configs/audio_task1.yaml 규약을 따른다
(sample_rate / duration / n_fft / hop_length / f_min / f_max).
GPU(torch/timm/torchaudio) 환경에서 동작.
"""
import os
import random
import numpy as np


# ----------------------------------------------------------------------
# AIS feature engineering  (정형 피처 8-dim)
# ----------------------------------------------------------------------
def build_ais_features(df):
    sog  = df["sog"].fillna(0).values.astype(np.float32)
    cog  = df["cog"].fillna(0).values.astype(np.float32)
    head = df["true_heading"].fillna(0).values.astype(np.float32)

    sog_invalid = (sog >= 100).astype(np.float32)   # 102.3 = AIS '데이터없음'
    sog = np.where(sog >= 100, 0.0, sog)
    head = np.clip(head, 0.0, 360.0)                 # heading=482 등 클리핑

    is_stop      = (sog < 0.5).astype(np.float32)
    head_missing = (head == 0).astype(np.float32)
    cog_r, head_r = np.deg2rad(cog), np.deg2rad(head)

    return np.stack([
        np.log1p(sog),
        np.cos(cog_r), np.sin(cog_r),
        np.cos(head_r), np.sin(head_r),
        is_stop, head_missing, sog_invalid,
    ], axis=1).astype(np.float32)


AIS_DIM = 8

AST_DEFAULT_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _is_ast(cfg_or_name):
    name = cfg_or_name.get("model_name", "") if isinstance(cfg_or_name, dict) else cfg_or_name
    return "ast" in str(name).lower()


# ----------------------------------------------------------------------
# AST dataset  — WAV → 16kHz → ASTFeatureExtractor (1024,128). use_npy 무시.
# ----------------------------------------------------------------------
def _make_ast_dataset(df, ais_feats, cfg, labels, train):
    import random
    import torch
    from torch.utils.data import Dataset
    import torchaudio
    from transformers import ASTFeatureExtractor

    audio_dir = cfg["_audio_dir"]
    ast_id = cfg.get("ast_model_id", AST_DEFAULT_ID)
    ast_sr = 16000
    fe = ASTFeatureExtractor.from_pretrained(ast_id)   # 16kHz/128mel/1024, AudioSet 정규화

    class _ASTDS(Dataset):
        def __init__(self):
            self.df = df.reset_index(drop=True)
            self.ais = ais_feats
            self.labels = labels
            self.train = train
            self._rs = {}

        def __len__(self):
            return len(self.df)

        def _feat(self, fn):
            npy_path = os.path.join("data/ast_npy", fn.replace(".wav", ".npy"))
            if os.path.exists(npy_path):
                return torch.from_numpy(np.load(npy_path)).float()
            wav, sr = torchaudio.load(os.path.join(audio_dir, fn))
            wav = wav.mean(0)
            if sr != ast_sr:
                if sr not in self._rs:
                    self._rs[sr] = torchaudio.transforms.Resample(sr, ast_sr)
                wav = self._rs[sr](wav)
            out = fe(wav.numpy(), sampling_rate=ast_sr, return_tensors="pt")
            return out["input_values"][0]

        def __getitem__(self, i):
            x = self._feat(self.df.iloc[i]["filename"])
            if self.train and random.random() < 0.5:       # 가벼운 time mask
                t = random.randint(0, max(0, x.shape[0] - 80)); x[t:t + 80, :] = x.mean()
            ais = torch.from_numpy(self.ais[i])
            if self.labels is not None:
                return x, ais, torch.tensor(self.labels[i], dtype=torch.long)
            return x, ais

    return _ASTDS()


# ----------------------------------------------------------------------
# AST model  — ASTModel 인코더(pooler) + AIS MLP 융합. forward 시그니처는 CNN과 동일.
# ----------------------------------------------------------------------
def _create_ast_model(model_name, num_classes, pretrained,
                      ais_dim=AIS_DIM, ais_hidden=64, proj=128):
    import torch
    import torch.nn as nn
    from transformers import ASTModel, ASTConfig

    ast_id = model_name if "/" in str(model_name) else AST_DEFAULT_ID

    class ASTShipNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.ast = ASTModel.from_pretrained(ast_id) if pretrained else ASTModel(ASTConfig())
            for p in self.ast.embeddings.parameters(): p.requires_grad = False
            for p in self.ast.encoder.layer[:8].parameters(): p.requires_grad = False
            feat = self.ast.config.hidden_size            # 768
            self.audio_proj = nn.Sequential(nn.Linear(feat, proj), nn.ReLU())
            self.ais_mlp = nn.Sequential(
                nn.Linear(ais_dim, ais_hidden), nn.ReLU(),
                nn.Linear(ais_hidden, ais_hidden), nn.ReLU())
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(proj + ais_hidden, 128), nn.ReLU(),
                nn.Linear(128, num_classes))

        def forward(self, x, ais):                         # x: (B,1024,128)
            pooled = self.ast(input_values=x).pooler_output    # (B, 768)
            a = self.audio_proj(pooled)
            m = self.ais_mlp(ais)
            return self.head(torch.cat([a, m], dim=1))

    return ASTShipNet()


# ----------------------------------------------------------------------
# Dataset  (mel npy 우선, WAV fallback)
# ----------------------------------------------------------------------
def make_dataset(df, ais_feats, cfg, labels=None, train=False):
    """torch Dataset 생성 (지연 import — torch 없는 환경 보호).

    config 키는 audio_task1.yaml 규약(sample_rate/duration/hop_length/...) 을 따른다.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    import torchaudio

    if _is_ast(cfg):                                # AST: 전용 16kHz fbank 경로
        return _make_ast_dataset(df, ais_feats, cfg, labels, train)

    sr        = cfg.get("sample_rate", 32000)
    duration  = cfg.get("duration", 5.0)
    hp_cut    = cfg.get("hp_cutoff", 20.0)
    n_mels    = cfg.get("n_mels", 128)
    n_fft     = cfg.get("n_fft", 2048)
    hop       = cfg.get("hop_length", 512)
    fmin      = cfg.get("f_min", 50)
    fmax      = cfg.get("f_max", 16000)
    use_npy   = cfg.get("use_npy", False)
    mel_dir   = cfg.get("mel_dir", "data/mel_npy")
    audio_dir = cfg["_audio_dir"]               # train.py 가 split 별로 주입
    target    = int(sr * duration)

    class _DS(Dataset):
        def __init__(self):
            self.df = df.reset_index(drop=True)
            self.ais = ais_feats
            self.labels = labels
            self.train = train
            if not use_npy:
                self.melspec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr, n_fft=n_fft, hop_length=hop,
                    n_mels=n_mels, f_min=fmin, f_max=fmax)
                self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        def __len__(self): return len(self.df)

        def _mel(self, fn):
            if use_npy:
                arr = np.load(os.path.join(mel_dir, fn.replace(".wav", ".npy")))
                mel = torch.from_numpy(arr).float()
                if mel.dim() == 2: mel = mel.unsqueeze(0)     # (1, n_mels, T)
                return mel
            wav, _sr = torchaudio.load(os.path.join(audio_dir, fn))
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            if _sr != sr: wav = torchaudio.functional.resample(wav, _sr, sr)
            wav = torchaudio.functional.highpass_biquad(wav, sr, hp_cut)
            wav = F.pad(wav, (0, target - wav.shape[1])) if wav.shape[1] < target else wav[:, :target]
            mel = self.to_db(self.melspec(wav))
            return (mel - mel.mean()) / (mel.std() + 1e-6)

        def __getitem__(self, i):
            mel = self._mel(self.df.iloc[i]["filename"])
            if self.train:                                    # SpecAugment
                if random.random() < 0.5:
                    t = random.randint(0, max(0, mel.shape[-1]-48)); mel[..., t:t+48] = mel.min()
                if random.random() < 0.5:
                    fch = random.randint(0, max(0, mel.shape[-2]-24)); mel[..., fch:fch+24, :] = mel.min()
            ais = torch.from_numpy(self.ais[i])
            if self.labels is not None:
                return mel, ais, torch.tensor(self.labels[i], dtype=torch.long)
            return mel, ais

    return _DS()


# ----------------------------------------------------------------------
# Model  (timm CNN 백본 + AIS MLP 융합)
# ----------------------------------------------------------------------
def create_audio_model(model_name="tf_efficientnet_b0_ns", num_classes=4,
                       pretrained=True, ais_dim=AIS_DIM, ais_hidden=64, proj=128):
    import torch
    import torch.nn as nn
    import timm

    if _is_ast(model_name):                         # AST: ASTModel + AIS 융합
        return _create_ast_model(model_name, num_classes, pretrained, ais_dim=ais_dim)

    class ShipNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, in_chans=1,
                num_classes=0, global_pool="avg")
            feat = self.backbone.num_features
            self.audio_proj = nn.Sequential(nn.Linear(feat, proj), nn.ReLU())
            self.ais_mlp = nn.Sequential(
                nn.Linear(ais_dim, ais_hidden), nn.ReLU(),
                nn.Linear(ais_hidden, ais_hidden), nn.ReLU())
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(proj + ais_hidden, 128), nn.ReLU(),
                nn.Linear(128, num_classes))

        def forward(self, mel, ais):
            a = self.audio_proj(self.backbone(mel))
            m = self.ais_mlp(ais)
            return self.head(torch.cat([a, m], dim=1))

    return ShipNet()


def class_weights(labels, n=4):
    import torch
    cnt = np.bincount(labels, minlength=n).astype(np.float32)
    w = cnt.sum() / (n * np.maximum(cnt, 1))
    return torch.tensor(w, dtype=torch.float32)


RECOMMENDED_BACKBONES = [
    "tf_efficientnet_b0_ns",   # 가벼운 baseline
    "eca_nfnet_l0",            # 음향 대회 단골
    "convnext_small",          # 강력
]
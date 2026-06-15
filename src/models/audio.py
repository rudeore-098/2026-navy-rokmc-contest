"""음향 모델 — AST(Audio Spectrogram Transformer) + AIS(MLP) 융합 분류기.

Task 1 전용. AST 백본만 사용. AudioSet pretrained.

핵심 설계 (대회 EDA 반영):
- SOG=102.3 = AIS 결측 아티팩트 → sog_invalid 플래그 + 값 제거 (음향은 정상)
- heading==0 다수가 미보고 → head_missing 플래그 / heading>360 클리핑
- 80% 정지 클립 → is_stop 플래그로 'AIS 신뢰도 낮음' 을 모델에 명시
- AST fbank 사전생성(scripts/precompute_ast.py) 우선 로드, 없으면 WAV fallback

config 키:
    model_name : AST model id (또는 AST_DEFAULT_ID)
    _audio_dir : train.py 가 split 별로 주입 (train_audio / task1_audio)
    ast_model_id : (optional) feature extractor id

GPU(torch/torchaudio/transformers) 환경에서 동작.

[변경 사항 - 2026-06-15]
- AIS feature: 8-d → 14-d (drift angle, hour, sog bins 추가)
- class_weights: inverse-freq → effective number (β=0.9999, Cui et al. 2019)
  · 기존 inverse-freq가 C_Passenger를 과예측하게 만들어 Macro F1 32%에서 정체
- AIS MLP 강화: hidden 64→128, BatchNorm + Dropout 추가, 3-layer
- Fusion head 강화: 2-layer → 3-layer + BatchNorm
- CNN(timm) 경로 제거 — AST 전용으로 단순화
"""
import os
import random
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# AIS feature engineering  (정형 피처 14-dim)
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

    # drift angle = cog - heading (조류·바람 영향)
    drift = ((cog - head + 180.0) % 360.0) - 180.0
    drift_r = np.deg2rad(drift)

    # SOG 구간 (작업/순항 구분 — A는 저속, D는 중속, B는 광범위)
    sog_low  = ((sog >= 0.5) & (sog < 5)).astype(np.float32)
    sog_mid  = ((sog >= 5) & (sog < 12)).astype(np.float32)
    sog_high = (sog >= 12).astype(np.float32)

    # hour-of-day (여객선 정기 항로 패턴)
    ts = pd.to_datetime(df["ais_timestamp"], errors="coerce")
    hour = ts.dt.hour.fillna(0).values.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)

    feats = np.stack([
        np.log1p(sog),                          # 0
        np.cos(cog_r), np.sin(cog_r),           # 1, 2
        np.cos(head_r), np.sin(head_r),         # 3, 4
        is_stop, head_missing, sog_invalid,     # 5, 6, 7
        np.cos(drift_r), np.sin(drift_r),       # 8, 9
        sog_low, sog_mid, sog_high,             # 10, 11, 12
        hour_sin,                                # 13
    ], axis=1).astype(np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


AIS_DIM = 14
AST_DEFAULT_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


# ----------------------------------------------------------------------
# Dataset — AST fbank npy 우선, 없으면 WAV → ASTFeatureExtractor fallback
# ----------------------------------------------------------------------
def make_dataset(df, ais_feats, cfg, labels=None, train=False):
    import torch
    from torch.utils.data import Dataset

    audio_dir = cfg["_audio_dir"]
    npy_dir = cfg.get("ast_npy_dir", "data/ast_npy")
    ast_sr = 16000

    # FeatureExtractor는 lazy 로드 — npy로 다 커버되면 영원히 안 불림
    _fe_cache = {}
    def _get_fe():
        if "fe" not in _fe_cache:
            from transformers import ASTFeatureExtractor
            ast_id = cfg.get("ast_model_id", AST_DEFAULT_ID)
            _fe_cache["fe"] = ASTFeatureExtractor.from_pretrained(ast_id)
        return _fe_cache["fe"]

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
            npy_path = os.path.join(npy_dir, fn.replace(".wav", ".npy"))
            if os.path.exists(npy_path):
                # fp16 저장본 → fp32 텐서
                return torch.from_numpy(np.load(npy_path)).float()
            # fallback
            import torchaudio
            wav, sr = torchaudio.load(os.path.join(audio_dir, fn))
            wav = wav.mean(0)
            if sr != ast_sr:
                if sr not in self._rs:
                    self._rs[sr] = torchaudio.transforms.Resample(sr, ast_sr)
                wav = self._rs[sr](wav)
            out = _get_fe()(wav.numpy(), sampling_rate=ast_sr, return_tensors="pt")
            return out["input_values"][0]

        def __getitem__(self, i):
            x = self._feat(self.df.iloc[i]["filename"])
            if self.train and random.random() < 0.5:
                t = random.randint(0, max(0, x.shape[0] - 80))
                x[t:t + 80, :] = x.mean()
            ais = torch.from_numpy(self.ais[i])
            if self.labels is not None:
                return x, ais, torch.tensor(self.labels[i], dtype=torch.long)
            return x, ais

    return _ASTDS()


# ----------------------------------------------------------------------
# Model — ASTModel(pooler) + AIS MLP 융합
# ----------------------------------------------------------------------
def create_audio_model(model_name=AST_DEFAULT_ID, num_classes=4,
                       pretrained=True, ais_dim=AIS_DIM, ais_hidden=128, proj=128):
    import torch
    import torch.nn as nn
    from transformers import ASTModel, ASTConfig

    ast_id = model_name if "/" in str(model_name) else AST_DEFAULT_ID

    class ASTShipNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.ast = ASTModel.from_pretrained(ast_id) if pretrained else ASTModel(ASTConfig())
            # AST 하위 layer freeze: embeddings + 첫 8개 encoder layer
            for p in self.ast.embeddings.parameters():
                p.requires_grad = False
            for p in self.ast.encoder.layer[:8].parameters():
                p.requires_grad = False

            feat = self.ast.config.hidden_size            # 768

            self.audio_proj = nn.Sequential(
                nn.Linear(feat, proj),
                nn.ReLU(),
            )

            # AIS MLP: BN + Dropout 3-layer (강화)
            self.ais_mlp = nn.Sequential(
                nn.Linear(ais_dim, ais_hidden),
                nn.BatchNorm1d(ais_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(ais_hidden, ais_hidden),
                nn.BatchNorm1d(ais_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(ais_hidden, ais_hidden),
                nn.ReLU(),
            )

            # Fusion head: 3-layer + BN (강화)
            fused_dim = proj + ais_hidden
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(fused_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            )

        def forward(self, x, ais):                         # x: (B,1024,128)
            pooled = self.ast(input_values=x).pooler_output    # (B, 768)
            a = self.audio_proj(pooled)
            m = self.ais_mlp(ais)
            return self.head(torch.cat([a, m], dim=1))

    return ASTShipNet()


# ----------------------------------------------------------------------
# Class weights — effective number (Cui et al. 2019)
# ----------------------------------------------------------------------
def class_weights(labels, n=4, beta=0.9999):
    """Effective number weighting.

    기존 inverse-freq (count.sum() / (n * count)) 는 train 분포 (C가 10.7%) 에서
    C 에 5.1배 weight 를 줘서 'C 로 찍자' 부작용 발생 (val 32%, C 과예측).

    Effective number 는 'unique sample 수' 개념으로 부드럽게 보정:
        w_c ∝ (1-β) / (1-β^n_c)

    β 선택 가이드 (이 데이터셋 N≈35k 기준):
        β=0.99   → 모든 weight ≈ 1.0 (가중 없음과 동일)
        β=0.999  → 모든 weight ≈ 1.0 (35k 규모에선 너무 약함)
        β=0.9999 → C/A 비율 ~2.7x (기존 5.1x 의 절반, 권장 시작점)
        β=0.99999 → C/A 비율 ~4x (강한 가중 — 부작용 다시 나타날 수 있음)

    val 점수 보고 조정:
      - C 과예측 여전 → β 낮추기 (0.9999 → 0.999)
      - 소수 클래스 recall 너무 낮음 → β 높이기 (0.9999 → 0.99999)
    """
    import torch
    cnt = np.bincount(labels, minlength=n).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    effective_num = 1.0 - np.power(beta, cnt)
    w = (1.0 - beta) / effective_num
    w = w / w.sum() * n     # 평균 1로 정규화 (loss scale 유지)
    return torch.tensor(w, dtype=torch.float32)
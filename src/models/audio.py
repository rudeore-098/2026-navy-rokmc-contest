"""Audio models — timm backbone + AIS fusion for ship acoustic tasks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance (Garcia-Garcia et al. 2017).

    weight: per-class tensor for additional imbalance correction.
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


class AudioModel(nn.Module):
    """timm CNN backbone + optional AIS late-fusion + embedding head + classifier.

    Parameters
    ----------
    model_name  : timm model id (e.g. "eca_nfnet_l0")
    num_classes : 4 for Task 1 ship-type / 362 for Task 2 ship-id stage
    in_chans    : 1 for mono log-mel spectrogram
    ais_dim     : AIS feature dim; 0 disables the AIS branch (Task 2 inference)
    embed_dim   : latent dim returned when return_embedding=True
    drop        : dropout before the embedding FC
    """

    def __init__(self, model_name: str, num_classes: int,
                 in_chans: int = 1, pretrained: bool = True,
                 ais_dim: int = 5, embed_dim: int = 512, drop: float = 0.2):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, in_chans=in_chans, global_pool="avg",
        )
        feat_dim = self.backbone.num_features

        self.ais_proj = nn.Linear(ais_dim, 32) if ais_dim > 0 else None
        in_dim = feat_dim + (32 if ais_dim > 0 else 0)

        self.embed_fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, spec, ais=None, return_embedding: bool = False):
        x = self.backbone(spec)                                    # (B, feat_dim)
        if self.ais_proj is not None and ais is not None:
            x = torch.cat([x, F.relu(self.ais_proj(ais))], dim=1)
        emb = self.embed_fc(x)                                     # (B, embed_dim)
        if return_embedding:
            return F.normalize(emb, dim=1)
        return self.classifier(emb)                                # (B, num_classes)


class ASTWrapper(nn.Module):
    """HuggingFace ASTModel 래퍼 — AudioModel과 동일한 forward 인터페이스.

    AudioSet 사전학습 가중치를 사용하므로 CNN timm 모델과 앙상블 시 다양성 확보.

    Parameters
    ----------
    pretrained  : HuggingFace 모델 ID 또는 로컬 경로
    num_classes : 출력 클래스 수
    ais_dim     : 5-dim AIS 피처; 0이면 AIS 브랜치 비활성화
    embed_dim   : 임베딩 차원 (return_embedding=True 시 반환)
    """

    def __init__(self, pretrained: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 num_classes: int = 4, ais_dim: int = 5,
                 embed_dim: int = 512, drop: float = 0.2):
        super().__init__()
        from transformers import ASTModel
        self.ast = ASTModel.from_pretrained(pretrained)
        feat_dim = self.ast.config.hidden_size  # 768 (base model)

        self.ais_proj = nn.Linear(ais_dim, 32) if ais_dim > 0 else None
        in_dim = feat_dim + (32 if ais_dim > 0 else 0)

        self.embed_fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, spec, ais=None, return_embedding: bool = False):
        # spec: (B, 1, n_mels, T) → AST expects (B, T, n_mels)
        x_input = spec.squeeze(1).transpose(1, 2)                  # (B, 313, 128)
        x = self.ast(input_values=x_input).last_hidden_state[:, 0, :]  # CLS (B, 768)
        if self.ais_proj is not None and ais is not None:
            x = torch.cat([x, F.relu(self.ais_proj(ais))], dim=1)
        emb = self.embed_fc(x)
        if return_embedding:
            return F.normalize(emb, dim=1)
        return self.classifier(emb)


def create_audio_model(cfg: dict, num_classes: int) -> nn.Module:
    """model_name == 'ast' → ASTWrapper, 그 외 → AudioModel(timm)."""
    model_name = cfg["model_name"]
    ais_dim    = cfg.get("ais_dim", 5)
    embed_dim  = cfg.get("embed_dim", 512)

    if model_name == "ast":
        return ASTWrapper(
            pretrained=cfg.get("ast_pretrained",
                               "MIT/ast-finetuned-audioset-10-10-0.4593"),
            num_classes=num_classes,
            ais_dim=ais_dim,
            embed_dim=embed_dim,
        )
    return AudioModel(
        model_name=model_name,
        num_classes=num_classes,
        in_chans=cfg.get("in_chans", 1),
        pretrained=cfg.get("pretrained", True),
        ais_dim=ais_dim,
        embed_dim=embed_dim,
    )

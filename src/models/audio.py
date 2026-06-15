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

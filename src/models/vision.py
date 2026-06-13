"""비전 모델 — timm 백본 래퍼.

ImageNet 사전학습 백본을 불러와 분류기 머리만 교체한다.
스펙트로그램(음향)도 이미지로 보면 그대로 재사용 가능.
"""


def create_vision_model(model_name: str = "tf_efficientnet_b0_ns",
                        num_classes: int = 2, pretrained: bool = True,
                        in_chans: int = 3):
    """timm 백본 생성. torch/timm이 설치된 환경(GPU)에서 동작."""
    import timm
    model = timm.create_model(
        model_name, pretrained=pretrained,
        num_classes=num_classes, in_chans=in_chans,
    )
    return model


# 권장 백본 (앙상블 시 계열 섞기):
#   tf_efficientnet_b0_ns ~ b3   가볍고 빠른 baseline
#   convnext_small               최신 CNN, 강력
#   eca_nfnet_l0                 음향 대회 단골
#   swin_tiny_patch4_window7_224 트랜스포머 계열 (다양성)
RECOMMENDED_BACKBONES = [
    "tf_efficientnet_b0_ns",
    "convnext_small",
    "eca_nfnet_l0",
    "swin_tiny_patch4_window7_224",
]

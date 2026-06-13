"""사전학습 가중치 미리 받기 — 오프라인 대회 대비.

집(인터넷 되는 환경)에서 미리 실행해 weights/에 캐싱해두면,
당일 인터넷이 차단돼도 백본을 로컬에서 불러올 수 있다.

사용: python weights/download_weights.py
"""
import os

# 대회에서 쓸 백본 목록 (vision.py의 RECOMMENDED_BACKBONES와 맞춤)
BACKBONES = [
    "tf_efficientnet_b0_ns",
    "convnext_small",
    "eca_nfnet_l0",
    "swin_tiny_patch4_window7_224",
]

WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    try:
        import timm
        import torch
    except ImportError:
        print("timm/torch가 필요합니다. 먼저 설치하세요: pip install timm torch")
        return

    for name in BACKBONES:
        print(f"다운로드: {name} ...")
        try:
            model = timm.create_model(name, pretrained=True)
            path = os.path.join(WEIGHTS_DIR, f"{name}.pth")
            torch.save(model.state_dict(), path)
            print(f"  저장: {path}")
        except Exception as e:
            print(f"  실패: {name} ({e})")

    print("\n완료. 오프라인 대회에선 다음처럼 로컬 로드:")
    print("  model = timm.create_model(name, pretrained=False)")
    print("  model.load_state_dict(torch.load('weights/{name}.pth'))")

    # AST/BEATs 등 음향 특화 모델도 여기서 미리 받아두면 좋다 (Hugging Face)
    print("\n음향 특화 모델(AST 등)은 transformers로 미리 캐싱:")
    print("  from transformers import ASTForAudioClassification")
    print("  ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')")


if __name__ == "__main__":
    main()

# weights/ — 사전학습 가중치 캐싱

오프라인 대회(인터넷 차단) 대비용. **집에서 미리 채워두세요.**

## 사용법

인터넷 되는 환경에서:

```bash
python weights/download_weights.py
```

그러면 `weights/`에 백본 가중치(.pth)가 저장됩니다. (git에는 안 올라감)

## 오프라인 대회에서 로드

```python
import timm, torch
model = timm.create_model("tf_efficientnet_b0_ns", pretrained=False)
model.load_state_dict(torch.load("weights/tf_efficientnet_b0_ns.pth"))
```

## 음향 특화 모델

AST 등은 Hugging Face로 미리 캐싱:

```python
from transformers import ASTForAudioClassification
ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
# → ~/.cache/huggingface 에 저장됨. 이 캐시를 서버로 옮기거나 Kaggle Dataset으로 업로드.
```

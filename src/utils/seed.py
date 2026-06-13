"""시드 고정 — 재현성의 기본. 모든 실험 시작 시 호출."""
import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """파이썬·numpy·(가능하면)torch의 난수를 모두 고정한다.

    재현성은 실험 기록의 전제다. 같은 설정인데 점수가 흔들리면 비교가 무의미해진다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 완전한 재현을 원하면 아래 두 줄을 켠다 (속도는 약간 느려질 수 있음)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    except Exception:
        # torch 미설치 또는 환경 문제 시 조용히 넘어감 (정형 파이프라인은 torch 불필요)
        pass

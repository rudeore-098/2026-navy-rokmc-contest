#!/usr/bin/env bash
# 서버 접속 후 한 방 세팅 스크립트
# 사용: bash setup.sh
set -e

echo "=== [1/4] 가상환경 생성 ==="
if command -v conda &> /dev/null && [ -f environment.yml ]; then
    echo "conda 환경 생성 (environment.yml)"
    conda env create -f environment.yml || conda env update -f environment.yml
    echo "→ conda activate comp 로 활성화하세요"
else
    echo "venv 생성"
    python3 -m venv .venv
    source .venv/bin/activate
    echo "=== [2/4] 의존성 설치 (버전 고정) ==="
    pip install --upgrade pip
    if [ -f requirements.lock ]; then
        pip install -r requirements.lock
    else
        pip install -r requirements.txt
    fi
fi

echo "=== [3/4] 데이터 디렉토리 확인 ==="
mkdir -p data experiments weights
echo "데이터는 data/README.md 의 경로 규칙대로 마운트/심볼릭링크 하세요"

echo "=== [4/4] 사전학습 가중치 다운로드 (선택) ==="
if [ -f weights/download_weights.py ]; then
    echo "오프라인 대회 대비: python weights/download_weights.py 를 집에서 미리 실행하세요"
fi

echo ""
echo "✅ 세팅 완료. 빠른 동작 확인:"
echo "   python -m src.train --config configs/tabular.yaml --exp exp_demo --demo"

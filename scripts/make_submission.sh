#!/usr/bin/env bash
# 학습 → 추론 → 제출까지 한 번에
# 사용: bash scripts/make_submission.sh configs/tabular.yaml exp_001 [weights]
set -e

CONFIG=${1:-configs/tabular.yaml}
EXP=${2:-exp_001}
WEIGHTS=${3:-}

echo "=== 학습 ==="
python -m src.train --config "$CONFIG" --exp "$EXP"

echo "=== 추론 & 제출 ==="
if [ -n "$WEIGHTS" ]; then
    python -m src.infer --config "$CONFIG" --exp "$EXP" --weights "$WEIGHTS"
else
    python -m src.infer --config "$CONFIG" --exp "$EXP"
fi

echo "✅ 완료: experiments/$EXP/submission.csv"

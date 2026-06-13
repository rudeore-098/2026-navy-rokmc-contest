# experiments/{exp_id}/ — 실험별 산출물

각 실험은 자체 폴더에 결과를 저장합니다. (대용량 .npy는 .gitignore)

## 폴더 구성

```
exp_001/
├── config.yaml / config.json   # 이 실험의 설정 (재현용)
├── oof_lgb.npy                 # LightGBM OOF 예측
├── oof_xgb.npy                 # XGBoost OOF 예측
├── oof_cat.npy                 # CatBoost OOF 예측
├── oof.npy                     # ← 3종 평균 OOF (앙상블의 핵심 입력)
├── test_*.npy                  # 각 모델의 test 예측
├── y_true.npy                  # 정답 (앙상블 가중치 계산용)
├── test_ids.npy                # 제출용 id
└── submission.csv              # 최종 제출 파일
```

## oof.npy가 핵심인 이유

`oof.npy`는 **누수 없는 검증 예측**입니다. 여러 실험의 oof.npy를 모아
`src/ensemble.py`에 넘기면, test에 일반화되는 앙상블 가중치를 찾을 수 있습니다.

```bash
python -m src.ensemble --task binary --exps exp_001 exp_002 exp_003
```

## 노션 DB 연결

폴더명(`exp_001`)을 노션 실험 DB의 `실험ID`(`EXP-1`)와 맞추면,
기록과 산출물을 1:1로 추적할 수 있습니다.

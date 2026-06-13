# 🏆 comp-template — AI 경진대회 팀 템플릿

당일 주제 공개형 대회를 위한 **범용 파이프라인 템플릿**입니다.
정형·비전·시계열을 config 하나로 전환하고, OOF 기반 앙상블까지 한 흐름으로 처리합니다.

---

## ⚡ 빠른 시작 (3분)

```bash
# 1. 세팅 (서버 접속 후 한 방)
bash setup.sh

# 2. 동작 확인 (합성 데이터로 전체 파이프라인 검증)
python -m src.train --config configs/tabular.yaml --exp exp_demo --demo

# 3. 추론 & 제출
python -m src.infer --config configs/tabular.yaml --exp exp_demo
```

`exp_demo/submission.csv`가 생기면 파이프라인이 정상 작동하는 겁니다.

---

## 📁 디렉토리 구조

```
comp-template/
├── README.md              # 이 파일
├── requirements.lock      # 버전 완전 고정 (당일 충돌 방지)
├── environment.yml        # conda용
├── setup.sh               # 한 방 세팅 스크립트
├── configs/               # task별 설정 (여기만 바꾸면 됨)
│   ├── base.yaml          # 공통 (시드, 경로)
│   ├── tabular.yaml       # 부스팅 3종 파라미터
│   ├── vision.yaml        # timm 백본
│   └── timeseries.yaml
├── data/                  # .gitignore — 마운트 경로 심볼릭 링크
├── weights/               # 사전학습 가중치 캐싱 (대회 전 미리 채움)
│   └── download_weights.py
├── src/
│   ├── data/{loaders,cv}.py        # 데이터 로드 + CV 전략
│   ├── models/{tabular,vision,timeseries}.py
│   ├── train.py           # 공통 학습 (config 받아 분기)
│   ├── infer.py           # 추론 + 제출
│   ├── ensemble.py        # OOF 기반 블렌딩
│   └── utils/{seed,metrics,logger}.py
├── notebooks/
│   ├── 00_eda_template.ipynb     # 당일 데이터만 갈아끼우기
│   └── 01_quick_baseline.ipynb
├── experiments/           # .gitignore — 실험별 산출물 (oof.npy 등)
└── scripts/make_submission.sh
```

---

## 🔄 작업 흐름

```
주제 공개
  → configs/{task}.yaml 수정 (task, 경로, 컬럼명)
  → notebooks/00_eda_template 로 데이터 점검
  → python -m src.train --config ... --exp exp_001   (OOF 생성)
  → python -m src.train --config ... --exp exp_002   (다른 설정)
  → python -m src.ensemble --exps exp_001 exp_002    (가중치 최적화)
  → python -m src.infer --config ... --exp exp_001   (제출 파일)
```

**핵심 산출물은 `experiments/{exp}/oof.npy`** 입니다. 누수 없는 검증 예측이라
여러 실험의 OOF를 모아 앙상블하는 게 점수를 끌어올리는 길입니다.

---

## 👥 역할 분담 (예시)

| 역할 | 담당 | 산출물 |
|------|------|--------|
| **EDA & 전처리** | A | `notebooks/00_eda`, `src/data/` |
| **모델 A (정형)** | B | `exp_tab_*` 실험들 |
| **모델 B (딥러닝)** | C | `exp_vis_*` 실험들 |
| **앙상블 & 제출** | D | `src/ensemble.py`, 최종 submission |

> 각자 다른 `--exp` 이름으로 실험을 돌리면 산출물이 안 섞입니다.
> 모두의 oof.npy가 모이면 D가 앙상블합니다.

---

## 📏 컨벤션

### 브랜치 전략
- `main`: 안정 버전 (항상 돌아가는 상태 유지)
- `exp/{이름}-{내용}`: 개인 실험 브랜치 (예: `exp/jin-lgbm-tuning`)
- 실험이 검증되면 PR로 main에 머지

### 커밋 메시지
```
[exp] lgbm num_leaves 튜닝 → OOF 0.921
[feat] CatBoost 래퍼 추가
[fix] GroupKFold 누수 버그 수정
[docs] README 역할분담 갱신
```

### 실험 ID 규칙 (노션 DB와 연결)
- 폴더명 `exp_001` ↔ 노션 실험 DB `EXP-1`
- 이렇게 맞추면 "노션에서 best로 기록된 실험"의 oof.npy를 바로 찾음

### 재현성 3원칙
1. **seed 항상 고정** (`seed_everything(42)`)
2. **한 번에 하나만 바꾸기** (무엇이 점수를 올렸는지 추적 가능)
3. **config는 JSON으로 저장** (train.py가 자동 저장)

---

## 🧪 실험 기록

`src/utils/logger.py`가 매 학습마다 `experiments/log.csv`에 자동 기록합니다.
팀 공유는 **노션 실험 DB**를 함께 쓰세요 (동시 편집·필터·보드뷰).
OOF와 LB 점수를 **나란히** 기록해 과적합을 잡는 게 핵심입니다.

---

## ⚠️ 당일 체크리스트

- [ ] `setup.sh` 실행 → `--demo`로 파이프라인 검증
- [ ] 데이터 심볼릭 링크 (`data/README.md` 참고)
- [ ] config의 `task`, 경로, 컬럼명 수정
- [ ] **평가지표 확인** → `src/utils/metrics.py`가 맞는지 (확률 vs 라벨!)
- [ ] baseline 먼저 제출 (양식 검증 + 기준점)
- [ ] 그룹 누수 가능성 점검 → 필요시 `group_col` 설정
- [ ] 오프라인 대회면 `weights/` 미리 채우기

---

## 🔗 관련 자료

이 템플릿은 다음 자료들과 함께 쓰도록 설계됐습니다:
- 정형 파이프라인 해설서 (OOF/앙상블 개념)
- 음향 파이프라인 + EDA (스펙트로그램, AST/BEATs)
- 실험 트래커 (CSV + Optuna + 노션 연동)

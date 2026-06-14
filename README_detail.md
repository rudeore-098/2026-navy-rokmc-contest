# 📘 comp-template 레포 사용 설명서 (당일 손볼 곳만)

레포가 파일이 많아서 복잡해 보이지만, **당일 실제로 건드리는 곳은 딱 3군데**예요.
나머지는 안 봐도 돌아갑니다. 이 문서는 그 3군데만 콕 집어 설명해요.

> 💡 **먼저 읽어두기:** 레포를 바로 쓰기 부담되면, `allinone_baseline.ipynb`(노트북 한 개)로
> 먼저 대회를 치르세요. 익숙해진 뒤 이 레포로 옮겨오면 됩니다. 순서가 중요해요.

---

## 🗺️ 전체 그림 — 어떻게 돌아가나

```
configs/tabular.yaml  ← ① 여기서 설정 바꾸고
        │
        ▼
python -m src.train   ← ② 이 명령으로 학습 (oof.npy 생성)
        │
        ▼
python -m src.infer   ← ③ 이 명령으로 제출 파일 생성
        │
        ▼
experiments/exp_001/submission.csv  ← 이걸 제출!
```

복잡한 `src/` 안의 파일들은 **②③ 명령이 알아서 불러다 써요.** 직접 열 일이 거의 없어요.

---

## ✋ 당일 손볼 곳 — 딱 3군데

### ① `configs/tabular.yaml` — 설정 (제일 중요)

이 파일에서 **3가지만** 본인 대회에 맞게 바꾸면 됩니다:

```yaml
task: binary            # ← (1) 문제 종류: binary / multiclass / regression

target_col: "target"    # ← (2) 정답 칸 이름 (대회 데이터 보고 맞추기)
id_col: "id"            # ← (3) 제출용 번호 칸 이름

paths:
  train: "data/train.csv"   # ← (4) 데이터 경로 (보통 그대로 두고 data/에 링크)
  test: "data/test.csv"
```

> 🔰 **task 고르는 법:**
> - 답이 둘 중 하나(0/1) → `binary`
> - 답이 셋 이상 → `multiclass`
> - 답이 숫자(가격 등) → `regression`

이게 전부예요. 모델 파라미터(`params_by_kind`)는 처음엔 안 건드려도 됩니다.

### ② 데이터 연결 — `data/` 폴더

데이터 파일을 `data/` 안에 넣거나 링크해요. 두 가지 방법:

**방법 A (간단):** 그냥 복사
```bash
cp /어디서받은/train.csv data/train.csv
cp /어디서받은/test.csv  data/test.csv
```

**방법 B (서버에서, 대용량):** 심볼릭 링크
```bash
ln -s /mnt/대회데이터경로/train.csv data/train.csv
ln -s /mnt/대회데이터경로/test.csv  data/test.csv
```

### ③ 평가지표 확인 — `src/utils/metrics.py` (가끔만)

대회가 **확률을 원하는지 라벨(0/1)을 원하는지** 확인하세요.
보통은 기본값으로 맞는데, 특이한 지표(F1 등)면 이 파일의 `get_metric`만 살짝 고치면 돼요.
처음엔 건드릴 일이 거의 없어요.

---

## ▶️ 실제 사용 순서 (복붙하면 됨)

```bash
# 0. (처음 한 번) 세팅
bash setup.sh

# 1. 동작 확인 — 가짜 데이터로 전체가 도는지 (실제 데이터 없어도 됨)
python -m src.train --config configs/tabular.yaml --exp exp_demo --demo

# 2. 진짜 학습 — 위 ①②를 마친 뒤
python -m src.train --config configs/tabular.yaml --exp exp_001

# 3. 제출 파일 만들기
python -m src.infer --config configs/tabular.yaml --exp exp_001
#    → experiments/exp_001/submission.csv 생성됨

# (선택) 여러 실험을 합치고 싶을 때
python -m src.ensemble --task binary --exps exp_001 exp_002
```

> 💡 `--exp exp_001` 부분은 실험마다 이름을 바꿔요 (`exp_002`, `exp_003`...).
> 그래야 결과가 안 섞이고, 나중에 여러 개를 앙상블할 수 있어요.

---

## 🧩 각 폴더가 뭐 하는 곳인지 (참고용, 안 외워도 됨)

| 폴더/파일 | 뭐 하는 곳 | 당일 건드림? |
|----------|----------|:---:|
| `configs/` | 설정 (task, 경로) | ✅ **여기** |
| `data/` | 데이터 두는 곳 | ✅ **여기** |
| `src/train.py` | 학습 명령 (자동) | ❌ |
| `src/infer.py` | 제출 명령 (자동) | ❌ |
| `src/models/` | 모델 코드 (자동) | ❌ |
| `src/utils/metrics.py` | 채점 방식 | 🔶 가끔 |
| `experiments/` | 결과 저장 (자동 생성) | ❌ |
| `notebooks/` | EDA·baseline 노트북 | 🔶 데이터 볼 때 |
| `weights/` | 사전학습 가중치 (비전용) | 🔶 비전 대회만 |

**✅ 표시 2개(configs, data)만 당일 손대면 돼요.** 나머지는 명령어가 알아서 합니다.

---

## ❓ 자주 막히는 것

**Q. `python -m src.train`이 "No module named src" 에러가 나요**
→ 레포 **맨 위 폴더**(comp-template/)에서 실행하세요. `cd comp-template` 먼저.

**Q. 데이터 칸 이름을 모르겠어요**
→ 노트북에서 `pd.read_csv("data/train.csv").head()`로 확인하고, 그 정답 칸 이름을 config의 `target_col`에 넣으세요.

**Q. 제출했더니 점수가 0이거나 에러예요**
→ 대회가 준 `sample_submission.csv`와 칸 이름·순서를 비교하세요. 다르면 안 맞아요.

**Q. 비전/음향 대회인데요?**
→ `configs/vision.yaml`을 쓰는데, 이건 GPU 환경(Kaggle/Colab)이 필요해요.
   학습 본체는 음향 파이프라인 노트북을 참고해 채워야 합니다. 정형보다 한 단계 어려워요.

---

## 🎯 핵심 요약

1. 복잡해 보여도 **당일 건드리는 건 `configs/`랑 `data/` 둘 뿐**이에요.
2. 학습·제출은 **명령어 2개**면 끝 (`src.train` → `src.infer`).
3. 부담되면 **`allinone_baseline.ipynb` 노트북으로 먼저** 시작하세요.
4. 익숙해진 만큼만 레포로 옮겨오면 됩니다. 한 번에 다 할 필요 없어요.
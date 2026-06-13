# data/ — 데이터 경로 규칙

이 폴더는 **git에 올리지 않습니다** (`.gitignore` 처리됨).
대용량 데이터는 서버 마운트 경로를 심볼릭 링크로 연결하세요.

## 경로 규칙

```
data/
├── train.csv              # 학습 데이터 (정답 포함)
├── test.csv               # 테스트 데이터 (정답 없음)
├── sample_submission.csv  # 제출 양식 (컬럼명/순서 확인용)
└── raw/                   # 원본 (이미지/오디오 파일 등)
    ├── train/
    └── test/
```

## 심볼릭 링크 예시

서버에 데이터가 `/mnt/comp_data`에 마운트돼 있다면:

```bash
ln -s /mnt/comp_data/train.csv data/train.csv
ln -s /mnt/comp_data/test.csv  data/test.csv
ln -s /mnt/comp_data/raw       data/raw
```

## 주의

- config의 `paths:` 항목이 이 경로를 가리킵니다. 다르면 config를 수정하세요.
- 제출 전 `sample_submission.csv`와 출력 컬럼명·순서를 반드시 대조하세요.

# Exp05: M2 fine search

Exp04 M2가 저장한 OOF 확률과 회귀 원본 예측을 재사용해 추가 모델 학습 없이
다음을 정밀 탐색합니다.

- hard gating threshold: `0.000`부터 `0.100`까지 `0.005` 간격
- 그룹별 scale: 기존 최적값 주변 및 그룹 1의 기존 상한 밖까지 확장
- 그룹별 bias: `100 kWh` 간격
- pooled OOF 목적과 Fold A/B 평균 목적을 별도로 평가
- 최종 진단 예측에 최적값을 적용한 제출 후보 2개 생성

## 실행

저장소 루트에서 실행합니다.

```bash
cd /home/work/baram/Baram-yena
export PYTHONPATH="$PWD:$PWD/baseline/src:/home/work/baram/Baram:$PYTHONPATH"

python3 -m exp_yena.exp05_m2_fine_search.src.run_search \
  --config exp_yena/exp05_m2_fine_search/configs/fine_search.yaml
```

## 산출물

```text
outputs/fine_search/
├── pooled_threshold_search.csv
├── fold_robust_threshold_search.csv
├── best_pooled.yaml
├── best_fold_robust.yaml
├── submission_exp05_pooled.csv
├── submission_exp05_fold_robust.csv
├── submission_diagnostics.csv
└── search_manifest.json
```

`best_pooled.yaml`은 전체 OOF 점수를 최적화합니다. `best_fold_robust.yaml`은
Fold A와 Fold B의 그룹별 점수를 동일 가중 평균하므로 시간 구간 간 안정성을 더
중시합니다. 두 제출을 모두 만들되, 리더보드 과적합 위험이 더 낮은
`submission_exp05_fold_robust.csv`를 우선 후보로 사용합니다.

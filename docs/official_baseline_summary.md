# 공식 RandomForest baseline 감사

## 원본

`[Baseline]_기상 예보 데이터 기반 RandomForest 풍력발전량 예측.ipynb`를 확인했다. 원본은 이동하거나 수정하지 않았다.

## 동작 요약

- 데이터: LDAPS/GFS train/test, labels, sample submission만 읽으며 SCADA와 `info.xlsx`는 사용하지 않는다.
- 시간: 모든 시간 문자열을 datetime으로 바꾸고 `forecast_kst_dtm == kst_dtm`으로 label과 기상을 결합한다. 명세상 label 시각은 1시간 발전량 집계 구간의 **종료 시각**이며 forecast 시각과 대응한다.
- grid: 각 모델의 모든 수치 기상 변수를 forecast 시각별 단순 평균한다. 공간 위치나 grid별 값은 버린다.
- feature: month/day/hour/dayofweek/weekend와 hour/month sin/cos, LDAPS/GFS 전체 변수 평균이다.
- label: label을 기준으로 left merge하며 target별 결측 행만 제외한다.
- 결측: 하나의 median imputer를 전체 train feature에 fit하고 test에는 transform한다.
- 모델: target별 `RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=8, max_features="sqrt", random_state=42, n_jobs=-1)`.
- validation/metric: validation과 metric 계산이 없다.
- 제출: sample의 key 순서를 보존하고 세 target을 예측한다. 예측은 0과 그룹 설비용량 사이로 clipping하고 UTF-8 BOM CSV로 저장한다.

## 누수·재현성 감사

train/test를 합쳐 통계를 계산하지 않고 target별 결측을 처리하는 점은 재사용할 수 있다. 그러나 `X_train`에서 target과 시각만 제거하므로 label merge 뒤 남은 `forecast_id`가 있다면 식별자가 feature가 될 수 있다(현재 train label에는 없어 실행상 생성되지는 않지만 표현이 취약하다). 더 중요한 한계는 시간 validation 부재, 모든 원 기상 변수를 무차별 사용, 공간 정보 손실, lead time 미사용, 제출 상한 clipping을 고정한 점이다. 공식 notebook에는 nMAE 또는 정산금획득률 공식이 없다.

공용 baseline은 공식 RF 파라미터를 그대로 제공하되 명시적 feature allow-list, 시간 split, train-only preprocessing, raw/postprocessed metric, optional upper clipping을 쓴다. 공식 feature를 그대로 재현하는 별도 모드는 아직 두지 않고, 공식 파라미터를 더 풍부한 공통 feature에 적용한다.

## 풍속 파생값 주의

`ws50_maxcomp = sqrt(u_max² + v_max²)`는 U/V 성분별 최대를 조합한 값이지 실제 시간상 최대 풍속이 아니다. `ws50_mincomp`도 동일한 성격이며, 기본 물리적 대표값은 각 성분 midpoint로 계산한 `ws50_mid`다.


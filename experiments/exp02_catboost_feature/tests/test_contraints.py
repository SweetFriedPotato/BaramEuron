# 단조 제약 설정이 모델에 올바르게 주입되었는지 검증
# experiments/exp01_xgboost_monotonic/tests/test_constraints.py

import pytest
from experiments.exp01_xgboost_monotonic.src.features import get_monotonic_constraints


def test_monotonic_constraints_mapping():
    # 입력 피처 목록과 단조 제약 설정 정의
    feature_names = ["temperature", "wind_speed_cubed", "humidity", "air_density"]
    constraint_config = {
        "wind_speed_cubed": 1,
        "air_density": 1
    }
    
    # 제약 조건 튜플 생성
    constraints = get_monotonic_constraints(feature_names, constraint_config)
    
    # 1. 반환된 결과가 튜플 형태인지 확인
    assert isinstance(constraints, tuple)
    
    # 2. 피처 개수와 튜플의 길이가 일치하는지 확인
    assert len(constraints) == len(feature_names)
    
    # 3. 제약이 걸린 피처는 1, 걸리지 않은 피처는 0으로 맵핑되었는지 검증
    # ["temperature"(0), "wind_speed_cubed"(1), "humidity"(0), "air_density"(1)]
    assert constraints == (0, 1, 0, 1)


def test_empty_constraints():
    feature_names = ["feat_a", "feat_b"]
    constraint_config = {}  # 제약이 아무것도 없을 때
    
    constraints = get_monotonic_constraints(feature_names, constraint_config)
    
    assert constraints == (0, 0)
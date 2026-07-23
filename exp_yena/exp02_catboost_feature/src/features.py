# XGBoost 입력용 피처 정렬 및 단조 제약 매핑
# experiments/exp01_xgboost_monotonic/src/features.py

def get_monotonic_constraints(feature_names: list, constraint_config: dict) -> list:
    """학습 데이터의 피처 순서에 맞게 단조 제약 튜플을 생성합니다.
    
    완전 일치(Exact Match)를 우선으로 탐색하며, 일치하지 않는 경우 피처명에 
    설정된 단조성 규칙 키워드가 포함(Substring)되어 있는지 패턴 매칭하여 유연하게 제약을 주입합니다.
    """
    if not constraint_config:
        return list(0 for _ in feature_names)

    constraints = []
    for col in feature_names:
        mapped_val = 0
        
        # 1단계: 완전 일치 검사
        if col in constraint_config:
            mapped_val = constraint_config[col]
        else:
            # 2단계: 부분 일치 검사 (예: 'wind_speed_cubed_lag_1'에 'wind_speed_cubed' 제약 주입)
            for rule_key, constraint_val in constraint_config.items():
                if rule_key in col:
                    mapped_val = constraint_val
                    break  # 첫 번째 매칭된 규칙을 우선 적용
                    
        constraints.append(mapped_val)
        
    return list(constraints)
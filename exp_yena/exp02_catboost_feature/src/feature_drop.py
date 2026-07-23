import pandas as pd
import os

DROP_LIST_FILE = "experiments/exp02_catboost_feature/configs/dropped_features_list.txt"

def drop_low_importance_features(report_path, threshold=0.05):
    print("=" * 60)
    print(f"피처 중요도 분석 및 노이즈 필터링 시작 (기준값: {threshold})")
    print("=" * 60)
    
    # 1. 리포트 파일 존재 여부 확인
    if not os.path.exists(report_path):
        print(f"에러: [{report_path}] 파일을 찾을 수 없습니다.")
        print("먼저 실험을 실행하여 'feature_importances_report.csv'를 생성해 주세요.")
        return None
        
    # 2. 데이터 로드
    df = pd.read_csv(report_path)
    total_features = len(df)
    
    # 3. 중요도 기준값(Threshold) 미만 피처 필터링
    drop_df = df[df['importance'] < threshold].sort_values(by='importance', ascending=True)
    keep_df = df[df['importance'] >= threshold]
    
    dropped_count = len(drop_df)
    kept_count = len(keep_df)
    
    print(f"총 피처 개수: {total_features}개")
    print(f"유지할 피처 (중요도 >= {threshold}): {kept_count}개")
    print(f"드롭할 피처 (중요도 <  {threshold}): {dropped_count}개")
    print("-" * 60)
    
    if dropped_count == 0:
        print(f"중요도가 {threshold} 미만인 피처가 없습니다! 드롭할 필요가 없습니다.")
        return list(keep_df['feature'])
        
    # 4. 드롭 대상 하위 피처 목록 출력 (최하위 10개 예시)
    print(f"드롭 대상 주요 하위 피처 (총 {dropped_count}개 중 일부):")
    for idx, row in drop_df.head(10).iterrows():
        print(f"  - {row['feature']:<65} | 중요도: {row['importance']:.6f}")
        
    if dropped_count > 10:
        print(f"  ... 외 {dropped_count - 10}개 피처 더 있음.")
        
    print("-" * 60)
    
    # 5. 드롭할 피처 목록 리스트 저장 (차후 전처리 파이프라인 연동용)
    drop_features_list = drop_df['feature'].tolist()
    keep_features_list = keep_df['feature'].tolist()
    
    # 텍스트 파일로 저장하여 config나 다른 코드에서 불러오기 쉽게 만듦
    drop_log_path = DROP_LIST_FILE
    os.makedirs(os.path.dirname(drop_log_path), exist_ok=True)
    
    with open(drop_log_path, 'w', encoding='utf-8') as f:
        for feat in drop_features_list:
            f.write(f"{feat}\n")
            
    print(f"드롭 대상 피처 목록 파일 저장 완료: {drop_log_path}")
    print("=" * 60)
    
    return keep_features_list

# ------------------------------------------------------------------
# 데이터프레임 적용 예시 (실제 코드 연동 시 참고용 함수)
# ------------------------------------------------------------------
def apply_drop_to_dataframe(df_train, df_test, drop_list):
    """
    실제 X_train, X_test 데이터프레임에서 리스트에 담긴 변수들을 제거하는 헬퍼 함수
    """
    # 데이터프레임에 실제로 존재하는 컬럼만 선별하여 제거 (KeyError 방지)
    cols_to_drop = [col for col in drop_list if col in df_train.columns]
    
    X_train_clean = df_train.drop(columns=cols_to_drop)
    X_test_clean = df_test.drop(columns=cols_to_drop) if df_test is not None else None
    
    print(f"데이터프레임 정제 완료: {len(cols_to_drop)}개 피처 삭제됨.")
    return X_train_clean, X_test_clean


if __name__ == "__main__":
    # 리포트 파일 경로 지정 (질문자님의 아웃풋 폴더 구조에 맞춤)
    REPORT_FILE = "experiments/exp02_catboost_feature/outputs/feature_importances_report.csv"
    
    # 현재 디렉토리에 파일이 바로 있다면 아래와 같이 지정 가능
    if not os.path.exists(REPORT_FILE) and os.path.exists("feature_importances_report.csv"):
        REPORT_FILE = "feature_importances_report.csv"
        
    # 함수 실행 -> 유지할 피처 이름 리스트 반환
    features_to_keep = drop_low_importance_features(REPORT_FILE, threshold=0.05)

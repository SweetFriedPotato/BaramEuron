import os
import argparse
import copy
import pathlib
import yaml
import pandas as pd
import numpy as np
import catboost as cb  # XGBoost를 CatBoost로 대체
from pathlib import Path

# 공용 프레임워크 및 공유 상수/메트릭 모듈 임포트
from baram.feature_builder import load_raw_feature_artifacts, get_features_for_group
from baram.validation import split_labeled_table
from baram.preprocessing import fit_tree_preprocessor
from baram.constants import TIME_COL as CONST_TIME_COL
from baram.data import load_sample_submission
from baram.submission import create_submission, postprocess
from shared.constants import CAPACITY_KWH
from .features import get_monotonic_constraints
from .feature_blocks import FeatureBlockPipeline  


FOLD_SPECS = [
    {
        "fold": "Fold A",
        "groups": (1, 2),
        "validation": {
            "group_1_2_train_start": "2022-01-01 01:00:00",
            "group_1_2_train_end": "2023-01-01 00:00:00",
            "valid_start": "2023-01-01 01:00:00",
            "valid_end": "2024-01-01 00:00:00",
        },
    },
    {
        "fold": "Fold B",
        "groups": (1, 2, 3),
        "validation": {
            "group_1_2_train_start": "2022-01-01 01:00:00",
            "group_3_train_start": "2023-01-01 01:00:00",
            "group_1_2_train_end": "2024-01-01 00:00:00",
            "group_3_train_end": "2024-01-01 00:00:00",
            "valid_start": "2024-01-01 01:00:00",
            "valid_end": "2025-01-01 00:00:00",
        },
    },
]


def get_feature_blocks_config(config):
    """Map YAML feature flags directly to FeatureBlockPipeline block names."""
    features = config.get("features", {})
    return {
        "wind_physics": features.get("wind_physics", False),
        "thermodynamic": features.get("thermodynamic", False),
        "forecast_disagreement": features.get("forecast_disagreement", False),
        "advanced_meteorology": features.get("advanced_meteorology", True),
    }


def calculate_group_metrics(y_true, y_pred, group_key):
    """Calculate the competition components for one group."""
    actual = np.asarray(y_true, dtype=float)
    forecast = np.asarray(y_pred, dtype=float)
    capacity = CAPACITY_KWH[group_key]
    valid = actual >= capacity * 0.10
    if not np.any(valid):
        return {"nmae": 1.0, "one_minus_nmae": 0.0, "ficr": 0.0, "total_score": 0.0}

    actual_valid = actual[valid]
    error_rate = np.abs(forecast[valid] - actual_valid) / capacity
    nmae = float(np.mean(error_rate))
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    max_settlement = float(np.sum(actual_valid * 4.0))
    ficr = 0.0 if max_settlement == 0 else float(np.sum(actual_valid * unit_price) / max_settlement)
    one_minus_nmae = 1.0 - nmae
    return {
        "nmae": nmae,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "total_score": 0.5 * one_minus_nmae + 0.5 * ficr,
    }


def calculate_oof_metrics(oof_predictions):
    """Aggregate all chronological OOF rows using the competition's group averaging."""
    group_metrics = {}
    for group_key, group_frame in oof_predictions.groupby("target"):
        group_metrics[group_key] = calculate_group_metrics(
            group_frame["y_true"], group_frame["prediction"], group_key
        )
    missing = sorted(set(CAPACITY_KWH) - set(group_metrics))
    if missing:
        raise ValueError(f"Missing OOF predictions for groups: {missing}")
    one_minus_nmae = float(np.mean([item["one_minus_nmae"] for item in group_metrics.values()]))
    ficr = float(np.mean([item["ficr"] for item in group_metrics.values()]))
    return {
        "total_score": 0.5 * one_minus_nmae + 0.5 * ficr,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }, group_metrics


def load_dropped_features(config):
    """Load feature names selected for removal by feature_drop.py."""
    drop_list_path = Path(
        config.get(
            "feature_drop_list",
            "experiments/exp02_catboost_feature/configs/dropped_features_list.txt",
        )
    )
    if not drop_list_path.exists():
        print(f"No feature drop list found at {drop_list_path}; using all features.")
        return set()

    with open(drop_list_path, "r", encoding="utf-8-sig") as f:
        dropped_features = {line.strip() for line in f if line.strip()}

    print(f"Loaded {len(dropped_features)} features to drop from: {drop_list_path}")
    return dropped_features


def apply_feature_drop(train_df, other_df, dropped_features):
    """Drop the same available feature columns from a train/validation or train/test pair."""
    columns_to_drop = [
        column for column in train_df.columns
        if column in dropped_features and column in other_df.columns
    ]
    train_clean = train_df.drop(columns=columns_to_drop)
    other_clean = other_df.drop(columns=columns_to_drop)

    if train_clean.shape[1] == 0:
        raise ValueError("Feature dropping removed every available feature.")

    return train_clean, other_clean, columns_to_drop


def parse_args():
    parser = argparse.ArgumentParser(description="Exp01: CatBoost Regressor with Meteor Re-engineering")
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the experiment config file"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=None, 
        help="CatBoost iterations (overrides config)"
    )
    parser.add_argument(
        "--output-root", 
        type=str, 
        default=None, 
        help="Path to save experiment outputs"
    )
    parser.add_argument(
        "--no-finalize", 
        action="store_true", 
        help="Skip submission generation, validation only"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if args.iterations is not None:
        if "model" not in config:
            config["model"] = {}
        if "params" not in config["model"]:
            config["model"]["params"] = {}
        config["model"]["params"]["iterations"] = args.iterations
    
    output_root = args.output_root if args.output_root is not None else config.get("output_root", "experiments/exp02_catboost_feature/outputs")
    os.makedirs(output_root, exist_ok=True)
    dropped_features = load_dropped_features(config)
    
    print("[1/4] Loading Raw Feature Artifacts...")
    raw_artifacts = load_raw_feature_artifacts(config)
    if isinstance(raw_artifacts, tuple):
        train_features = raw_artifacts[0]
        test_features = raw_artifacts[1]
    else:
        train_features = raw_artifacts
        test_features = None
        
    print("[1/4] Loading Target Labels...")
    time_col_name = CONST_TIME_COL
    
    train_labels_path = pathlib.Path(config["data"]["train_dir"]) / "train_labels.csv"
    if not train_labels_path.exists():
        train_labels_path = pathlib.Path(config["data"]["root"]) / "train/train_labels.csv"
        
    targets_df = pd.read_csv(train_labels_path, encoding="utf-8-sig")
    
    if "kst_dtm" in targets_df.columns:
        targets_df = targets_df.rename(columns={"kst_dtm": time_col_name})
    elif "datetime" in targets_df.columns:
        targets_df = targets_df.rename(columns={"datetime": time_col_name})
    elif "timestamp" in targets_df.columns:
        targets_df = targets_df.rename(columns={"timestamp": time_col_name})
        
    targets_df[time_col_name] = pd.to_datetime(targets_df[time_col_name])
    
    oof_frames = []
    fold_metric_rows = []
    best_iterations_by_group = {1: [], 2: [], 3: []}
    importance_list = []
    
    print("[2/4] Starting Fold Validation...")
    validation_runs = [
        (spec["fold"], group_id, spec["validation"])
        for spec in FOLD_SPECS
        for group_id in spec["groups"]
    ]
    for fold_name, group_id, validation_config in validation_runs:
        print(f"--- Processing {fold_name} | Group {group_id} ---")
        
        group_features = get_features_for_group(train_features, group_id).copy()
        target_col = f"kpx_group_{group_id}"
        
        group_features[time_col_name] = pd.to_datetime(group_features[time_col_name])
        group_data = pd.merge(group_features, targets_df[[time_col_name, target_col]], on=time_col_name, how="inner")
        if group_data[time_col_name].duplicated().any():
            raise ValueError(
                f"Expected one row per prediction timestamp for group {group_id}, "
                "but duplicate timestamps were found."
            )
        
        fold_config = copy.deepcopy(config)
        fold_config["fold"] = fold_name
        fold_config["validation"] = copy.deepcopy(validation_config)
        train_mask, val_mask = split_labeled_table(group_data, target_col, fold_config)
        
        train_df = group_data.loc[train_mask].reset_index(drop=True)
        val_df = group_data.loc[val_mask].reset_index(drop=True)
        if train_df.empty or val_df.empty:
            raise ValueError(
                f"Empty split for {fold_name} / group {group_id}: "
                f"train={len(train_df)}, validation={len(val_df)}"
            )
        print(
            f"Train: {train_df[time_col_name].min()} -> {train_df[time_col_name].max()} "
            f"({len(train_df)} rows) | Validation: {val_df[time_col_name].min()} -> "
            f"{val_df[time_col_name].max()} ({len(val_df)} rows)"
        )
        
        targets_to_drop = [c for c in group_data.columns if c.startswith("kpx_group_")]
        cols_to_drop = [time_col_name] + targets_to_drop
        
        X_train = train_df.drop(columns=cols_to_drop, errors="ignore")
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=cols_to_drop, errors="ignore")
        y_val = val_df[target_col]
        
        # 기상 역학 및 풍력 공식 물리 엔진 블록 통합 구성
        blocks_config = get_feature_blocks_config(config)
        
        pipeline = FeatureBlockPipeline(
            blocks=blocks_config, 
            group_id=group_id,
            wind_config=config.get("wind_physics", {})
        )
        
        X_train_processed = pipeline.fit_transform(X_train)
        X_val_processed = pipeline.transform(X_val)
        
        preprocessor, X_train_imputed_arr, X_val_imputed_arr, feature_names = fit_tree_preprocessor(
            X_train_processed, X_val_processed, config=config
        )
        
        X_train_imputed = pd.DataFrame(X_train_imputed_arr, columns=feature_names)
        X_val_imputed = pd.DataFrame(X_val_imputed_arr, columns=feature_names)

        X_train_imputed, X_val_imputed, dropped_columns = apply_feature_drop(
            X_train_imputed, X_val_imputed, dropped_features
        )
        if dropped_columns:
            print(f"Dropped {len(dropped_columns)} features for {fold_name} / group {group_id}.")
        
        feature_names = X_train_imputed.columns.tolist()
        
        # CatBoost 형식에 맞는 단조성 제약 구조 배열 연산 처리
        model_params = config['model']['params'].copy()
        
        # XGBoost용 파라미터가 config에 혼재되어 있는 경우를 대비한 네이밍 변환 및 기본값 매핑
        if 'n_estimators' in model_params:
            model_params['iterations'] = model_params.pop('n_estimators')
        if 'iterations' not in model_params:
            model_params['iterations'] = 2000
            
        monotonic_features = config.get('monotonic_features', {})
        if monotonic_features:
            model_params['monotone_constraints'] = get_monotonic_constraints(feature_names, monotonic_features)
        else:
            model_params.pop('monotone_constraints', None)
        
        # 하위 컷오프 확인 프로세스 편의를 위한 시드 및 스레드 디폴트 처리
        model_params['random_seed'] = model_params.get('random_seed', 42)
        model_params['verbose'] = False
        
        try:
            # CatBoost GPU 훈련 백엔드 테스트 프로토콜
            test_params = model_params.copy()
            test_params['task_type'] = 'GPU'
            test_params['iterations'] = 5
            model = cb.CatBoostRegressor(**test_params)
            model.fit(X_train_imputed.iloc[:10], y_train.iloc[:10])
            
            # 테스트 통과 시 실제 메인 파라미터 설정 적용
            model_params['task_type'] = 'GPU'
            model = cb.CatBoostRegressor(**model_params)
        except Exception as error:
            if config.get('require_gpu', False):
                raise RuntimeError("CatBoost GPU validation failed; CPU fallback is disabled for this run.") from error
            print("Warning: CatBoost GPU training failed. Falling back to CPU.")
            model_params['task_type'] = 'CPU'
            model = cb.CatBoostRegressor(**model_params)
            
        # CatBoost 고유 객체 Pool 구성 및 오버핏 방지를 위한 Early Stopping 연동 
        train_pool = cb.Pool(X_train_imputed, y_train)
        val_pool = cb.Pool(X_val_imputed, y_val)
        
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=150,
            verbose=False
        )
        
        raw_val_predictions = model.predict(X_val_imputed)
        group_key = f"kpx_group_{group_id}"
        val_predictions = postprocess(
            raw_val_predictions, group_key, config.get("postprocess", {})
        )
        val_trues = y_val.to_numpy()
        best_iteration = int(model.tree_count_)
        best_iterations_by_group[group_id].append(best_iteration)

        fold_metrics = calculate_group_metrics(val_trues, val_predictions, group_key)
        fold_metric_rows.append({
            "fold": fold_name,
            "group_id": group_id,
            "target": group_key,
            "train_start": str(train_df[time_col_name].min()),
            "train_end": str(train_df[time_col_name].max()),
            "valid_start": str(val_df[time_col_name].min()),
            "valid_end": str(val_df[time_col_name].max()),
            "train_rows": len(train_df),
            "valid_rows": len(val_df),
            "best_iteration": best_iteration,
            **fold_metrics,
        })
        capacity = CAPACITY_KWH[group_key]
        oof_frames.append(pd.DataFrame({
            time_col_name: val_df[time_col_name].to_numpy(),
            "fold": fold_name,
            "group_id": group_id,
            "target": group_key,
            "y_true": val_trues,
            "raw_prediction": raw_val_predictions,
            "prediction": val_predictions,
            "absolute_error": np.abs(val_predictions - val_trues),
            "error_rate": np.abs(val_predictions - val_trues) / capacity,
            "valid_for_metric": val_trues >= capacity * 0.10,
        }))
        print(
            f"{fold_name} / group {group_id}: score={fold_metrics['total_score']:.5f}, "
            f"1-NMAE={fold_metrics['one_minus_nmae']:.5f}, "
            f"FICR={fold_metrics['ficr']:.5f}, best_iteration={best_iteration}"
        )
        
        # 하위 변수 컷오프 분석 목적의 피처 임포턴스 데이터 취합 기록
        fold_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.get_feature_importance(),
            "fold": fold_name,
            "group_id": group_id,
        })
        importance_list.append(fold_importance)
        
    print("\n[3/4] Running Custom Metric Evaluation...")
    oof_predictions = pd.concat(oof_frames, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    metrics, group_metrics = calculate_oof_metrics(oof_predictions)
    print(f"Total Score     : {metrics['total_score']:.5f}")
    print(f"1 - NMAE         : {metrics['one_minus_nmae']:.5f}")
    print(f"FICR             : {metrics['ficr']:.5f}")
    
    oof_predictions.to_csv(
        Path(output_root) / "oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fold_metrics_df.to_csv(
        Path(output_root) / "fold_group_metrics.csv", index=False, encoding="utf-8-sig"
    )
    best_iterations_df = fold_metrics_df[
        ["fold", "group_id", "target", "best_iteration"]
    ].copy()
    best_iterations_df.to_csv(
        Path(output_root) / "best_iterations.csv", index=False, encoding="utf-8-sig"
    )
    validation_results = {
        **metrics,
        "group_metrics": group_metrics,
        "fold_group_metrics": fold_metric_rows,
    }
    with open(Path(output_root) / "val_results.txt", "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(validation_results, allow_unicode=True, sort_keys=False))
        
    # 모든 폴드의 중요도를 평균내어 어떤 변수가 노이즈이고 하위 순위인지 리스트 파일로 저장
    full_importance_df = (
        pd.concat(importance_list)
        .groupby("feature")[["importance"]]
        .mean()
        .sort_values(by="importance", ascending=False)
    )
    full_importance_df.to_csv(f"{output_root}/feature_importances_report.csv", encoding="utf-8-sig")
    print(f"Saved feature importances report to: {output_root}/feature_importances_report.csv")
        
    if args.no_finalize:
        print("Option '--no-finalize' detected. Skipping submission generation.")
        return
        
    print("\n[4/4] Finalizing Submission (Full Training & Inference)...")
    sample_sub = load_sample_submission(config)
    predictions = {}
    
    for group_id in [1, 2, 3]:
        group_key = f"kpx_group_{group_id}"
        print(f"Re-training and Inferencing: {group_key}...")
        
        group_train_features = get_features_for_group(train_features, group_id).copy()
        group_train_features[time_col_name] = pd.to_datetime(group_train_features[time_col_name])
        
        group_test_features = get_features_for_group(test_features, group_id).copy()
        group_test_features[time_col_name] = pd.to_datetime(group_test_features[time_col_name])
        
        group_data = pd.merge(group_train_features, targets_df[[time_col_name, group_key]], on=time_col_name, how="inner")
        
        mask = group_data[group_key].notna()
        full_train_df = group_data.loc[mask].reset_index(drop=True)
        
        targets_to_drop = [c for c in group_data.columns if c.startswith("kpx_group_")]
        cols_to_drop = [time_col_name] + targets_to_drop
        
        X_full_train = full_train_df.drop(columns=cols_to_drop, errors="ignore")
        y_full_train = full_train_df[group_key]
        X_test = group_test_features.drop(columns=[time_col_name], errors="ignore")
        
        blocks_config = get_feature_blocks_config(config)
        
        pipeline = FeatureBlockPipeline(
            blocks=blocks_config, 
            group_id=group_id,
            wind_config=config.get("wind_physics", {})
        )
        
        X_full_train_processed = pipeline.fit_transform(X_full_train)
        X_test_processed = pipeline.transform(X_test)
        
        preprocessor, X_train_imputed_arr, X_test_imputed_arr, feature_names = fit_tree_preprocessor(
            X_full_train_processed, X_test_processed, config=config
        )
        
        X_train_imputed = pd.DataFrame(X_train_imputed_arr, columns=feature_names)
        X_test_imputed = pd.DataFrame(X_test_imputed_arr, columns=feature_names)

        X_train_imputed, X_test_imputed, dropped_columns = apply_feature_drop(
            X_train_imputed, X_test_imputed, dropped_features
        )
        if dropped_columns:
            print(f"Dropped {len(dropped_columns)} features for final group {group_id}.")

        feature_names = X_train_imputed.columns.tolist()
        
        model_params = config['model']['params'].copy()
        
        if 'n_estimators' in model_params:
            model_params['iterations'] = model_params.pop('n_estimators')
        if 'iterations' not in model_params:
            model_params['iterations'] = 2000
        iteration_multiplier = float(config.get("final_iteration_multiplier", 1.0))
        selected_iterations = max(
            1,
            int(round(np.median(best_iterations_by_group[group_id]) * iteration_multiplier)),
        )
        model_params['iterations'] = selected_iterations
        print(
            f"Using {selected_iterations} final iterations for group {group_id} "
            f"from fold values {best_iterations_by_group[group_id]} "
            f"(multiplier={iteration_multiplier})."
        )
            
        monotonic_features = config.get('monotonic_features', {})
        if monotonic_features:
            model_params['monotone_constraints'] = get_monotonic_constraints(feature_names, monotonic_features)
        else:
            model_params.pop('monotone_constraints', None)
        model_params['random_seed'] = model_params.get('random_seed', 42)
        model_params['verbose'] = False
        
        try:
            test_params = model_params.copy()
            test_params['task_type'] = 'GPU'
            test_params['iterations'] = 5
            model = cb.CatBoostRegressor(**test_params)
            model.fit(X_train_imputed.iloc[:10], y_full_train.iloc[:10])
            
            model_params['task_type'] = 'GPU'
            model = cb.CatBoostRegressor(**model_params)
        except Exception as error:
            if config.get('require_gpu', False):
                raise RuntimeError("CatBoost GPU final training failed; CPU fallback is disabled for this run.") from error
            model_params['task_type'] = 'CPU'
            model = cb.CatBoostRegressor(**model_params)
            
        model.fit(X_train_imputed, y_full_train, verbose=False)
        
        test_preds_cap = model.predict(X_test_imputed)
        test_preds_cap_processed = postprocess(test_preds_cap, group_key, config.get("postprocess", {}))
        
        predictions[group_key] = test_preds_cap_processed
        
    sub_dir = Path(output_root) / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_filepath = sub_dir / "submission_catboost_advanced.csv"
    
    print(f"Writing submission file to: {sub_filepath}")
    create_submission(sample_sub, predictions, path=sub_filepath)
    print(f"Experiment complete. Outputs and finalized submission saved at: {output_root}")


if __name__ == "__main__":
    main()

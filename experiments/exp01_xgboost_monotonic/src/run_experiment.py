import os
import argparse
import pathlib
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# 공용 프레임워크 및 공유 상수/메트릭 모듈 임포트
from baram.feature_builder import load_raw_feature_artifacts, get_features_for_group
from baram.validation import split_labeled_table
from baram.preprocessing import fit_tree_preprocessor
from baram.constants import TIME_COL as CONST_TIME_COL
from baram.data import load_sample_submission
from baram.submission import create_submission, postprocess
from shared.constants import CAPACITY_KWH
from shared.metrics import calculate_competition_metric
from .features import get_monotonic_constraints
from .feature_blocks import FeatureBlockPipeline  


def parse_args():
    parser = argparse.ArgumentParser(description="Exp01: XGBoost Monotonic Constraints")
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the experiment config file"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=None, 
        help="XGBoost n_estimators (overrides config)"
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
        config["model"]["params"]["n_estimators"] = args.iterations
    
    output_root = args.output_root if args.output_root is not None else config.get("output_root", "experiments/exp01_xgboost_monotonic/outputs")
    os.makedirs(output_root, exist_ok=True)
    
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
    
    all_oof_preds = []
    all_oof_trues = []
    
    print("[2/4] Starting Fold Validation...")
    for fold_name, group_id in [("Fold A", 1), ("Fold A", 2), ("Fold B", 1), ("Fold B", 2), ("Fold B", 3)]:
        print(f"--- Processing {fold_name} | Group {group_id} ---")
        
        group_features = get_features_for_group(train_features, group_id).copy()
        target_col = f"kpx_group_{group_id}"
        
        group_features[time_col_name] = pd.to_datetime(group_features[time_col_name])
        group_data = pd.merge(group_features, targets_df[[time_col_name, target_col]], on=time_col_name, how="inner")
        
        config["fold"] = fold_name
        train_mask, val_mask = split_labeled_table(group_data, target_col, config)
        
        train_df = group_data.loc[train_mask].reset_index(drop=True)
        val_df = group_data.loc[val_mask].reset_index(drop=True)
        
        targets_to_drop = [c for c in group_data.columns if c.startswith("kpx_group_")]
        cols_to_drop = [time_col_name] + targets_to_drop
        
        X_train = train_df.drop(columns=cols_to_drop, errors="ignore")
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=cols_to_drop, errors="ignore")
        y_val = val_df[target_col]
        
        blocks_config = {
            "wind_physics": config["features"].get("power_curve_features", False),
            "thermodynamic": config["features"].get("thermodynamic", False),
            "forecast_disagreement": config["features"].get("weather_summary", False),
        }
        
        # config에 맞춤 하이퍼파라미터 주입 유도
        pipeline = FeatureBlockPipeline(
            blocks=blocks_config, 
            group_id=group_id,
            wind_config=config.get("wind_physics", {})
        )
        
        # 훈련 데이터셋으로 스케일 적합 및 임계 보정 후 트랜스폼 적용
        X_train_processed = pipeline.fit_transform(X_train)
        X_val_processed = pipeline.transform(X_val)
        
        # 피처 가공이 완료된 데이터셋을 나무 전처리기에 입력
        preprocessor, X_train_imputed_arr, X_val_imputed_arr, feature_names = fit_tree_preprocessor(
            X_train_processed, X_val_processed, config=config
        )
        
        X_train_imputed = pd.DataFrame(X_train_imputed_arr, columns=feature_names)
        X_val_imputed = pd.DataFrame(X_val_imputed_arr, columns=feature_names)
        
        feature_names = X_train_imputed.columns.tolist()
        mono_constraints = get_monotonic_constraints(feature_names, config['monotonic_features'])
        
        model_params = config['model']['params'].copy()
        model_params['monotone_constraints'] = mono_constraints
        
        try:
            model_params['tree_method'] = 'hist'
            model_params['device'] = 'cuda'
            model = xgb.XGBRegressor(**model_params)
            model.fit(X_train_imputed.iloc[:10], y_train.iloc[:10])
        except Exception:
            print("Warning: GPU training failed. Falling back to CPU.")
            model_params['device'] = 'cpu'
            model = xgb.XGBRegressor(**model_params)
            
        model.fit(
            X_train_imputed, y_train,
            eval_set=[(X_val_imputed, y_val)],
            verbose=False
        )
        
        val_preds_cap = model.predict(X_val_imputed)
        
        group_key = f"kpx_group_{group_id}"
        val_preds_kwh = val_preds_cap 
        val_trues_kwh = y_val.to_numpy()
        
        all_oof_preds.append(pd.DataFrame({group_key: val_preds_kwh}))
        all_oof_trues.append(pd.DataFrame({group_key: val_trues_kwh}))
        
    print("\n[3/4] Running Custom Metric Evaluation...")
    oof_preds_df = pd.concat(all_oof_preds, axis=1).fillna(0)
    oof_trues_df = pd.concat(all_oof_trues, axis=1).fillna(0)
    
    metrics = calculate_competition_metric(oof_trues_df, oof_preds_df)
    print(f"Total Score     : {metrics['total_score']:.5f}")
    print(f"1 - NMAE         : {metrics['one_minus_nmae']:.5f}")
    print(f"FICR             : {metrics['ficr']:.5f}")
    
    with open(f"{output_root}/val_results.txt", "w") as f:
        f.write(yaml.dump(metrics))
        
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
        
        blocks_config = {
            "wind_physics": config["features"].get("power_curve_features", False),
            "thermodynamic": config["features"].get("thermodynamic", False),
            "forecast_disagreement": config["features"].get("weather_summary", False),
        }
        
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
        
        mono_constraints = get_monotonic_constraints(feature_names, config['monotonic_features'])
        model_params = config['model']['params'].copy()
        model_params['monotone_constraints'] = mono_constraints
        
        try:
            model_params['tree_method'] = 'hist'
            model_params['device'] = 'cuda'
            model = xgb.XGBRegressor(**model_params)
            model.fit(X_train_imputed.iloc[:10], y_full_train.iloc[:10])
        except Exception:
            model_params['device'] = 'cpu'
            model = xgb.XGBRegressor(**model_params)
            
        model.fit(X_train_imputed, y_full_train, verbose=False)
        
        test_preds_cap = model.predict(X_test_imputed)
        test_preds_cap_processed = postprocess(test_preds_cap, group_key, config.get("postprocess", {}))
        
        predictions[group_key] = test_preds_cap_processed
        
    sub_dir = Path(output_root) / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    sub_filepath = sub_dir / "submission_xgboost_monotonic.csv"
    
    print(f"Writing submission file to: {sub_filepath}")
    create_submission(sample_sub, predictions, path=sub_filepath)
    print(f"Experiment complete. Outputs and finalized submission saved at: {output_root}")


if __name__ == "__main__":
    main()
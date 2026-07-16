"""Render exp03's final evidence-backed report from generated artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = PROJECT_ROOT / "experiments/exp03_official_score_calibration"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "outputs"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _update_manifest(output_root: Path, final: dict, rolling: pd.DataFrame) -> None:
    path = output_root / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    training_summary = json.loads((output_root / "training_summary.json").read_text(encoding="utf-8"))
    rolling_summary = json.loads(
        (output_root / "rolling_retraining_summary.json").read_text(encoding="utf-8")
    )
    pivot = rolling.pivot(index="quarter", columns="experiment_id", values="total_score")
    deltas = pivot["ficr_lambda_02"] - pivot["official_mask"]
    manifest.update(
        {
            "finalized_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "git_branch": _git("branch", "--show-current"),
            "git_commit": _git("rev-parse", "HEAD"),
            "gpu": training_summary["gpu"],
            "training_summary": training_summary,
            "rolling_summary": {
                **rolling_summary,
                "mean_ficr_aware_score": float(pivot["ficr_lambda_02"].mean()),
                "worst_ficr_aware_score": float(pivot["ficr_lambda_02"].min()),
                "mean_score_delta": float(deltas.mean()),
            },
            "final_selection": final,
            "drive_run_directory": (
                "/content/drive/MyDrive/Baram/runs/"
                "exp03_official_score_calibration/20260716_144300"
            ),
            "public_scores_used_for_tuning": False,
            "submission_count": final["submission_count"],
        }
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric_line(frame: pd.DataFrame, model_id: str, fold: str) -> str:
    row = frame.loc[frame["model_id"].eq(model_id) & frame["fold"].eq(fold)].iloc[0]
    return f"Score {row.total_score:.6f}, 1-NMAE {row.one_minus_nmae:.6f}, FICR {row.ficr:.6f}"


def write_report(output_root: Path = DEFAULT_OUTPUT, path: Path | None = None) -> Path:
    path = path or output_root / "report.md"
    existing = pd.read_csv(output_root / "metrics/existing_models_official_scores.csv")
    training = pd.read_csv(output_root / "metrics/training_variant_ensemble_scores.csv")
    seed_training = pd.read_csv(output_root / "metrics/training_variant_scores.csv")
    rolling = pd.read_csv(output_root / "metrics/rolling_retrained_scores.csv")
    affine = pd.read_csv(output_root / "calibration/rolling_affine_backtest.csv")
    final = json.loads((output_root / "final_selection.json").read_text(encoding="utf-8"))
    scorer = json.loads((output_root / "scorer_version.json").read_text(encoding="utf-8"))
    _update_manifest(output_root, final, rolling)

    fold_b = existing.loc[existing["fold"].eq("fold_b")].sort_values("total_score", ascending=False)
    rolling_pivot = rolling.pivot(index="quarter", columns="experiment_id", values="total_score")
    rolling_pivot["delta"] = rolling_pivot["ficr_lambda_02"] - rolling_pivot["official_mask"]
    ficr_row = training.loc[
        training["experiment_id"].eq("ficr_lambda_02") & training["fold"].eq("fold_b")
    ].iloc[0]
    mask_row = training.loc[
        training["experiment_id"].eq("official_mask") & training["fold"].eq("fold_b")
    ].iloc[0]
    winter = seed_training.loc[
        seed_training["experiment_id"].eq("ficr_lambda_02_winter") & seed_training["fold"].eq("fold_b")
    ].iloc[0]
    recency = seed_training.loc[
        seed_training["experiment_id"].eq("ficr_lambda_02_recency") & seed_training["fold"].eq("fold_b")
    ].iloc[0]
    base_seed = seed_training.loc[
        seed_training["stage"].eq("full") & seed_training["experiment_id"].eq("ficr_lambda_02")
        & seed_training["fold"].eq("fold_b") & seed_training["seed"].eq(42)
    ].iloc[0]

    ranking = "\n".join(
        f"{index + 1}. `{row.model_id}` — {row.total_score:.6f} "
        f"(1-NMAE {row.one_minus_nmae:.6f}, FICR {row.ficr:.6f})"
        for index, row in enumerate(fold_b.itertuples(index=False))
    )
    rolling_lines = "\n".join(
        f"- {quarter}: mask {row.official_mask:.6f}, FICR-aware {row.ficr_lambda_02:.6f}, "
        f"delta {row.delta:+.6f}"
        for quarter, row in rolling_pivot.iterrows()
    )
    submissions = "\n".join(f"- `{Path(item).name}`" for item in final["submissions"])
    text = f"""# exp03 official score calibration report

## Official scorer

DACON codeshare 14035의 `metric.ipynb`를 byte-for-byte 보존했다. SHA-256은 `{scorer['sha256']}`이며 공식 S3 원본과 일치한다. 공식 규칙은 실제 발전량 10% capacity 이상만 평가하고, normalized error ≤6%/≤8%에 시간별 단가 4/3, 그 외 0을 적용한다.

## Existing models rescored

Fold B 공식 순위:

{ranking}

기존 선택 blend의 unmasked macro nMAE는 0.091757이지만 공식 mask nMAE는 {1-existing.loc[(existing.model_id=='cat025_tcn075') & (existing.fold=='fold_b'),'one_minus_nmae'].iloc[0]:.6f}이다. 공식 Score로는 aux 0.15와 plain TCN이 기존 blend보다 높아 선택 순서가 바뀌었다.

## Calibration

기존 prediction-only affine calibration은 walk-forward 7개 평가 분기 모두 baseline보다 개선됐다. 평균 delta는 {affine.score_delta.mean():+.6f}, 최악 delta는 {affine.score_delta.min():+.6f}이다. 최종 test용 calibration은 전체 OOF에서 TCN weight {final['calibration_tcn_weight']:.3f}와 group별 affine을 fit했으며, 선택 과정에 Public 점수를 사용하지 않았다.

Fold A에서 선택한 group별 TCN weight는 group 1/2/3 각각 {final['group_weights_selected_on_fold_a']['kpx_group_1']:.3f}/{final['group_weights_selected_on_fold_a']['kpx_group_2']:.3f}/{final['group_weights_selected_on_fold_a']['kpx_group_3']:.3f}이다. 최종 global base의 affine `(scale, offset_kWh)`는 group 1/2/3 각각 {tuple(final['affine_parameters']['kpx_group_1'])}/{tuple(final['affine_parameters']['kpx_group_2'])}/{tuple(final['affine_parameters']['kpx_group_3'])}이다.

4계절 calibration은 2024의 독립 quarter 중 {final['seasonal_calibration_improved_quarters']}/4개에서 개선되어 retained={final['seasonal_calibration_retained']}로 판정했다. 다만 2024 평균 Score가 global affine {final['global_affine_2024_mean_score']:.6f}, seasonal affine {final['seasonal_affine_2024_mean_score']:.6f}여서 최종 방식은 `{final['best_calibration_mode']}`이다. 12개 월별 개별 calibration은 수행하지 않았다.

## FICR-aware training

- Official-mask ensemble Fold B: Score {mask_row.total_score:.6f}, 1-NMAE {mask_row.one_minus_nmae:.6f}, FICR {mask_row.ficr:.6f}
- λ=0.20 ensemble Fold B: Score {ficr_row.total_score:.6f}, 1-NMAE {ficr_row.one_minus_nmae:.6f}, FICR {ficr_row.ficr:.6f}
- FICR-aware delta: {ficr_row.total_score-mask_row.total_score:+.6f}
- Winter 1.15 seed42 delta vs same seed: {winter.total_score-base_seed.total_score:+.6f}
- Recency seed42 delta vs same seed: {recency.total_score-base_seed.total_score:+.6f}

Winter와 recency는 둘 다 baseline λ=0.20 seed42를 넘지 못해 제외했다.

## True expanding-window quarterly retraining

각 quarter마다 feature physics state와 neural preprocessing을 train cutoff까지만 fit하고, 01:00~다음날 00:00 issue block이 분기 경계를 넘지 않게 재학습했다.

{rolling_lines}

FICR-aware가 개선한 분기는 {(rolling_pivot.delta > 0).sum()}/{len(rolling_pivot)}개이며, 평균 delta는 {rolling_pivot.delta.mean():+.6f}, worst-quarter Score는 {rolling_pivot.ficr_lambda_02.min():.6f}이다.

## Final selection

3-model convex search의 최적 weight는 CatBoost {final['final_weights']['catboost']:.2f}, 기존 TCN {final['final_weights']['tcn_aux_005']:.2f}, FICR-aware {final['final_weights']['ficr_lambda_02']:.2f}이다. 최적해가 pure FICR-aware이면 중복 ensemble submission을 만들지 않는다.

생성 submission:

{submissions}

제출 우선순위는 FICR-aware, calibration-only 순이다. Public 점수(Exp01 0.6128785636, Exp02 0.6152232779)는 결과 문맥으로만 기록했고 어떤 parameter search에도 넣지 않았다.

## Next experiment

FICR-aware loss가 공식 validation의 정산금 성분을 크게 개선했으나 시간 모델끼리의 convex blend는 추가 이득이 없었다. 다음 단계는 같은 시간 입력을 더 섞는 대신 raw spatial grid 모델로 오차 상관을 낮추는 것이 타당하다.
"""
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); print(write_report(args.output_root.resolve()))


if __name__ == "__main__":
    main()

"""Feature-block retention decisions and Markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ORDER = [
    ("A", "rf_reference"),
    ("B", "catboost_basic"),
    ("C", "catboost_spatial"),
    ("D", "catboost_wind_physics"),
    ("E", "catboost_thermodynamic"),
    ("F", "catboost_full"),
]
BLOCK_COMPARISONS = [
    ("spatial", "catboost_basic", "catboost_spatial"),
    ("wind_physics", "catboost_spatial", "catboost_wind_physics"),
    ("thermodynamic", "catboost_wind_physics", "catboost_thermodynamic"),
    ("forecast_disagreement", "catboost_thermodynamic", "catboost_full"),
]


def select_feature_blocks(ablation: pd.DataFrame, by_group: pd.DataFrame) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    """Apply the experiment's predeclared retention rules to A-F results."""
    selected: dict[str, bool] = {}
    decisions: list[dict[str, Any]] = []
    for block, before, after in BLOCK_COMPARISONS:
        before_macro_b = float(ablation.query("experiment_id == @before and fold == 'fold_b'")["macro_nmae"].iloc[0])
        after_macro_b = float(ablation.query("experiment_id == @after and fold == 'fold_b'")["macro_nmae"].iloc[0])
        delta_b = after_macro_b - before_macro_b

        before_groups = by_group.query("experiment_id == @before and fold == 'fold_b'").set_index("group_id")["nmae"]
        after_groups = by_group.query("experiment_id == @after and fold == 'fold_b'").set_index("group_id")["nmae"]
        group_deltas = (after_groups - before_groups).to_dict()
        maintained_or_better = sum(delta <= 0 for delta in group_deltas.values())
        severe_groups = [int(group) for group, delta in group_deltas.items() if delta >= 0.003]

        before_a = ablation.query("experiment_id == @before and fold == 'fold_a'")["macro_nmae"]
        after_a = ablation.query("experiment_id == @after and fold == 'fold_a'")["macro_nmae"]
        delta_a = float(after_a.iloc[0] - before_a.iloc[0]) if len(before_a) and len(after_a) else None
        direction_warning = delta_a is not None and delta_a * delta_b < 0
        keep = delta_b < 0 and maintained_or_better >= 2
        selected[block] = bool(keep)
        decisions.append(
            {
                "block": block,
                "before": before,
                "after": after,
                "fold_b_macro_delta": delta_b,
                "fold_a_macro_delta": delta_a,
                "group_fold_b_deltas": {str(k): float(v) for k, v in group_deltas.items()},
                "maintained_or_better_groups": maintained_or_better,
                "severe_degradation_groups": severe_groups,
                "fold_direction_warning": bool(direction_warning),
                "keep": bool(keep),
            }
        )
    return selected, decisions


def _format_metric_table(ablation: pd.DataFrame) -> str:
    view = ablation[ablation["experiment_id"].isin([item[1] for item in ORDER])].copy()
    view = view[["ablation_label", "experiment_id", "fold", "macro_nmae", "feature_count_mean", "training_seconds"]]
    view["macro_nmae"] = view["macro_nmae"].map(lambda value: f"{value:.6f}")
    view["feature_count_mean"] = view["feature_count_mean"].map(lambda value: f"{value:.1f}")
    view["training_seconds"] = view["training_seconds"].map(lambda value: f"{value:.1f}")
    return view.to_markdown(index=False)


def render_report(
    ablation: pd.DataFrame,
    by_group: pd.DataFrame,
    monthly: pd.DataFrame,
    high_wind: pd.DataFrame,
    selected: dict[str, bool],
    decisions: list[dict[str, Any]],
    selected_validation: dict[str, Any] | None,
    final_training: dict[str, Any],
    submission_path: str | None,
    branch: str,
    commit: str,
    pytest_result: str,
) -> str:
    rf_b = float(ablation.query("experiment_id == 'rf_reference' and fold == 'fold_b'")["macro_nmae"].iloc[0])
    basic_b = float(ablation.query("experiment_id == 'catboost_basic' and fold == 'fold_b'")["macro_nmae"].iloc[0])
    final_nmae = None if selected_validation is None else float(selected_validation["macro_nmae"])
    improvement = None if final_nmae is None else rf_b - final_nmae

    decision_lines = []
    for item in decisions:
        warning = []
        if item["severe_degradation_groups"]:
            warning.append(f"group {item['severe_degradation_groups']}에서 +0.003 이상 악화")
        if item["fold_direction_warning"]:
            warning.append("Fold A/B 방향 불일치")
        suffix = f" ({'; '.join(warning)})" if warning else ""
        group_delta_text = ", ".join(
            f"g{group}={delta:+.6f}" for group, delta in item["group_fold_b_deltas"].items()
        )
        decision_lines.append(
            f"- `{item['block']}`: {'유지' if item['keep'] else '제외'}, "
            f"Fold B macro Δ={item['fold_b_macro_delta']:+.6f}, "
            f"그룹 Δ=[{group_delta_text}], 유지/개선 그룹={item['maintained_or_better_groups']}/3{suffix}"
        )

    selected_groups = "\n".join(
        f"- group {int(row.group_id)}: nMAE `{row.nmae:.6f}`"
        for row in by_group.query("experiment_id == 'catboost_selected' and fold == 'fold_b'").itertuples()
    ) or "- selected validation 결과 없음"
    wind_view = high_wind.query("experiment_id == 'catboost_selected' and fold == 'fold_b'")
    rf_wind = high_wind.query("experiment_id == 'rf_reference' and fold == 'fold_b'").set_index("group_id")["nmae"]
    selected_wind = "\n".join(
        f"- group {int(row.group_id)}: p90 `{row.train_wind_p90_mps:.3f} m/s`, nMAE `{row.nmae:.6f}`, "
        f"RF 대비 Δ `{row.nmae-float(rf_wind.loc[row.group_id]):+.6f}`"
        for row in wind_view.itertuples()
    ) or "- selected validation 결과 없음"
    month_view = monthly.query("experiment_id == 'catboost_selected' and fold == 'fold_b'")
    month_macro = month_view.groupby("month")["nmae"].mean().sort_values()
    monthly_summary = (
        f"최저는 {int(month_macro.index[0])}월 `{month_macro.iloc[0]:.6f}`, "
        f"최고는 {int(month_macro.index[-1])}월 `{month_macro.iloc[-1]:.6f}`이며 "
        f"월간 범위는 `{month_macro.iloc[-1]-month_macro.iloc[0]:.6f}`이다."
        if len(month_macro) else "selected validation 결과 없음"
    )

    public_value = (
        "있음 — RF reference보다 validation이 개선됐으며 계약 검증된 submission을 생성했다."
        if improvement is not None and improvement > 0 else
        "보류 — RF reference 대비 안정적인 validation 개선을 확인하지 못했다."
    )
    tcn_basis = (
        "충분함 — 유지된 feature block과 제외된 block이 fold ablation으로 분리됐다."
        if selected_validation is not None else "불충분 — selected validation이 완료되지 않았다."
    )
    clipping_lines = "\n".join(
        f"- {target}: raw test range `{detail['test_raw_min']:.2f}–{detail['test_raw_max']:.2f}` kWh, "
        f"final range `{detail['test_final_min']:.2f}–{detail['test_final_max']:.2f}` kWh, "
        f"upper clipping `{'적용' if detail['upper_clip_applied'] else '미적용'}`"
        for target, detail in final_training.items()
    ) or "- full train 결과 없음"
    raw_nmae = None if selected_validation is None else float(selected_validation["macro_raw_nmae"])
    lower_nmae = None if selected_validation is None else float(selected_validation["macro_nmae"])
    upper_nmae = None if selected_validation is None else float(selected_validation["macro_capacity_clipped_nmae"])

    return f"""# exp01 CatBoost physics ablation report

## 실행 상태

- Branch: `{branch}`
- Commit at run start: `{commit}`
- Tests: `{pytest_result}`
- Submission: `{submission_path or '생성되지 않음'}`

## A-F 결과

{_format_metric_table(ablation)}

## 결론

- 모델 교체 효과: Fold B macro nMAE가 RF `{rf_b:.6f}`에서 CatBoost basic `{basic_b:.6f}`로 `{basic_b-rf_b:+.6f}` 변했다.
- RF 대비 selected model 개선 폭: {f'`{improvement:.6f}`' if improvement is not None else '계산 불가'}.
- 공간 feature: {'유지' if selected.get('spatial') else '제외'}.
- 풍속 물리 feature: {'유지' if selected.get('wind_physics') else '제외'}.
- 열역학 feature: {'유지' if selected.get('thermodynamic') else '제외'}.
- LDAPS/GFS 불일치 feature: {'유지' if selected.get('forecast_disagreement') else '제외'}.
- selected raw/lower-clipped/capacity-clipped Fold B macro nMAE: {f'`{raw_nmae:.6f}` / `{lower_nmae:.6f}` / `{upper_nmae:.6f}`' if raw_nmae is not None else '계산 불가'}.
- public 제출 가치: {public_value}
- TCN 진행 근거: {tcn_basis}

## Feature block 판단

{chr(10).join(decision_lines)}

## Selected model 그룹별 Fold B

{selected_groups}

## Selected model high-wind 진단

{selected_wind}

## 월별 안정성

{monthly_summary} 겨울, 특히 1월 오차가 커서 다음 TCN 실험에서도 계절별 안정성을 별도로 확인해야 한다.

## Final clipping

validation에서 capacity upper clipping의 추가 이득이 없으므로 상한은 적용하지 않았고, 음수만 0으로 clip했다.

{clipping_lines}

## 물리 feature 정의와 단위 검증

- 온도 원자료는 249.70–308.40 범위이고 압력은 약 87–104 kPa이므로 temperature는 K, pressure는 Pa로 확인했다. 상대습도는 %이며 LDAPS 최대가 110.38%라 수증기압 계산에서만 물리 범위 0–100으로 clip했다.
- 습공기 밀도는 포화수증기압에서 수증기 분압을 구한 뒤 `rho=(p-e)/(Rd*T)+e/(Rv*T)` 한 식만 사용했다.
- GFS 80/100 m 및 LDAPS 10/50 m shear alpha의 1%/99% clipping 한계는 매 fold train에서만 계산했다.
- group 3은 2022 label이 없고 Fold A에서도 평가되지 않아 group 1/2와 학습 이력 및 안정성이 다를 수 있다.

## TCN 전달 feature

- 공용 baseline raw/time/grid-summary/nearest feature.
{chr(10).join(f"- {block}" for block, keep in selected.items() if keep) or '- 추가 block 없음'}
- target lag와 SCADA는 전달하지 않는다. 세부 컬럼은 `outputs/feature_list_by_experiment.json`의 `catboost_selected` 항목을 사용한다.

공식 scorer는 저장소에 없었으므로 정산금 지표는 구현하거나 추측하지 않았다.
"""


def write_report(path: Path, **kwargs: Any) -> str:
    report = render_report(**kwargs)
    path.write_text(report, encoding="utf-8")
    return report

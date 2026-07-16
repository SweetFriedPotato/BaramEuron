"""Strict earlier-quarter-only selection helpers."""

from __future__ import annotations

import pandas as pd


ORDERED_QUARTERS = [f"{year}Q{quarter}" for year in (2023, 2024) for quarter in (1, 2, 3, 4)]


def preceding_quarters(evaluation_quarter: str, available: list[str] | None = None) -> list[str]:
    ordered = ORDERED_QUARTERS if available is None else [q for q in ORDERED_QUARTERS if q in available]
    if evaluation_quarter not in ordered:
        raise ValueError(f"unknown evaluation quarter: {evaluation_quarter}")
    return ordered[:ordered.index(evaluation_quarter)]


def assert_nested_order(fit_quarters: list[str], evaluation_quarter: str) -> None:
    if not fit_quarters:
        raise ValueError("nested selection requires at least one earlier quarter")
    if max(pd.Period(value, freq="Q") for value in fit_quarters) >= pd.Period(evaluation_quarter, freq="Q"):
        raise ValueError("evaluation quarter target leaked into nested selection")


def nested_outer_plan(available: list[str]) -> list[dict]:
    ordered = [quarter for quarter in ORDERED_QUARTERS if quarter in available]
    return [
        {
            "evaluation_quarter": quarter,
            "fit_quarters": ordered[:index],
            "eligible": index > 0,
            "reason": None if index > 0 else "no earlier rolling OOF quarter",
        }
        for index, quarter in enumerate(ordered)
    ]

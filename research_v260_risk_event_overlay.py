from __future__ import annotations

import numpy as np
import pandas as pd


RULE = {
    "stk_shock": "observe only; no trading action",
    "stk_high_shock": "new-entry score tiebreak on the first visible session only",
    "stk_alert": "observe only; no trading action",
    "held_position_exit": "unchanged; an event never forces a sale",
}


EVENT_COMPONENTS = ("ordinary_1d", "severe_1d", "severe_5d", "alert_active")
LEGACY_BLOCK_COMPONENTS = ("ordinary_1d", "severe_5d", "alert_active")


def validate_frozen_contract(contract: dict) -> None:
    rules = contract.get("rules", {})
    expected = {
        "ordinary_abnormal_volatility": {
            "action": "observe_only",
            "forced_exit": False,
        },
        "severe_abnormal_volatility": {
            "action": "new_entry_score_tiebreak",
            "event_window_sessions": 1,
            "margin_source": "base_candidate.replacement_advantage",
            "changes_eligibility": False,
            "forced_exit": False,
        },
        "exchange_focus_security": {
            "action": "observe_only",
            "forced_exit": False,
        },
    }
    actual = {name: rules.get(name) for name in expected}
    if actual != expected:
        raise ValueError("frozen risk-event contract does not match shared implementation")
    if rules.get("missing_or_unknown_event_state") != "fail_closed_for_diagnostic_input":
        raise ValueError("risk-event missing-state rule drifted")
    coverage = contract.get("diagnostic_coverage", {})
    expected_coverage = {
        "source_event_min_dates": {
            "stk_shock": "20260303",
            "stk_high_shock": "20260209",
            "stk_alert": "20260210",
        },
        "ordinary_first_visible_session": "20260304",
        "severe_first_visible_session": "20260210",
        "exchange_alert_first_visible_session": "20260211",
        "event_metric_start": "20260210",
        "precoverage_state": "unknown_not_zero",
    }
    if coverage != expected_coverage:
        raise ValueError("risk-event diagnostic coverage drifted")


def build_event_masks(
    dates: np.ndarray,
    stocks: np.ndarray,
    features: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict]:
    required = {
        "signal_date",
        "stock_code",
        "shock_count_1d",
        "high_shock_count_1d",
        "high_shock_count_5d",
        "alert_active_count",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"missing risk-event fields: {missing}")
    if features.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("risk-event feature keys must be unique")

    date_index = {str(value): index for index, value in enumerate(dates)}
    stock_index = {str(value): index for index, value in enumerate(stocks)}
    shape = (len(dates), len(stocks))
    masks = {
        name: np.zeros(shape, dtype=np.bool_)
        for name in EVENT_COMPONENTS
    }
    matched = 0
    for row in features[
        [
            "signal_date",
            "stock_code",
            "shock_count_1d",
            "high_shock_count_1d",
            "high_shock_count_5d",
            "alert_active_count",
        ]
    ].itertuples(index=False):
        d_idx = date_index.get(str(row.signal_date))
        s_idx = stock_index.get(str(row.stock_code))
        if d_idx is None or s_idx is None:
            continue
        try:
            ordinary_count = int(row.shock_count_1d)
            severe_1d_count = int(row.high_shock_count_1d)
            severe_count = int(row.high_shock_count_5d)
            alert_count = int(row.alert_active_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("risk-event counts must be finite integers") from exc
        if min(ordinary_count, severe_1d_count, severe_count, alert_count) < 0:
            raise ValueError("risk-event counts cannot be negative")
        masks["ordinary_1d"][d_idx, s_idx] = ordinary_count > 0
        masks["severe_1d"][d_idx, s_idx] = severe_1d_count > 0
        masks["severe_5d"][d_idx, s_idx] = severe_count > 0
        masks["alert_active"][d_idx, s_idx] = alert_count > 0
        matched += 1
    return masks, {
        "rule": RULE,
        "source_rows": int(len(features)),
        "matched_context_rows": int(matched),
        "ordinary_1d_stock_dates": int(masks["ordinary_1d"].sum()),
        "severe_1d_stock_dates": int(masks["severe_1d"].sum()),
        "severe_5d_stock_dates": int(masks["severe_5d"].sum()),
        "alert_active_stock_dates": int(masks["alert_active"].sum()),
    }


def compose_overlay_masks(
    components: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    missing = sorted(set(LEGACY_BLOCK_COMPONENTS) - set(components))
    if missing:
        raise ValueError(f"missing risk-event components: {missing}")
    arrays = {
        name: np.asarray(components[name], dtype=np.bool_)
        for name in LEGACY_BLOCK_COMPONENTS
    }
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("risk-event component masks do not align")
    entry = arrays["ordinary_1d"] | arrays["severe_5d"] | arrays["alert_active"]
    maintenance = arrays["severe_5d"] | arrays["alert_active"]
    return entry, maintenance


def score_tiebreak_order(
    score: np.ndarray,
    base_order: np.ndarray,
    components: dict[str, np.ndarray],
    margin: float,
) -> np.ndarray:
    if "severe_1d" not in components:
        raise ValueError("missing severe_1d risk-event component")
    values = np.asarray(score, dtype=np.float64)
    order = np.asarray(base_order)
    events = np.asarray(components["severe_1d"], dtype=np.bool_)
    if values.shape != order.shape or values.shape != events.shape:
        raise ValueError("score, order and severe-event mask must align")
    if values.ndim != 2:
        raise ValueError("event tiebreak requires two-dimensional daily rankings")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("event tiebreak margin must be finite and non-negative")

    result = np.empty_like(order)
    for row_index in range(order.shape[0]):
        adjusted = values[row_index] - float(margin) * events[row_index]
        original = [int(index) for index in order[row_index]]
        result[row_index] = np.asarray(
            sorted(
                original,
                key=lambda index: (
                    -round(float(adjusted[index]), 12)
                    if np.isfinite(adjusted[index])
                    else np.inf,
                    bool(events[row_index, index]),
                ),
            ),
            dtype=order.dtype,
        )
    return result


def build_block_matrix(
    dates: np.ndarray,
    stocks: np.ndarray,
    features: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    components, audit = build_event_masks(dates, stocks, features)
    block = components["severe_5d"].copy()
    return block, {
        "rule": RULE,
        "component_stock_dates": {
            "ordinary_shock_1d": audit["ordinary_1d_stock_dates"],
            "severe_shock_5d": audit["severe_5d_stock_dates"],
            "exchange_alert_active": audit["alert_active_stock_dates"],
        },
        "blocked_stock_dates": int(block.sum()),
        "blocked_signal_dates": int(np.any(block, axis=1).sum()),
    }


def entry_ranked_without_events(ranked: list[int], blocked_row: np.ndarray) -> list[int]:
    return [idx for idx in ranked if not bool(blocked_row[idx])]


def maintenance_buy_allowed(index: int, blocked_row: np.ndarray) -> bool:
    return not bool(blocked_row[index])

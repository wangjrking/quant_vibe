from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as drawdown
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_weak2_drawdown_position_attribution_20260822 as source
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_drawdown_position_age_attribution_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def age_band(age_sessions: int) -> str:
    age = int(age_sessions)
    if age < 0:
        raise ValueError("position age cannot be negative")
    if age <= 4:
        return "age_1_4"
    if age <= 9:
        return "age_5_9"
    if age <= 19:
        return "age_10_19"
    return "age_20_plus"


def reconstruct_mark_ages(observations: list[dict], actions: pd.DataFrame) -> list[dict]:
    dates = [str(item["buy_date"]) for item in observations]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError("observation dates must be unique and ordered")
    date_index = {value: index for index, value in enumerate(dates)}
    action_groups = {
        str(buy_date): group
        for buy_date, group in actions.groupby("buy_date", sort=False)
    }
    positions: set[str] = set()
    entries: dict[str, str] = {}
    rows: list[dict] = []

    for observation in observations:
        buy_date = str(observation["buy_date"])
        marks = observation.get("mark_records", [])
        mark_stocks = {str(item["stock_code"]) for item in marks}
        if mark_stocks != positions:
            raise RuntimeError("mark records do not match reconstructed pre-trade positions")
        for mark in marks:
            stock_code = str(mark["stock_code"])
            entry_date = entries.get(stock_code)
            if entry_date is None:
                raise RuntimeError("marked position has no reconstructed entry date")
            age_sessions = date_index[buy_date] - date_index[entry_date]
            rows.append(
                {
                    "buy_date": buy_date,
                    "stock_code": stock_code,
                    "entry_date": entry_date,
                    "age_sessions": int(age_sessions),
                    "age_band": age_band(age_sessions),
                    "mark_pnl": float(mark["mark_pnl"]),
                    "current_open_available": bool(mark["current_open_available"]),
                }
            )

        group = action_groups.get(buy_date)
        if group is not None:
            for action in group.itertuples(index=False):
                stock_code = str(action.stock_code)
                if str(action.action) == "SELL":
                    if np.isclose(float(action.target_pct), 0.0, rtol=0.0, atol=1e-12):
                        positions.discard(stock_code)
                        entries.pop(stock_code, None)
                elif str(action.action) == "BUY":
                    if stock_code not in positions:
                        positions.add(stock_code)
                        entries[stock_code] = buy_date
                else:
                    raise ValueError(f"unsupported action: {action.action}")
        post_positions = set(observation.get("positions_after_trades", {}))
        if post_positions != positions:
            raise RuntimeError("actions do not reconstruct post-trade positions")
    return rows


def summarize_episode(rows: list[dict], peak_date: str, trough_date: str) -> dict:
    selected = [
        row
        for row in rows
        if str(peak_date) < str(row["buy_date"]) <= str(trough_date)
    ]
    if not selected:
        raise ValueError("drawdown episode has no position marks")
    result = {}
    gross_negative = sum(min(float(row["mark_pnl"]), 0.0) for row in selected)
    for band in ("age_1_4", "age_5_9", "age_10_19", "age_20_plus"):
        subset = [row for row in selected if row["age_band"] == band]
        negative = sum(min(float(row["mark_pnl"]), 0.0) for row in subset)
        positive = sum(max(float(row["mark_pnl"]), 0.0) for row in subset)
        result[band] = {
            "position_days": int(len(subset)),
            "unique_stocks": int(len({row["stock_code"] for row in subset})),
            "net_mark_pnl": float(negative + positive),
            "gross_negative_mark_pnl": float(negative),
            "gross_positive_mark_pnl": float(positive),
            "share_of_episode_gross_negative": (
                float(negative / gross_negative) if gross_negative < 0 else 0.0
            ),
            "negative_position_day_ratio": (
                float(sum(float(row["mark_pnl"]) < 0 for row in subset) / len(subset))
                if subset
                else 0.0
            ),
        }
    return {
        "peak_date_exclusive": str(peak_date),
        "trough_date_inclusive": str(trough_date),
        "position_days": int(len(selected)),
        "unique_stocks": int(len({row["stock_code"] for row in selected})),
        "gross_negative_mark_pnl": float(gross_negative),
        "age_bands": result,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is unavailable for position-age attribution")
    policy = checkpoint["selected_policy"]
    context = harness.load_context(policy)
    daily, actions, observations, _, _, _ = source.run_with_observer(context, policy)
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(metrics[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
            "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("position-age attribution baseline drifted")
    episode = drawdown.maximum_drawdown_episode(daily)
    rows = reconstruct_mark_ages(observations, actions)
    summary = summarize_episode(rows, episode["peak_date"], episode["trough_date"])
    repeated_rows = reconstruct_mark_ages(observations, actions)
    deterministic = rows == repeated_rows
    if not deterministic:
        raise RuntimeError("position-age attribution replay failed")

    dominant_share = max(
        value["share_of_episode_gross_negative"]
        for value in summary["age_bands"].values()
    )
    diagnosis = (
        "age_concentrated_drawdown"
        if dominant_share >= 0.60
        else "drawdown_spans_multiple_position_ages"
    )
    result = {
        "status": "position_age_attribution_complete_2026_not_opened",
        "maximum_drawdown_episode": episode,
        "position_age_attribution": summary,
        "diagnosis": diagnosis,
        "rule_implication": (
            "do not add a position-age hard exit from this evidence"
            if diagnosis == "drawdown_spans_multiple_position_ages"
            else "age concentration requires a separate robustness test before any rule change"
        ),
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "position_age_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_rank_sizing_residual_cash_sweep_20260823 as sweep
import research_v260_lowrisk_score_tuning_20260821 as research_base


REPORTS = REPO / "quant/data_file/reports"
SOURCE_CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
EQUALWEIGHT_CHECKPOINT = sizing.CHECKPOINT
OUTPUT_ROOT = (
    REPORTS
    / "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823"
)
COST_LEVELS = (0.0030, 0.0040, 0.0065)
START_OFFSETS = (0, 5, 20, 60)
CANDIDATE_ID = "fixed10_global_rank11_to_9_entry_buy_day_cash_sweep"


def action_integrity(actions) -> dict:
    keys = ["signal_date", "buy_date", "action", "stock_code"]
    return {
        "unique_action_keys": bool(not actions.duplicated(keys).any()),
        "valid_action_types": bool(
            actions["action"].isin({"BUY", "SELL"}).all()
        ),
        "no_bj": bool(
            not actions["stock_code"].astype(str).str.endswith(".BJ").any()
        ),
        "finite_execution_open": bool(
            np.isfinite(actions["execution_open_raw"].to_numpy(dtype=float)).all()
        ),
    }


def main() -> None:
    source = json.loads(SOURCE_CHECKPOINT.read_text(encoding="utf-8"))
    equalweight_source = json.loads(
        EQUALWEIGHT_CHECKPOINT.read_text(encoding="utf-8")
    )
    if (
        source.get("validation_2026_opened") is not False
        or equalweight_source.get("validation_2026_opened") is not False
    ):
        raise PermissionError("2026 entered residual-cash checkpoint")
    policy = copy.deepcopy(source["selected_policy"])
    policy["residual_cash_sweep"] = {
        "enabled": True,
        "trigger": "after_buy_trade",
        "recipient": "most_underweight_existing_buyable_holding",
        "target_pct": "gross_target_divided_by_max_positions",
        "board_lot_shares": 100,
        "max_orders_per_trade_day": None,
        "new_names_allowed": False,
        "forced_sales_allowed": False,
        "equalweight_and_full_investment_are_soft_directions": True,
    }
    context = sizing.width_tools.harness.load_context(policy)
    if context.access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("pre-2026 checkpoint boundary drift")

    cost_metrics, frames = {}, {}
    for cost in COST_LEVELS:
        daily, actions = sweep.run_with_runtime(
            context,
            policy,
            cost,
            sweep=True,
            rank_sizing=True,
            sweep_trigger="buy_trade",
        )
        key = f"{cost:.4f}"
        cost_metrics[key] = sizing.evaluate(daily, actions)
        frames[key] = (daily, actions)

    daily, actions = frames["0.0030"]
    start_offset_metrics = {
        str(offset): round1.evaluate_run(
            daily,
            actions,
            research_base.FIRST_BUY
            if offset == 0
            else robustness.window_start_date(context.arrays, offset),
            round1.DEVELOPMENT_END,
            target_positions=sizing.TARGET_POSITIONS,
        )
        for offset in START_OFFSETS
    }
    repeat_daily, repeat_actions = sweep.run_with_runtime(
        context,
        policy,
        sizing.BASELINE_COST,
        sweep=True,
        rank_sizing=True,
        sweep_trigger="buy_trade",
    )
    deterministic = {
        "daily": round1.frame_hash(daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(actions) == round1.frame_hash(repeat_actions),
    }
    integrity = action_integrity(actions)
    integrity.update(
        {
            "pre2026_boundary": context.access["logical_max_date"]
            == round1.DEVELOPMENT_END,
            "finite_metrics": all(
                np.isfinite(cost_metrics["0.0030"][key])
                for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown")
            ),
            "deterministic": all(deterministic.values()),
            "cash_nonnegative": bool((daily["invested_ratio"] <= 1.000001).all()),
        }
    )
    if not all(integrity.values()):
        raise RuntimeError("residual-cash checkpoint execution integrity failed")

    candidate = cost_metrics["0.0030"]
    equalweight = equalweight_source["current_best_equalweight"]
    prior_rank = source["metrics_0_30pct"]
    result = {
        "status": "residual_cash_sweep_profit_candidate_frozen_pre2026_validation_not_opened",
        "source_strategy": source["source_strategy"],
        "selected_candidate": CANDIDATE_ID,
        "selected_policy": policy,
        "selection_reason": (
            "spending board-lot residual cash on existing buyable underweight holdings "
            "only when a buy already occurred raises pre-2026 net cumulative return "
            "at the primary 0.30% cost across all tested capital scales and slightly "
            "reduces turnover; higher-cost sensitivity is disclosed, while equalweight "
            "and full investment remain soft economic directions"
        ),
        "metrics_0_30pct": candidate,
        "cost_metrics": cost_metrics,
        "start_offset_metrics": start_offset_metrics,
        "candidate_minus_prior_rank_candidate": sizing.checkpoint_tools.metric_delta(
            candidate, prior_rank
        ),
        "candidate_minus_equalweight": sizing.checkpoint_tools.metric_delta(
            candidate, equalweight
        ),
        "hard_execution_integrity_only": integrity,
        "profit_and_risk_disclosures_are_not_integrity_gates": True,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "pre2026_checkpoint.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate": CANDIDATE_ID,
                "cumulative_return": candidate["cumulative_return"],
                "cagr": candidate["cagr"],
                "sharpe": candidate["sharpe"],
                "average_invested_ratio": candidate["average_invested_ratio"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

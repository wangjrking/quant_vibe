from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_current_weak_pressure_age_boundary_20260822 as age_boundary
import research_v260_fixed10_entry_beta_tail_guard_20260822 as percentile
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_pressure_regime_trigger_20260822 as trigger
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_risk_event_overlay import (
    build_event_masks,
    compose_overlay_masks,
)


RISK_SIGNAL_DB = (
    REPO / "quant/data_file/production_assets/duckdb/l2_stock_risk_signal.duckdb"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_tushare_risk_event_entry_gate_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
COST_LEVELS = (0.0030, 0.0040, 0.0065)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pre2026_features(min_date: str):
    connection = duckdb.connect(str(RISK_SIGNAL_DB), read_only=True)
    try:
        columns = {
            row[1] for row in connection.execute(
                "PRAGMA table_info('stock_risk_signal')"
            ).fetchall()
        }
        required = {
            "signal_date", "stock_code", "shock_count_1d",
            "high_shock_count_5d", "alert_active_count",
        }
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"risk-event L2 schema missing: {missing}")
        frame = connection.execute(
            """
            SELECT signal_date, stock_code, shock_count_1d,
                   high_shock_count_5d, alert_active_count
            FROM stock_risk_signal
            WHERE signal_date >= ? AND signal_date <= ?
            ORDER BY signal_date, stock_code
            """,
            [str(min_date), "20251231"],
        ).df()
    finally:
        connection.close()
    if frame.duplicated(["signal_date", "stock_code"]).any():
        raise RuntimeError("risk-event L2 keys are not unique")
    if frame["stock_code"].astype(str).str.endswith(".BJ").any():
        raise RuntimeError("risk-event L2 includes Beijing exchange rows")
    return frame


def event_masks(dates: np.ndarray, stocks: np.ndarray, frame) -> tuple[dict, dict]:
    masks, audit = build_event_masks(dates, stocks, frame)
    audit["source_rows_pre2026"] = audit.pop("source_rows")
    return masks, audit


def run(
    context,
    policy: dict,
    event_block,
    priority_matrix,
    cost: float,
    maintenance_event_block=None,
):
    strong = regime.strong_market_mask(context.score, context.protocol)
    threshold = float(policy["maintenance_topup_requires_score"])
    maintenance_block = (
        ~np.isfinite(context.score)
        | (context.score < threshold)
        | np.asarray(
            event_block
            if maintenance_event_block is None
            else maintenance_event_block,
            dtype=np.bool_,
        )
    )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        age_boundary.pressure_age_schedule(strong, 10),
        cost,
        entry_block_mask_override=event_block,
        maintenance_buy_block_mask_override=maintenance_block,
        pressure_trigger_override=trigger.pressure_trigger_schedule(strong),
        score_sell_priority_override=priority_matrix,
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered Tushare risk-event development")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    strong = regime.strong_market_mask(context.score, context.protocol)
    weak2 = confirmed.confirmed_weak_mask(strong, 2)
    volatility = defensive.trailing_log_volatility(
        context.arrays["close_qfq"], defensive.VOLATILITY_WINDOW
    )
    volatility_rank = percentile.cross_sectional_percent_rank(volatility)
    priority_matrix = priority.sell_priority_matrix(
        context.score, volatility_rank, ~weak2, 0.05
    )
    frame = load_pre2026_features(str(context.arrays["dates"][0]))
    component, risk_audit = event_masks(
        context.arrays["dates"], context.arrays["stocks"], frame
    )
    empty = np.zeros(context.score.shape, dtype=np.bool_)
    layered_entry, layered_maintenance = compose_overlay_masks(component)
    masks = {
        "risk_event_gate_off": (empty, empty),
        "control_ordinary_1d": (component["ordinary_1d"], empty),
        "control_severe_5d": (component["severe_5d"], component["severe_5d"]),
        "control_alert_active": (component["alert_active"], component["alert_active"]),
        "candidate_layered_gate": (layered_entry, layered_maintenance),
    }
    results, cache = {}, {}
    for candidate_id, (event_block, maintenance_event_block) in masks.items():
        cost_metrics = {}
        for cost in COST_LEVELS:
            daily, actions = run(
                context,
                policy,
                event_block,
                priority_matrix,
                cost,
                maintenance_event_block=maintenance_event_block,
            )
            cost_metrics[f"{cost:.4f}"] = round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            )
            if cost == round1.BASELINE_COST:
                cache[candidate_id] = (daily, actions)
        daily, actions = cache[candidate_id]
        results[candidate_id] = {
            "blocked_stock_dates": int(event_block.sum()),
            "blocked_signal_dates": int(np.any(event_block, axis=1).sum()),
            "cost_metrics": cost_metrics,
            "train_2022_2024": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, "20241231"
            ),
            "holdout_2025": round1.evaluate_run(
                daily, actions, "20250102", round1.DEVELOPMENT_END
            ),
            "start_offset_metrics": {
                str(offset): round1.evaluate_run(
                    daily,
                    actions,
                    research_base.FIRST_BUY
                    if offset == 0
                    else robustness.window_start_date(context.arrays, offset),
                    round1.DEVELOPMENT_END,
                )
                for offset in (0, 5, 20, 60)
            },
        }

    baseline_id = "risk_event_gate_off"
    candidate_id = "candidate_layered_gate"
    baseline, candidate = results[baseline_id], results[candidate_id]
    expected = checkpoint["current_best_equalweight"]
    equivalent = {
        key: bool(np.isclose(
            baseline["cost_metrics"]["0.0030"][key], expected[key],
            rtol=0.0, atol=1e-12,
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalent.values()):
        raise RuntimeError("Tushare risk-event baseline drifted")
    current = candidate["cost_metrics"]["0.0030"]
    reference = baseline["cost_metrics"]["0.0030"]
    gates = {
        "training_cagr_improved": candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"],
        "training_sharpe_improved": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "full_period_cumulative_not_worse": current["cumulative_return"]
        >= reference["cumulative_return"],
        "full_period_sharpe_not_worse": current["sharpe"] >= reference["sharpe"],
        "drawdown_not_worse": current["max_drawdown"] <= reference["max_drawdown"],
        "forward_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["cost_metrics"]["0.0065"]["cumulative_return"]
        >= baseline["cost_metrics"]["0.0065"]["cumulative_return"],
        "turnover_not_worse": current["turnover_annualized"]
        <= reference["turnover_annualized"],
        "all_start_offsets_profitable": min(
            item["cumulative_return"]
            for item in candidate["start_offset_metrics"].values()
        ) > 0.0,
        "all_years_positive": min(current["annual_returns"].values()) > 0.0,
        "exactly10_full_period": current["full_10_position_ratio"] == 1.0,
    }
    selected = candidate_id if all(gates.values()) else baseline_id
    repeat_daily, repeat_actions = run(
        context,
        policy,
        masks[selected][0],
        priority_matrix,
        round1.BASELINE_COST,
        maintenance_event_block=masks[selected][1],
    )
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0]) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("Tushare risk-event replay failed")
    result = {
        "status": "candidate_accepted_pre2026_validation_not_opened"
        if selected == candidate_id
        else "candidate_rejected_pre2026_validation_not_opened",
        "primary_rule": {
            "ordinary_abnormal_volatility": "pause new entry for 1 session",
            "severe_abnormal_volatility": "pause new entry/top-up for 5 sessions",
            "exchange_focus_security": "pause new entry/top-up while officially active",
            "existing_holdings": "unchanged original score-based exit",
        },
        "results": results,
        "candidate_gates": gates,
        "selected_candidate": selected,
        "selected_policy": {
            **policy,
            "tushare_risk_event_entry_gate": None
            if selected == baseline_id else "layered_1d_5d_alert_active",
        },
        "risk_asset": {
            "path": str(RISK_SIGNAL_DB),
            "table": "stock_risk_signal",
            "sha256": sha256_file(RISK_SIGNAL_DB),
            **risk_audit,
            "query_max_date": "20251231",
        },
        "checkpoint_equivalence": equivalent,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

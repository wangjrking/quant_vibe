from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_drawdown_position_attribution_20260822"
)


def observation_hash(observations: list[dict]) -> str:
    payload = json.dumps(
        observations, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def aggregate_episode(
    observations: list[dict],
    peak_date: str,
    trough_date: str,
) -> dict:
    episode = [
        item
        for item in observations
        if peak_date < str(item["buy_date"]) <= trough_date
    ]
    if not episode:
        raise ValueError("drawdown episode has no observations")
    mark_rows = [
        {"buy_date": item["buy_date"], **record}
        for item in episode
        for record in item["mark_records"]
    ]
    frame = pd.DataFrame(mark_rows)
    if frame.empty:
        raise ValueError("drawdown episode has no held-position marks")
    by_stock = (
        frame.groupby("stock_code", as_index=False)
        .agg(
            mark_pnl=("mark_pnl", "sum"),
            holding_days=("buy_date", "count"),
            negative_days=("mark_pnl", lambda values: int((values < 0).sum())),
            positive_days=("mark_pnl", lambda values: int((values > 0).sum())),
        )
        .sort_values(["mark_pnl", "stock_code"], ascending=[True, True])
    )
    by_stock["loss_amount"] = (-by_stock["mark_pnl"]).clip(lower=0.0)
    negative_loss = float(by_stock["loss_amount"].sum())
    top_losses = by_stock[by_stock["loss_amount"] > 0].head(10).copy()

    def concentration(count: int) -> float:
        if negative_loss <= 0:
            return 0.0
        return float(top_losses.head(count)["loss_amount"].sum() / negative_loss)

    mark_pnl = float(frame["mark_pnl"].sum())
    mark_equity_change = float(
        sum(item["equity_before_trades"] - item["previous_equity"] for item in episode)
    )
    trading_effect = float(
        sum(
            item["equity_after_trades"] - item["equity_before_trades"]
            for item in episode
        )
    )
    total_equity_change = float(
        episode[-1]["equity_after_trades"] - episode[0]["previous_equity"]
    )
    top3_share = concentration(3)
    negative_stock_count = int((by_stock["mark_pnl"] < 0).sum())
    classification = (
        "broad_based_full_investment_drawdown"
        if top3_share < 0.50 and negative_stock_count >= 10
        else "concentrated_position_drawdown"
    )
    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "observation_days": len(episode),
        "held_position_mark_rows": len(frame),
        "unique_held_stocks": int(frame["stock_code"].nunique()),
        "negative_stock_count": negative_stock_count,
        "positive_stock_count": int((by_stock["mark_pnl"] > 0).sum()),
        "zero_stock_count": int((by_stock["mark_pnl"] == 0).sum()),
        "gross_negative_mark_pnl": -negative_loss,
        "gross_positive_mark_pnl": float(
            by_stock.loc[by_stock["mark_pnl"] > 0, "mark_pnl"].sum()
        ),
        "net_mark_pnl": mark_pnl,
        "mark_equity_change": mark_equity_change,
        "mark_accounting_difference": mark_equity_change - mark_pnl,
        "trading_and_cost_effect": trading_effect,
        "total_equity_change": total_equity_change,
        "top1_negative_concentration": concentration(1),
        "top3_negative_concentration": top3_share,
        "top5_negative_concentration": concentration(5),
        "classification": classification,
        "worst_stock_mark_contributors": [
            {
                "stock_code": str(row.stock_code),
                "mark_pnl": float(row.mark_pnl),
                "holding_days": int(row.holding_days),
                "negative_days": int(row.negative_days),
                "positive_days": int(row.positive_days),
                "share_of_gross_negative": (
                    float(row.loss_amount / negative_loss)
                    if negative_loss > 0
                    else 0.0
                ),
            }
            for row in top_losses.itertuples(index=False)
        ],
    }


def run_with_observer() -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = checkpoint["selected_policy"]
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered position attribution")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    observations: list[dict] = []
    daily, actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=np.zeros(score.shape, dtype=np.bool_),
        maintenance_buy_block_mask_override=quality.maintenance_quality_block(
            score, True
        ),
        position_observer=observations.append,
    )
    return daily, actions, observations, {
        "checkpoint": checkpoint,
        "rules": rules,
        "manifests": manifests,
        "access": access,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    daily, actions, observations, context = run_with_observer()
    episode = attribution.maximum_drawdown_episode(daily)
    aggregate = aggregate_episode(
        observations, episode["peak_date"], episode["trough_date"]
    )
    expected = context["checkpoint"]["current_best_equalweight"]
    actual = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    checkpoint_equivalence = {
        key: bool(np.isclose(actual[key], expected[key], rtol=0.0, atol=1e-12))
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
        raise RuntimeError("position observer changed checkpoint behavior")
    if abs(aggregate["mark_accounting_difference"]) > 1e-6:
        raise RuntimeError("position mark attribution does not reconcile")

    _, _, repeated_observations, _ = run_with_observer()
    deterministic = observation_hash(observations) == observation_hash(
        repeated_observations
    )
    if not deterministic:
        raise RuntimeError("position attribution replay failed")

    result = {
        "status": "pre2026_diagnostic_2026_not_opened",
        "source_strategy": context["rules"]["strategy_id"],
        "candidate_policy": context["checkpoint"]["selected_policy"],
        "maximum_drawdown_episode": episode,
        "position_attribution": aggregate,
        "interpretation": (
            "A broad-based classification favors improving cross-sectional defense "
            "while staying fully invested; a concentrated classification favors a "
            "position-level holding rule."
        ),
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "observation_hash": observation_hash(observations),
        "data_access": context["access"],
        "source_manifests": context["manifests"],
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "drawdown_position_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

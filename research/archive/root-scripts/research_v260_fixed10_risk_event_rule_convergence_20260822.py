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

import research_v260_fixed10_abnormal_announcement_proxy_20260822 as proxy
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_recheck_harness_20260822 as pressure_harness
import research_v260_fixed10_severe_event_temporal_robustness_20260822 as robustness
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_risk_event_rule_convergence_20260822"
)
CHECKPOINT_PATH = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_position_pressure_exit_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
CASES = (
    "risk_event_off",
    "severe_pause5",
    "severe_pause5_warning_quality",
    "severe_pause5_ordinary_warning_quality",
)


def compose_entry_masks(
    score: np.ndarray,
    ordinary: np.ndarray,
    severe: np.ndarray,
    warning: np.ndarray,
    quality_threshold: float,
) -> dict[str, np.ndarray]:
    arrays = [
        np.asarray(value, dtype=np.bool_)
        for value in (ordinary, severe, warning)
    ]
    score_array = np.asarray(score, dtype=float)
    if any(value.shape != score_array.shape for value in arrays):
        raise ValueError("event masks must align with score matrix")
    if not np.isfinite(quality_threshold):
        raise ValueError("quality threshold must be finite")
    weak_score = ~np.isfinite(score_array) | (score_array < float(quality_threshold))
    ordinary_mask, severe_mask, warning_mask = arrays
    empty = np.zeros(score_array.shape, dtype=np.bool_)
    return {
        "risk_event_off": empty,
        "severe_pause5": severe_mask.copy(),
        "severe_pause5_warning_quality": severe_mask | (warning_mask & weak_score),
        "severe_pause5_ordinary_warning_quality": (
            severe_mask | ((ordinary_mask | warning_mask) & weak_score)
        ),
    }


def run_case(context, policy: dict, entry_mask: np.ndarray):
    threshold = policy.get("maintenance_topup_requires_score")
    maintenance_block = (
        context.empty_block
        if threshold is None
        else ~np.isfinite(context.score) | (context.score < float(threshold))
    )
    common = dict(
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=cadence.rebalance_schedule(
            len(context.arrays["dates"]),
            int(policy["portfolio_rebalance_interval_days"]),
        ),
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=entry_mask,
        maintenance_buy_block_mask_override=maintenance_block,
        score_sell_pressure_trigger_override=policy["score_sell_pressure_trigger"],
        score_sell_pressure_limit_override=policy["score_sell_pressure_limit"],
        score_sell_pressure_confirmation_days_override=policy.get(
            "score_sell_pressure_confirmation_days", 1
        ),
    )
    daily, actions = round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        **common,
    )
    stress_daily, stress_actions = round1.run_fixed10(
        context.harness,
        context.arrays,
        context.protocol,
        context.definition,
        context.score,
        context.order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.STRESS_COST,
        record_actions=True,
        **common,
    )
    return {
        "metrics_0_30pct": round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
        ),
        "train_2022_2024": round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        ),
        "holdout_2025": round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
        ),
        "metrics_0_65pct": round1.evaluate_run(
            stress_daily,
            stress_actions,
            research_base.FIRST_BUY,
            round1.DEVELOPMENT_END,
        ),
    }, daily, actions


def selection_gates(candidate: dict, baseline: dict, direct: dict) -> dict[str, bool]:
    return {
        "train_sharpe_improved": candidate["train_2022_2024"]["sharpe"]
        > baseline["train_2022_2024"]["sharpe"],
        "train_cagr_improved": candidate["train_2022_2024"]["cagr"]
        > baseline["train_2022_2024"]["cagr"],
        "holdout_2025_not_worse": candidate["holdout_2025"]["cumulative_return"]
        >= baseline["holdout_2025"]["cumulative_return"],
        "stress_not_worse": candidate["metrics_0_65pct"]["cumulative_return"]
        >= baseline["metrics_0_65pct"]["cumulative_return"],
        "drawdown_not_worse": candidate["metrics_0_30pct"]["max_drawdown"]
        <= baseline["metrics_0_30pct"]["max_drawdown"],
        "full_10_positions": candidate["metrics_0_30pct"]["full_10_position_ratio"]
        == 1.0,
        "direct_evidence_not_single_episode": (
            direct["distinct_signal_date_count"] >= 3
            and len(direct["affected_years"]) >= 2
        ),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation is not available for rule convergence")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = pressure_harness.load_context(policy)
    features = proxy.load_features()
    ordinary, ordinary_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features,
        ordinary_window=1, severe_window=0, warning_window=0,
    )
    severe, severe_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features,
        ordinary_window=0, severe_window=5, warning_window=0,
    )
    warning, warning_audit = proxy.announcement_block_matrix(
        context.arrays["dates"], context.arrays["stocks"], features,
        ordinary_window=0, severe_window=0, warning_window=5,
    )
    masks = compose_entry_masks(
        context.score,
        ordinary,
        severe,
        warning,
        float(policy["sell_score_below"]),
    )
    results, cache = {}, {}
    for case_id in CASES:
        item, daily, actions = run_case(context, policy, masks[case_id])
        results[case_id] = item
        cache[case_id] = (daily, actions)

    baseline = results["risk_event_off"]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(
            baseline["metrics_0_30pct"][key], expected[key], rtol=0.0, atol=1e-12
        ))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(checkpoint_equivalence.values()):
        raise RuntimeError("risk-event convergence baseline drifted")

    direct_interventions = {}
    gates = {}
    for case_id in CASES[1:]:
        blocked = robustness.blocked_baseline_buys(
            cache["risk_event_off"][1],
            context.arrays["dates"],
            context.arrays["stocks"],
            masks[case_id],
        )
        years = sorted({str(value)[:4] for value in blocked["signal_date"]})
        direct = {
            "blocked_baseline_buy_count": int(len(blocked)),
            "distinct_signal_date_count": int(blocked["signal_date"].nunique()),
            "distinct_stock_count": int(blocked["stock_code"].nunique()),
            "affected_years": years,
        }
        direct_interventions[case_id] = direct
        gates[case_id] = selection_gates(results[case_id], baseline, direct)

    passing = [case_id for case_id in CASES[1:] if all(gates[case_id].values())]
    selected = max(
        passing,
        key=lambda key: (
            results[key]["train_2022_2024"]["sharpe"],
            results[key]["train_2022_2024"]["cagr"],
        ),
        default="risk_event_off",
    )
    repeat, repeat_daily, repeat_actions = run_case(context, policy, masks[selected])
    deterministic = {
        "daily": round1.frame_hash(cache[selected][0])
        == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(cache[selected][1])
        == round1.frame_hash(repeat_actions),
        "metrics": all(
            repeat["metrics_0_30pct"][key]
            == results[selected]["metrics_0_30pct"][key]
            for key in ("cagr", "sharpe", "max_drawdown")
        ),
    }
    if not all(deterministic.values()):
        raise RuntimeError("risk-event convergence replay failed")

    result = {
        "status": "risk_event_rule_convergence_complete_2026_not_opened",
        "source_strategy": context.rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "candidate_budget": list(CASES),
        "rule_semantics": {
            "ordinary_abnormal": (
                "default pass-through; the broad candidate only raises entry quality "
                "to the existing 0.85 score threshold for one session"
            ),
            "severe_abnormal": "pause new entry for five official sessions",
            "exchange_focus_proxy": (
                "while proxy warning is active, require the existing 0.85 score threshold"
            ),
            "held_positions": "never force exit; existing score and execution rules remain authoritative",
            "maintenance_topups": "unchanged",
            "new_fitted_numeric_parameters": 0,
        },
        "proxy_component_audit": {
            "ordinary_1d": ordinary_audit,
            "severe_5d": severe_audit,
            "warning_5d": warning_audit,
        },
        "results": results,
        "direct_interventions": direct_interventions,
        "selection_gates": gates,
        "selected_candidate": selected,
        "changed_from_checkpoint": selected != "risk_event_off",
        "interpretation": (
            "The historical announcement asset is a mechanism proxy, not a substitute "
            "for the three Tushare interfaces. A rule is retained only when it improves "
            "2022-2024 and does not regress 2025, stress cost, drawdown, or evidence breadth."
        ),
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": context.access,
        "source_manifests": context.manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

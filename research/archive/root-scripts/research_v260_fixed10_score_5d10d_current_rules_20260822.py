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

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_equalweight_maintenance_20260822 as maintenance
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_hold_robustness_20260822 as robustness
import research_v260_fixed10_maintenance_quality_gate_20260822 as quality
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_renewal_maintenance_20260822 as renewal
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_risk_event_v110 as runtime


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_score_5d10d_current_rules_20260822"
)
CANDIDATE_IDS = (
    "score_10d",
    "score_5d50_10d50",
    "score_5d50_10d50_entry_only",
)
SMOOTHING_WINDOW = 7
RAW_ALPHA = 0.10
MIN_CAGR_IMPROVEMENT = 0.005
MIN_SHARPE_IMPROVEMENT = 0.01
MIN_DRAWDOWN_IMPROVEMENT = 0.005


def smooth_score(raw: np.ndarray, window: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=np.float64)
    finite = np.isfinite(values)
    sums = np.vstack(
        [
            np.zeros((1, values.shape[1]), dtype=np.float64),
            np.cumsum(np.where(finite, values, 0.0), axis=0),
        ]
    )
    counts = np.vstack(
        [
            np.zeros((1, values.shape[1]), dtype=np.int32),
            np.cumsum(finite, axis=0, dtype=np.int32),
        ]
    )
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - int(window), 0)
    smooth = np.divide(
        sums[ends] - sums[starts],
        counts[ends] - counts[starts],
        out=np.full_like(sums[ends], np.nan),
        where=(counts[ends] - counts[starts]) > 0,
    )
    score = (float(alpha) * values + (1.0 - float(alpha)) * smooth).astype(
        np.float32
    )
    order = np.argsort(
        -np.nan_to_num(score, nan=-np.inf), axis=1, kind="stable"
    )
    return score, order


def candidate_score(
    candidate_id: str,
    arrays: dict[str, np.ndarray],
    baseline_score: np.ndarray,
    baseline_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_id == "score_10d":
        return baseline_score, baseline_order
    if candidate_id in {
        "score_5d50_10d50",
        "score_5d50_10d50_entry_only",
    }:
        raw = 0.50 * arrays["rank_5d"] + 0.50 * arrays["rank_10d"]
        return smooth_score(raw, SMOOTHING_WINDOW, RAW_ALPHA)
    raise ValueError(f"unknown candidate: {candidate_id}")


def materially_better(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["cagr"] - baseline["cagr"] >= MIN_CAGR_IMPROVEMENT
        or candidate["sharpe"] - baseline["sharpe"] >= MIN_SHARPE_IMPROVEMENT
        or baseline["max_drawdown"] - candidate["max_drawdown"]
        >= MIN_DRAWDOWN_IMPROVEMENT
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    policy = copy.deepcopy(checkpoint["selected_policy"])
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered current score development")

    definition = harness.production_definition(protocol)
    baseline_score, baseline_order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    active = cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    entry_block = np.zeros(baseline_score.shape, dtype=np.bool_)
    results = {}
    run_cache = {}
    score_cache = {}

    for candidate_id in CANDIDATE_IDS:
        score, order = candidate_score(
            candidate_id, arrays, baseline_score, baseline_order
        )
        maintenance_block = quality.maintenance_quality_block(
            score, policy.get("maintenance_topup_requires_score") is not None
        )
        common = dict(
            simulator=runtime.simulate,
            portfolio_rebalance_active_override=active,
            portfolio_rebalance_min_weight_deviation_override=policy[
                "portfolio_rebalance_min_weight_deviation"
            ],
            entry_block_mask_override=entry_block,
            maintenance_buy_block_mask_override=maintenance_block,
            exit_score_override=(
                baseline_score
                if candidate_id == "score_5d50_10d50_entry_only"
                else None
            ),
        )
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
            **common,
        )
        stress_daily, stress_actions = round1.run_fixed10(
            harness,
            arrays,
            protocol,
            definition,
            score,
            order,
            policy,
            round1.DEVELOPMENT_END,
            slip=round1.STRESS_COST,
            record_actions=True,
            **common,
        )
        train = round1.evaluate_run(
            daily, actions, research_base.FIRST_BUY, "20241231"
        )
        holdout = round1.evaluate_run(
            daily, actions, "20250102", round1.DEVELOPMENT_END
        )
        windows = {
            str(offset): round1.evaluate_run(
                daily,
                actions,
                robustness.window_start_date(arrays, offset),
                round1.DEVELOPMENT_END,
            )
            for offset in (5, 20, 60)
        }
        results[candidate_id] = {
            "score_formula": (
                {"rank_10d": 1.0}
                if candidate_id == "score_10d"
                else {"rank_5d": 0.50, "rank_10d": 0.50}
            ),
            "exit_score_formula": (
                {"rank_10d": 1.0}
                if candidate_id
                in {"score_10d", "score_5d50_10d50_entry_only"}
                else {"rank_5d": 0.50, "rank_10d": 0.50}
            ),
            "score_smoothing": {
                "window": SMOOTHING_WINDOW,
                "raw_alpha": RAW_ALPHA,
            },
            "metrics_0_30pct": round1.evaluate_run(
                daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
            ),
            "metrics_0_65pct": round1.evaluate_run(
                stress_daily,
                stress_actions,
                research_base.FIRST_BUY,
                round1.DEVELOPMENT_END,
            ),
            "train_2022_2024": train,
            "holdout_2025_not_used_for_selection": holdout,
            "window_start_metrics": windows,
            "min_window_cagr": float(min(item["cagr"] for item in windows.values())),
            "action_diagnostics": maintenance.maintenance_action_metrics(actions),
            "maximum_drawdown_episode": attribution.maximum_drawdown_episode(daily),
        }
        run_cache[candidate_id] = (daily, actions)
        score_cache[candidate_id] = (score, order, maintenance_block)

    baseline_id = "score_10d"
    current = results[baseline_id]["metrics_0_30pct"]
    expected = checkpoint["current_best_equalweight"]
    checkpoint_equivalence = {
        key: bool(np.isclose(current[key], expected[key], rtol=0.0, atol=1e-12))
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
        raise RuntimeError("current score baseline drifted from checkpoint")

    ranked_id = max(results, key=lambda key: renewal.selection_key(results[key]))
    material_improvement = bool(
        ranked_id != baseline_id
        and materially_better(
            results[ranked_id]["train_2022_2024"],
            results[baseline_id]["train_2022_2024"],
        )
    )
    selected_id = ranked_id if material_improvement else baseline_id
    selected_daily, selected_actions = run_cache[selected_id]
    selected_score, selected_order, selected_maintenance_block = score_cache[selected_id]
    repeat_daily, repeat_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        selected_score,
        selected_order,
        policy,
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=runtime.simulate,
        portfolio_rebalance_active_override=active,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
        entry_block_mask_override=entry_block,
        maintenance_buy_block_mask_override=selected_maintenance_block,
        exit_score_override=(
            baseline_score
            if selected_id == "score_5d50_10d50_entry_only"
            else None
        ),
    )
    deterministic = {
        "daily": round1.frame_hash(selected_daily) == round1.frame_hash(repeat_daily),
        "actions": round1.frame_hash(selected_actions)
        == round1.frame_hash(repeat_actions),
    }
    if not all(deterministic.values()):
        raise RuntimeError("current score deterministic replay failed")

    result = {
        "status": "development_candidate_frozen_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "selection_boundary": [research_base.FIRST_BUY, "20241231"],
        "forward_confirmation_boundary": ["20250102", round1.DEVELOPMENT_END],
        "candidate_budget": list(CANDIDATE_IDS),
        "only_change": (
            "replace the 10-day score with an equal 5-day/10-day score; all holding, "
            "exit, execution, and portfolio rules remain unchanged"
        ),
        "results": results,
        "walk_forward_selection_evidence": robustness.walk_forward_selections(results),
        "selected_candidate_id": selected_id,
        "selected_reason": (
            "material aggregate 2022-2024 improvement"
            if material_improvement
            else "candidate did not clear the predeclared practical improvement gate"
        ),
        "practical_improvement_gate": {
            "minimum_cagr_improvement": MIN_CAGR_IMPROVEMENT,
            "minimum_sharpe_improvement": MIN_SHARPE_IMPROVEMENT,
            "minimum_drawdown_improvement": MIN_DRAWDOWN_IMPROVEMENT,
            "raw_ranked_candidate_id": ranked_id,
            "passed": material_improvement,
        },
        "selected_policy": {
            **copy.deepcopy(policy),
            "score_formula": results[selected_id]["score_formula"],
            "score_smoothing": results[selected_id]["score_smoothing"],
        },
        "checkpoint_equivalence": checkpoint_equivalence,
        "deterministic_replay": deterministic,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "development_result.json", result)
    print(
        json.dumps(
            {
                "selected": selected_id,
                "ranked": ranked_id,
                "practical_gate": result["practical_improvement_gate"],
                "results": {
                    key: {
                        "train": value["train_2022_2024"],
                        "holdout": value["holdout_2025_not_used_for_selection"],
                    }
                    for key, value in results.items()
                },
            },
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()

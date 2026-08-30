import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from research_v260_production_rule_simplification_ablation_20260823 import (
    classify_rules,
)


def metric(daily: str, actions: str, cumulative: float = 1.0) -> dict:
    return {
        "daily_sha256": daily,
        "actions_sha256": actions,
        "cumulative_return": cumulative,
    }


def test_dead_rules_require_identical_daily_and_actions_at_both_costs() -> None:
    baseline = {
        "0.003": metric("d1", "a1"),
        "0.0065": metric("d2", "a2"),
    }
    results = {
        "production": baseline,
        "remove_position_width_warmup": {
            "0.003": metric("d1", "a1"),
            "0.0065": metric("d2", "a2"),
        },
        "remove_weak_gross_gate": {
            "0.003": metric("d1", "a1"),
            "0.0065": metric("d2", "a2"),
        },
    }
    actual = classify_rules(results)["development_no_op_cleanup_candidates"]
    assert actual["position_width_warmup"]["verified"] is True
    assert actual["weak_gross_gate"]["verified"] is True


def test_action_drift_prevents_dead_rule_classification() -> None:
    results = {
        "production": {
            "0.003": metric("d1", "a1"),
            "0.0065": metric("d2", "a2"),
        },
        "remove_position_width_warmup": {
            "0.003": metric("d1", "changed"),
            "0.0065": metric("d2", "a2"),
        },
        "remove_weak_gross_gate": {
            "0.003": metric("d1", "a1"),
            "0.0065": metric("d2", "a2"),
        },
    }
    actual = classify_rules(results)["development_no_op_cleanup_candidates"]
    assert actual["position_width_warmup"]["verified"] is False
    assert actual["weak_gross_gate"]["verified"] is True


def test_classification_rejects_the_year_concentrated_40pct_rule() -> None:
    baseline = {
        "0.003": metric("d1", "a1"),
        "0.0065": metric("d2", "a2"),
    }
    results = {
        "production": baseline,
        "remove_position_width_warmup": baseline,
        "remove_weak_gross_gate": baseline,
    }
    rejected = classify_rules(results)["reject_as_overfit"]
    assert "strong_market_40pct_target" in rejected
    assert "fixed10_layered_exit_refill_stack" in rejected

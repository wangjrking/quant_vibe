from __future__ import annotations

import sys
from pathlib import Path


MAIN_ROOT = Path(r"D:/work/quant/quant_mcp/quant/main")
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_pre2026_search_saturation_20260822 as subject


def fixtures():
    checkpoint = {
        "selected_candidate": subject.CANDIDATE_ID,
        "selected_policy": {
            "entry_rank_sizing": {"equalweight_is_soft_reference": True},
            "residual_cash_sweep": {
                "equalweight_and_full_investment_are_soft_directions": True,
            },
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    historical_checkpoint = {
        "validation_2026_opened": False,
        "production_modified": False,
    }
    generalization = {
        "rules": {"a": {"classification": "robust_core_rule"}},
        "unsupported_rules": [],
        "validation_2026_opened": False,
        "production_modified": False,
    }
    turnover = {
        "turnover_attribution": {
            "replacement_turnover_share": 0.98,
            "equalweight_maintenance_turnover_share": 0.02,
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    exit_confirmation = {
        "selected_candidate": "score_exit_confirmation_1d",
        "validation_2026_opened": False,
        "production_modified": False,
    }
    event_policy = {
        "event_overlay_role": "diagnostic_only_not_selection_candidate",
        "validation_2026_opened": False,
        "production_modified": False,
    }
    pairwise = {
        "dominating_pair_removals": [],
        "validation_2026_opened": False,
        "production_modified": False,
    }
    return (
        checkpoint,
        historical_checkpoint,
        generalization,
        turnover,
        exit_confirmation,
        event_policy,
        pairwise,
    )


def test_saturation_gates_pass_for_closed_search():
    gates = subject.saturation_gates(*fixtures())
    assert all(gates.values())


def test_saturation_gates_detect_open_or_contaminated_search():
    values = list(fixtures())
    values[3]["turnover_attribution"]["replacement_turnover_share"] = 0.80
    values[5]["validation_2026_opened"] = True
    gates = subject.saturation_gates(*values)
    assert gates["replacement_turnover_dominates"] is False
    assert gates["validation_2026_unopened"] is False

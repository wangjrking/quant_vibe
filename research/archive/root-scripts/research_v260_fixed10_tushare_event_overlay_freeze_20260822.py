from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_tushare_event_overlay_freeze_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
L1_MANIFEST = (
    REPO
    / "quant/data_file/experimental_assets/tushare_stock_risk_events_v1/manifest.json"
)
L2_PUBLICATION = (
    REPO
    / "quant/data_file/reports/l1_tushare_stock_risk_events_production_publish_20260822.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation was already opened")
    if not L1_MANIFEST.exists() or not L2_PUBLICATION.exists():
        raise FileNotFoundError("risk-event metadata evidence is incomplete")

    contract = {
        "status": "frozen_before_one_shot_2026_validation",
        "contract_id": "fixed10_tushare_severe_score_tiebreak_diagnostic_v2",
        "base_candidate": checkpoint["selected_candidate"],
        "source_endpoints": {
            "ordinary_abnormal_volatility": "stk_shock",
            "severe_abnormal_volatility": "stk_high_shock",
            "exchange_focus_security": "stk_alert",
        },
        "pit_timing": (
            "effective on the first official open session strictly after the source "
            "event publication date; exact stock_code and session only"
        ),
        "diagnostic_coverage": {
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
        },
        "rules": {
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
            "existing_holdings": (
                "unchanged frozen score exit, T+1, tradability and replacement rules"
            ),
            "missing_or_unknown_event_state": "fail_closed_for_diagnostic_input",
        },
        "one_shot_validation_arms": [
            "production_strategy_reference",
            "fixed10_current_candidate_without_event_overlay",
            "fixed10_event_overlay_diagnostic_only",
        ],
        "decision_order": {
            "fixed10_base": "judge against production with already frozen gates",
            "event_overlay": (
                "report return, Sharpe and drawdown deltas versus the same fixed10 "
                "base; diagnostic only and cannot change candidate acceptance"
            ),
            "no_post_validation_tuning": True,
        },
        "event_overlay_role": "diagnostic_only_not_selection_candidate",
        "pre2026_business_row_evidence": {
            "ordinary_event_rows": 0,
            "severe_event_rows": 0,
            "exchange_alert_rows": 0,
            "interpretation": (
                "direct event history is unavailable before 2026, so the overlay is "
                "reported in validation but cannot be selected from that validation"
            ),
        },
        "pre2026_proxy_research": {
            "path": str(
                REPO
                / "quant/data_file/reports/"
                "strategy_agent_v260_fixed10_risk_event_score_tiebreak_20260822/"
                "development_result.json"
            ),
            "interpretation": (
                "ordinary and exchange-alert trading rules were rejected; the severe "
                "one-session score tiebreak remains diagnostic because train and 2025 "
                "support disagree"
            ),
        },
        "metadata_evidence": {
            "l1_manifest": str(L1_MANIFEST),
            "l1_manifest_sha256": sha256(L1_MANIFEST),
            "l2_publication_report": str(L2_PUBLICATION),
            "l2_publication_report_sha256": sha256(L2_PUBLICATION),
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "event_overlay_contract.json", contract)
    print(json.dumps({
        "status": contract["status"],
        "contract_id": contract["contract_id"],
        "rules": contract["rules"],
        "validation_2026_opened": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

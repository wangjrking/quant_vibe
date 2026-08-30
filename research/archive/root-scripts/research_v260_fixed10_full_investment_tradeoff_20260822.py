from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORTS = REPO / "quant/data_file/reports"
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_full_investment_tradeoff_20260822"
)
EXPOSURE = (
    REPORTS
    / "strategy_agent_v260_fixed10_current_vs_production_exposure_attribution_20260822/"
    "exposure_attribution.json"
)
POSITION = (
    REPORTS
    / "strategy_agent_v260_fixed10_current_drawdown_position_attribution_20260822/"
    "position_attribution.json"
)
STRUCTURE = (
    REPORTS
    / "strategy_agent_v260_fixed10_current_portfolio_structure_diagnostic_20260822/"
    "portfolio_structure.json"
)
RISK_SCORECARD = (
    REPORTS
    / "strategy_agent_v260_fixed10_prevalidation_risk_scorecard_20260822/"
    "prevalidation_risk_scorecard.json"
)


def contribution_share(part: float, total: float) -> float:
    if total == 0.0:
        raise ValueError("total contribution cannot be zero")
    return float(part / total)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sources = {
        "exposure": json.loads(EXPOSURE.read_text(encoding="utf-8")),
        "position": json.loads(POSITION.read_text(encoding="utf-8")),
        "structure": json.loads(STRUCTURE.read_text(encoding="utf-8")),
        "risk": json.loads(RISK_SCORECARD.read_text(encoding="utf-8")),
    }
    if any(item.get("validation_2026_opened") is not False for item in sources.values()):
        raise PermissionError("2026 validation is not closed in tradeoff sources")

    exposure = sources["exposure"]
    position = sources["position"]["position_attribution"]
    structure = sources["structure"]
    risk = sources["risk"]
    total_excess = exposure["candidate_minus_production_total_log_return"]
    low_mid = exposure["candidate_minus_production_low_and_mid_exposure_log_return"]
    high = exposure["candidate_minus_production_high_exposure_log_return"]
    candidate_mdd = risk["pre2026_metrics"]["candidate"]["max_drawdown"]
    payload = {
        "status": "full_investment_tradeoff_confirmed_2026_not_opened",
        "return_edge_attribution": {
            "total_log_excess": total_excess,
            "low_or_mid_production_exposure_log_excess": low_mid,
            "production_already_full_log_excess": high,
            "share_from_low_or_mid_production_exposure": contribution_share(
                low_mid, total_excess
            ),
            "share_from_production_already_full": contribution_share(
                high, total_excess
            ),
            "production_below_50pct_bin": exposure["exposure_bins"][
                "production_below_50pct"
            ],
        },
        "drawdown_structure": {
            "maximum_drawdown": candidate_mdd,
            "buffer_to_35pct_gate": float(0.35 - candidate_mdd),
            "classification": position["classification"],
            "unique_held_stocks": position["unique_held_stocks"],
            "negative_stock_count": position["negative_stock_count"],
            "top1_negative_concentration": position[
                "top1_negative_concentration"
            ],
            "top5_negative_concentration": position[
                "top5_negative_concentration"
            ],
            "correlation_drawdown_minus_all": structure["correlation"][
                "drawdown_minus_all_mean"
            ],
            "drawdown_industry_concentration_materially_higher": structure[
                "structural_flags"
            ]["drawdown_industry_concentration_materially_higher"],
            "drawdown_size_percentile_materially_lower": structure[
                "structural_flags"
            ]["drawdown_size_percentile_materially_lower"],
        },
        "constraint_compatibility": {
            "exactly10_equalweight_full_investment": True,
            "cash_regime_allowed": False,
            "single_name_or_industry_patch_supported": False,
            "exposure_reduction_would_address_main_risk": True,
            "exposure_reduction_conflicts_with_objective": True,
        },
        "decision": {
            "candidate_role": "aggressive_return_candidate",
            "do_not_add_narrow_risk_patch": True,
            "retain_full_investment_objective": True,
            "interpretation": (
                "the return edge is primarily compensation for forcing capital into "
                "the market when production holds cash; broad drawdown cannot be fixed "
                "honestly by a stock, industry or announcement-specific patch"
            ),
            "candidate_changed": False,
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "full_investment_tradeoff.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "return_edge_attribution": payload["return_edge_attribution"],
        "constraint_compatibility": payload["constraint_compatibility"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

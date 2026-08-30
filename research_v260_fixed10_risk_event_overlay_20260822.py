from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import duckdb
import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as base
import research_v260_fixed10_maintenance_quality_gate_20260822 as maintenance_round
import research_v260_fixed10_portfolio_rebalance_band_20260822 as rebalance
from research_v260_runtime import fixed10_risk_event_v110 as event_runtime
from research_v260_risk_event_overlay import RULE, build_block_matrix


EVENT_ROOT = REPO / "quant/data_file/experimental_assets/tushare_stock_risk_events_v1"
FEATURE_DB = EVENT_ROOT / "l2_stock_risk_signal.duckdb"
EVENT_DB = EVENT_ROOT / "l2_stock_risk_events.duckdb"
OUTPUT_ROOT = REPO / "quant/data_file/reports/strategy_agent_v260_fixed10_risk_event_overlay_20260822"


def frame_hash(frame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def load_features():
    connection = duckdb.connect(str(FEATURE_DB), read_only=True)
    try:
        return connection.execute(
            """
            SELECT *
            FROM stock_risk_signal
            WHERE signal_date < '20260101'
            ORDER BY signal_date, stock_code
            """
        ).df()
    finally:
        connection.close()


def pre2026_coverage() -> dict:
    connection = duckdb.connect(str(EVENT_DB), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT source_api, count(*) AS rows, count(DISTINCT stock_code) AS stocks,
                   min(event_date) AS min_date, max(event_date) AS max_date
            FROM stock_risk_events
            WHERE event_date < '20260101'
            GROUP BY source_api
            ORDER BY source_api
            """
        ).fetchdf()
    finally:
        connection.close()
    by_api = {
        row.source_api: {
            "rows": int(row.rows),
            "stocks": int(row.stocks),
            "min_date": row.min_date,
            "max_date": row.max_date,
        }
        for row in rows.itertuples(index=False)
    }
    for api in ("stk_shock", "stk_high_shock", "stk_alert"):
        by_api.setdefault(
            api,
            {"rows": 0, "stocks": 0, "min_date": None, "max_date": None},
        )
    return by_api


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(
        (
            REPO
            / "quant/data_file/reports/strategy_agent_v260_fixed10_pre2026_checkpoint_20260822/pre2026_checkpoint.json"
        ).read_text(encoding="utf-8")
    )
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 validation was already opened")

    harness, protocol, rules, manifests, arrays, access = base.load_arrays(
        base.DEVELOPMENT_END
    )
    if access["logical_max_date"] != base.DEVELOPMENT_END:
        raise PermissionError("pre-2026 boundary drift")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    policy = checkpoint["selected_policy"]
    schedule = rebalance.cadence.rebalance_schedule(
        len(arrays["dates"]), policy["portfolio_rebalance_interval_days"]
    )
    features = load_features()
    block, event_audit = build_block_matrix(arrays["dates"], arrays["stocks"], features)
    if np.any(block):
        raise PermissionError("pre-2026 event rows unexpectedly available; candidate must be re-reviewed")

    common = dict(
        portfolio_rebalance_active_override=schedule,
        portfolio_rebalance_min_weight_deviation_override=policy[
            "portfolio_rebalance_min_weight_deviation"
        ],
    )
    empty_block = np.zeros(score.shape, dtype=np.bool_)
    quality_block = maintenance_round.maintenance_quality_block(
        score, policy.get("maintenance_topup_requires_score") is not None
    )
    current_daily, current_actions = base.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        policy,
        base.DEVELOPMENT_END,
        slip=base.BASELINE_COST,
        record_actions=True,
        simulator=event_runtime.simulate,
        entry_block_mask_override=empty_block,
        maintenance_buy_block_mask_override=quality_block,
        **common,
    )
    candidate_daily, candidate_actions = base.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        policy,
        base.DEVELOPMENT_END,
        slip=base.BASELINE_COST,
        record_actions=True,
        simulator=event_runtime.simulate,
        entry_block_mask_override=block,
        maintenance_buy_block_mask_override=quality_block,
        **common,
    )
    equivalence = {
        "daily": frame_hash(current_daily) == frame_hash(candidate_daily),
        "actions": frame_hash(current_actions) == frame_hash(candidate_actions),
    }
    if not all(equivalence.values()):
        raise RuntimeError("event runtime changed behavior when the event mask was empty")

    robustness_path = (
        REPO
        / "quant/data_file/reports/"
        "strategy_agent_v260_fixed10_severe_event_temporal_robustness_20260822/"
        "development_result.json"
    )
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if robustness.get("supported") is not False:
        raise RuntimeError("severe-event robustness decision is not frozen as rejected")
    result = {
        "status": "candidate_rejected_pre2026_insufficient_direct_interventions",
        "candidate": "v260_fixed10_equalweight_risk_event_overlay_v1",
        "source_strategy": rules["strategy_id"],
        "base_fixed10_policy": policy,
        "only_change": RULE,
        "design_reasoning": {
            "entry_pause": "only severe abnormal-volatility announcements pause first entry for five sessions",
            "maintenance_topup": "unchanged; action decomposition found no incremental effect from blocking top-ups",
            "no_forced_exit": "abnormal-volatility events can also describe strong winners; the existing score exit remains authoritative",
            "ordinary_and_alert_passthrough": "the pre-2026 announcement proxy found ordinary events and generic warning gates harmful",
            "no_score_penalty": "the Tushare interfaces have no usable pre-2026 A-share history, so a fitted penalty would be unsupported",
        },
        "pre2026_proxy_evidence": {
            "path": str(
                REPO
                / "quant/data_file/reports/strategy_agent_v260_fixed10_abnormal_announcement_proxy_20260822/development_result.json"
            ),
            "selected_mechanism": "severe_5d_only",
        },
        "temporal_robustness_evidence": {
            "path": str(robustness_path),
            "direct_intervention_count": robustness["direct_interventions"][
                "blocked_baseline_buy_count"
            ],
            "supported": robustness["supported"],
            "decision": (
                "reject: the apparent improvement depends on only two direct "
                "entry interventions and is too concentrated for final validation"
            ),
        },
        "pre2026_event_coverage": pre2026_coverage(),
        "pre2026_block_matrix": event_audit,
        "pre2026_business_result_available": False,
        "reason_no_development_result": "all usable no-BJ L2 events begin in 2026",
        "empty_mask_reference_equivalence": equivalence,
        "eligible_for_final_2026_validation": False,
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    (OUTPUT_ROOT / "development_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

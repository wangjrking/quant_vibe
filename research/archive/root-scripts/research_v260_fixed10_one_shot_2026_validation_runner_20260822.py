from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
import traceback
from dataclasses import replace
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
import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_one_shot_2026_validation_protocol_freeze_20260822 as protocol_freeze
import research_v260_fixed10_portfolio_rebalance_cadence_20260822 as cadence
import research_v260_fixed10_pressure_extra_age_guard_20260822 as age_guard
import research_v260_fixed10_pressure_recheck_harness_20260822 as pressure_harness
import research_v260_fixed10_regime_exit_threshold_20260822 as regime
import research_v260_fixed10_tushare_risk_event_entry_gate_20260822 as risk_event
import research_v260_fixed10_weak_market_defensive_rerank_20260822 as defensive
import research_v260_fixed10_weak_market_volatility_sell_priority_20260822 as priority
import research_v260_lowrisk_score_tuning_20260821 as research_base
import research_v260_risk_event_overlay as event_overlay
from research_v260_runtime import fixed10_residual_cash_sweep_v112 as sweep_runtime


REPORTS = REPO / "quant/data_file/reports"
PROTOCOL = (
    REPORTS
    / "strategy_agent_v260_fixed10_one_shot_2026_validation_protocol_20260822/"
    "one_shot_validation_protocol.json"
)
OUTPUT_ROOT = REPORTS / "strategy_agent_v260_fixed10_one_shot_2026_validation_20260822"
START_CLAIM = OUTPUT_ROOT / "validation_claim.lock"
START_MARKER = OUTPUT_ROOT / "validation_started.json"
RESULT_PATH = OUTPUT_ROOT / "result.json"
FAILURE_PATH = OUTPUT_ROOT / "failure.json"
OUTPUT_ARTIFACT_NAMES = (
    "production_daily.csv",
    "production_actions.csv",
    "fixed10_daily.csv",
    "fixed10_actions.csv",
    "fixed10_stress_0_65pct_daily.csv",
    "fixed10_stress_0_65pct_actions.csv",
    "fixed10_event_overlay_daily.csv",
    "fixed10_event_overlay_actions.csv",
)
EXECUTE_CONFIRMATION = "fixed10_pre2026_frozen_one_shot_2026_v3"
L2_MARKET_DB = REPO / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
L2_MARKET_TABLE = "STOCK_DAILY_DATA"
REQUIRED_L2_COLUMNS = {
    "trade_date", "stock_code", "amount", "total_mv", "turnover_rate",
    "close_qfq", "list_date", "ST_TYPE", "ST_TYPE_name", "name", "open",
    "pre_close",
}
REQUIRED_EVENT_COLUMNS = {
    "signal_date", "stock_code", "shock_count_1d", "high_shock_count_1d",
    "high_shock_count_5d", "alert_active_count",
}
REQUIRED_DAILY_OUTPUT_COLUMNS = {
    "date", "return", "equity", "turnover", "invested_ratio", "positions", "trades",
}
REQUIRED_ACTION_OUTPUT_COLUMNS = {
    "signal_date", "buy_date", "action", "stock_code", "target_pct",
    "execution_open_raw",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_frozen_candidate(
    context,
    policy: dict,
    extra_age,
    cost: float,
    sell_priority,
    end_date: str,
    exit_score_override=None,
):
    contract = policy.get("entry_rank_sizing", {})
    if contract != {
        "application": "new_entries_only",
        "ranking": "global_existing_frozen_score_order_before_execution_refill",
        "positions": 10,
        "top_multiplier": 1.1,
        "bottom_multiplier": 0.9,
        "schedule": "linear_descending",
        "reference_top10_multiplier_sum": 10.0,
        "refill_outside_global_top10_multiplier": 1.0,
        "actual_entry_batch_gross_neutral": False,
        "equalweight_is_soft_reference": True,
    }:
        raise RuntimeError("validation entry-rank sizing contract drifted")
    cash_sweep = policy.get("residual_cash_sweep", {})
    if cash_sweep != {
        "enabled": True,
        "trigger": "after_buy_trade",
        "recipient": "most_underweight_existing_buyable_holding",
        "target_pct": "gross_target_divided_by_max_positions",
        "board_lot_shares": 100,
        "max_orders_per_trade_day": None,
        "new_names_allowed": False,
        "forced_sales_allowed": False,
        "equalweight_and_full_investment_are_soft_directions": True,
    }:
        raise RuntimeError("validation residual-cash sweep contract drifted")
    multipliers = sizing.entry_rank_multipliers(
        context.order,
        positions=int(contract["positions"]),
        top_multiplier=float(contract["top_multiplier"]),
        bottom_multiplier=float(contract["bottom_multiplier"]),
    )
    return age_guard.run_policy_at_cost(
        context,
        policy,
        extra_age,
        cost,
        score_sell_priority_override=sell_priority,
        simulator=functools.partial(
            sweep_runtime.simulate,
            candidate_target_multiplier_override=multipliers,
            residual_cash_sweep_override=True,
            residual_cash_sweep_max_orders_override=cash_sweep[
                "max_orders_per_trade_day"
            ],
            residual_cash_sweep_trigger_override="buy_trade",
        ),
        target_positions_override=int(contract["positions"]),
        target_gross_override=1.0,
        exit_score_override=exit_score_override,
        end_date=end_date,
    )


def one_shot_artifact_paths() -> tuple[Path, ...]:
    return (
        START_CLAIM,
        START_MARKER,
        RESULT_PATH,
        FAILURE_PATH,
        *(OUTPUT_ROOT / name for name in OUTPUT_ARTIFACT_NAMES),
    )


def one_shot_already_started() -> bool:
    return any(path.exists() for path in one_shot_artifact_paths())


def atomic_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale atomic CSV temporary exists: {temporary}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_and_select_output_arm(
    name: str,
    daily,
    actions,
    start_date: str,
    end_date: str,
    expected_dates: tuple[str, ...] | None = None,
    signal_date_by_buy_date: dict[str, str] | None = None,
):
    missing_daily = sorted(REQUIRED_DAILY_OUTPUT_COLUMNS - set(daily.columns))
    missing_actions = sorted(REQUIRED_ACTION_OUTPUT_COLUMNS - set(actions.columns))
    if missing_daily or missing_actions:
        raise RuntimeError(
            f"validation output schema incomplete for {name}: "
            f"daily={missing_daily}, actions={missing_actions}"
        )
    daily_dates = daily["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    selected_daily = daily[
        daily_dates.between(start_date, end_date)
    ].copy()
    selected_daily["date"] = daily_dates[daily_dates.between(start_date, end_date)]
    action_dates = actions["buy_date"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    selected_actions = actions[
        action_dates.between(start_date, end_date)
    ].copy()
    selected_actions["buy_date"] = action_dates[
        action_dates.between(start_date, end_date)
    ]
    selected_actions["signal_date"] = selected_actions["signal_date"].astype(
        str
    ).str.replace(r"\.0$", "", regex=True)
    if selected_daily.empty:
        raise RuntimeError(f"validation daily output is empty for {name}")
    if (
        selected_daily["date"].min() != start_date
        or selected_daily["date"].max() != end_date
        or selected_daily["date"].duplicated().any()
    ):
        raise RuntimeError(f"validation daily date contract failed for {name}")
    if expected_dates is not None and tuple(selected_daily["date"]) != expected_dates:
        raise RuntimeError(f"validation daily calendar completeness failed for {name}")
    numeric_daily = (
        "return", "equity", "turnover", "invested_ratio", "positions", "trades",
    )
    for column in numeric_daily:
        values = np.asarray(selected_daily[column], dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"validation daily nonfinite {column} for {name}")
    if (
        (selected_daily["equity"].astype(float) <= 0).any()
        or (selected_daily["turnover"].astype(float) < 0).any()
        or not selected_daily["invested_ratio"].astype(float).between(0, 1.000001).all()
        or (selected_daily["positions"].astype(float) < 0).any()
        or (selected_daily["trades"].astype(float) < 0).any()
    ):
        raise RuntimeError(f"validation daily numeric bounds failed for {name}")
    if not np.equal(
        selected_daily["positions"].astype(float) % 1, 0
    ).all() or not np.equal(
        selected_daily["trades"].astype(float) % 1, 0
    ).all():
        raise RuntimeError(f"validation daily integer contract failed for {name}")
    if len(selected_actions):
        action_keys = ["signal_date", "buy_date", "action", "stock_code"]
        if selected_actions.duplicated(action_keys).any():
            raise RuntimeError(f"validation action keys are not unique for {name}")
        if not selected_actions["action"].isin({"BUY", "SELL"}).all():
            raise RuntimeError(f"validation action type is invalid for {name}")
        if selected_actions["stock_code"].astype(str).str.endswith(".BJ").any():
            raise RuntimeError(f"validation actions contain Beijing rows for {name}")
        if expected_dates is not None and not set(selected_actions["buy_date"]).issubset(
            expected_dates
        ):
            raise RuntimeError(f"validation action date is outside calendar for {name}")
        if signal_date_by_buy_date is not None:
            expected_signals = selected_actions["buy_date"].map(
                signal_date_by_buy_date
            )
            if expected_signals.isna().any() or not (
                selected_actions["signal_date"] == expected_signals
            ).all():
                raise RuntimeError(f"validation action T+1 mapping failed for {name}")
        for column in ("target_pct", "execution_open_raw"):
            values = np.asarray(selected_actions[column], dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"validation action nonfinite {column} for {name}")
        if (
            (selected_actions["target_pct"].astype(float) < 0).any()
            or (selected_actions["execution_open_raw"].astype(float) <= 0).any()
        ):
            raise RuntimeError(f"validation action numeric bounds failed for {name}")
    return selected_daily, selected_actions


def expected_validation_dates(all_dates, start_date: str, end_date: str) -> tuple[str, ...]:
    normalized = [str(value).removesuffix(".0") for value in all_dates]
    selected = tuple(value for value in normalized if start_date <= value <= end_date)
    if (
        not selected
        or selected[0] != start_date
        or selected[-1] != end_date
        or tuple(sorted(selected)) != selected
        or len(set(selected)) != len(selected)
    ):
        raise RuntimeError("validation source calendar contract failed")
    return selected


def expected_t1_signal_map(
    all_dates, expected_dates: tuple[str, ...]
) -> dict[str, str]:
    normalized = tuple(str(value).removesuffix(".0") for value in all_dates)
    if tuple(sorted(normalized)) != normalized or len(set(normalized)) != len(normalized):
        raise RuntimeError("validation source calendar contract failed")
    index = {value: position for position, value in enumerate(normalized)}
    mapping = {}
    for buy_date in expected_dates:
        position = index.get(buy_date)
        if position is None or position == 0:
            raise RuntimeError("validation T+1 predecessor is unavailable")
        mapping[buy_date] = normalized[position - 1]
    return mapping


def persist_validation_outputs(
    arms: tuple,
    start_date: str,
    end_date: str,
    expected_dates: tuple[str, ...],
    signal_date_by_buy_date: dict[str, str],
) -> None:
    for name, daily, actions in arms:
        selected_daily, selected_actions = validate_and_select_output_arm(
            name,
            daily,
            actions,
            start_date,
            end_date,
            expected_dates,
            signal_date_by_buy_date,
        )
        atomic_csv(selected_daily, OUTPUT_ROOT / f"{name}_daily.csv")
        atomic_csv(selected_actions, OUTPUT_ROOT / f"{name}_actions.csv")


def output_artifact_sha256s() -> dict[str, str]:
    paths = {name: OUTPUT_ROOT / name for name in OUTPUT_ARTIFACT_NAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"validation output artifact missing: {missing[0]}")
    return {name: sha256(path) for name, path in paths.items()}


def validate_execute_confirmation(execute: bool, confirmation: str | None) -> None:
    if execute and confirmation != EXECUTE_CONFIRMATION:
        raise PermissionError("one-shot validation confirmation does not match")


def validate_protocol_derivation(protocol: dict, checkpoint: dict, event_contract: dict) -> None:
    expected = protocol_freeze.build_protocol(checkpoint, event_contract)
    if protocol != expected:
        raise RuntimeError("validation protocol drifted from frozen source derivation")


def evaluate_base_candidate_gates(
    base: dict,
    production: dict,
    stress: dict,
    deterministic: dict,
    decision_contract: dict,
) -> dict[str, bool]:
    if decision_contract.get("primary") != "cumulative_return_above_production":
        raise ValueError("unsupported validation primary decision")
    thresholds = decision_contract["machine_thresholds"]
    return {
        "cumulative_return_above_production": base["cumulative_return"]
        > production["cumulative_return"],
        "deterministic_replay": (
            all(deterministic.values())
            if thresholds["deterministic_replay_required"]
            else True
        ),
    }


def build_failure_record(protocol: dict, exc: Exception) -> dict:
    return {
        "status": "failed_after_one_shot_validation_started",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "start_marker_sha256": (
            sha256(START_MARKER) if START_MARKER.is_file() else None
        ),
        "validation_boundary": protocol["validation_boundary"],
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "validation_2026_opened": START_MARKER.exists(),
        "production_modified": False,
        "trading_triggered": False,
        "retry_allowed": False,
    }


def load_and_validate_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "one_shot_2026_validation_protocol_frozen_not_executed":
        raise RuntimeError("validation protocol status is not frozen")
    bindings = protocol["source_bindings"]
    for path_key, digest_key in (
        ("candidate_checkpoint", "candidate_checkpoint_sha256"),
        ("event_overlay_contract", "event_overlay_contract_sha256"),
    ):
        path = Path(bindings[path_key])
        if sha256(path) != bindings[digest_key]:
            raise RuntimeError(f"validation source binding drifted: {path_key}")
    if protocol["validation_2026_opened"]:
        raise RuntimeError("validation protocol already reports 2026 opened")
    checkpoint = json.loads(
        Path(bindings["candidate_checkpoint"]).read_text(encoding="utf-8")
    )
    event_contract = json.loads(
        Path(bindings["event_overlay_contract"]).read_text(encoding="utf-8")
    )
    validate_protocol_derivation(protocol, checkpoint, event_contract)
    event_overlay.validate_frozen_contract(event_contract)
    validate_event_metadata(event_contract)
    return protocol


def validate_event_metadata(event_contract: dict) -> None:
    metadata = event_contract.get("metadata_evidence", {})
    l1_manifest_path = Path(metadata.get("l1_manifest", ""))
    publication_path = Path(metadata.get("l2_publication_report", ""))
    if not l1_manifest_path.is_file() or not publication_path.is_file():
        raise FileNotFoundError("risk-event metadata evidence is unavailable")
    if sha256(l1_manifest_path) != metadata.get("l1_manifest_sha256"):
        raise RuntimeError("risk-event L1 manifest drifted")
    if sha256(publication_path) != metadata.get("l2_publication_report_sha256"):
        raise RuntimeError("risk-event publication report drifted")
    manifest = json.loads(l1_manifest_path.read_text(encoding="utf-8"))
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    expected_apis = {"stk_shock", "stk_high_shock", "stk_alert"}
    if manifest.get("status") != "candidate_l1_l2_ready":
        raise RuntimeError("risk-event full pull is not complete")
    if set(manifest.get("source_apis", [])) != expected_apis:
        raise RuntimeError("risk-event full pull does not cover all three APIs")
    requested = manifest.get("requested_full_range", {})
    if str(requested.get("end", "")) < "20260820":
        raise RuntimeError("risk-event full pull does not cover validation end")
    if publication.get("status") != "production_l1_published":
        raise RuntimeError("risk-event L1 publication is not complete")
    if set(publication.get("tables", {})) != expected_apis:
        raise RuntimeError("risk-event publication does not cover all three APIs")
    coverage = event_contract["diagnostic_coverage"]
    source_min_dates = coverage["source_event_min_dates"]
    for api in sorted(expected_apis):
        quality = publication["tables"][api].get("quality", {})
        if str(quality.get("min_date", "")) != str(source_min_dates[api]):
            raise RuntimeError(f"risk-event source coverage drifted: {api}")
        if int(quality.get("duplicate_key_groups", -1)) != 0:
            raise RuntimeError(f"risk-event source duplicate keys: {api}")
        if int(quality.get("bj_rows", -1)) != 0:
            raise RuntimeError(f"risk-event source contains Beijing rows: {api}")
    if not risk_event.RISK_SIGNAL_DB.is_file():
        raise FileNotFoundError("risk-event L2 signal asset is unavailable")


def table_columns(database: Path, table: str) -> list[str]:
    if not database.is_file():
        raise FileNotFoundError(f"validation database is unavailable: {database}")
    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
            """,
            [str(table)],
        ).fetchall()
    finally:
        connection.close()
    columns = [str(row[0]) for row in rows]
    if not columns:
        raise RuntimeError(f"validation table is unavailable: {database}::{table}")
    return columns


def require_columns(
    actual: list[str], required: set[str], asset_name: str
) -> None:
    missing = sorted(required - set(actual))
    if missing:
        raise RuntimeError(f"validation schema incomplete for {asset_name}: {missing}")


def validate_validation_asset_metadata(protocol: dict) -> dict:
    end_date = str(protocol["validation_boundary"]["end"])
    harness = research_base.load_harness()
    harness.END_DATE = end_date
    _, _, manifests = harness.validate_control_plane()
    l4_assets = {}
    for label, manifest_path in harness.MANIFESTS.items():
        database, table, payload = harness.resolve_manifest(manifest_path)
        required = {
            "trade_date", "stock_code", "pred_prob", "score_source",
            f"executable_{label}_open_return",
        }
        columns = table_columns(database, table)
        require_columns(columns, required, f"L4 {label}")
        if str(payload.get("max_trade_date", "")) < end_date:
            raise RuntimeError(f"L4 {label} manifest does not cover validation end")
        l4_assets[label] = {
            "manifest": str(manifest_path).replace("\\", "/"),
            "manifest_sha256": sha256(manifest_path),
            "database": str(database).replace("\\", "/"),
            "table": table,
            "manifest_min_trade_date": str(payload.get("min_trade_date")),
            "manifest_max_trade_date": str(payload.get("max_trade_date")),
            "required_columns_present": True,
        }
    l2_columns = table_columns(L2_MARKET_DB, L2_MARKET_TABLE)
    require_columns(l2_columns, REQUIRED_L2_COLUMNS, "L2 market")
    event_columns = table_columns(risk_event.RISK_SIGNAL_DB, "stock_risk_signal")
    require_columns(event_columns, REQUIRED_EVENT_COLUMNS, "risk-event signal")
    return {
        "status": "validation_asset_metadata_ready_without_business_rows",
        "control_plane_manifest_count": len(manifests),
        "l4_assets": l4_assets,
        "l2_market": {
            "database": str(L2_MARKET_DB).replace("\\", "/"),
            "table": L2_MARKET_TABLE,
            "required_columns_present": True,
        },
        "risk_event_signal": {
            "database": str(risk_event.RISK_SIGNAL_DB).replace("\\", "/"),
            "table": "stock_risk_signal",
            "required_columns_present": True,
        },
        "business_row_query_count": 0,
        "validation_2026_opened": False,
    }


def make_context(end_date: str, runtime_contract: dict):
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(end_date)
    score_definition = runtime_contract["score_construction"]
    score, order = harness.v95.score_pair(
        arrays,
        float(score_definition["weight_5d"]),
        int(score_definition["smooth_window"]),
        float(score_definition["current_weight"]),
    )
    return pressure_harness.PressureContext(
        harness=harness,
        protocol=protocol,
        definition=harness.production_definition(protocol),
        rules=rules,
        manifests=manifests,
        arrays=arrays,
        access=access,
        score=score,
        order=order,
        active=cadence.rebalance_schedule(len(arrays["dates"]), 20),
        maintenance_block=np.zeros(score.shape, dtype=np.bool_),
        empty_block=np.zeros(score.shape, dtype=np.bool_),
    )


def load_event_features(start_date: str, end_date: str):
    digest_before = sha256(risk_event.RISK_SIGNAL_DB)
    connection = duckdb.connect(str(risk_event.RISK_SIGNAL_DB), read_only=True)
    try:
        frame = connection.execute(
            """
            SELECT signal_date, stock_code, shock_count_1d,
                   high_shock_count_1d,
                   high_shock_count_5d, alert_active_count
            FROM stock_risk_signal
            WHERE signal_date >= ? AND signal_date <= ?
            ORDER BY signal_date, stock_code
            """,
            [str(start_date), str(end_date)],
        ).df()
    finally:
        connection.close()
    digest_after = sha256(risk_event.RISK_SIGNAL_DB)
    if digest_before != digest_after:
        raise RuntimeError("risk-event L2 signal asset changed during validation read")
    if frame.duplicated(["signal_date", "stock_code"]).any():
        raise RuntimeError("validation event keys are not unique")
    if frame["stock_code"].astype(str).str.endswith(".BJ").any():
        raise RuntimeError("validation event source contains Beijing exchange rows")
    return frame, digest_before


def evaluate(daily, actions, start_date: str, end_date: str) -> dict:
    return round1.evaluate_run(daily, actions, start_date, end_date)


def atomic_start_marker(protocol: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if one_shot_already_started():
        raise FileExistsError("one-shot validation was already started")
    claim_payload = {
        "status": "exclusive_validation_claim_acquired",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "pid": os.getpid(),
    }
    descriptor = os.open(
        START_CLAIM,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(claim_payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        round1.atomic_json(START_MARKER, {
            "status": "validation_started_no_retry",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256(PROTOCOL),
            "validation_boundary": protocol["validation_boundary"],
            "exclusive_claim_sha256": sha256(START_CLAIM),
            "risk_event_signal_path": str(risk_event.RISK_SIGNAL_DB),
            "risk_event_signal_sha256_before_business_read": sha256(
                risk_event.RISK_SIGNAL_DB
            ),
        })
    except Exception:
        if not START_MARKER.exists() and START_CLAIM.exists():
            START_CLAIM.unlink()
        raise


def execute_validation(protocol: dict) -> dict:
    boundary = protocol["validation_boundary"]
    start_date, end_date = boundary["start"], boundary["end"]
    validate_validation_asset_metadata(protocol)
    atomic_start_marker(protocol)
    runtime_contract = protocol["runtime_contract"]
    context = make_context(end_date, runtime_contract)
    if context.access["logical_max_date"] != end_date:
        raise RuntimeError("validation data endpoint does not match frozen end date")
    policy = protocol["development_boundary"]["candidate_policy"]
    strong = regime.strong_market_mask(context.score, context.protocol)
    priority_contract = runtime_contract["confirmed_weak_sell_priority"]
    pressure_contract = runtime_contract["pressure_regime"]
    confirmed_weak = confirmed.confirmed_weak_mask(
        strong, int(priority_contract["confirmation_days"])
    )
    extra_age = age_boundary.pressure_age_schedule(
        strong, int(pressure_contract["weak_extra_min_age"])
    )
    volatility_rank = percentile.cross_sectional_percent_rank(
        defensive.trailing_log_volatility(
            context.arrays["close_qfq"],
            int(priority_contract["trailing_volatility_window"]),
        )
    )
    sell_priority = priority.sell_priority_matrix(
        context.score,
        volatility_rank,
        ~confirmed_weak,
        float(priority_contract["volatility_percentile_penalty"]),
    )

    production_daily, production_actions = research_base.run_shell(
        context.harness,
        context.arrays,
        context.protocol,
        context.score,
        context.order,
        context.definition,
        "production_shell",
        end_date,
        actions=True,
    )
    base_daily, base_actions = run_frozen_candidate(
        context,
        policy,
        extra_age,
        round1.BASELINE_COST,
        sell_priority,
        end_date,
    )
    stress_daily, stress_actions = run_frozen_candidate(
        context,
        policy,
        extra_age,
        round1.STRESS_COST,
        sell_priority,
        end_date,
    )
    repeat_daily, repeat_actions = run_frozen_candidate(
        context,
        policy,
        extra_age,
        round1.BASELINE_COST,
        sell_priority,
        end_date,
    )

    event_frame, event_source_sha256 = load_event_features(start_date, end_date)
    coverage = protocol["event_overlay_runtime"]["diagnostic_coverage"]
    coverage_start = str(coverage["event_metric_start"])
    if not (start_date <= coverage_start <= end_date):
        raise RuntimeError("event diagnostic coverage is outside validation boundary")
    if coverage.get("precoverage_state") != "unknown_not_zero":
        raise RuntimeError("event precoverage state is not fail-closed")
    components, event_audit = event_overlay.build_event_masks(
        context.arrays["dates"], context.arrays["stocks"], event_frame
    )
    event_order = event_overlay.score_tiebreak_order(
        context.score,
        context.order,
        components,
        float(policy["replacement_advantage"]),
    )
    event_context = replace(context, order=event_order)
    overlay_daily, overlay_actions = run_frozen_candidate(
        event_context,
        policy,
        extra_age,
        round1.BASELINE_COST,
        sell_priority,
        end_date,
        exit_score_override=context.score,
    )
    overlay_repeat_daily, overlay_repeat_actions = run_frozen_candidate(
        event_context,
        policy,
        extra_age,
        round1.BASELINE_COST,
        sell_priority,
        end_date,
        exit_score_override=context.score,
    )

    metrics = {
        "production_strategy_reference": evaluate(
            production_daily, production_actions, start_date, end_date
        ),
        "fixed10_current_candidate_without_event_overlay": evaluate(
            base_daily, base_actions, start_date, end_date
        ),
        "fixed10_current_candidate_stress_0_65pct": evaluate(
            stress_daily, stress_actions, start_date, end_date
        ),
        "fixed10_current_candidate_with_frozen_event_overlay": evaluate(
            overlay_daily, overlay_actions, start_date, end_date
        ),
    }
    event_coverage_metrics = {
        "fixed10_without_event_tiebreak": evaluate(
            base_daily, base_actions, coverage_start, end_date
        ),
        "fixed10_with_severe_event_tiebreak": evaluate(
            overlay_daily, overlay_actions, coverage_start, end_date
        ),
    }
    deterministic = {
        "base_daily": round1.frame_hash(base_daily) == round1.frame_hash(repeat_daily),
        "base_actions": round1.frame_hash(base_actions) == round1.frame_hash(repeat_actions),
        "overlay_daily": round1.frame_hash(overlay_daily)
        == round1.frame_hash(overlay_repeat_daily),
        "overlay_actions": round1.frame_hash(overlay_actions)
        == round1.frame_hash(overlay_repeat_actions),
    }
    base = metrics["fixed10_current_candidate_without_event_overlay"]
    production = metrics["production_strategy_reference"]
    stress = metrics["fixed10_current_candidate_stress_0_65pct"]
    overlay = event_coverage_metrics["fixed10_with_severe_event_tiebreak"]
    overlay_base = event_coverage_metrics["fixed10_without_event_tiebreak"]
    base_gates = evaluate_base_candidate_gates(
        base,
        production,
        stress,
        deterministic,
        protocol["base_candidate_decision"],
    )
    overlay_diagnostic = {
        key: float(overlay[key] - overlay_base[key])
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    overlay_diagnostic["exactly10_position_ratio"] = overlay[
        "full_10_position_ratio"
    ]
    status = "passed_research_candidate" if all(base_gates.values()) else "rejected_on_2026"
    overlay_status = "diagnostic_only_no_selection_decision"

    validation_dates = expected_validation_dates(
        context.arrays["dates"], start_date, end_date
    )
    signal_date_by_buy_date = expected_t1_signal_map(
        context.arrays["dates"], validation_dates
    )
    persist_validation_outputs((
        ("production", production_daily, production_actions),
        ("fixed10", base_daily, base_actions),
        ("fixed10_stress_0_65pct", stress_daily, stress_actions),
        ("fixed10_event_overlay", overlay_daily, overlay_actions),
    ), start_date, end_date, validation_dates, signal_date_by_buy_date)
    output_sha256s = output_artifact_sha256s()

    result = {
        "status": status,
        "event_overlay_status": overlay_status,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "start_marker_sha256": sha256(START_MARKER),
        "output_artifact_sha256": output_sha256s,
        "validation_boundary": [start_date, end_date],
        "metrics": metrics,
        "event_overlay_coverage_metrics": event_coverage_metrics,
        "base_candidate_gates": base_gates,
        "event_overlay_diagnostic_delta_vs_base": overlay_diagnostic,
        "event_audit": {
            **event_audit,
            "diagnostic_coverage": coverage,
            "precoverage_treated_as_no_event": False,
        },
        "event_source": {
            "path": str(risk_event.RISK_SIGNAL_DB),
            "sha256": event_source_sha256,
        },
        "deterministic_replay": deterministic,
        "validation_access": context.access,
        "validation_2026_opened": True,
        "production_modified": False,
        "trading_triggered": False,
    }
    round1.atomic_json(RESULT_PATH, result)
    return result


def preflight(protocol: dict) -> dict:
    asset_metadata = validate_validation_asset_metadata(protocol)
    return {
        "status": "build_preflight_passed_2026_not_opened",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "validation_boundary": protocol["validation_boundary"],
        "runtime_contract": protocol["runtime_contract"],
        "event_source_exists": risk_event.RISK_SIGNAL_DB.exists(),
        "event_source_sha256": sha256(risk_event.RISK_SIGNAL_DB),
        "validation_asset_metadata": asset_metadata,
        "one_shot_already_started": one_shot_already_started(),
        "validation_2026_opened": False,
        "production_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    args = parser.parse_args()
    validate_execute_confirmation(args.execute, args.confirmation)
    protocol = load_and_validate_protocol()
    if args.execute:
        started_before = one_shot_already_started()
        try:
            result = execute_validation(protocol)
        except Exception as exc:
            if not started_before and START_MARKER.exists() and not FAILURE_PATH.exists():
                round1.atomic_json(FAILURE_PATH, build_failure_record(protocol, exc))
            raise
    else:
        result = preflight(protocol)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

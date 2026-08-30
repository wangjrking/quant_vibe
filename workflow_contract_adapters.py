from __future__ import annotations

from typing import Any

from workflow_contract import build_layer_handoff_contract, validate_layer_handoff_contract


def _bool_gate(name: str, passed: bool, detail: str | None = None, severity: str = "blocker") -> dict[str, Any]:
    item = {"name": name, "passed": bool(passed), "severity": severity}
    if detail:
        item["detail"] = detail
    return item


def _all_alignment_zero(alignment_section: dict[str, Any]) -> bool:
    if not isinstance(alignment_section, dict):
        return False
    for key, value in alignment_section.items():
        if not isinstance(value, dict):
            continue
        for field_name, field_value in value.items():
            if isinstance(field_value, bool):
                continue
            if "count" in field_name or "duplicate" in field_name:
                if int(field_value or 0) != 0:
                    return False
    return True


def _asset_ref(path: Any, table: Any = None) -> str:
    if not path:
        return ""
    if table:
        return f"{path}::{table}"
    return str(path)


def _non_empty_strings(items: list[Any]) -> list[str]:
    return [str(item) for item in items if str(item or "").strip()]


def _zero_or_empty(value: Any) -> bool:
    if value in (None, "", [], {}):
        return True
    if isinstance(value, (int, float)):
        return int(value) == 0
    return False


def build_l1_contract_from_report(
    report: dict[str, Any],
    *,
    workflow_run_id: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    target_trade_date = str(report.get("target_trade_date") or "")
    source_gate = report.get("source_gate") or {}
    ready_for_audit_review = bool(report.get("ready_for_audit_review"))
    allow_next = bool(report.get("allow_l2_continue"))
    gate_pass = bool(source_gate.get("gate_pass"))
    alignment_ok = _all_alignment_zero(report.get("alignment") or {})

    if allow_next:
        status = "audit_passed"
    elif ready_for_audit_review:
        status = "ready_for_audit_review"
    elif gate_pass:
        status = "blocked"
    else:
        status = "source_not_ready"

    payload = build_layer_handoff_contract(
        workflow_run_id=workflow_run_id,
        layer="L1",
        target_trade_date=target_trade_date,
        status=status,
        ready_for_audit_review=ready_for_audit_review,
        allow_next_layer_continue=allow_next,
        active_input_assets=["tushare source APIs"],
        active_output_assets=[str(report.get("active_l1_asset") or "")] if report.get("active_l1_asset") else [],
        gate_checks=[
            _bool_gate(
                "source_gate_pass",
                gate_pass,
                detail=f"daily_rows={source_gate.get('daily_rows')} stk_factor_rows={source_gate.get('stk_factor_rows')}",
            ),
            _bool_gate(
                "alignment_zero_mismatch",
                alignment_ok,
                detail="source/raw parquet/active DuckDB mismatch counts and duplicate groups should all be zero",
            ),
            _bool_gate(
                "no_bj_universe",
                str(report.get("universe_rule") or "").lower() == "no_bj",
                detail=f"universe_rule={report.get('universe_rule')}",
            ),
        ],
        handoff_constraints=[str(item) for item in (report.get("governance_notes") or [])],
        evidence_paths=list(evidence_paths or []),
        residual_risk=[],
        boundaries={
            "no_l2_trigger": not allow_next,
            "no_l3_l8_trigger": True,
            "daily_data_drives_calendar": True,
        },
        layer_payload=report,
    )
    return payload


def build_l2_contract_from_report(
    report: dict[str, Any],
    *,
    workflow_run_id: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    target_trade_date = str(report.get("target_trade_date") or "")
    governance = report.get("governance") or {}
    target_qfq_price_nulls = report.get("target_qfq_price_nulls") or {}
    indicator_mismatch = report.get("target_indicator_sample_mismatch") or {}

    qfq_price_ok = all(int(value or 0) == 0 for value in target_qfq_price_nulls.values())
    indicator_ok = all(int(value or 0) == 0 for value in indicator_mismatch.values())
    duplicate_ok = int(report.get("duplicate_key_groups") or 0) == 0
    no_bj_ok = bool(governance.get("no_bj"))
    duckdb_only_ok = bool(governance.get("duckdb_only"))
    processing_scope = str(report.get("processing_scope") or "full_history")
    formula_field = (
        "target_qfq_formula_mismatch_count"
        if processing_scope == "target_trade_date_only"
        else "full_price_qfq_formula_mismatch_count"
    )
    formula_ok = int(report.get(formula_field) or 0) == 0

    payload = build_layer_handoff_contract(
        workflow_run_id=workflow_run_id,
        layer="L2",
        target_trade_date=target_trade_date,
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=[str(report.get("l1_report") or "")] if report.get("l1_report") else [],
        active_output_assets=[
            f"{report.get('l2_asset')}::{report.get('table')}"
            if report.get("l2_asset") and report.get("table")
            else str(report.get("l2_asset") or "")
        ],
        gate_checks=[
            _bool_gate("duplicate_key_groups_zero", duplicate_ok, detail=f"duplicate_key_groups={report.get('duplicate_key_groups')}"),
            _bool_gate("target_qfq_price_nulls_zero", qfq_price_ok, detail=str(target_qfq_price_nulls)),
            _bool_gate(
                "target_qfq_formula_mismatch_zero"
                if processing_scope == "target_trade_date_only"
                else "full_price_qfq_formula_mismatch_zero",
                formula_ok,
                detail=f"scope={processing_scope} mismatch_count={report.get(formula_field)}",
            ),
            _bool_gate("target_indicator_sample_mismatch_zero", indicator_ok, detail=str(indicator_mismatch), severity="warning"),
            _bool_gate("no_bj_enforced", no_bj_ok, detail=f"governance.no_bj={governance.get('no_bj')}"),
            _bool_gate("duckdb_only_enforced", duckdb_only_ok, detail=f"governance.duckdb_only={governance.get('duckdb_only')}"),
        ],
        handoff_constraints=[
            "daily_data 仍是交易日历与股票集合驱动表",
            "adj_factor 不得反向驱动交易日历",
            "L3 必须继承 no-BJ、DuckDB-only 和显式 qfq 契约",
        ],
        evidence_paths=list(evidence_paths or []),
        residual_risk=[],
        boundaries={
            "touches_l3_or_downstream": False,
            "processing_scope": processing_scope,
            "qfq_price_fields_recomputed": processing_scope != "target_trade_date_only",
            "qfq_price_fields_incremental": processing_scope == "target_trade_date_only",
            "non_qfq_fields_policy": (report.get("scope") or {}).get("non_qfq_fields"),
        },
        layer_payload=report,
    )
    return payload


def build_l3_contract_from_report(
    report: dict[str, Any],
    *,
    workflow_run_id: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    target_trade_date = str(report.get("target_date") or report.get("target_trade_date") or "")
    feature_audit = report.get("feature_audit") or {}
    label_audit = report.get("label_audit") or {}
    legacy_staging = report.get("legacy_staging_opt_in") or {}

    feature_asset = _asset_ref(feature_audit.get("duckdb_path"), feature_audit.get("table_name"))
    label_asset = _asset_ref(label_audit.get("duckdb_path"), label_audit.get("table_name"))
    feature_target_ok = int(feature_audit.get("target_date_rows") or 0) > 0
    feature_dup_ok = int(feature_audit.get("duplicate_rows") or 0) == 0
    label_dup_ok = int(label_audit.get("duplicate_rows") or 0) == 0
    no_bj_ok = int(feature_audit.get("bj_rows") or 0) == 0 and int(feature_audit.get("target_date_bj_rows") or 0) == 0 and int(label_audit.get("bj_rows") or 0) == 0
    qfq_ok = (
        _zero_or_empty(feature_audit.get("has_naked_price_columns"))
        and _zero_or_empty(feature_audit.get("has_legacy_gtja_columns"))
        and bool(feature_audit.get("has_qfq_price_columns"))
        and bool(feature_audit.get("has_qfq_gtja_columns"))
    )
    label_not_forward = str(label_audit.get("max_trade_date") or "") < target_trade_date

    return build_layer_handoff_contract(
        workflow_run_id=workflow_run_id,
        layer="L3",
        target_trade_date=target_trade_date,
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=_non_empty_strings([report.get("feature_target_part_path"), report.get("raw_parts_dir")]),
        active_output_assets=_non_empty_strings([feature_asset, label_asset]),
        gate_checks=[
            _bool_gate("feature_target_date_rows_positive", feature_target_ok, detail=f"target_date_rows={feature_audit.get('target_date_rows')}"),
            _bool_gate("feature_duplicate_rows_zero", feature_dup_ok, detail=f"duplicate_rows={feature_audit.get('duplicate_rows')}"),
            _bool_gate("label_duplicate_rows_zero", label_dup_ok, detail=f"duplicate_rows={label_audit.get('duplicate_rows')}"),
            _bool_gate("no_bj_enforced", no_bj_ok, detail=f"feature_bj={feature_audit.get('bj_rows')} label_bj={label_audit.get('bj_rows')}"),
            _bool_gate("explicit_qfq_schema", qfq_ok, detail="naked qfq-derived columns must be absent; *_qfq columns must exist"),
            _bool_gate("label_not_forward_filled", label_not_forward, detail=f"label_max_trade_date={label_audit.get('max_trade_date')}"),
        ],
        handoff_constraints=[
            "L4 只能读取 active L3 DuckDB 一表一文件资产",
            "最新日 feature freshness 不等于最新日标签成熟",
            "前复权价格、技术字段和 GTJA 因子必须显式使用 _qfq 字段名",
        ],
        evidence_paths=list(evidence_paths or []),
        residual_risk=[
            {
                "severity": "P2",
                "item": "feature_rewrite_mode",
                "detail": str((report.get("feature_rewrite") or {}).get("mode") or ""),
            }
        ],
        boundaries={
            "staging_path": report.get("workspace_dir"),
            "staging_opt_in": legacy_staging,
            "staging_is_not_active_route": True,
            "no_training": True,
            "no_prediction": True,
        },
        layer_payload=report,
    )


def build_l4_contract_from_report(
    report: dict[str, Any],
    *,
    workflow_run_id: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    target_trade_date = str(report.get("target_date") or report.get("target_trade_date") or "")
    inputs = report.get("inputs") or {}
    outputs = report.get("outputs") or []
    output_assets = []
    latest_rows_ok = True
    duplicate_ok = True
    null_pred_ok = True
    no_bj_ok = True
    for item in outputs:
        if not isinstance(item, dict):
            continue
        output_assets.append(_asset_ref(item.get("duckdb_path"), item.get("table")))
        stats = item.get("stats") or {}
        latest_rows_ok = latest_rows_ok and int(stats.get("latest_day_rows") or 0) > 0
        duplicate_ok = duplicate_ok and int(stats.get("duplicate_key_groups") or 0) == 0
        null_pred_ok = null_pred_ok and int(stats.get("null_pred_prob") or 0) == 0
        no_bj_ok = no_bj_ok and int(stats.get("latest_day_bj_rows") or 0) == 0

    boundaries = report.get("boundaries") or {}
    no_boundary_violation = all(bool(boundaries.get(key)) for key in ("no_training", "no_tuning", "no_signal", "no_backtest"))
    allow_next = bool(report.get("allow_l5_continue"))

    return build_layer_handoff_contract(
        workflow_run_id=workflow_run_id,
        layer="L4",
        target_trade_date=target_trade_date,
        status="audit_passed" if allow_next else "ready_for_audit_review",
        ready_for_audit_review=not allow_next,
        allow_next_layer_continue=allow_next,
        active_input_assets=_non_empty_strings([inputs.get("factor_asset"), inputs.get("label_asset")]),
        active_output_assets=_non_empty_strings(output_assets),
        gate_checks=[
            _bool_gate("latest_day_rows_positive", latest_rows_ok),
            _bool_gate("duplicate_key_groups_zero", duplicate_ok),
            _bool_gate("pred_prob_not_null", null_pred_ok),
            _bool_gate("no_bj_predictions", no_bj_ok),
            _bool_gate("no_training_tuning_signal_backtest", no_boundary_violation, detail=str(boundaries)),
        ],
        handoff_constraints=[
            "L5/L6 只能读取 active formal DuckDB manifest",
            "最新日预测刷新不代表最新日标签已成熟可评价",
            "不得回退 MODEL_PREDICTIONS.db、research-only 资产或 legacy odb.db",
        ],
        evidence_paths=list(evidence_paths or []),
        residual_risk=[{"severity": "P2", "item": "label_maturity", "detail": str((inputs.get("factor_status") or {}).get("label_max_trade_date") or "")}],
        boundaries=boundaries,
        layer_payload=report,
    )


def build_l5_contract_from_report(
    report: dict[str, Any],
    *,
    workflow_run_id: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    output_paths = report.get("output_paths") or {}
    old_chain = report.get("old_chain_read_check") or {}
    duplicate_ok = int(report.get("duplicate_signal_stock_keys") or 0) == 0
    signal_ok = int(report.get("signal_row_count") or 0) > 0 and int(report.get("signal_stock_count") or 0) > 0
    no_legacy_ok = not bool(old_chain.get("legacy_odb_used")) and not bool(old_chain.get("model_predictions_sqlite_used")) and not bool(old_chain.get("research_only_used"))

    return build_layer_handoff_contract(
        workflow_run_id=workflow_run_id,
        layer="L5",
        target_trade_date=str(report.get("signal_date") or report.get("target_trade_date") or ""),
        status="ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=False,
        active_input_assets=[str(path) for path in (report.get("input_manifests") or {}).values()],
        active_output_assets=_non_empty_strings([output_paths.get("latest_signal_csv"), output_paths.get("latest_status_json")]),
        gate_checks=[
            _bool_gate("signal_rows_positive", signal_ok, detail=f"rows={report.get('signal_row_count')} stocks={report.get('signal_stock_count')}"),
            _bool_gate("duplicate_signal_stock_keys_zero", duplicate_ok, detail=f"duplicate={report.get('duplicate_signal_stock_keys')}"),
            _bool_gate("legacy_inputs_not_used", no_legacy_ok, detail=str(old_chain)),
        ],
        handoff_constraints=[
            "正式信号只能来自 production_signals latest 文件",
            "candidate/blended 明细只能作为 reports 证据，不得当正式信号",
            "买入日硬门控未完成前不得进入交易执行",
        ],
        evidence_paths=list(evidence_paths or []),
        residual_risk=list(report.get("residual_risks") or []),
        boundaries={"no_trade_execution": True, "no_backtest": True, "no_strategy_param_change": True},
        layer_payload=report,
    )


def build_l6_contract_from_report(
    report: dict[str, Any],
    *,
    workflow_run_id: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    output_paths = report.get("output_paths") or {}
    l7_handoff = report.get("l7_handoff") or {}
    hard_gate_complete = bool(report.get("buy_day_hard_gate_complete"))
    allow_l7 = bool(l7_handoff.get("allowed_now")) or int(report.get("signal_row_count") or 0) > 0

    return build_layer_handoff_contract(
        workflow_run_id=workflow_run_id,
        layer="L6",
        target_trade_date=str(report.get("signal_date") or report.get("target_trade_date") or ""),
        status="pending_buy_day_hard_gate" if not hard_gate_complete else "ready_for_audit_review",
        ready_for_audit_review=True,
        allow_next_layer_continue=allow_l7,
        active_input_assets=_non_empty_strings([output_paths.get("latest_signal_csv"), report.get("strategy_manifest_path"), report.get("trading_rules_path")]),
        active_output_assets=_non_empty_strings([output_paths.get("archive_latest_signal_csv"), output_paths.get("full_history_signal_csv"), output_paths.get("latest_status_json")]),
        gate_checks=[
            _bool_gate("buy_day_hard_gate_not_bypassed", not hard_gate_complete, detail=f"buy_day_hard_gate_complete={hard_gate_complete}", severity="warning"),
            _bool_gate("l7_handoff_defined", bool(l7_handoff), detail=str(l7_handoff)),
            _bool_gate("production_signal_exists", int(report.get("signal_row_count") or 0) > 0, detail=f"signal_row_count={report.get('signal_row_count')}"),
        ],
        handoff_constraints=[
            "L7 只能读取正式 latest signal/status",
            "L7 必须在 buy_date 平台/实时行情可用后重跑买入日硬门控",
            "当前信号发布不等于自动交易授权",
        ],
        evidence_paths=list(evidence_paths or []),
        residual_risk=list(report.get("residual_risks") or []),
        boundaries={"no_auto_trade": True, "requires_buy_day_hard_gate": True},
        layer_payload=report,
    )


def build_l7_contract_from_report(
    report: dict[str, Any],
    *,
    workflow_run_id: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    latest_status = report.get("latest_status") or {}
    hard_gate = report.get("buy_day_hard_gate") or {}
    integrity = report.get("integrity_checks") or {}
    hard_gate_complete = bool(hard_gate.get("buy_day_hard_gate_complete"))
    ready_for_human = bool(hard_gate.get("ready_for_human_confirmation_execution"))
    if ready_for_human:
        status = "ready_for_human_confirmation_execution"
    elif hard_gate_complete:
        status = "pending_user_approval"
    else:
        status = "pending_buy_day_hard_gate"

    integrity_ok = all(bool(value) for value in integrity.values()) if integrity else False
    duplicate_ok = int(report.get("duplicate_signal_stock_keys") or 0) == 0

    return build_layer_handoff_contract(
        workflow_run_id=workflow_run_id,
        layer="L7",
        target_trade_date=str(report.get("signal_date") or latest_status.get("signal_date") or ""),
        status=status,
        ready_for_audit_review=True,
        allow_next_layer_continue=True,
        active_input_assets=_non_empty_strings([((report.get("source_assets") or {}).get("latest_csv")), ((report.get("source_assets") or {}).get("latest_status_json"))]),
        active_output_assets=_non_empty_strings([report.get("snapshot_availability_path"), hard_gate.get("summary_path")]),
        gate_checks=[
            _bool_gate("delivery_integrity_checks_pass", integrity_ok, detail=str(integrity)),
            _bool_gate("duplicate_signal_stock_keys_zero", duplicate_ok, detail=f"duplicate={report.get('duplicate_signal_stock_keys')}"),
            _bool_gate("buy_day_hard_gate_state_recorded", bool(hard_gate.get("status")), detail=str(hard_gate)),
        ],
        handoff_constraints=[
            "L8 只能登记非执行态，除非 L7 已达到 ready_for_human_confirmation_execution",
            "买入日硬门控通过也只代表可人工确认执行，不代表自动交易授权",
            "不得读取 reports candidate/blended 作为正式交易输入",
        ],
        evidence_paths=list(evidence_paths or []),
        residual_risk=[{"severity": "P2", "item": "buy_day_hard_gate", "detail": str(hard_gate)}],
        boundaries={"no_auto_trade": True, "requires_user_approval": True},
        layer_payload=report,
    )


def build_l8_contract_from_report(
    report: dict[str, Any],
    *,
    workflow_run_id: str,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    status = str(report.get("status") or "pending")
    if status not in {"pending", "in_progress", "reviewing", "blocked", "completed"}:
        status = "reviewing"
    agents = report.get("agents") or []
    all_agent_statuses = [str(item.get("status") or "") for item in agents if isinstance(item, dict)]
    has_l1_l8 = len(agents) >= 8

    return build_layer_handoff_contract(
        workflow_run_id=workflow_run_id or str(report.get("workflow_id") or ""),
        layer="L8",
        target_trade_date=str(report.get("target_trade_date") or report.get("workflow_id") or ""),
        status=status,
        ready_for_audit_review=status in {"reviewing", "completed"},
        allow_next_layer_continue=False,
        active_input_assets=[],
        active_output_assets=[],
        gate_checks=[
            _bool_gate("workflow_monitor_has_agents", has_l1_l8, detail=f"agent_count={len(agents)}"),
            _bool_gate("requires_user_approval_recorded", isinstance(report.get("requires_user_approval"), bool), detail=f"requires_user_approval={report.get('requires_user_approval')}"),
            _bool_gate("agent_statuses_recorded", bool(all_agent_statuses), detail=str(all_agent_statuses)),
        ],
        handoff_constraints=[
            "L8 只负责发布登记和工作流状态治理，不代表自动交易执行",
            "生产资产状态必须以 active route / registry / manifest / evidence report 为准",
            "任何 pending_buy_day_hard_gate 都不得被解释为 pending_user_approval",
        ],
        evidence_paths=list(evidence_paths or []),
        residual_risk=list(report.get("open_risks") or []),
        boundaries={"no_business_script": True, "no_auto_trade": True, "governance_only": True},
        layer_payload=report,
    )


def validate_built_contract(payload: dict[str, Any], expected_layer: str) -> list[str]:
    return validate_layer_handoff_contract(payload, expected_layer=expected_layer)

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
STRATEGY_ID = "candidate_dynamic_h2m3_sl06dd0612_pos76_tp060_v20260624"
ARCHIVE_DIR = MAIN / "strategy_library" / "exploration" / STRATEGY_ID

BASE_REPORT_DIR = (
    DATA
    / "reports"
    / "strategy_agent_goal_high_annual_20260623"
    / "top1_no_delist_rerun_20260624"
    / "diversification_tune_20260624"
    / "strict_sync_liquidity_neighborhood_20260624"
    / "formal_horizon_entry_confirmation_20260624"
)
VALIDATION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_sl06_dd0612_pos76_tp060_validation_20260624"
TAKEPROFIT_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_pos76_takeprofit_20260624"
SIGNAL_FILE = TAKEPROFIT_DIR / "signals" / "pos76_sl06_dd0612_tp060.csv"
STRATEGY_MAIN = BASE_REPORT_DIR.parent.parent / "strict_sync_strategy_snapshot_20260624" / "main.py"

FORMAL_MANIFESTS = {
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _metric(summary: dict, tag: str) -> dict:
    for row in summary["time_slices"]:
        if row.get("tag") == tag:
            return row
    raise KeyError(tag)


def _latest_signal(rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("empty signal file")
    return sorted(rows, key=lambda row: (row.get("signal_date") or "", row.get("buy_date") or ""))[-1]


def _prediction_meta() -> dict:
    assets = {}
    for horizon, path in FORMAL_MANIFESTS.items():
        manifest = _read_json(path)
        assets[horizon] = {
            "manifest_path": str(path),
            "approval_status": manifest.get("approval_status"),
            "db_path": manifest.get("db_path"),
            "table": manifest.get("table"),
            "min_trade_date": manifest.get("min_trade_date"),
            "max_trade_date": manifest.get("max_trade_date"),
            "row_count": manifest.get("row_count"),
            "trade_days": manifest.get("trade_days"),
            "null_pred_prob": manifest.get("null_pred_prob"),
            "duplicate_keys": manifest.get("duplicate_keys"),
            "pred_prob_sha256": manifest.get("pred_prob_sha256"),
        }
    return {
        "schema_version": 1,
        "source_mode": "formal_l4_3d5d10d_only",
        "approval_required": "approved_for_l5",
        "legacy_odb_forbidden_by_default": True,
        "assets": assets,
        "score_inputs_used_by_strategy": ["rank_10d", "rank_5d", "rank_3d"],
        "entry_score_formula": "0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d",
    }


def _trading_rules() -> dict:
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "strategy_name": "动态持有 Top1 三周期融合 pos76 6%止盈增强",
        "status": "production_candidate_pending_audit_not_published",
        "score_rule": {
            "rank_weight_10d": 0.78,
            "rank_weight_5d": 0.12,
            "rank_weight_3d": 0.10,
            "rank_weight_1d": 0.0,
            "exit_score_basis": "10d_core_score_table",
        },
        "selection_rule": {
            "top_n": 1,
            "refill_after_buy_day_rejection": True,
            "exclude_bj": True,
            "exclude_st": True,
            "exclude_delisting": True,
            "skip_open_limit_up_buy": True,
            "calendar_or_date_filter": False,
            "industry_filter": False,
            "latest_state_filter_for_history": False,
        },
        "position_rule": {
            "max_positions": 1,
            "target_position_pct": 0.76,
            "cash_buffer": 0.99,
            "position_weight_mode": "single_stock_with_account_drawdown_scaling",
        },
        "holding_rule": {
            "holding_days": 2,
            "max_holding_days": 3,
            "open_daily_score_exit": True,
            "score_exit_entry_ratio": 0.97,
            "score_continue_entry_ratio": 0.98,
            "min_holding_days_before_score_exit": 1,
        },
        "risk_rule": {
            "intraday_stop_loss_pct": 0.06,
            "take_profit_pct": 0.06,
            "account_drawdown_scale_enabled": True,
            "equity_dd_soft_trigger": 0.06,
            "equity_dd_hard_trigger": 0.12,
            "equity_dd_recover_trigger": 0.03,
            "equity_dd_soft_scale": 0.80,
            "equity_dd_hard_scale": 0.60,
            "equity_dd_resize_existing": False,
        },
        "execution_rule": {
            "platform": "juejin",
            "initial_cash": 600000,
            "slippage_ratio": 0.0015,
            "market_db_path": str(DATA / "STOCK_DAILY_DATA.db"),
            "signal_date_rule": "使用 T 日 formal L4 分数生成 T+1 开盘交易信号",
        },
    }


def _validation_payload(summary: dict, audit: dict) -> dict:
    full = _metric(summary, "time_full")
    recent60 = _metric(summary, "time_slice_recent60")
    ytd = _metric(summary, "time_slice_2026ytd")
    stress_rows = _read_csv(VALIDATION_DIR / "stress_summary.csv")
    nearby_rows = _read_csv(VALIDATION_DIR / "nearby_summary.csv")
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "validation_platform": "juejin",
        "validated_at": "2026-06-24",
        "status": "production_candidate_pending_audit_not_published",
        "backtest_window": {"start": full["start"], "end": full["end"]},
        "metrics": {
            "pnl_ratio": full["pnl_ratio"],
            "annual_return": full["annual"],
            "sharpe": full["sharpe"],
            "max_drawdown": full["max_drawdown"],
            "win_ratio": full["win_ratio"],
            "open_count": full["open_count"],
            "close_count": full["close_count"],
            "recent60_annual": recent60["annual"],
            "recent60_sharpe": recent60["sharpe"],
            "recent60_max_drawdown": recent60["max_drawdown"],
            "ytd_annual": ytd["annual"],
        },
        "low_path_dependency": {
            row["anchor"]: {
                "annual_min": row["annual_min"],
                "annual_median": row["annual_median"],
                "annual_max": row["annual_max"],
                "max_drawdown_max": row["max_drawdown_max"],
            }
            for row in nearby_rows
        },
        "contribution_stress": {
            row["variant"]: {
                "annual_return": row.get("full_annual"),
                "sharpe": row.get("full_sharpe"),
                "max_drawdown": row.get("full_max_drawdown"),
                "late_min_annual": row.get("late_min_annual"),
            }
            for row in stress_rows
        },
        "hard_gate_audit": audit,
        "strict_admission": {
            "passed": False,
            "reason": "策略侧验证通过主要强准入检查，但仍需审计智能体复核和用户发布审批；当前仅为探索候选。",
            "strong_gate_note": "收益提升不能单独决定准入；本候选同时保留止盈邻域、低路径依赖、贡献压力和硬过滤证据。",
        },
    }


def _manifest_payload(validation: dict, signal_rows: list[dict]) -> dict:
    full = validation["metrics"]
    latest = _latest_signal(signal_rows)
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "name": "动态持有 Top1 三周期融合 pos76 6%止盈增强",
        "code_name": "dynamic_h2_m3_c098_w78_5d12_3d10_pos76_e097_mh1_sl06_dd0612_tp060",
        "status": "production_candidate_pending_audit_not_published",
        "created_at": "2026-06-24",
        "created_by": "strategy-agent",
        "definition": "基于 formal L4 10D/5D/3D 资产，按 0.78*10D rank + 0.12*5D rank + 0.10*3D rank 选 Top1，目标仓位 76%，动态持有 2-3 日，日内 6% 止损，账户回撤缩放，并新增 6% 止盈。",
        "input_contract": {
            "source_mode": "formal_l4_3d5d10d_only",
            "approval_required": "approved_for_l5",
            "legacy_odb_forbidden_by_default": True,
        },
        "validation": {
            "platform": "juejin",
            "annual_return": full["annual_return"],
            "sharpe": full["sharpe"],
            "max_drawdown": full["max_drawdown"],
            "pnl_ratio": full["pnl_ratio"],
            "win_ratio": full["win_ratio"],
            "open_count": full["open_count"],
            "close_count": full["close_count"],
        },
        "admission_status": {
            "strict_production_admission_passed": False,
            "candidate_status": "production_candidate_pending_audit",
            "reason": "策略侧已完成掘金验证、止盈邻域、低路径依赖、贡献压力和硬过滤审计，但尚未完成审计智能体复核与用户发布审批。",
            "known_risks": [
                "recent60 年化低于无止盈主线，属于持续性折中项，需要审计复核。",
                "当前候选信号仅覆盖到 signal_date=20260622 / buy_date=20260623。",
                "当前归档位于 exploration，不是 production.current。",
            ],
        },
        "current_signal": {
            "latest_signal_date": latest.get("signal_date"),
            "latest_buy_date": latest.get("buy_date"),
            "latest_candidate_signal": "signals/latest_candidate_signal.csv",
            "status_file": "signals/latest_candidate_signal_status.json",
            "status": "research_candidate_signal_not_production",
        },
        "governance": {
            "is_current_l5": False,
            "is_only_registered_l5_production_strategy": False,
            "formal_performance_requires_juejin": True,
            "exclude_bj": True,
            "exclude_st": True,
            "exclude_delisting": True,
            "skip_open_limit_up_buy": True,
            "production": False,
            "requires_audit_before_production": True,
            "requires_user_approval_before_registry_update": True,
        },
    }


def _admission_precheck(validation: dict) -> dict:
    metrics = validation["metrics"]
    audit = validation["hard_gate_audit"]
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "checked_at": "2026-06-24",
        "checked_by": "strategy-agent",
        "status": "production_candidate_pending_audit_not_published",
        "current_registry_unchanged": True,
        "production_current_after_precheck": _read_json(MAIN / "strategy_library" / "registry.json")["production"]["current"],
        "strong_gate_precheck": {
            "formal_l4_only": {"status": "pass", "evidence": "predictions/prediction_table_meta.json"},
            "legacy_odb_forbidden_by_default": {"status": "pass", "evidence": "strategy_manifest.json"},
            "hard_filter_audit": {
                "status": "pass",
                "bj_rows": audit["bj_rows"],
                "buy_st_rows": audit["buy_st_rows"],
                "buy_delist_rows": audit["buy_delist_rows"],
                "open_limit_up_buy_rows": audit["open_limit_up_buy_rows"],
                "evidence": "research/hard_gate_audit.json",
            },
            "juejin_validation": {
                "status": "pass",
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "evidence": "validation.json",
            },
            "takeprofit_neighborhood": {
                "status": "pass_with_recent60_tradeoff",
                "best_case": "pos76_sl06_dd0612_tp060",
                "evidence": "research/takeprofit_report.md",
            },
            "time_slice_continuity": {"status": "pass_with_recent60_tradeoff", "evidence": "research/time_slices.csv"},
            "low_path_dependency": {"status": "pass", "evidence": "research/nearby_summary.csv"},
            "contribution_stress": {
                "status": "pass",
                "drop_top_1pct_annual_return": validation["contribution_stress"]["drop_top_1pct"]["annual_return"],
                "evidence": "research/stress_summary.csv",
            },
            "production_archive_contract": {
                "status": "partial",
                "reason": "已生成探索区候选归档，但未移入 production，未更新 registry，未完成审计复核。",
            },
            "latest_signal_contract": {
                "status": "partial",
                "latest_signal_date": audit["latest_signal_date"],
                "latest_buy_date": audit["latest_buy_date"],
                "reason": "候选信号不是生产信号，且尚未生成 20260623->20260624 信号。",
            },
        },
        "required_before_production_publish": [
            "审计智能体复核本候选归档和证据路径。",
            "用户明确批准发布为生产策略。",
            "按生产目录规范生成 production 归档目录，不覆盖旧生产结果。",
            "更新 registry.json，使 production.current 指向新生产策略。",
            "使用批准生产入口生成最新生产信号，并记录信号状态。",
        ],
    }


def _write_clean_summary(validation: dict) -> None:
    metrics = validation["metrics"]
    lines = [
        "# tp060 候选准入摘要",
        "",
        "## 当前结论",
        "",
        "该候选在当前 `pos76 + sl06 + dd0612` 主线基础上增加通用 6% 止盈。掘金完整验证显示全周期收益明显提升，最大回撤略降，低路径依赖和贡献压力表现通过；但 recent60 年化低于无止盈主线，因此仍需审计复核，不直接发布生产。",
        "",
        "## 掘金指标",
        "",
        f"- 年化：`{float(metrics['annual_return']):.2%}`",
        f"- Sharpe：`{float(metrics['sharpe']):.2f}`",
        f"- 最大回撤：`{float(metrics['max_drawdown']):.2%}`",
        f"- recent60 年化：`{float(metrics['recent60_annual']):.2%}`",
        f"- 2026YTD 年化：`{float(metrics['ytd_annual']):.2%}`",
        "",
        "## 强准入判断",
        "",
        "- formal L4 输入：通过。",
        "- 北交所、ST/风险警示、退市、开盘涨停买入：审计命中均为 0。",
        "- 低路径依赖：三个近期开仓锚点附近最差年化均大于 100%。",
        "- 贡献压力：去最高 1% 代理贡献信号后年化仍为 463.39%。",
        "- 持续性折中：recent60 年化从无止盈主线的 44.79% 降至 34.95%，需审计复核。",
        "",
        "## 状态",
        "",
        "`production_candidate_pending_audit_not_published`，未修改 `registry.json`，未发布生产，未生成生产信号。",
    ]
    (ARCHIVE_DIR / "research" / "tp060_admission_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    signal_rows = _read_csv(SIGNAL_FILE)
    summary = _read_json(VALIDATION_DIR / "validation_summary.json")
    audit = _read_json(VALIDATION_DIR / "hard_gate_audit.json")
    validation = _validation_payload(summary, audit)

    if ARCHIVE_DIR.exists():
        shutil.rmtree(ARCHIVE_DIR)

    _copy(SIGNAL_FILE, ARCHIVE_DIR / "signals" / "historical_sl06_dd0612_pos76_tp060.csv")
    _write_csv(ARCHIVE_DIR / "signals" / "latest_candidate_signal.csv", [_latest_signal(signal_rows)])
    _write_json(
        ARCHIVE_DIR / "signals" / "latest_candidate_signal_status.json",
        {
            "status": "research_candidate_signal_not_production",
            "latest_signal_date": _latest_signal(signal_rows).get("signal_date"),
            "latest_buy_date": _latest_signal(signal_rows).get("buy_date"),
        },
    )

    for name in [
        "validation_summary.json",
        "hard_gate_audit.json",
        "time_slices.csv",
        "nearby_summary.csv",
        "stress_summary.csv",
        "validation_report.md",
    ]:
        _copy(VALIDATION_DIR / name, ARCHIVE_DIR / "research" / name)
    for name in ["summary.csv", "summary_by_annual.csv", "takeprofit_report.md", "hard_gate_audit.json"]:
        dst = "takeprofit_hard_gate_audit.json" if name == "hard_gate_audit.json" else name
        _copy(TAKEPROFIT_DIR / name, ARCHIVE_DIR / "research" / dst)
    for log_name in [
        "time_full.log",
        "time_slice_recent60.log",
        "stress_base_full_20240605.log",
        "stress_drop_top_1pct_full_20240605.log",
    ]:
        _copy(VALIDATION_DIR / "logs" / log_name, ARCHIVE_DIR / "backtests" / log_name)
    _copy(STRATEGY_MAIN, ARCHIVE_DIR / "code_snapshot" / "juejin_main.py")
    for script in [
        "tune_dynamic_h2_m3_pos76_takeprofit_20260624.py",
        "validate_dynamic_h2_m3_sl06_dd0612_pos76_tp060_20260624.py",
        "archive_dynamic_h2_m3_sl06_dd0612_pos76_tp060_candidate_20260624.py",
    ]:
        _copy(MAIN / script, ARCHIVE_DIR / "code_snapshot" / script)

    _write_json(ARCHIVE_DIR / "predictions" / "prediction_table_meta.json", _prediction_meta())
    _write_json(ARCHIVE_DIR / "trading_rules.json", _trading_rules())
    _write_json(ARCHIVE_DIR / "validation.json", validation)
    _write_json(ARCHIVE_DIR / "strategy_manifest.json", _manifest_payload(validation, signal_rows))
    _write_json(ARCHIVE_DIR / "admission_precheck.json", _admission_precheck(validation))
    _write_clean_summary(validation)
    print(json.dumps({"archive_dir": str(ARCHIVE_DIR), "strategy_id": STRATEGY_ID}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

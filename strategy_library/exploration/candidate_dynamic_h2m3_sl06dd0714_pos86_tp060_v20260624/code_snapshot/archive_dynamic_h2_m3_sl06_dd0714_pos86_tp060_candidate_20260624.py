from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
STRATEGY_ID = "candidate_dynamic_h2m3_sl06dd0714_pos86_tp060_v20260624"
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
VALIDATION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_sl06_dd0714_pos86_tp060_validation_20260624"
HIGH_POSITION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_dd0714_tp060_high_position_20260624"
SIGNAL_FILE = HIGH_POSITION_DIR / "signals" / "dd0714_tp060_pos86.csv"
STRATEGY_MAIN = BASE_REPORT_DIR.parent.parent / "strict_sync_strategy_snapshot_20260624" / "main.py"

FORMAL_MANIFESTS = {
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty csv rows for {path}")
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


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def metric(summary: dict, tag: str) -> dict:
    for row in summary["time_slices"]:
        if row.get("tag") == tag:
            return row
    raise KeyError(tag)


def latest_signal(rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("signal file is empty")
    return sorted(rows, key=lambda row: (row.get("signal_date") or "", row.get("buy_date") or ""))[-1]


def prediction_meta() -> dict:
    assets = {}
    for horizon, path in FORMAL_MANIFESTS.items():
        manifest = read_json(path)
        assets[horizon] = {
            "manifest_path": str(path),
            "approval_status": manifest.get("approval_status"),
            "source_type": manifest.get("source_type"),
            "db_path": manifest.get("db_path"),
            "table": manifest.get("table"),
            "market_db_path": manifest.get("market_db_path"),
            "min_trade_date": manifest.get("min_trade_date"),
            "max_trade_date": manifest.get("max_trade_date"),
            "row_count": manifest.get("row_count"),
            "trade_days": manifest.get("trade_days"),
            "stock_count": manifest.get("stock_count"),
            "null_pred_prob": manifest.get("null_pred_prob"),
            "duplicate_keys": manifest.get("duplicate_keys"),
        }
    return {
        "schema_version": 1,
        "source_mode": "formal_l4_3d5d10d_only",
        "approval_required": "approved_for_l5",
        "legacy_odb_forbidden_by_default": True,
        "assets": assets,
        "entry_score_formula": "0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d",
        "score_inputs_used_by_strategy": ["rank_10d", "rank_5d", "rank_3d"],
    }


def trading_rules() -> dict:
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "strategy_name": "动态持有 Top1 三周期融合 pos86 止损止盈回撤缩放候选",
        "status": "exploration_candidate_pending_audit_not_published",
        "score_rule": {
            "rank_weight_10d": 0.78,
            "rank_weight_5d": 0.12,
            "rank_weight_3d": 0.10,
            "rank_weight_1d": 0.0,
            "entry_score_formula": "0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d",
            "exit_score_basis": "10d_core_score_table",
        },
        "selection_rule": {
            "top_n": 1,
            "max_positions": 1,
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
            "target_position_pct": 0.86,
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
            "equity_dd_soft_trigger": 0.07,
            "equity_dd_hard_trigger": 0.14,
            "equity_dd_recover_trigger": 0.04,
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


def validation_payload(summary: dict, audit: dict) -> dict:
    full = metric(summary, "time_full")
    recent60 = metric(summary, "time_slice_recent60")
    recent120 = metric(summary, "time_slice_recent120")
    ytd = metric(summary, "time_slice_2026ytd")
    stress_rows = read_csv(VALIDATION_DIR / "stress_summary.csv")
    nearby_rows = read_csv(VALIDATION_DIR / "nearby_summary.csv")
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "validation_platform": "juejin",
        "validated_at": "2026-06-24",
        "status": "exploration_candidate_pending_audit_not_published",
        "backtest_window": {"start": full["start"], "end": full["end"]},
        "metrics": {
            "pnl_ratio": full["pnl_ratio"],
            "annual_return": full["annual"],
            "sharpe": full["sharpe"],
            "max_drawdown": full["max_drawdown"],
            "win_ratio": full["win_ratio"],
            "open_count": full["open_count"],
            "close_count": full["close_count"],
            "avg_invested_pct": full["avg_invested_pct"],
            "ge80_ratio": full["ge80_ratio"],
            "recent60_annual": recent60["annual"],
            "recent60_sharpe": recent60["sharpe"],
            "recent60_max_drawdown": recent60["max_drawdown"],
            "recent120_annual": recent120["annual"],
            "recent120_sharpe": recent120["sharpe"],
            "ytd_annual": ytd["annual"],
            "ytd_sharpe": ytd["sharpe"],
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
            "reason": "收益指标仅为弱准入。该候选硬过滤、近期开仓、低路径依赖和贡献压力测试表现较好，但最大回撤为 39.94%，非常接近 40% 风险边界，且尚未完成审计智能体复核和用户生产发布审批。",
            "strong_gate_note": "敏感性、持续性、低路径依赖、无未来信息泄漏、硬过滤和生产归档契约为强准入项。",
        },
    }


def manifest_payload(validation: dict, signal_rows: list[dict]) -> dict:
    metrics = validation["metrics"]
    latest = latest_signal(signal_rows)
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "name": "动态持有 Top1 三周期融合 pos86 止损止盈回撤缩放候选",
        "code_name": "dynamic_h2_m3_c098_w78_5d12_3d10_pos86_e097_mh1_sl06_dd0714_tp060",
        "status": "exploration_candidate_pending_audit_not_published",
        "created_at": "2026-06-24",
        "created_by": "strategy-agent",
        "definition": "基于 formal L4 10D/5D/3D 资产，按 0.78*10D rank + 0.12*5D rank + 0.10*3D rank 选 Top1，目标仓位 86%，动态持有 2-3 日，日内 6% 止损，6% 止盈，账户回撤缩放 soft=7% / hard=14% / scale=0.80/0.60。",
        "validation": {
            "platform": "juejin",
            "annual_return": metrics["annual_return"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "recent60_annual": metrics["recent60_annual"],
            "ytd_annual": metrics["ytd_annual"],
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
            "production": False,
            "formal_performance_requires_juejin": True,
            "legacy_odb_forbidden_by_default": True,
            "exclude_bj": True,
            "exclude_st": True,
            "exclude_delisting": True,
            "skip_open_limit_up_buy": True,
            "requires_audit_before_production": True,
            "requires_user_approval_before_registry_update": True,
        },
    }


def admission_precheck(validation: dict) -> dict:
    metrics = validation["metrics"]
    audit = validation["hard_gate_audit"]
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "checked_at": "2026-06-24",
        "checked_by": "strategy-agent",
        "status": "exploration_candidate_pending_audit_not_published",
        "current_registry_unchanged": True,
        "production_current_after_precheck": read_json(MAIN / "strategy_library" / "registry.json")["production"]["current"],
        "weak_admission_metrics": {
            "annual_return": metrics["annual_return"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "note": "收益指标是弱准入，不能单独决定生产发布。",
        },
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
            "time_slice_continuity": {
                "status": "pass_with_recent60_weakness",
                "recent60_annual": metrics["recent60_annual"],
                "recent120_annual": metrics["recent120_annual"],
                "evidence": "research/time_slices.csv",
            },
            "low_path_dependency": {"status": "pass", "evidence": "research/nearby_summary.csv"},
            "contribution_stress": {
                "status": "pass",
                "drop_top_1pct_annual_return": validation["contribution_stress"]["drop_top_1pct"]["annual_return"],
                "evidence": "research/stress_summary.csv",
            },
            "parameter_sensitivity": {
                "status": "pass_with_drawdown_edge_risk",
                "chosen_case": "dd0714_tp060_pos86",
                "evidence": "research/high_position_summary.csv",
                "reason": "pos88/pos90 年化更高但最大回撤超过 40%，pos86 是回撤红线内的收益增强点；不过 39.94% 贴近红线，需要审计复核。",
            },
            "production_archive_contract": {
                "status": "partial",
                "reason": "已生成 exploration 候选归档；未移入 production，未更新 registry，未生成生产信号。",
            },
        },
        "production_publish_blockers": [
            "需要审计智能体复核强准入证据，尤其是最大回撤 39.94% 贴近 40% 的边界风险。",
            "需要用户明确批准生产发布和 registry 切换。",
            "需要按生产目录规范生成 production 归档，不覆盖旧生产业务结果。",
            "需要使用批准后的生产入口生成最新生产信号。",
        ],
    }


def write_report(validation: dict) -> None:
    metrics = validation["metrics"]
    lines = [
        "# pos86 止盈回撤缩放候选准入摘要",
        "",
        "## 当前结论",
        "",
        "该候选在 pos84 基础上把目标仓位提高到 86%，其余规则保持同口径：formal L4 10D/5D/3D 融合、Top1、动态持有 2-3 日、6% 止损、6% 止盈、账户回撤缩放 dd0714。掘金完整验证显示收益提升明显，硬过滤和近期开仓检查干净。",
        "",
        "但收益指标只是弱准入。该版本最大回撤为 39.94%，非常接近 40% 风险边界，因此当前只能归档为探索候选，不能直接声明生产准入通过。",
        "",
        "## 掘金指标",
        "",
        f"- 年化收益：`{float(metrics['annual_return']):.2%}`",
        f"- Sharpe：`{float(metrics['sharpe']):.2f}`",
        f"- 最大回撤：`{float(metrics['max_drawdown']):.2%}`",
        f"- recent60 年化：`{float(metrics['recent60_annual']):.2%}`",
        f"- recent120 年化：`{float(metrics['recent120_annual']):.2%}`",
        f"- 2026YTD 年化：`{float(metrics['ytd_annual']):.2%}`",
        "",
        "## 强准入判断",
        "",
        "- formal L4 输入：通过。",
        "- 北交所、ST/风险警示、退市、开盘涨停买入：审计命中均为 0。",
        "- 低路径依赖：2025-07、2025-10、2026-01 三组近期开仓锚点最差年化均大于 100%。",
        "- 贡献压力：去掉最高 1% 代理贡献信号后年化仍为 642.05%。",
        "- 参数敏感性：pos88/pos90 收益更高但回撤超过 40%，pos86 是回撤红线内的收益增强点，属于通过但贴边。",
        "- 持续性风险：recent60 年化为 38.75%，弱于全周期；需继续观察和审计复核。",
        "",
        "## 当前状态",
        "",
        "`exploration_candidate_pending_audit_not_published`。未修改 `registry.json`，未发布生产策略，未生成生产信号。",
    ]
    (ARCHIVE_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if ARCHIVE_DIR.exists():
        raise FileExistsError(f"archive already exists: {ARCHIVE_DIR}")

    signal_rows = read_csv(SIGNAL_FILE)
    summary = read_json(VALIDATION_DIR / "validation_summary.json")
    audit = read_json(VALIDATION_DIR / "hard_gate_audit.json")
    validation = validation_payload(summary, audit)

    copy_file(SIGNAL_FILE, ARCHIVE_DIR / "signals" / "candidate_signals.csv")
    write_csv(ARCHIVE_DIR / "signals" / "latest_candidate_signal.csv", [latest_signal(signal_rows)])
    write_json(
        ARCHIVE_DIR / "signals" / "latest_candidate_signal_status.json",
        {
            "status": "research_candidate_signal_not_production",
            "latest_signal_date": latest_signal(signal_rows).get("signal_date"),
            "latest_buy_date": latest_signal(signal_rows).get("buy_date"),
        },
    )

    for name in [
        "validation_summary.json",
        "hard_gate_audit.json",
        "time_slices.csv",
        "nearby_summary.csv",
        "nearby_cold_starts.csv",
        "stress_summary.csv",
        "stress_cases.csv",
        "proxy_summary.json",
        "proxy_signal_contribution.csv",
        "validation_report.md",
    ]:
        copy_file(VALIDATION_DIR / name, ARCHIVE_DIR / "research" / name)

    for name in ["summary.csv", "summary_by_annual.csv", "detail.csv", "high_position_report.md", "hard_gate_audit.json"]:
        archive_name = "high_position_hard_gate_audit.json" if name == "hard_gate_audit.json" else f"high_position_{name}"
        copy_file(HIGH_POSITION_DIR / name, ARCHIVE_DIR / "research" / archive_name)

    for log_name in [
        "time_full.log",
        "time_slice_recent60.log",
        "time_slice_2026ytd.log",
        "stress_base_full_20240605.log",
        "stress_drop_top_1pct_full_20240605.log",
    ]:
        copy_file(VALIDATION_DIR / "logs" / log_name, ARCHIVE_DIR / "backtests" / log_name)

    copy_file(STRATEGY_MAIN, ARCHIVE_DIR / "code_snapshot" / "juejin_main.py")
    for script in [
        "tune_dynamic_h2_m3_dd0714_tp060_high_position_20260624.py",
        "validate_dynamic_h2_m3_sl06_dd0714_pos86_tp060_20260624.py",
        "archive_dynamic_h2_m3_sl06_dd0714_pos86_tp060_candidate_20260624.py",
    ]:
        copy_file(MAIN / script, ARCHIVE_DIR / "code_snapshot" / script)

    write_json(ARCHIVE_DIR / "predictions" / "prediction_table_meta.json", prediction_meta())
    write_json(ARCHIVE_DIR / "trading_rules.json", trading_rules())
    write_json(ARCHIVE_DIR / "validation.json", validation)
    write_json(ARCHIVE_DIR / "strategy_manifest.json", manifest_payload(validation, signal_rows))
    write_json(ARCHIVE_DIR / "admission_precheck.json", admission_precheck(validation))
    write_report(validation)

    print(json.dumps({"strategy_id": STRATEGY_ID, "archive_dir": str(ARCHIVE_DIR)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

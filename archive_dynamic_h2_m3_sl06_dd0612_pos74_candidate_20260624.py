from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
STRATEGY_ID = "candidate_dynamic_h2m3_sl06dd0612_pos74_v20260624"
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
VALIDATION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_sl06_dd0612_pos74_validation_20260624"
POSITION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_sl06_dd0612_position_20260624"
SIGNAL_FILE = POSITION_DIR / "signals" / "sl06_dd0612_pos74.csv"
STRATEGY_MAIN = (
    BASE_REPORT_DIR.parent.parent
    / "strict_sync_strategy_snapshot_20260624"
    / "main.py"
)

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


def _latest_signal(rows: list[dict]) -> dict:
    if not rows:
        raise RuntimeError("empty signal file")
    return sorted(rows, key=lambda row: (row.get("signal_date") or "", row.get("buy_date") or ""))[-1]


def _metric(summary: dict, tag: str) -> dict:
    for row in summary["time_slices"]:
        if row.get("tag") == tag:
            return row
    raise KeyError(tag)


def _copy_evidence_files() -> None:
    _copy(SIGNAL_FILE, ARCHIVE_DIR / "signals" / "historical_sl06_dd0612_pos74.csv")
    rows = _read_csv(SIGNAL_FILE)
    latest = _latest_signal(rows)
    _write_csv(ARCHIVE_DIR / "signals" / "latest_candidate_signal.csv", [latest])

    for name in [
        "validation_summary.json",
        "hard_gate_audit.json",
        "time_slices.csv",
        "nearby_summary.csv",
        "stress_summary.csv",
        "validation_report.md",
    ]:
        _copy(VALIDATION_DIR / name, ARCHIVE_DIR / "research" / name)
    for name in ["summary.csv", "summary_by_annual.csv", "position_report.md"]:
        _copy(POSITION_DIR / name, ARCHIVE_DIR / "research" / name)

    for log_name in [
        "time_full.log",
        "time_slice_recent60.log",
        "stress_base_full_20240605.log",
        "stress_drop_top_1pct_full_20240605.log",
    ]:
        _copy(VALIDATION_DIR / "logs" / log_name, ARCHIVE_DIR / "backtests" / log_name)

    _copy(STRATEGY_MAIN, ARCHIVE_DIR / "code_snapshot" / "juejin_main.py")
    for script in [
        "tune_dynamic_h2_m3_sl06_dd0612_position_20260624.py",
        "validate_dynamic_h2_m3_sl06_dd0612_pos74_20260624.py",
        "tune_dynamic_h2_m3_intraday_risk_20260624.py",
        "validate_dynamic_h2_m3_sl06_dd0612_20260624.py",
    ]:
        _copy(MAIN / script, ARCHIVE_DIR / "code_snapshot" / script)


def _build_prediction_meta() -> dict:
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


def _build_trading_rules() -> dict:
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "strategy_name": "动态持有 Top1 三周期融合 pos74",
        "status": "production_candidate_pending_audit_not_published",
        "score_rule": {
            "rank_weight_10d": 0.78,
            "rank_weight_5d": 0.12,
            "rank_weight_3d": 0.10,
            "rank_weight_1d": 0.0,
            "score_table_for_exit": "score_div_top1_w90_5d10_h5_e099",
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
            "target_position_pct": 0.74,
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
            "account_drawdown_scale_enabled": True,
            "equity_dd_soft_trigger": 0.06,
            "equity_dd_hard_trigger": 0.12,
            "equity_dd_recover_trigger": 0.03,
            "equity_dd_soft_scale": 0.80,
            "equity_dd_hard_scale": 0.60,
            "equity_dd_resize_existing": False,
            "take_profit_pct": None,
        },
        "execution_rule": {
            "platform": "juejin",
            "initial_cash": 600000,
            "slippage_ratio": 0.0015,
            "market_db_path": str(DATA / "STOCK_DAILY_DATA.db"),
            "signal_date_rule": "使用 T 日 formal L4 分数生成 T+1 开盘交易信号。",
        },
    }


def _build_validation(summary: dict, audit: dict) -> dict:
    full = _metric(summary, "time_full")
    recent60 = _metric(summary, "time_slice_recent60")
    ytd = _metric(summary, "time_slice_2026ytd")
    stress = _read_csv(VALIDATION_DIR / "stress_summary.csv")
    stress_by_variant = {row["variant"]: row for row in stress}
    nearby = _read_csv(VALIDATION_DIR / "nearby_summary.csv")
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
            for row in nearby
        },
        "contribution_stress": {
            variant: {
                "annual_return": row.get("full_annual"),
                "sharpe": row.get("full_sharpe"),
                "max_drawdown": row.get("full_max_drawdown"),
                "late_min_annual": row.get("late_min_annual"),
                "reason": row.get("reason"),
            }
            for variant, row in stress_by_variant.items()
        },
        "hard_gate_audit": audit,
        "strict_admission": {
            "passed": False,
            "candidate_status": "production_candidate_pending_audit",
            "reason": "策略侧强准入验证已显著改善，但尚未完成生产归档审计复核，且未获用户明确发布批准。",
        },
        "evidence": {
            "validation_summary": "research/validation_summary.json",
            "hard_gate_audit": "research/hard_gate_audit.json",
            "gm_full_log": "backtests/time_full.log",
            "gm_recent60_log": "backtests/time_slice_recent60.log",
            "historical_signals": "signals/historical_sl06_dd0612_pos74.csv",
        },
    }


def _build_reproduction() -> dict:
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "status": "production_candidate_pending_audit_not_published",
        "python_commands": [
            ".venv\\Scripts\\python.exe quant\\main\\tune_dynamic_h2_m3_sl06_dd0612_position_20260624.py",
            ".venv\\Scripts\\python.exe quant\\main\\validate_dynamic_h2_m3_sl06_dd0612_pos74_20260624.py",
        ],
        "juejin_strategy_dir": str(
            BASE_REPORT_DIR.parent.parent / "strict_sync_strategy_snapshot_20260624"
        ),
        "source_signal": str(SIGNAL_FILE),
        "source_validation_dir": str(VALIDATION_DIR),
        "source_position_dir": str(POSITION_DIR),
        "notes": [
            "该归档是探索区候选归档，不修改 registry.json，不替换当前生产策略。",
            "正式生产发布前必须由审计智能体复核，并由用户明确批准。",
        ],
    }


def _build_manifest(summary: dict, latest: dict) -> dict:
    full = _metric(summary, "time_full")
    return {
        "schema_version": 1,
        "strategy_id": STRATEGY_ID,
        "name": "动态持有 Top1 三周期融合 pos74",
        "code_name": "dynamic_h2_m3_c098_w78_5d12_3d10_pos74_e097_mh1_sl06_dd0612",
        "status": "production_candidate_pending_audit_not_published",
        "created_at": "2026-06-24",
        "created_by": "strategy-agent",
        "definition": "基于 formal L4 10D/5D/3D 资产，按 0.78*10D rank + 0.12*5D rank + 0.10*3D rank 选 Top1，目标仓位 74%，动态持有 2-3 日，日内 6% 止损，并启用账户回撤缩放。",
        "input_contract": {
            "source_mode": "formal_l4_3d5d10d_only",
            "approval_required": "approved_for_l5",
            "legacy_odb_forbidden_by_default": True,
        },
        "validation": {
            "platform": "juejin",
            "annual_return": full["annual"],
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
            "reason": "策略侧已完成主要强准入验证，但尚未完成审计智能体复核和用户发布批准。",
            "known_risks": [
                "去最高 1% 代理信号后最大回撤升至约 40.37%，仍需披露尾部贡献压力。",
                "当前候选信号仅覆盖到 signal_date=20260622 / buy_date=20260623；尚未生成 20260623->20260624 候选信号。",
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


def _write_report(summary: dict, validation: dict, latest: dict) -> None:
    full = validation["metrics"]
    stress = validation["contribution_stress"]["drop_top_1pct"]
    lines = [
        "# 动态持有 Top1 三周期融合 pos74 候选归档",
        "",
        "## 当前状态",
        "",
        "该目录是探索区生产候选归档，状态为 `production_candidate_pending_audit_not_published`。本次未修改 `registry.json`，未替换当前生产策略，未生成生产信号。",
        "",
        "## 核心规则",
        "",
        "- 输入资产：formal L4 10D / 5D / 3D，均要求 `approved_for_l5`。",
        "- 入场分数：`0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d`。",
        "- 每日最多买入：`Top1`。",
        "- 目标仓位：`74%`。",
        "- 持有规则：第 `2` 个交易日开始检查，满足延持条件最多持有到第 `3` 个交易日。",
        "- 分数退出：10D 核心分数低于入场分数 `0.97`。",
        "- 日内风控：浮亏达到 `6%` 触发止损。",
        "- 账户回撤缩放：软触发 `6%`，硬触发 `12%`，缩放比例 `0.80 / 0.60`。",
        "- 硬过滤：不买北交所，不买 ST / 风险警示，不买退市标记，不买开盘涨停。",
        "",
        "## 掘金结果",
        "",
        f"- 年化收益：`{float(full['annual_return']):.2%}`",
        f"- Sharpe：`{float(full['sharpe']):.2f}`",
        f"- 最大回撤：`{float(full['max_drawdown']):.2%}`",
        f"- recent60 年化：`{float(full['recent60_annual']):.2%}`",
        f"- 2026YTD 年化：`{float(full['ytd_annual']):.2%}`",
        f"- 去最高 1% 代理信号后年化：`{float(stress['annual_return']):.2%}`",
        "",
        "## 信号覆盖",
        "",
        f"- 最新候选信号日：`{latest.get('signal_date')}`",
        f"- 最新候选买入日：`{latest.get('buy_date')}`",
        "- 该信号不是生产信号，不得直接交交易智能体执行。",
        "",
        "## 残留风险",
        "",
        "- 尚未完成审计智能体复核。",
        "- 尚未获得用户明确发布批准。",
        "- 尚未更新 `registry.json`，因此不是当前生产策略。",
        "- 去最高 1% 代理信号后回撤仍会上升，尾部贡献压力需要继续披露。",
        "",
        "## 证据路径",
        "",
        "- `strategy_manifest.json`",
        "- `trading_rules.json`",
        "- `validation.json`",
        "- `reproduction_v20260624.json`",
        "- `predictions/prediction_table_meta.json`",
        "- `research/validation_summary.json`",
        "- `research/hard_gate_audit.json`",
        "- `backtests/time_full.log`",
    ]
    (ARCHIVE_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    for path in [SIGNAL_FILE, VALIDATION_DIR / "validation_summary.json", VALIDATION_DIR / "hard_gate_audit.json"]:
        if not path.exists():
            raise FileNotFoundError(path)

    _copy_evidence_files()
    signal_rows = _read_csv(SIGNAL_FILE)
    latest = _latest_signal(signal_rows)
    summary = _read_json(VALIDATION_DIR / "validation_summary.json")
    audit = _read_json(VALIDATION_DIR / "hard_gate_audit.json")
    validation = _build_validation(summary, audit)

    _write_json(ARCHIVE_DIR / "strategy_manifest.json", _build_manifest(summary, latest))
    _write_json(ARCHIVE_DIR / "trading_rules.json", _build_trading_rules())
    _write_json(ARCHIVE_DIR / "validation.json", validation)
    _write_json(ARCHIVE_DIR / "reproduction_v20260624.json", _build_reproduction())
    _write_json(ARCHIVE_DIR / "predictions" / "prediction_table_meta.json", _build_prediction_meta())
    _write_json(
        ARCHIVE_DIR / "signals" / "latest_candidate_signal_status.json",
        {
            "schema_version": 1,
            "status": "research_candidate_signal_not_production",
            "latest_signal_date": latest.get("signal_date"),
            "latest_buy_date": latest.get("buy_date"),
            "production_signal": False,
            "blocked_for_production_delivery": True,
            "reason": "探索区生产候选信号，尚未完成审计复核和用户发布批准。",
        },
    )
    _write_report(summary, validation, latest)
    print(json.dumps({"archive_dir": str(ARCHIVE_DIR), "status": "ok"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
STRATEGY_ID = "candidate_dynamic_h2m3_sl06dd0712_pos90_tp060_v20260624"
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
VALIDATION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_sl06_dd0712_pos90_tp060_validation_20260624"
DD_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_pos90_tp060_dd_neighborhood_20260624"
SIGNAL_FILE = DD_DIR / "signals" / "tp060_pos90_dd0712_s7050.csv"
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
        "strategy_name": "动态持有 Top1 三周期融合 pos90 收益回撤改善候选",
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
            "target_position_pct": 0.90,
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
            "equity_dd_hard_trigger": 0.12,
            "equity_dd_recover_trigger": 0.03,
            "equity_dd_soft_scale": 0.70,
            "equity_dd_hard_scale": 0.50,
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


def validation_payload(summary: dict) -> dict:
    full = metric(summary, "time_full")
    recent60 = metric(summary, "time_slice_recent60")
    recent120 = metric(summary, "time_slice_recent120")
    ytd = metric(summary, "time_slice_2026ytd")
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
            for row in summary["nearby_summary"]
        },
        "contribution_stress": {
            row["variant"]: {
                "annual_return": row.get("full_annual"),
                "sharpe": row.get("full_sharpe"),
                "max_drawdown": row.get("full_max_drawdown"),
                "late_min_annual": row.get("late_min_annual"),
                "reason": row.get("reason"),
            }
            for row in summary["stress_summary"]
        },
        "hard_gate_audit": summary["audit"],
        "strong_admission_status": {
            "status": "candidate_pending_audit_not_production",
            "passed_local_precheck": True,
            "reason": "收益指标属于弱准入。该候选完整掘金验证、低路径依赖、贡献压力和硬过滤预检通过；但仍需审计复核和用户生产发布审批。",
            "residual_risks": [
                "recent60 年化仍明显低于全周期年化，需要审计复核持续性解释。",
                "尚未写入 production registry，不能称为当前生产策略。",
            ],
        },
    }


def report_text(validation: dict, latest: dict) -> str:
    metrics = validation["metrics"]
    stress = validation["contribution_stress"]
    return "\n".join(
        [
            "# dd0712 pos90 收益回撤改善候选归档",
            "",
            "## 当前结论",
            "",
            "该版本是探索候选，不是生产策略。它在 `pos90` 高收益主线基础上，将账户回撤缩放调整为 `soft=7% / hard=12% / scale=0.70/0.50`。掘金结果显示年化提升且最大回撤下降，但仍需审计复核。",
            "",
            "## 掘金结果",
            "",
            f"- 年化收益：`{float(metrics['annual_return']):.2%}`",
            f"- Sharpe：`{float(metrics['sharpe']):.2f}`",
            f"- 最大回撤：`{float(metrics['max_drawdown']):.2%}`",
            f"- recent60 年化：`{float(metrics['recent60_annual']):.2%}`",
            f"- recent120 年化：`{float(metrics['recent120_annual']):.2%}`",
            f"- 2026YTD 年化：`{float(metrics['ytd_annual']):.2%}`",
            "",
            "## 强准入状态",
            "",
            "- 硬过滤预检通过：北交所、ST/风险警示、退市、开盘涨停买入命中均为 `0`。",
            "- 低路径依赖预检通过：三个近期开仓锚点最差年化均为正且大于 100%。",
            f"- 贡献压力通过：去最高 1% 代理信号后年化仍为 `{float(stress['drop_top_1pct']['annual_return']):.2%}`。",
            "- 未通过生产发布：尚未审计复核，尚未写入生产 registry。",
            "",
            "## 最新信号覆盖",
            "",
            f"- 最新 signal_date：`{latest.get('signal_date')}`",
            f"- 最新 buy_date：`{latest.get('buy_date')}`",
            f"- 最新标的：`{latest.get('stock_code')}` / `{latest.get('name')}`",
            "",
            "## 证据路径",
            "",
            f"- 验证目录：`{VALIDATION_DIR}`",
            f"- 源信号：`{SIGNAL_FILE}`",
            f"- 归档目录：`{ARCHIVE_DIR}`",
            "",
        ]
    )


def main() -> int:
    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(VALIDATION_DIR)
    if not SIGNAL_FILE.exists():
        raise FileNotFoundError(SIGNAL_FILE)
    summary = read_json(VALIDATION_DIR / "validation_summary.json")
    validation = validation_payload(summary)
    signal_rows = read_csv(SIGNAL_FILE)
    latest = latest_signal(signal_rows)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(ARCHIVE_DIR / "prediction_table_meta.json", prediction_meta())
    write_json(ARCHIVE_DIR / "trading_rules.json", trading_rules())
    write_json(ARCHIVE_DIR / "validation.json", validation)
    write_json(
        ARCHIVE_DIR / "strategy_manifest.json",
        {
            "schema_version": 1,
            "strategy_id": STRATEGY_ID,
            "status": "exploration_candidate_pending_audit_not_published",
            "created_at": "2026-06-24",
            "validation_platform": "juejin",
            "source_signal_file": str(SIGNAL_FILE),
            "validation_dir": str(VALIDATION_DIR),
            "registry_updated": False,
            "production_published": False,
        },
    )
    (ARCHIVE_DIR / "report.md").write_text(report_text(validation, latest), encoding="utf-8")

    copy_file(SIGNAL_FILE, ARCHIVE_DIR / "signals" / SIGNAL_FILE.name)
    copy_file(VALIDATION_DIR / "validation_report.md", ARCHIVE_DIR / "backtests" / "validation_report.md")
    copy_file(VALIDATION_DIR / "validation_summary.json", ARCHIVE_DIR / "backtests" / "validation_summary.json")
    copy_file(VALIDATION_DIR / "time_slices.csv", ARCHIVE_DIR / "backtests" / "time_slices.csv")
    copy_file(VALIDATION_DIR / "nearby_summary.csv", ARCHIVE_DIR / "backtests" / "nearby_summary.csv")
    copy_file(VALIDATION_DIR / "stress_summary.csv", ARCHIVE_DIR / "backtests" / "stress_summary.csv")
    copy_file(VALIDATION_DIR / "hard_gate_audit.json", ARCHIVE_DIR / "backtests" / "hard_gate_audit.json")
    copy_file(DD_DIR / "summary.csv", ARCHIVE_DIR / "research_grid" / "dd_neighborhood_summary.csv")
    copy_file(DD_DIR / "dd_neighborhood_report.md", ARCHIVE_DIR / "research_grid" / "dd_neighborhood_report.md")
    if STRATEGY_MAIN.exists():
        copy_file(STRATEGY_MAIN, ARCHIVE_DIR / "code_snapshot" / "main.py")
    copy_file(Path(__file__), ARCHIVE_DIR / "code_snapshot" / Path(__file__).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

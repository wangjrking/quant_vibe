from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"

STRATEGY_ID = "candidate_dynamic_h2m3_sl06dd0712_pos8975_c975_tp070_scale7555_v20260625"
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
BOUNDARY_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625"
VALIDATION_DIR = BASE_REPORT_DIR / "dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_validation_20260625"
SIGNAL_FILE = BOUNDARY_DIR / "signals" / "pos8975_scale7555.csv"
STRATEGY_MAIN = BASE_REPORT_DIR.parent.parent / "strict_sync_strategy_snapshot_20260624" / "main.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def latest_signal(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: (str(row.get("signal_date") or ""), str(row.get("buy_date") or "")))


def metrics_from_summary(summary: dict) -> dict:
    time_rows = summary["time_slices"]
    full = next(row for row in time_rows if row.get("slice") == "full")
    recent60 = next(row for row in time_rows if row.get("slice") == "slice_recent60")
    recent120 = next(row for row in time_rows if row.get("slice") == "slice_recent120")
    ytd = next(row for row in time_rows if row.get("slice") == "slice_2026ytd")
    return {
        "annual_return": float(full["annual"]),
        "pnl_ratio": float(full["pnl_ratio"]),
        "sharpe": float(full["sharpe"]),
        "max_drawdown": float(full["max_drawdown"]),
        "win_ratio": float(full["win_ratio"]),
        "open_count": int(float(full["open_count"])),
        "close_count": int(float(full["close_count"])),
        "avg_invested_pct": float(full["avg_invested_pct"]),
        "recent60_annual": float(recent60["annual"]),
        "recent120_annual": float(recent120["annual"]),
        "ytd_annual": float(ytd["annual"]),
    }


def trading_rules() -> dict:
    return {
        "strategy_id": STRATEGY_ID,
        "strategy_name": "动态持有 Top1 高仓位边界收益候选",
        "status": "exploration_candidate_pending_audit_not_published",
        "model_input": {
            "source": "formal L4 3D/5D/10D rank cache",
            "entry_weights": {"10d": 0.78, "5d": 0.12, "3d": 0.10},
            "uses_1d": False,
        },
        "stock_pool_filters": {
            "exclude_bj": True,
            "exclude_st_risk_warning": True,
            "exclude_delist": True,
            "skip_open_limit_up_buy": True,
            "no_industry_filter": True,
            "no_month_or_date_exclusion": True,
            "no_latest_state_backfill": True,
        },
        "position_rule": {
            "max_positions": 1,
            "target_position_pct": 0.8975,
            "cash_buffer_mode": "gm_cash_buffer_0p99",
        },
        "holding_rule": {
            "holding_days": 2,
            "max_holding_days": 3,
            "score_exit_entry_ratio": 0.970,
            "score_continue_entry_ratio": 0.975,
            "min_holding_days_before_score_exit": 1,
        },
        "risk_rule": {
            "intraday_stop_loss_pct": 0.06,
            "take_profit_pct": 0.07,
            "equity_drawdown_risk_mode": True,
            "dd_soft_trigger": 0.07,
            "dd_hard_trigger": 0.12,
            "dd_recover_trigger": 0.03,
            "dd_soft_scale": 0.75,
            "dd_hard_scale": 0.55,
        },
        "execution_rule": {
            "signal_date_rule": "使用 T 日 formal L4 分数生成 T+1 开盘交易研究信号",
            "force_sell_market_order": True,
            "juejin_backtest_slippage_ratio": 0.0015,
            "initial_cash": 600000,
        },
        "governance_note": "风格暴露和风格漂移只作为提示项，不作为单独强准入；正式生产发布仍需用户批准和审计复核。",
    }


def report_text(metrics: dict, latest: dict, audit: dict) -> str:
    return "\n".join(
        [
            "# pos8975_scale7555 研究候选归档",
            "",
            "## 当前结论",
            "",
            "该版本是探索候选，不是生产策略。它在 c975_tp070 基础上只微调目标仓位和账户回撤缩放：目标仓位 `89.75%`，账户回撤 soft/hard 缩放 `75%/55%`。",
            "",
            "生产 registry 未修改，正式发布仍需要用户批准和审计复核。",
            "",
            "## 掘金验证结果",
            "",
            f"- 年化收益：`{metrics['annual_return']:.2%}`",
            f"- 累计收益：`{metrics['pnl_ratio']:.2%}`",
            f"- Sharpe：`{metrics['sharpe']:.2f}`",
            f"- 最大回撤：`{metrics['max_drawdown']:.2%}`",
            f"- recent60 年化：`{metrics['recent60_annual']:.2%}`",
            f"- recent120 年化：`{metrics['recent120_annual']:.2%}`",
            f"- 2026YTD 年化：`{metrics['ytd_annual']:.2%}`",
            f"- 平均仓位：`{metrics['avg_invested_pct']:.2%}`",
            "",
            "## 硬过滤审计",
            "",
            f"- 审计文件数：`{audit['audited_files']}`",
            f"- 失败文件数：`{audit['failed_files']}`",
            f"- 买入日行情缺失：`{audit['buy_join_missing']}`",
            f"- 北交所命中：`{audit['bj_rows']}`",
            f"- ST / 风险警示命中：`{audit['buy_st_rows']}`",
            f"- 退市命中：`{audit['buy_delist_rows']}`",
            f"- 开盘涨停买入命中：`{audit['open_limit_up_buy_rows']}`",
            "",
            "## 最新研究信号",
            "",
            f"- 最新 signal_date：`{latest.get('signal_date')}`",
            f"- 最新 buy_date：`{latest.get('buy_date')}`",
            f"- 最新标的：`{latest.get('stock_code')}` / `{latest.get('name')}`",
            "",
            "## 证据路径",
            "",
            f"- 验证目录：`{VALIDATION_DIR}`",
            f"- 边界搜索目录：`{BOUNDARY_DIR}`",
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
    audit = read_json(VALIDATION_DIR / "hard_gate_audit.json")
    signal_rows = read_csv(SIGNAL_FILE)
    latest = latest_signal(signal_rows)
    metrics = metrics_from_summary(summary)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        ARCHIVE_DIR / "strategy_manifest.json",
        {
            "schema_version": 1,
            "strategy_id": STRATEGY_ID,
            "status": "exploration_candidate_pending_audit_not_published",
            "created_at": "2026-06-25",
            "validation_platform": "juejin",
            "production_registry_updated": False,
            "production_published": False,
            "source_signal_file": str(SIGNAL_FILE),
            "validation_dir": str(VALIDATION_DIR),
            "boundary_search_dir": str(BOUNDARY_DIR),
            "style_exposure_and_drift": "risk_prompt_only_not_hard_gate",
        },
    )
    write_json(ARCHIVE_DIR / "trading_rules.json", trading_rules())
    write_json(
        ARCHIVE_DIR / "validation.json",
        {
            "metrics": metrics,
            "hard_gate_audit": {
                "audited_files": audit["audited_files"],
                "failed_files": audit["failed_files"],
                "buy_join_missing": audit["buy_join_missing"],
                "bj_rows": audit["bj_rows"],
                "buy_st_rows": audit["buy_st_rows"],
                "buy_delist_rows": audit["buy_delist_rows"],
                "open_limit_up_buy_rows": audit["open_limit_up_buy_rows"],
            },
            "nearby_summary": summary["nearby_summary"],
            "stress_summary": summary["stress_summary"],
            "status": "exploration_candidate_pending_audit_not_published",
        },
    )
    (ARCHIVE_DIR / "report.md").write_text(report_text(metrics, latest, audit), encoding="utf-8")

    copy_file(SIGNAL_FILE, ARCHIVE_DIR / "signals" / SIGNAL_FILE.name)
    copy_file(VALIDATION_DIR / "validation_report.md", ARCHIVE_DIR / "backtests" / "validation_report.md")
    copy_file(VALIDATION_DIR / "validation_summary.json", ARCHIVE_DIR / "backtests" / "validation_summary.json")
    copy_file(VALIDATION_DIR / "time_slices.csv", ARCHIVE_DIR / "backtests" / "time_slices.csv")
    copy_file(VALIDATION_DIR / "nearby_summary.csv", ARCHIVE_DIR / "backtests" / "nearby_summary.csv")
    copy_file(VALIDATION_DIR / "stress_summary.csv", ARCHIVE_DIR / "backtests" / "stress_summary.csv")
    copy_file(VALIDATION_DIR / "hard_gate_audit.json", ARCHIVE_DIR / "backtests" / "hard_gate_audit.json")
    copy_file(BOUNDARY_DIR / "summary_by_admission.csv", ARCHIVE_DIR / "research_grid" / "high_pos_boundary_summary_by_admission.csv")
    copy_file(BOUNDARY_DIR / "high_pos_boundary_report.md", ARCHIVE_DIR / "research_grid" / "high_pos_boundary_report.md")
    if STRATEGY_MAIN.exists():
        copy_file(STRATEGY_MAIN, ARCHIVE_DIR / "code_snapshot" / "main.py")
    copy_file(Path(__file__), ARCHIVE_DIR / "code_snapshot" / Path(__file__).name)
    copy_file(
        MAIN / "validate_dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_20260625.py",
        ARCHIVE_DIR / "code_snapshot" / "validate_dynamic_h2_m3_sl06_dd0712_pos8975_scale7555_20260625.py",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

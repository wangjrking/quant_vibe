from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from export_dynamic_top1_formal_signals import export_signals


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
MODEL_DB = DATA / "model_predictions" / "MODEL_PREDICTIONS.db"

PRODUCTION_ROOT = MAIN / "strategy_library" / "production"
REGISTRY_PATH = MAIN / "strategy_library" / "registry.json"
PRODUCTION_SIGNALS_DIR = DATA / "production_signals"
PRODUCTION_TASKS = MAIN / "config" / "production_tasks.json"
PRODUCTION_TASKS_EXAMPLE = MAIN / "config" / "production_tasks.example.json"

SOURCE_STRATEGY_ID = "prod_dynamic_h2m3_pos8975_scale7555_v20260625"
TARGET_STRATEGY_ID = "prod_dynamic_top1_amt9p5w_dd08_115_v20260625"
SOURCE_DIR = PRODUCTION_ROOT / SOURCE_STRATEGY_ID
TARGET_DIR = PRODUCTION_ROOT / TARGET_STRATEGY_ID

VALIDATION_DIR = DATA / "reports" / "strategy_agent_amt9p5w_s7555_dd08_115_current_validation_20260625"
FULL_SIGNAL_FILE = DATA / "reports" / "strategy_agent_prod_filter_amount_interpolate_20260625" / "signals" / "prod_filter_amt9p5w_mv20w.csv"

MANIFESTS = {
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _table_summary(table: str) -> dict[str, Any]:
    conn = sqlite3.connect(str(MODEL_DB))
    try:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                COUNT(DISTINCT trade_date) AS trade_days,
                COUNT(DISTINCT stock_code) AS stock_count,
                MIN(trade_date) AS min_trade_date,
                MAX(trade_date) AS max_trade_date,
                SUM(CASE WHEN pred_prob IS NULL THEN 1 ELSE 0 END) AS null_pred_prob
            FROM "{table}"
            """
        ).fetchone()
        latest_date = str(row[4])
        latest_rows = conn.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE trade_date = ?',
            (latest_date,),
        ).fetchone()[0]
        dup = conn.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT trade_date, stock_code
                FROM "{table}"
                GROUP BY trade_date, stock_code
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "row_count": int(row[0]),
        "trade_days": int(row[1]),
        "stock_count": int(row[2]),
        "min_trade_date": str(row[3]),
        "max_trade_date": latest_date,
        "latest_day_rows": int(latest_rows),
        "null_pred_prob": int(row[5] or 0),
        "duplicate_key_groups": int(dup),
    }


def _prediction_table_meta() -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for key, manifest_path in MANIFESTS.items():
        manifest = _read_json(manifest_path)
        db_path = (manifest_path.parent / manifest["db_path"]).resolve()
        assets[key] = {
            "manifest_path": str(manifest_path),
            "approval_status": manifest.get("approval_status"),
            "source_type": manifest.get("source_type"),
            "db_path": str(db_path),
            "table": manifest["table"],
            "summary": _table_summary(manifest["table"]),
        }
    return {
        "schema_version": 1,
        "strategy_id": TARGET_STRATEGY_ID,
        "created_at": "2026-06-25",
        "formal_l4_assets": assets,
        "legacy_odb_allowed": False,
    }


def _build_strategy_manifest(signal_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "strategy_id": TARGET_STRATEGY_ID,
        "strategy_name": "单票动态Top1 量额95k dd08_115",
        "status": "production",
        "created_at": "2026-06-25",
        "published_at": "2026-06-25",
        "published_by": "strategy-agent",
        "validation_platform": "juejin",
        "production_registry_updated": True,
        "production_published": True,
        "source_strategy_id": SOURCE_STRATEGY_ID,
        "validation_dir": str(VALIDATION_DIR),
        "full_history_signal_file": str(FULL_SIGNAL_FILE),
        "input_contract": {
            "formal_manifest_3d": str(MANIFESTS["3d"]),
            "formal_manifest_5d": str(MANIFESTS["5d"]),
            "formal_manifest_10d": str(MANIFESTS["10d"]),
            "market_db_path": str(DATA / "STOCK_DAILY_DATA.db"),
            "allow_legacy": False,
        },
        "current_signal": {
            "signal_date": signal_status["signal_date"],
            "buy_date": signal_status["buy_date"],
            "archive_file": str(TARGET_DIR / "signals" / f"latest_signal_{signal_status['signal_date']}_for_{signal_status['buy_date']}.csv"),
            "latest_file": str(PRODUCTION_SIGNALS_DIR / f"{TARGET_STRATEGY_ID}_latest.csv"),
            "status_file": str(TARGET_DIR / "signals" / f"latest_signal_status_{signal_status['signal_date']}_for_{signal_status['buy_date']}.json"),
            "buy_day_hard_gate_complete": bool(signal_status["buy_day_hard_gate_complete"]),
            "buy_day_realtime_checks_required": not bool(signal_status["buy_day_hard_gate_complete"]),
        },
    }


def _build_trading_rules(signal_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": TARGET_STRATEGY_ID,
        "strategy_name": "单票动态Top1 量额95k dd08_115",
        "status": "production",
        "model_input": {
            "source": "formal L4 3D/5D/10D rank cache",
            "weight_name": "w78_5d12_3d10",
            "entry_weights": {"10d": 0.78, "5d": 0.12, "3d": 0.10},
            "uses_1d": False,
        },
        "selection_rule": {
            "filter_name": "amt9p5w_mv20w_no_bj_st_delist",
            "exclude_bj": True,
            "exclude_st_risk_warning": True,
            "exclude_delist": True,
            "skip_open_limit_up_buy": True,
            "no_industry_filter": True,
            "no_month_or_date_exclusion": True,
            "no_latest_state_backfill": True,
            "amount_min": 95000.0,
            "total_mv_min": 200000.0,
            "total_mv_max": None,
            "turnover_min": None,
        },
        "position_rule": {
            "max_positions": 1,
            "target_position_pct": 0.8975,
            "cash_buffer_mode": "gm_cash_buffer_0p99",
        },
        "holding_rule": {
            "holding_name": "h2_m3_c0975",
            "holding_days": 2,
            "max_holding_days": 3,
            "score_exit_entry_ratio": 0.97,
            "score_continue_entry_ratio": 0.975,
            "min_holding_days_before_score_exit": 1,
        },
        "risk_rule": {
            "intraday_stop_loss_pct": 0.06,
            "take_profit_pct": 0.07,
            "equity_drawdown_risk_mode": True,
            "dd_soft_trigger": 0.08,
            "dd_hard_trigger": 0.115,
            "dd_recover_trigger": 0.03,
            "dd_soft_scale": 0.75,
            "dd_hard_scale": 0.55,
        },
        "execution_rule": {
            "signal_date_rule": "使用 T 日 formal L4 分数生成 T+1 开盘交易信号",
            "force_sell_market_order": True,
            "juejin_backtest_slippage_ratio": 0.0015,
            "initial_cash": 600000,
            "current_signal_status": "hard_gate_complete" if signal_status["buy_day_hard_gate_complete"] else "pending_buy_day_market_audit",
            "signal_exporter": "export_dynamic_top1_formal_signals.py",
        },
        "governance_note": "不买北交所、不买 ST/风险警示、不买退市、不买开盘涨停；风格暴露与风格漂移仅作提示项。",
    }


def _build_validation(signal_status: dict[str, Any]) -> dict[str, Any]:
    summary = _read_json(VALIDATION_DIR / "validation_summary.json")
    audit = _read_json(VALIDATION_DIR / "hard_gate_audit.json")
    full = next(item for item in summary["time_slices"] if item["slice"] == "full")
    recent60 = next(item for item in summary["time_slices"] if item["slice"] == "slice_recent60")
    recent120 = next(item for item in summary["time_slices"] if item["slice"] == "slice_recent120")
    ytd = next(item for item in summary["time_slices"] if item["slice"] == "slice_2026ytd")
    return {
        "metrics": {
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
        },
        "hard_gate_audit": {
            "audited_files": audit["audited_files"],
            "failed_files": audit["failed_files"],
            "buy_join_missing": audit["buy_join_missing"],
            "bj_rows": audit["bj_rows"],
            "buy_st_rows": audit["buy_st_rows"],
            "buy_delist_rows": audit["buy_delist_rows"],
            "open_limit_up_buy_rows": audit["open_limit_up_buy_rows"],
            "latest_signal_date": audit["latest_signal_date"],
            "latest_buy_date": audit["latest_buy_date"],
        },
        "nearby_summary": summary["nearby_summary"],
        "stress_summary": summary["stress_summary"],
        "status": "production",
        "validation_platform": "juejin",
        "production_publication": {
            "published_at": "2026-06-25",
            "user_approved": True,
            "buy_day_hard_gate_complete_for_latest_signal": bool(signal_status["buy_day_hard_gate_complete"]),
        },
    }


def _build_report(signal_status: dict[str, Any]) -> str:
    validation = _build_validation(signal_status)
    metrics = validation["metrics"]
    nearby = validation["nearby_summary"]
    lines = [
        "# 单票动态Top1 量额95k dd08_115",
        "",
        "## 当前结论",
        "",
        "这是当前已通过策略准入的生产替换版。它沿用 formal L4 3D/5D/10D 模型输入，不使用 1D，不使用行业/月度/日期限制，只在单票动态 Top1 链路上把候选池收紧到 `amount >= 95000`，并把账户回撤软阈值提升到 `8%`、硬阈值保持 `11.5%`。",
        "",
        "## 核心规则",
        "",
        "- 模型打分：`0.78 * rank_10d + 0.12 * rank_5d + 0.10 * rank_3d`",
        "- 候选过滤：不买北交所、不买 ST/风险警示、不买退市、不买开盘涨停",
        "- 流动性过滤：`amount >= 95000`，`total_mv >= 200000`",
        "- 持仓规则：单票，目标仓位 `89.75%`",
        "- 持有规则：至少 2 天，最多 3 天，`score_exit_entry_ratio = 0.97`，`score_continue_entry_ratio = 0.975`",
        "- 风控规则：止损 `6%`，止盈 `7%`，账户回撤 soft/hard `8% / 11.5%`，缩放 `75% / 55%`",
        "",
        "## 掘金验证",
        "",
        f"- 全周期年化：`{metrics['annual_return']:.2%}`",
        f"- 全周期累计收益：`{metrics['pnl_ratio']:.2%}`",
        f"- Sharpe：`{metrics['sharpe']:.2f}`",
        f"- 最大回撤：`{metrics['max_drawdown']:.2%}`",
        f"- recent120 年化：`{metrics['recent120_annual']:.2%}`",
        f"- recent60 年化：`{metrics['recent60_annual']:.2%}`",
        f"- 2026YTD 年化：`{metrics['ytd_annual']:.2%}`",
        f"- 平均持仓率：`{metrics['avg_invested_pct']:.2%}`",
        "",
        "## 低路径依赖复核",
        "",
    ]
    for item in nearby:
        lines.append(
            f"- 锚点 `{item['anchor']}`：近邻空仓启动最差年化 `{float(item['annual_min']):.2%}`，中位年化 `{float(item['annual_median']):.2%}`"
        )
    lines.extend(
        [
            "",
            "## 最新生产信号",
            "",
            f"- signal_date：`{signal_status['signal_date']}`",
            f"- buy_date：`{signal_status['buy_date']}`",
            f"- buy_day_hard_gate_complete：`{bool(signal_status['buy_day_hard_gate_complete'])}`",
            "",
            "## 证据路径",
            "",
            f"- 验证目录：`{VALIDATION_DIR}`",
            f"- 全量信号：`{FULL_SIGNAL_FILE}`",
            f"- 最新信号：`{PRODUCTION_SIGNALS_DIR / f'{TARGET_STRATEGY_ID}_latest.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_registry(signal_status: dict[str, Any]) -> None:
    registry = _read_json(REGISTRY_PATH)
    current = registry.get("production", {}).get("current")
    old_items = registry.get("production", {}).get("strategies", [])
    historical = registry.setdefault("historical_production", {}).setdefault("retained_directories", [])
    if current and current != TARGET_STRATEGY_ID:
        old_item = next((item for item in old_items if item.get("strategy_id") == current), None)
        if old_item and not any(item.get("strategy_id") == current for item in historical):
            historical.insert(
                0,
                {
                    "strategy_id": old_item["strategy_id"],
                    "path": old_item["path"],
                    "status": "historical_archive",
                    "reason": "用户于 2026-06-25 批准以 amt9p5w_s7555_dd08_115 替换当前生产策略，旧生产版本下架保留复现。",
                },
            )
    metrics = _build_validation(signal_status)["metrics"]
    registry["updated_at"] = "2026-06-25"
    registry["production"] = {
        "current": TARGET_STRATEGY_ID,
        "strategies": [
            {
                "strategy_id": TARGET_STRATEGY_ID,
                "name": "单票动态Top1 量额95k dd08_115",
                "path": f"strategy_library/production/{TARGET_STRATEGY_ID}",
                "status": "production",
                "published_at": "2026-06-25",
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "validation_platform": "juejin",
            }
        ],
    }
    _write_json(REGISTRY_PATH, registry)


def _update_production_tasks() -> None:
    for path in [PRODUCTION_TASKS, PRODUCTION_TASKS_EXAMPLE]:
        payload = _read_json(path)
        payload["strategies"] = [
            {
                "strategy_id": TARGET_STRATEGY_ID,
                "enabled": True,
                "prediction_manifest": "config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json",
                "description": "当前唯一 L5 生产策略：单票动态Top1 amt9p5w dd08_115。只读取 approved_for_l5 formal L4 3D/5D/10D 资产，自动生成最新信号。",
                "steps": [
                    {
                        "name": "export_signals",
                        "command": [
                            "{python}",
                            "export_dynamic_top1_formal_signals.py",
                            "--strategy-dir",
                            "strategy_library/production/{strategy_id}",
                            "--output",
                            "{signal_dir}/{strategy_id}_latest.csv",
                            "--status-output",
                            "{signal_dir}/{strategy_id}_latest_status.json",
                        ],
                    }
                ],
            }
        ]
        _write_json(path, payload)


def main() -> int:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(SOURCE_DIR)
    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(VALIDATION_DIR)
    if not FULL_SIGNAL_FILE.exists():
        raise FileNotFoundError(FULL_SIGNAL_FILE)

    _copy_tree(SOURCE_DIR, TARGET_DIR)
    bootstrap_status = {
        "signal_date": "20260624",
        "buy_date": "20260625",
        "buy_day_hard_gate_complete": False,
    }
    _write_json(TARGET_DIR / "strategy_manifest.json", _build_strategy_manifest(bootstrap_status))
    _write_json(TARGET_DIR / "trading_rules.json", _build_trading_rules(bootstrap_status))
    signal_output = TARGET_DIR / "signals" / "signals_latest.csv"
    status_output = TARGET_DIR / "signals" / "latest_signal_status_auto.json"
    export_result = export_signals(
        strategy_dir=TARGET_DIR,
        output=signal_output,
        status_output=status_output,
        signal_date="20260624",
        buy_date="20260625",
    )
    signal_status = export_result["status"]
    signal_rows = export_result["rows"]

    archive_signal = TARGET_DIR / "signals" / f"latest_signal_{signal_status['signal_date']}_for_{signal_status['buy_date']}.csv"
    archive_status = TARGET_DIR / "signals" / f"latest_signal_status_{signal_status['signal_date']}_for_{signal_status['buy_date']}.json"
    shutil.copy2(signal_output, archive_signal)
    shutil.copy2(status_output, archive_status)
    shutil.copy2(signal_output, PRODUCTION_SIGNALS_DIR / f"{TARGET_STRATEGY_ID}_latest.csv")
    shutil.copy2(FULL_SIGNAL_FILE, TARGET_DIR / "signals" / FULL_SIGNAL_FILE.name)

    _write_json(TARGET_DIR / "strategy_manifest.json", _build_strategy_manifest(signal_status))
    _write_json(TARGET_DIR / "trading_rules.json", _build_trading_rules(signal_status))
    _write_json(TARGET_DIR / "validation.json", _build_validation(signal_status))
    _write_json(TARGET_DIR / "prediction_table_meta.json", _prediction_table_meta())
    _write_json(
        TARGET_DIR / "reproduction_v20260625.json",
        {
            "schema_version": 1,
            "strategy_id": TARGET_STRATEGY_ID,
            "published_at": "2026-06-25",
            "publisher": "strategy-agent",
            "signal_date": signal_status["signal_date"],
            "buy_date": signal_status["buy_date"],
            "publish_script": str(MAIN / "publish_amt9p5w_s7555_dd08_115_to_l5_20260625.py"),
            "signal_export_script": str(MAIN / "export_dynamic_top1_formal_signals.py"),
            "validation_dir": str(VALIDATION_DIR),
            "full_history_signal_file": str(FULL_SIGNAL_FILE),
            "latest_signal_file": str(PRODUCTION_SIGNALS_DIR / f"{TARGET_STRATEGY_ID}_latest.csv"),
            "notes": [
                "只读取 approved_for_l5 formal L4 3D/5D/10D 资产。",
                "不使用 1D，不使用行业/月度/日期限制。",
                "正式收益结论以掘金验证为准。",
            ],
        },
    )
    (TARGET_DIR / "report.md").write_text(_build_report(signal_status), encoding="utf-8")

    _update_registry(signal_status)
    _update_production_tasks()

    print(
        json.dumps(
            {
                "strategy_id": TARGET_STRATEGY_ID,
                "signal_status": signal_status,
                "latest_signal_file": str(PRODUCTION_SIGNALS_DIR / f"{TARGET_STRATEGY_ID}_latest.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

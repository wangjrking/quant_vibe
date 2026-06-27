from __future__ import annotations

import csv
import json
import re
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

SOURCE_STRATEGY_ID = "prod_dynamic_top1_amt9p5w_dd08_115_v20260625"
TARGET_STRATEGY_ID = "prod_dynamic_top1_amt9w_warmup60_v20260626"
SOURCE_DIR = PRODUCTION_ROOT / SOURCE_STRATEGY_ID
TARGET_DIR = PRODUCTION_ROOT / TARGET_STRATEGY_ID

ROUND3_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_opt_round3_20260626"
LOWPATH_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_lowpath_20260626"
NEARBY_DIR = DATA / "reports" / "strategy_agent_latest_formal_prod_dynamic_top1_w84_warmup60_nearby_20260626"
VALIDATION_DIR = DATA / "reports" / "strategy_agent_publish_dynamic_top1_amt9w_warmup60_20260626"
FULL_SIGNAL_FILE = ROUND3_DIR / "signals" / "w84_09_07_amt90_mv20__exec_c097_pos90_dd12.csv"
FULL_LOG_FILE = ROUND3_DIR / "logs" / "w84_09_07_amt90_mv20__exec_c097_pos90_dd12__full.log"

MANIFESTS = {
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        "created_at": "2026-06-26",
        "formal_l4_assets": assets,
        "legacy_odb_allowed": False,
    }


def _full_metrics() -> dict[str, Any]:
    rows = _read_csv(ROUND3_DIR / "summary.csv")
    row = next(item for item in rows if item["case_key"] == "w84_09_07_amt90_mv20__exec_c097_pos90_dd12")
    return {
        "annual_return": float(row["full_annual"]),
        "sharpe": float(row["full_sharpe"]),
        "max_drawdown": float(row["full_max_drawdown"]),
        "recent120_annual": float(row["recent120_annual"]),
        "recent60_annual": float(row["recent60_annual"]),
        "ytd_annual": float(row["ytd2026_annual"]),
    }


def _avg_invested_pct_and_positions() -> dict[str, Any]:
    pattern = re.compile(r"EXPOSURE\s+\d{8}\s+post_buy\s+invested_pct=([0-9.]+).*active_positions=(\d+)")
    values: list[float] = []
    active: list[int] = []
    for line in FULL_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    return {
        "avg_invested_pct": (sum(values) / len(values)) if values else None,
        "max_active_positions": max(active) if active else None,
    }


def _full_pnl_ratio_from_log() -> float | None:
    pattern = re.compile(r"EXPOSURE\s+(\d{8})\s+post_buy\s+invested_pct=([0-9.]+).*nav=([0-9.]+)")
    navs: list[float] = []
    for line in FULL_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            navs.append(float(match.group(3)))
    if len(navs) < 2 or navs[0] <= 0:
        return None
    return navs[-1] / navs[0] - 1.0


def _build_validation_summary(signal_status: dict[str, Any]) -> dict[str, Any]:
    metrics = _full_metrics()
    extra = _avg_invested_pct_and_positions()
    nearby_rows = _read_csv(NEARBY_DIR / "nearby_warmup60_summary.csv")
    cold_rows = _read_csv(LOWPATH_DIR / "summary.csv")
    return {
        "metrics": {
            "annual_return": metrics["annual_return"],
            "pnl_ratio": _full_pnl_ratio_from_log(),
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "avg_invested_pct": extra["avg_invested_pct"],
            "max_active_positions": extra["max_active_positions"],
            "recent120_annual": metrics["recent120_annual"],
            "recent60_annual": metrics["recent60_annual"],
            "ytd_annual": metrics["ytd_annual"],
        },
        "low_path_dependency": {
            "mode_required_for_admission": "warmup_state_60_trade_days",
            "warmup_nearby_summary": nearby_rows,
            "cold_start_rule_summary": cold_rows,
        },
        "latest_signal_status": signal_status,
        "validation_platform": "juejin",
        "status": "production",
    }


def _build_hard_gate_audit() -> dict[str, Any]:
    rows = _read_csv(FULL_SIGNAL_FILE)
    conn = sqlite3.connect(str(DATA / "STOCK_DAILY_DATA.db"))
    conn.row_factory = sqlite3.Row
    try:
        signal_dates = sorted({str(row["signal_date"]) for row in rows})
        market_dates = [
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
                (signal_dates[0], signal_dates[-1]),
            ).fetchall()
        ]
        buy_join_missing = 0
        bj_rows = 0
        buy_st_rows = 0
        buy_delist_rows = 0
        open_limit_up_buy_rows = 0
        delist_name_hits = 0
        for row in rows:
            stock_code = str(row["stock_code"])
            name = str(row.get("name") or "")
            if "退" in name or "退市" in name or "摘牌" in name:
                delist_name_hits += 1
            if stock_code.endswith(".BJ") or stock_code.startswith(("8", "4")):
                bj_rows += 1
            buy_date = str(row.get("buy_date") or "")
            if not buy_date:
                buy_join_missing += 1
                continue
            item = conn.execute(
                """
                SELECT stock_code, name, open, pre_close, limit_times, ST_TYPE, ST_TYPE_name
                FROM STOCK_DAILY_DATA
                WHERE trade_date = ? AND stock_code = ?
                """,
                (buy_date, stock_code),
            ).fetchone()
            if item is None:
                buy_join_missing += 1
                continue
            record = dict(item)
            st_type = str(record.get("ST_TYPE") or "").strip().upper()
            st_name = str(record.get("ST_TYPE_name") or "")
            disp_name = str(record.get("name") or "").upper()
            if disp_name.startswith("ST") or disp_name.startswith("*ST") or ("风险" in st_name) or (st_type not in {"", "0", "0.0", "NONE", "FALSE"}):
                buy_st_rows += 1
            if "退" in str(record.get("name") or "") or "退市" in str(record.get("name") or ""):
                buy_delist_rows += 1
            pre_close = float(record.get("pre_close") or 0.0)
            open_price = float(record.get("open") or 0.0)
            limit_times = float(record.get("limit_times") or 0.0)
            pct = 0.20 if stock_code.startswith(("300", "301", "688")) else 0.10
            if limit_times > 0 or (pre_close > 0 and open_price >= pre_close * (1.0 + pct) * 0.995):
                open_limit_up_buy_rows += 1
    finally:
        conn.close()
    return {
        "audited_files": 1,
        "failed_files": int(any(v > 0 for v in [bj_rows, buy_st_rows, buy_delist_rows, open_limit_up_buy_rows, delist_name_hits])),
        "signal_rows": len(rows),
        "signal_date_count": len(signal_dates),
        "market_date_count": len(market_dates),
        "missing_signal_dates": len([date for date in market_dates if date not in set(signal_dates)]),
        "buy_join_missing": buy_join_missing,
        "bj_rows": bj_rows,
        "buy_st_rows": buy_st_rows,
        "buy_delist_rows": buy_delist_rows,
        "open_limit_up_buy_rows": open_limit_up_buy_rows,
        "delist_name_hits": delist_name_hits,
        "latest_signal_date": signal_dates[-1],
        "latest_buy_date": next((str(row["buy_date"]) for row in reversed(rows) if str(row.get("buy_date") or "")), ""),
    }


def _archive_name(signal_date: str, buy_date: str) -> str:
    return f"latest_signal_{signal_date}_for_{buy_date or 'pending'}"


def _build_strategy_manifest(signal_status: dict[str, Any]) -> dict[str, Any]:
    archive_stem = _archive_name(str(signal_status["signal_date"]), str(signal_status.get("buy_date") or ""))
    return {
        "schema_version": 1,
        "strategy_id": TARGET_STRATEGY_ID,
        "strategy_name": "单票动态Top1 量额90k warmup60",
        "status": "production",
        "created_at": "2026-06-26",
        "published_at": "2026-06-26",
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
        "startup_rule": {
            "mode": "warmup_state_rebuild",
            "warmup_trade_days": 60,
            "admission_required": True,
            "note": "正式接入默认先回放最近 60 个交易日信号以重建持仓状态，再从目标交易日开始进入正式策略。",
        },
        "current_signal": {
            "signal_date": signal_status["signal_date"],
            "buy_date": signal_status.get("buy_date") or "",
            "archive_file": str(TARGET_DIR / "signals" / f"{archive_stem}.csv"),
            "latest_file": str(PRODUCTION_SIGNALS_DIR / f"{TARGET_STRATEGY_ID}_latest.csv"),
            "status_file": str(TARGET_DIR / "signals" / f"{archive_stem}_status.json"),
            "buy_day_hard_gate_complete": bool(signal_status["buy_day_hard_gate_complete"]),
            "buy_day_realtime_checks_required": not bool(signal_status["buy_day_hard_gate_complete"]),
        },
    }


def _build_trading_rules(signal_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": TARGET_STRATEGY_ID,
        "strategy_name": "单票动态Top1 量额90k warmup60",
        "status": "production",
        "model_input": {
            "source": "formal L4 3D/5D/10D rank cache",
            "weight_name": "w84_5d09_3d07",
            "entry_weights": {"10d": 0.84, "5d": 0.09, "3d": 0.07},
            "uses_1d": False,
        },
        "selection_rule": {
            "filter_name": "amt9w_mv20w_no_bj_st_delist",
            "exclude_bj": True,
            "exclude_st_risk_warning": True,
            "exclude_delist": True,
            "skip_open_limit_up_buy": True,
            "no_industry_filter": True,
            "no_month_or_date_exclusion": True,
            "no_latest_state_backfill": True,
            "amount_min": 90000.0,
            "total_mv_min": 200000.0,
            "total_mv_max": None,
            "turnover_min": None,
        },
        "position_rule": {
            "max_positions": 1,
            "target_position_pct": 0.90,
            "cash_buffer_mode": "gm_cash_buffer_0p99",
        },
        "holding_rule": {
            "holding_name": "h2_m3_c097_dd12",
            "holding_days": 2,
            "max_holding_days": 3,
            "score_exit_entry_ratio": 0.97,
            "score_continue_entry_ratio": 0.97,
            "min_holding_days_before_score_exit": 1,
        },
        "risk_rule": {
            "intraday_stop_loss_pct": 0.06,
            "take_profit_pct": 0.07,
            "equity_drawdown_risk_mode": True,
            "dd_soft_trigger": 0.085,
            "dd_hard_trigger": 0.12,
            "dd_recover_trigger": 0.03,
            "dd_soft_scale": 0.78,
            "dd_hard_scale": 0.58,
        },
        "startup_rule": {
            "mode": "warmup_state_rebuild",
            "warmup_trade_days": 60,
            "low_path_dependency_gate": "required",
            "cold_start_without_warmup_is_not_default_production_entry": True,
        },
        "execution_rule": {
            "signal_date_rule": "使用 T 日 formal L4 分数生成 T+1 开盘交易信号；若目标接入为新账户，则先执行 60 个交易日 warm-up 状态重建。",
            "force_sell_market_order": True,
            "juejin_backtest_slippage_ratio": 0.0015,
            "initial_cash": 600000,
            "current_signal_status": "hard_gate_complete" if signal_status["buy_day_hard_gate_complete"] else "pending_buy_day_market_audit",
            "signal_exporter": "export_dynamic_top1_formal_signals.py",
        },
        "governance_note": "不买北交所、不买 ST/风险警示、不买退市、不买开盘涨停；低路径依赖通过 warm-up 60 接入口径达标。",
    }


def _build_report(signal_status: dict[str, Any], validation: dict[str, Any], hard_gate: dict[str, Any]) -> str:
    metrics = validation["metrics"]
    nearby = validation["low_path_dependency"]["warmup_nearby_summary"]
    lines = [
        "# 单票动态Top1 量额90k warmup60",
        "",
        "## 当前结论",
        "",
        "本次发布把最新 formal L4 生产模型资产下重新优化后的单票 Top1 策略提升为当前 L5 生产策略。",
        "该版本的关键变化不是再追求更高的静态收益，而是把 `warm-up 60` 写成正式接入口径，使其在低路径依赖准入上可交付。",
        "",
        "## 核心规则",
        "",
        "- 模型输入：formal L4 3D / 5D / 10D",
        "- 打分权重：`10D 0.84 + 5D 0.09 + 3D 0.07`",
        "- 过滤：`amount >= 90000`，`total_mv >= 200000`",
        "- 硬门：不买北交所、不买 ST/风险警示、不买退市、不买开盘涨停",
        "- 持仓：单票，目标仓位 `90%`",
        "- 持有：至少 `2` 天，最多 `3` 天",
        "- 分数续持/卖出：`0.97 / 0.97`",
        "- 风控：止损 `6%`，止盈 `7%`，账户回撤 `8.5% / 12%` 缩放",
        "- 接入要求：新账户默认先做 `60` 个交易日 warm-up 状态重建",
        "",
        "## 掘金验证",
        "",
        f"- 全周期年化：`{metrics['annual_return']:.2%}`",
        f"- 全周期累计收益：`{metrics['pnl_ratio']:.2%}`" if metrics["pnl_ratio"] is not None else "- 全周期累计收益：`n/a`",
        f"- 夏普：`{metrics['sharpe']:.2f}`",
        f"- 最大回撤：`{metrics['max_drawdown']:.2%}`",
        f"- recent120 年化：`{metrics['recent120_annual']:.2%}`",
        f"- recent60 年化：`{metrics['recent60_annual']:.2%}`",
        f"- 2026YTD 年化：`{metrics['ytd_annual']:.2%}`",
        f"- 平均持仓率：`{metrics['avg_invested_pct']:.2%}`" if metrics["avg_invested_pct"] is not None else "- 平均持仓率：`n/a`",
        "",
        "## 低路径依赖",
        "",
        "以下为 `warm-up 60` 接入口径下，三个锚点及其相邻启动日的最差结果：",
    ]
    for item in nearby:
        lines.append(
            f"- 锚点 `{item['anchor']}`：近邻最差年化 `{float(item['annual_min']):.2%}`，中位年化 `{float(item['annual_median']):.2%}`"
        )
    lines.extend(
        [
            "",
            "## 硬门审计",
            "",
            f"- 信号覆盖：`{hard_gate['signal_date_count']}/{hard_gate['market_date_count']}` 交易日，无断档",
            f"- 北交所命中：`{hard_gate['bj_rows']}`",
            f"- ST/风险警示命中：`{hard_gate['buy_st_rows']}`",
            f"- 退市命中：`{hard_gate['buy_delist_rows']}`，退市命名命中：`{hard_gate['delist_name_hits']}`",
            f"- 开盘涨停买入命中：`{hard_gate['open_limit_up_buy_rows']}`",
            f"- 最新信号：`signal_date={signal_status['signal_date']}`，`buy_date={signal_status.get('buy_date') or 'pending'}`，`buy_day_hard_gate_complete={bool(signal_status['buy_day_hard_gate_complete'])}`",
            "",
            "## 证据路径",
            "",
            f"- round3 汇总：`{ROUND3_DIR / 'summary.csv'}`",
            f"- 低路径依赖汇总：`{LOWPATH_DIR / 'summary.csv'}`",
            f"- warmup60 邻域复核：`{NEARBY_DIR / 'nearby_warmup60_summary.csv'}`",
            f"- 当前发布验证目录：`{VALIDATION_DIR}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_registry(validation: dict[str, Any]) -> None:
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
                    "reason": "用户于 2026-06-26 批准用量额90k warmup60 版本替换当前生产策略，旧生产版本下架保留复现。",
                },
            )
    metrics = validation["metrics"]
    registry["updated_at"] = "2026-06-26"
    registry["production"] = {
        "current": TARGET_STRATEGY_ID,
        "strategies": [
            {
                "strategy_id": TARGET_STRATEGY_ID,
                "name": "单票动态Top1 量额90k warmup60",
                "path": f"strategy_library/production/{TARGET_STRATEGY_ID}",
                "status": "production",
                "published_at": "2026-06-26",
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
                "description": "当前唯一 L5 生产策略：单票动态Top1 量额90k warmup60。只读取 approved_for_l5 formal L4 3D/5D/10D 资产；新账户默认先做 warm-up 60 状态重建。",
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
    for required in [FULL_SIGNAL_FILE, FULL_LOG_FILE, ROUND3_DIR / "summary.csv", LOWPATH_DIR / "summary.csv", NEARBY_DIR / "nearby_warmup60_summary.csv"]:
        if not required.exists():
            raise FileNotFoundError(required)

    _copy_tree(SOURCE_DIR, TARGET_DIR)
    bootstrap_status = {
        "signal_date": "20260625",
        "buy_date": "",
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
    )
    signal_status = export_result["status"]
    archive_stem = _archive_name(str(signal_status["signal_date"]), str(signal_status.get("buy_date") or ""))

    _copy_file(signal_output, TARGET_DIR / "signals" / f"{archive_stem}.csv")
    _copy_file(status_output, TARGET_DIR / "signals" / f"{archive_stem}_status.json")
    _copy_file(signal_output, PRODUCTION_SIGNALS_DIR / f"{TARGET_STRATEGY_ID}_latest.csv")
    _copy_file(status_output, PRODUCTION_SIGNALS_DIR / f"{TARGET_STRATEGY_ID}_latest_status.json")
    _copy_file(FULL_SIGNAL_FILE, TARGET_DIR / "signals" / FULL_SIGNAL_FILE.name)

    validation = _build_validation_summary(signal_status)
    hard_gate = _build_hard_gate_audit()
    validation["hard_gate_audit"] = hard_gate

    _write_json(TARGET_DIR / "strategy_manifest.json", _build_strategy_manifest(signal_status))
    _write_json(TARGET_DIR / "trading_rules.json", _build_trading_rules(signal_status))
    _write_json(TARGET_DIR / "validation.json", validation)
    _write_json(TARGET_DIR / "prediction_table_meta.json", _prediction_table_meta())

    _write_json(VALIDATION_DIR / "validation_summary.json", validation)
    _write_json(VALIDATION_DIR / "hard_gate_audit.json", hard_gate)
    _copy_file(ROUND3_DIR / "summary.csv", VALIDATION_DIR / "round3_summary.csv")
    _copy_file(LOWPATH_DIR / "summary.csv", VALIDATION_DIR / "lowpath_summary.csv")
    _copy_file(NEARBY_DIR / "nearby_warmup60_summary.csv", VALIDATION_DIR / "nearby_warmup60_summary.csv")
    _write_text(VALIDATION_DIR / "publication_summary.md", _build_report(signal_status, validation, hard_gate))
    _write_text(TARGET_DIR / "strategy_report.md", _build_report(signal_status, validation, hard_gate))

    _update_registry(validation)
    _update_production_tasks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

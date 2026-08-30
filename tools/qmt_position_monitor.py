"""Monitor current QMT positions against the active production strategy.

This script is read-only. It reads the production strategy registry, latest
official signal file, QMT holdings, and realtime ticks, then writes a short
monitoring report plus JSON evidence under quant/data_file/runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from build_manual_trade_package import (  # noqa: E402
    load_latest_signal_batch,
)
from project_paths import resolve_data_path, resolve_project_path  # noqa: E402
from strategy_asset_route import load_current_production_strategy_context  # noqa: E402
from tools.qmt_holdings_api import build_report as build_qmt_report  # noqa: E402


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_REGISTRY_FILE = "strategy_library/registry.json"
DEFAULT_SIGNAL_DIR = "production_signals"
DEFAULT_STRATEGY_ROOT = "strategy_library/production"
DEFAULT_OUTPUT_DIR = "runtime/trading_agent/position_monitor"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def is_trading_window(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    current = now.time()
    return time(9, 30) <= current <= time(11, 30) or time(13, 0) <= current <= time(15, 0)


def load_signal_rows(signal_path: Path) -> list[dict[str, str]]:
    return load_latest_signal_batch(signal_path)


def load_current_strategy(
    *,
    registry_file: str | Path = DEFAULT_REGISTRY_FILE,
    signal_dir: str | Path = DEFAULT_SIGNAL_DIR,
    strategy_root: str | Path = DEFAULT_STRATEGY_ROOT,
    strategy_id: str | None = None,
) -> dict[str, Any]:
    return load_current_production_strategy_context(
        strategy_id=strategy_id,
        registry_file=registry_file,
        signal_dir=signal_dir,
        strategy_root=strategy_root,
    )


def normalize_tick(raw_tick: dict[str, Any] | None) -> dict[str, Any]:
    tick = raw_tick or {}
    bid_prices = tick.get("bidPrice") if isinstance(tick.get("bidPrice"), list) else []
    ask_prices = tick.get("askPrice") if isinstance(tick.get("askPrice"), list) else []
    bid_volumes = tick.get("bidVol") if isinstance(tick.get("bidVol"), list) else []
    ask_volumes = tick.get("askVol") if isinstance(tick.get("askVol"), list) else []
    last = _float(tick.get("lastPrice") or tick.get("last") or tick.get("price"), 0.0)
    prev_close = _float(tick.get("lastClose") or tick.get("preClose"), 0.0)
    open_price = _float(tick.get("open"), 0.0)
    high = _float(tick.get("high"), 0.0)
    low = _float(tick.get("low"), 0.0)
    bid1 = _float(bid_prices[0] if bid_prices else tick.get("bidPrice1"), 0.0)
    ask1 = _float(ask_prices[0] if ask_prices else tick.get("askPrice1"), 0.0)
    bid1_volume = _float(bid_volumes[0] if bid_volumes else tick.get("bidVol1"), 0.0)
    ask1_volume = _float(ask_volumes[0] if ask_volumes else tick.get("askVol1"), 0.0)
    pct_chg = (last / prev_close - 1.0) if last and prev_close else None
    intraday_chg = (last / open_price - 1.0) if last and open_price else None
    amplitude = ((high - low) / prev_close) if high and low and prev_close else None
    spread = ((ask1 - bid1) / last) if ask1 and bid1 and last else None
    return {
        "last_price": last,
        "prev_close": prev_close,
        "open": open_price,
        "high": high,
        "low": low,
        "pct_chg": pct_chg,
        "intraday_chg": intraday_chg,
        "amplitude": amplitude,
        "amount": _float(tick.get("amount"), 0.0),
        "volume": _float(tick.get("volume"), 0.0),
        "bid1": bid1,
        "ask1": ask1,
        "bid1_volume": bid1_volume,
        "ask1_volume": ask1_volume,
        "spread": spread,
        "stock_status": tick.get("stockStatus"),
        "raw_keys": sorted(str(key) for key in tick.keys()),
    }


def fetch_ticks(stock_codes: list[str]) -> dict[str, dict[str, Any]]:
    if not stock_codes:
        return {}
    from xtquant import xtdata

    xtdata.connect()
    raw_ticks = xtdata.get_full_tick(stock_codes)
    return {code: normalize_tick(raw_ticks.get(code)) for code in stock_codes}


def build_monitor_snapshot(
    *,
    strategy_context: dict[str, Any],
    holdings_summary: dict[str, Any],
    ticks: dict[str, dict[str, Any]],
    now: datetime,
) -> dict[str, Any]:
    signal_rows = strategy_context["signal_rows"]
    target_by_code = {
        row.get("stock_code", ""): {
            "stock_code": row.get("stock_code", ""),
            "name": row.get("name", ""),
            "rank": _int(row.get("rank"), 999999),
            "target_pct": _float(row.get("target_pct")),
            "signal_date": row.get("signal_date", ""),
            "buy_date": row.get("buy_date", ""),
        }
        for row in signal_rows
        if row.get("stock_code")
    }
    positions = [
        row
        for row in holdings_summary.get("positions", [])
        if _int(row.get("volume")) > 0 or _float(row.get("market_value")) > 0
    ]
    positions_by_code = {row.get("stock_code"): row for row in positions if row.get("stock_code")}
    asset = holdings_summary.get("asset") or {}
    total_asset = _float(asset.get("total_asset"))
    market_value = _float(asset.get("market_value"))
    cash = _float(asset.get("cash"))

    position_rows: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    for position in positions:
        code = position.get("stock_code") or ""
        tick = ticks.get(code, {})
        actual_pct = (_float(position.get("market_value")) / total_asset) if total_asset else 0.0
        target_pct = target_by_code.get(code, {}).get("target_pct", 0.0)
        position_profit_rate = _float(position.get("profit_rate"))
        row = {
            "stock_code": code,
            "name": position.get("instrument_name") or target_by_code.get(code, {}).get("name", ""),
            "volume": _int(position.get("volume")),
            "can_use_volume": _int(position.get("can_use_volume")),
            "market_value": _float(position.get("market_value")),
            "position_profit": _float(position.get("position_profit")),
            "position_profit_rate": position_profit_rate,
            "actual_pct": actual_pct,
            "target_pct": target_pct,
            "target_gap": actual_pct - target_pct,
            "in_target_pool": code in target_by_code,
            "tick": tick,
        }
        position_rows.append(row)

        pct_chg = tick.get("pct_chg")
        amplitude = tick.get("amplitude")
        spread = tick.get("spread")
        if pct_chg is not None and abs(pct_chg) >= 0.05:
            alerts.append({"level": "warning", "stock_code": code, "type": "single_stock_pct", "message": f"{code} 日涨跌幅 {_pct(pct_chg)}"})
        if amplitude is not None and amplitude >= 0.08:
            alerts.append({"level": "warning", "stock_code": code, "type": "amplitude", "message": f"{code} 日内振幅 {_pct(amplitude)}"})
        if position_profit_rate <= -0.03:
            alerts.append({"level": "warning", "stock_code": code, "type": "floating_loss", "message": f"{code} 持仓浮亏 {_pct(position_profit_rate)}"})
        if position_profit_rate >= 0.05:
            alerts.append({"level": "info", "stock_code": code, "type": "floating_profit", "message": f"{code} 持仓浮盈 {_pct(position_profit_rate)}"})
        if spread is not None and spread >= 0.005:
            alerts.append({"level": "warning", "stock_code": code, "type": "spread", "message": f"{code} 买卖一价差 {_pct(spread)}"})
        if code in target_by_code and abs(actual_pct - target_pct) >= 0.02:
            alerts.append({"level": "warning", "stock_code": code, "type": "position_gap", "message": f"{code} 仓位偏离 {_pct(actual_pct - target_pct)}"})
        if code not in target_by_code:
            alerts.append({"level": "warning", "stock_code": code, "type": "should_sell", "message": f"{code} 不在最新目标池"})

    target_missing = sorted(set(target_by_code) - set(positions_by_code))
    for code in target_missing:
        alerts.append({"level": "warning", "stock_code": code, "type": "should_buy", "message": f"{code} 在目标池但当前未持有"})

    if total_asset and market_value / total_asset < 0.90:
        alerts.append({"level": "info", "type": "low_exposure", "message": f"持仓市值占总资产 {_pct(market_value / total_asset)}"})

    industry_weight = defaultdict(float)
    market_weight = defaultdict(float)
    for row in signal_rows:
        industry = row.get("industry") or ""
        market = row.get("market") or ""
        if industry:
            industry_weight[industry] += _float(row.get("target_pct"))
        if market:
            market_weight[market] += _float(row.get("target_pct"))

    position_rows.sort(key=lambda row: row["position_profit"])
    max_loss = position_rows[0] if position_rows else None
    max_profit = position_rows[-1] if position_rows else None
    if any(alert["type"] in {"should_buy", "should_sell", "position_gap", "floating_loss", "single_stock_pct"} for alert in alerts):
        conclusion = "重点关注"
    elif alerts:
        conclusion = "收盘前复核"
    else:
        conclusion = "继续持有"

    return {
        "generated_at": now.isoformat(),
        "is_trading_window": is_trading_window(now),
        "strategy": {
            "strategy_id": strategy_context["strategy_entry"].get("strategy_id"),
            "name": strategy_context["strategy_entry"].get("name"),
            "version": strategy_context["strategy_manifest"].get("production_version")
            or strategy_context["strategy_entry"].get("version")
            or strategy_context["strategy_entry"].get("published_at"),
            "signal_file": strategy_context["signal_path"],
            "strategy_dir": strategy_context["strategy_dir"],
        },
        "signal": {
            "signal_date": signal_rows[0].get("signal_date", "") if signal_rows else "",
            "buy_date": signal_rows[0].get("buy_date", "") if signal_rows else "",
            "target_pool": sorted(target_by_code.values(), key=lambda row: row["rank"]),
        },
        "account": {
            "account_id": holdings_summary.get("account_id"),
            "account_type": holdings_summary.get("account_type"),
            "total_asset": total_asset,
            "cash": cash,
            "market_value": market_value,
            "market_value_ratio": market_value / total_asset if total_asset else 0.0,
        },
        "positions": position_rows,
        "consistency": {
            "target_count": len(target_by_code),
            "position_count": len(positions_by_code),
            "target_missing_positions": target_missing,
            "non_target_positions": sorted(set(positions_by_code) - set(target_by_code)),
        },
        "exposure": {
            "target_pct_sum": sum(_float(row.get("target_pct")) for row in signal_rows),
            "industry_weight_from_signal": dict(sorted(industry_weight.items())),
            "market_weight_from_signal": dict(sorted(market_weight.items())),
        },
        "alerts": alerts,
        "max_profit_position": max_profit,
        "max_loss_position": max_loss,
        "conclusion": conclusion,
        "note": "执行监督和风险跟踪报告；盘中短时波动不是新的正式换仓信号，不生成未经审批的实盘指令。",
    }


def render_markdown(snapshot: dict[str, Any]) -> str:
    account = snapshot["account"]
    strategy = snapshot["strategy"]
    signal = snapshot["signal"]
    consistency = snapshot["consistency"]
    positions = snapshot["positions"]
    alerts = snapshot["alerts"]

    lines = [
        "# 盘中持仓监控报告",
        "",
        f"- 当前时间：{snapshot['generated_at']}",
        f"- 策略：{strategy['strategy_id']} / {strategy.get('version') or ''}",
        f"- 最新正式信号：signal_date={signal['signal_date']}，buy_date={signal['buy_date']}",
        f"- 账户总资产：{account['total_asset']:.2f}",
        f"- 持仓市值：{account['market_value']:.2f}（{_pct(account['market_value_ratio'])}）",
        f"- 可用资金：{account['cash']:.2f}",
        f"- 建议结论：{snapshot['conclusion']}",
        "",
        "## 持仓盈亏",
        "| 股票 | 名称 | 市值 | 仓位 | 目标仓位 | 浮盈亏 | 持仓收益 | 日涨跌 | 日内振幅 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in positions:
        tick = row.get("tick") or {}
        lines.append(
            "| {code} | {name} | {mv:.2f} | {actual} | {target} | {profit:.2f} | {profit_rate} | {pct} | {amp} |".format(
                code=row.get("stock_code") or "",
                name=row.get("name") or "",
                mv=_float(row.get("market_value")),
                actual=_pct(row.get("actual_pct")),
                target=_pct(row.get("target_pct")),
                profit=_float(row.get("position_profit")),
                profit_rate=_pct(row.get("position_profit_rate")),
                pct=_pct(tick.get("pct_chg")),
                amp=_pct(tick.get("amplitude")),
            )
        )

    lines.extend(
        [
            "",
            "## 一致性检查",
            f"- 应买未买：{', '.join(consistency['target_missing_positions']) if consistency['target_missing_positions'] else '无'}",
            f"- 应卖未卖：{', '.join(consistency['non_target_positions']) if consistency['non_target_positions'] else '无'}",
            f"- 最大盈利股：{(snapshot.get('max_profit_position') or {}).get('stock_code') or '无'}",
            f"- 最大拖累股：{(snapshot.get('max_loss_position') or {}).get('stock_code') or '无'}",
            "",
            "## 异常提示",
        ]
    )
    if alerts:
        for alert in alerts:
            lines.append(f"- [{alert.get('level')}] {alert.get('message')}")
    else:
        lines.append("- 无")
    lines.extend(["", f"> {snapshot['note']}", ""])
    return "\n".join(lines)


def write_outputs(snapshot: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    generated = datetime.fromisoformat(snapshot["generated_at"])
    base_dir = resolve_data_path(output_dir) / generated.strftime("%Y%m%d")
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = "qmt_position_monitor_" + generated.strftime("%Y%m%d_%H%M%S")
    json_path = base_dir / f"{stem}.json"
    md_path = base_dir / f"{stem}.md"
    latest_json_path = base_dir / "latest.json"
    latest_md_path = base_dir / "latest.md"
    json_text = json.dumps(snapshot, ensure_ascii=False, indent=2)
    md_text = render_markdown(snapshot)
    json_path.write_text(json_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")
    latest_md_path.write_text(md_text, encoding="utf-8")
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "latest_json_path": str(latest_json_path),
        "latest_md_path": str(latest_md_path),
    }


def build_runtime_snapshot(args: argparse.Namespace, now: datetime) -> dict[str, Any]:
    strategy_context = load_current_strategy(
        registry_file=args.registry_file,
        signal_dir=args.signal_dir,
        strategy_root=args.strategy_root,
        strategy_id=args.strategy_id or None,
    )
    qmt_report = build_qmt_report(
        account_id=args.account_id,
        account_type=args.account_type,
        userdata_path=args.userdata_path or None,
        connect_retries=args.connect_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    holdings_summary = qmt_report.get("summary") or {}
    if not holdings_summary.get("ok"):
        raise RuntimeError(f"QMT holdings API failed: {holdings_summary.get('error') or qmt_report}")

    signal_codes = [row.get("stock_code", "") for row in strategy_context["signal_rows"] if row.get("stock_code")]
    position_codes = [
        row.get("stock_code", "")
        for row in holdings_summary.get("positions", [])
        if row.get("stock_code") and (_int(row.get("volume")) > 0 or _float(row.get("market_value")) > 0)
    ]
    tick_codes = sorted(set(signal_codes) | set(position_codes))
    ticks = fetch_ticks(tick_codes)
    snapshot = build_monitor_snapshot(
        strategy_context=strategy_context,
        holdings_summary=holdings_summary,
        ticks=ticks,
        now=now,
    )
    snapshot["qmt_source"] = {
        "source": holdings_summary.get("source"),
        "xtdata_ok": bool(qmt_report.get("xtdata", {}).get("ok")),
        "xttrader_ok": bool(qmt_report.get("xttrader", {}).get("ok")),
    }
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor QMT positions against current production strategy.")
    parser.add_argument("--account-id", required=True, help="QMT account id, e.g. 8883530863")
    parser.add_argument("--account-type", default="STOCK")
    parser.add_argument("--userdata-path", default="")
    parser.add_argument("--connect-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument("--strategy-id", default="")
    parser.add_argument("--registry-file", default=DEFAULT_REGISTRY_FILE)
    parser.add_argument("--signal-dir", default=DEFAULT_SIGNAL_DIR)
    parser.add_argument("--strategy-root", default=DEFAULT_STRATEGY_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Run even outside A-share trading windows.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Stdout format. Files are always written as both markdown and json.",
    )
    parser.add_argument(
        "--exit-nonzero-on-alert",
        action="store_true",
        help="Return exit code 1 when alerts exist. Default keeps exit code 0 for automations.",
    )
    args = parser.parse_args()

    now = datetime.now(SHANGHAI_TZ)
    if not args.force and not is_trading_window(now):
        message = {
            "generated_at": now.isoformat(),
            "is_trading_window": False,
            "message": "当前不在 A 股交易监控时段，未展开监控报告。",
        }
        print(json.dumps(message, ensure_ascii=False, indent=2) if args.format == "json" else message["message"])
        return 0

    snapshot = build_runtime_snapshot(args, now)
    outputs = write_outputs(snapshot, args.output_dir)
    snapshot["outputs"] = outputs
    if args.format == "json":
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(snapshot))
        print(f"\n报告文件：{outputs['latest_md_path']}")
    if args.exit_nonzero_on_alert and snapshot.get("alerts"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

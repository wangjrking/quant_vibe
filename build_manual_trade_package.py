from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from project_paths import resolve_data_path, resolve_project_path


def load_registry(registry_path: str | Path) -> dict[str, Any]:
    path = resolve_project_path(registry_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def current_production_strategy(registry: dict[str, Any], strategy_id: str | None = None) -> dict[str, Any]:
    production = registry.get("production", {})
    selected_id = strategy_id or production.get("current")
    if not selected_id:
        raise ValueError("production.current missing in strategy registry")
    for item in production.get("strategies", []):
        if item.get("strategy_id") == selected_id:
            return item
    raise ValueError(f"strategy_id not found in production registry: {selected_id}")


def load_strategy_manifest(strategy_dir: str | Path) -> dict[str, Any]:
    path = Path(strategy_dir) / "strategy_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_latest_signal_file(
    strategy_id: str,
    *,
    signal_dir: str | Path | None = None,
    strategy_dir: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if signal_dir is not None:
        candidates.append(resolve_data_path(signal_dir) / f"{strategy_id}_latest.csv")
    if strategy_dir is not None:
        base = Path(strategy_dir)
        candidates.extend(
            [
                base / "signals" / "signals_latest.csv",
                base / "signals" / "production_signals.csv",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"latest signal file not found for {strategy_id}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_latest_signal_batch(signal_path: str | Path) -> list[dict[str, str]]:
    path = Path(signal_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"signal file is empty: {path}")
    latest_signal_date = max((row.get("signal_date") or "") for row in rows)
    latest_rows = [row for row in rows if (row.get("signal_date") or "") == latest_signal_date]
    latest_rows.sort(key=lambda row: int(row.get("rank") or "999999"))
    return latest_rows


def load_holdings_summary_from_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_holdings_summary_from_qmt(
    account_id: str,
    *,
    account_type: str = "STOCK",
    userdata_path: str | None = None,
    connect_retries: int = 3,
    retry_sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    from tools.qmt_holdings_api import build_report

    report = build_report(
        account_id=account_id,
        account_type=account_type,
        userdata_path=userdata_path,
        connect_retries=connect_retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    summary = report.get("summary") or {}
    if not summary.get("ok"):
        raise RuntimeError(f"QMT holdings API failed: {summary.get('error') or report}")
    return summary


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


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _looks_garbled_display_name(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return True
    if "�" in text:
        return True
    if text.count("?") >= max(2, len(text) // 3):
        return True
    return False


def _select_strategy_display_name(strategy_manifest: dict[str, Any], strategy_entry: dict[str, Any]) -> str:
    candidates = [
        strategy_manifest.get("name") or "",
        strategy_entry.get("name") or "",
        strategy_manifest.get("code_name") or "",
        strategy_entry.get("strategy_id") or "",
    ]
    for candidate in candidates:
        if not _looks_garbled_display_name(candidate):
            return candidate.strip()
    return (strategy_manifest.get("code_name") or strategy_entry.get("strategy_id") or "").strip()


def _sell_shares_for_gap(current_shares: int, current_pct: float, gap_pct: float) -> int:
    if current_shares <= 0 or current_pct <= 0 or gap_pct <= 0:
        return 0
    raw = current_shares * min(gap_pct / current_pct, 1.0)
    rounded = int(raw // 100) * 100
    return rounded or min(current_shares, 100)


def build_trade_package(
    *,
    registry: dict[str, Any],
    strategy_manifest: dict[str, Any],
    signal_rows: list[dict[str, str]],
    holdings_summary: dict[str, Any],
    platform: str,
    account_label: str,
    executor: str = "",
    review_status: str = "未审核",
    rebalance_tolerance: float = 0.02,
    signal_source_file: str = "",
) -> dict[str, Any]:
    strategy_entry = current_production_strategy(registry, strategy_manifest.get("strategy_id"))
    if not signal_rows:
        raise ValueError("signal_rows cannot be empty")

    asset = holdings_summary.get("asset") or {}
    total_asset = _float(asset.get("total_asset"))
    cash = _float(asset.get("cash"))
    market_value = _float(asset.get("market_value"))
    positions = holdings_summary.get("positions") or []
    holdings_by_code = {row.get("stock_code"): row for row in positions if row.get("stock_code")}

    buy_actions: list[dict[str, Any]] = []
    sell_actions: list[dict[str, Any]] = []
    hold_actions: list[dict[str, Any]] = []

    for signal in sorted(signal_rows, key=lambda row: int(row.get("rank") or "999999")):
        stock_code = signal.get("stock_code") or ""
        current = holdings_by_code.pop(stock_code, None)
        target_pct = _float(signal.get("target_pct"))
        rank = _int(signal.get("rank"))
        planned_amount = round(total_asset * target_pct, 2) if total_asset else 0.0
        if current is None:
            buy_actions.append(
                {
                    "action": "BUY",
                    "stock_code": stock_code,
                    "stock_name": signal.get("name") or "",
                    "rank": rank,
                    "target_pct": target_pct,
                    "current_position_pct": 0.0,
                    "current_shares": 0,
                    "available_shares": 0,
                    "planned_amount": planned_amount,
                    "planned_shares": "",
                    "time_window": "09:30-09:35",
                    "price_rule": "开盘价附近限价",
                    "reason": "最新目标池新增标的",
                    "notes": "",
                }
            )
            continue

        current_market_value = _float(current.get("market_value"))
        current_pct = (current_market_value / total_asset) if total_asset else 0.0
        pct_gap = target_pct - current_pct
        current_shares = _int(current.get("volume"))
        available_shares = _int(current.get("can_use_volume"))
        base_row = {
            "stock_code": stock_code,
            "stock_name": signal.get("name") or current.get("instrument_name") or "",
            "rank": rank,
            "target_pct": target_pct,
            "current_position_pct": current_pct,
            "current_shares": current_shares,
            "available_shares": available_shares,
            "time_window": "09:30-09:35",
            "price_rule": "开盘价附近限价",
            "planned_shares": "",
            "notes": "",
        }
        if pct_gap > rebalance_tolerance:
            buy_actions.append(
                {
                    **base_row,
                    "action": "BUY",
                    "planned_amount": round(total_asset * pct_gap, 2) if total_asset else 0.0,
                    "reason": "已在目标池，但当前仓位低于目标仓位",
                }
            )
        elif -pct_gap > rebalance_tolerance:
            sell_gap = current_pct - target_pct
            sell_actions.append(
                {
                    **base_row,
                    "action": "SELL",
                    "planned_amount": round(total_asset * sell_gap, 2) if total_asset else 0.0,
                    "planned_shares": _sell_shares_for_gap(current_shares, current_pct, sell_gap),
                    "sell_reason": "已在目标池，但当前仓位高于目标仓位",
                    "reason": "已在目标池，但当前仓位高于目标仓位",
                }
            )
        else:
            hold_actions.append(
                {
                    **base_row,
                    "action": "HOLD",
                    "planned_amount": 0.0,
                    "hold_reason": "已在最新目标池，且仓位偏差在容忍范围内",
                    "reason": "已在最新目标池，且仓位偏差在容忍范围内",
                }
            )

    for leftover in sorted(holdings_by_code.values(), key=lambda row: str(row.get("stock_code") or "")):
        current_market_value = _float(leftover.get("market_value"))
        current_pct = (current_market_value / total_asset) if total_asset else 0.0
        current_shares = _int(leftover.get("volume"))
        available_shares = _int(leftover.get("can_use_volume"))
        sell_actions.append(
            {
                "action": "SELL",
                "stock_code": leftover.get("stock_code") or "",
                "stock_name": leftover.get("instrument_name") or "",
                "rank": "",
                "target_pct": 0.0,
                "current_position_pct": current_pct,
                "current_shares": current_shares,
                "available_shares": available_shares,
                "planned_amount": round(current_market_value, 2),
                "planned_shares": available_shares or current_shares,
                "time_window": "09:30-09:35",
                "price_rule": "开盘价附近限价",
                "sell_reason": "不在最新目标池",
                "reason": "不在最新目标池",
                "notes": "",
            }
        )

    signal_date = signal_rows[0].get("signal_date") or ""
    buy_date = signal_rows[0].get("buy_date") or ""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    package = {
        "created_at": created_at,
        "strategy": {
            "strategy_id": strategy_manifest.get("strategy_id") or strategy_entry.get("strategy_id") or "",
            "name": _select_strategy_display_name(strategy_manifest, strategy_entry),
            "production_version": strategy_manifest.get("production_version")
            or strategy_entry.get("version")
            or "",
            "version": strategy_manifest.get("version") or strategy_entry.get("version") or "",
        },
        "signal": {
            "signal_date": signal_date,
            "buy_date": buy_date,
            "source_file": signal_source_file,
        },
        "account": {
            "platform": platform,
            "account_label": account_label,
            "account_id": holdings_summary.get("account_id") or "",
            "account_type": holdings_summary.get("account_type") or "",
            "total_asset": total_asset,
            "cash": cash,
            "market_value": market_value,
        },
        "summary": {
            "buy_count": len(buy_actions),
            "sell_count": len(sell_actions),
            "hold_count": len(hold_actions),
            "review_status": review_status,
            "executor": executor,
        },
        "actions": {
            "buy": buy_actions,
            "sell": sell_actions,
            "hold": hold_actions,
        },
    }
    return package


def render_markdown(package: dict[str, Any]) -> str:
    strategy = package["strategy"]
    signal = package["signal"]
    account = package["account"]
    summary = package["summary"]
    buys = package["actions"]["buy"]
    sells = package["actions"]["sell"]
    holds = package["actions"]["hold"]

    sell_rows = "\n".join(
        [
            f"| {idx} | {row.get('stock_code', '')} | {row.get('stock_name', '')} | {row.get('current_shares', '')} | "
            f"{row.get('available_shares', '')} | {row.get('reason') or row.get('sell_reason', '')} | 卖出 | {row.get('time_window', '09:30-09:35')} | "
            f"{row.get('price_rule', '开盘价附近限价')} | {row.get('planned_shares', '') or ''} | {row.get('notes', '')} |"
            for idx, row in enumerate(sells, start=1)
        ]
    ) or "|  |  |  |  |  |  |  |  |  |  |  |"

    buy_rows = "\n".join(
        [
            f"| {idx} | {row.get('stock_code', '')} | {row.get('stock_name', '')} | {row.get('rank', '')} | {_pct(_float(row.get('target_pct')))} | "
            f"{_float(row.get('planned_amount')):.2f} | {row.get('planned_shares', '')} | {row.get('time_window', '09:30-09:35')} | "
            f"{row.get('price_rule', '开盘价附近限价')} | 非停牌 / 非涨停 / 现金充足 | {row.get('notes', '')} |"
            for idx, row in enumerate(buys, start=1)
        ]
    ) or "|  |  |  |  |  |  |  |  |  |  |  |"

    hold_rows = "\n".join(
        [
            f"| {row.get('stock_code', '')} | {row.get('stock_name', '')} | {_pct(_float(row.get('current_position_pct')))} | "
            f"{row.get('reason') or row.get('hold_reason', '')} | 继续持有 |"
            for row in holds
        ]
    ) or "|  |  |  |  |  |"

    return (
        "# 次日交易执行单\n\n"
        "## 一、任务信息\n"
        f"- 策略名称：{strategy['name']}\n"
        f"- 策略 ID：{strategy['strategy_id']}\n"
        f"- 策略版本：{strategy['production_version']}\n"
        f"- 信号生成日（signal_date）：{signal['signal_date']}\n"
        f"- 计划执行日（trade_date / buy_date）：{signal['buy_date']}\n"
        f"- 账户类型：{account['account_label']}\n"
        f"- 执行平台：{account['platform']}\n"
        f"- 当前总资产：{account['total_asset']:.2f}\n"
        f"- 当前可用资金：{account['cash']:.2f}\n"
        f"- 当前持仓市值：{account['market_value']:.2f}\n"
        f"- 执行人：{summary.get('executor') or ''}\n"
        f"- 审核状态：{summary.get('review_status') or '未审核'}\n\n"
        "## 二、策略执行摘要\n"
        f"- 今日动作摘要：买入 {summary['buy_count']} 只，卖出 {summary['sell_count']} 只，继续持有 {summary['hold_count']} 只\n"
        "- 执行原则：先卖后买，优先释放现金，再按目标仓位建仓\n"
        "- 下单时段：09:30-09:35 首轮执行，09:35-09:45 补单校正\n"
        "- 价格口径：开盘价附近限价\n"
        f"- 正式信号文件：{signal.get('source_file') or ''}\n\n"
        "## 三、卖出建议\n"
        "| 序号 | 股票代码 | 股票名称 | 当前持仓 | 可卖数量 | 卖出原因 | 建议动作 | 下单时间 | 委托价格规则 | 计划卖出股数 | 备注 |\n"
        "|---|---|---|---:|---:|---|---|---|---|---:|---|\n"
        f"{sell_rows}\n\n"
        "## 四、买入建议\n"
        "| 序号 | 股票代码 | 股票名称 | 信号排名 | 目标仓位 | 建议买入金额 | 建议股数 | 下单时间 | 委托价格规则 | 买入前检查 | 备注 |\n"
        "|---|---|---|---:|---:|---:|---:|---|---|---|---|\n"
        f"{buy_rows}\n\n"
        "## 五、继续持有建议\n"
        "| 股票代码 | 股票名称 | 当前仓位 | 持有原因 | 明日动作 |\n"
        "|---|---|---:|---|---|\n"
        f"{hold_rows}\n"
    )


def build_signal_rows(package: dict[str, Any]) -> list[dict[str, Any]]:
    signal = package["signal"]
    rows: list[dict[str, Any]] = []
    for bucket in ("sell", "buy", "hold"):
        for row in package["actions"][bucket]:
            action = row.get("action")
            if not action:
                action = {"sell": "SELL", "buy": "BUY", "hold": "HOLD"}[bucket]
            rows.append(
                {
                    "signal_date": signal["signal_date"],
                    "buy_date": signal["buy_date"],
                    "action": action,
                    "stock_code": row.get("stock_code", ""),
                    "stock_name": row.get("stock_name", ""),
                    "rank": row.get("rank", ""),
                    "target_pct": row.get("target_pct", 0.0),
                    "current_position_pct": row.get("current_position_pct", 0.0),
                    "current_shares": row.get("current_shares", 0),
                    "available_shares": row.get("available_shares", 0),
                    "planned_amount": row.get("planned_amount", 0.0),
                    "planned_shares": row.get("planned_shares", ""),
                    "time_window": row.get("time_window", "09:30-09:35"),
                    "price_rule": row.get("price_rule", "开盘价附近限价"),
                    "trigger_rule": "次日开盘执行",
                    "tradability_check": "非停牌;非涨停;开盘价有效",
                    "reason": row.get("reason") or row.get("sell_reason") or row.get("hold_reason", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def write_trade_package(package: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    strategy_id = package["strategy"]["strategy_id"]
    signal_date = package["signal"]["signal_date"]
    buy_date = package["signal"]["buy_date"]
    folder = target_dir / f"{strategy_id}_{signal_date}_for_{buy_date}"
    folder.mkdir(parents=True, exist_ok=True)

    ticket_path = folder / "next_day_manual_trade_ticket.md"
    csv_path = folder / "next_day_manual_trade_signals.csv"
    json_path = folder / "trade_package_summary.json"

    ticket_path.write_text(render_markdown(package), encoding="utf-8")

    rows = build_signal_rows(package)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "signal_date",
                "buy_date",
                "action",
                "stock_code",
                "stock_name",
                "rank",
                "target_pct",
                "current_position_pct",
                "current_shares",
                "available_shares",
                "planned_amount",
                "planned_shares",
                "time_window",
                "price_rule",
                "trigger_rule",
                "tradability_check",
                "reason",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_dir": str(folder),
        "ticket_path": str(ticket_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


def build_from_runtime(
    *,
    strategy_id: str | None = None,
    registry_file: str | Path = "strategy_library/registry.json",
    signal_dir: str | Path = "production_signals",
    strategy_root: str | Path = "strategy_library/production",
    holdings_json: str | Path | None = None,
    account_id: str | None = None,
    account_type: str = "STOCK",
    userdata_path: str | None = None,
    platform: str = "QMT",
    account_label: str = "模拟盘",
    executor: str = "",
    review_status: str = "未审核",
    rebalance_tolerance: float = 0.02,
) -> dict[str, Any]:
    registry = load_registry(registry_file)
    strategy_entry = current_production_strategy(registry, strategy_id)
    strategy_dir = resolve_project_path(Path(strategy_root) / strategy_entry["strategy_id"])
    strategy_manifest = load_strategy_manifest(strategy_dir)
    signal_file = resolve_latest_signal_file(
        strategy_entry["strategy_id"],
        signal_dir=signal_dir,
        strategy_dir=strategy_dir,
    )
    signal_rows = load_latest_signal_batch(signal_file)
    if holdings_json:
        holdings_summary = load_holdings_summary_from_json(holdings_json)
    elif account_id:
        holdings_summary = load_holdings_summary_from_qmt(
            account_id,
            account_type=account_type,
            userdata_path=userdata_path,
        )
    else:
        raise ValueError("either holdings_json or account_id is required")
    return build_trade_package(
        registry=registry,
        strategy_manifest=strategy_manifest,
        signal_rows=signal_rows,
        holdings_summary=holdings_summary,
        platform=platform,
        account_label=account_label,
        executor=executor,
        review_status=review_status,
        rebalance_tolerance=rebalance_tolerance,
        signal_source_file=str(signal_file),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build next-day manual trade package from current L5 signals.")
    parser.add_argument("--strategy-id", default="", help="Optional explicit production strategy id.")
    parser.add_argument("--registry-file", default="strategy_library/registry.json")
    parser.add_argument("--signal-dir", default="production_signals")
    parser.add_argument("--strategy-root", default="strategy_library/production")
    parser.add_argument("--holdings-json", default="", help="Use a local holdings summary json instead of QMT API.")
    parser.add_argument("--account-id", default="", help="QMT account id.")
    parser.add_argument("--account-type", default="STOCK")
    parser.add_argument("--userdata-path", default="")
    parser.add_argument("--platform", default="QMT")
    parser.add_argument("--account-label", default="模拟盘")
    parser.add_argument("--executor", default="")
    parser.add_argument("--review-status", default="未审核")
    parser.add_argument("--rebalance-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--output-dir",
        default="runtime/trading_agent/manual_trade_packages",
        help="Relative to data_file unless absolute path is provided.",
    )
    args = parser.parse_args()

    package = build_from_runtime(
        strategy_id=args.strategy_id or None,
        registry_file=args.registry_file,
        signal_dir=args.signal_dir,
        strategy_root=args.strategy_root,
        holdings_json=args.holdings_json or None,
        account_id=args.account_id or None,
        account_type=args.account_type,
        userdata_path=args.userdata_path or None,
        platform=args.platform,
        account_label=args.account_label,
        executor=args.executor,
        review_status=args.review_status,
        rebalance_tolerance=args.rebalance_tolerance,
    )
    output_dir = resolve_data_path(args.output_dir)
    outputs = write_trade_package(package, output_dir)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

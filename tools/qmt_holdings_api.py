"""Read current QMT holdings through official Python APIs.

This tool only uses QMT/xtquant APIs. It does not inspect the UI.

Execution order:
1. Probe xtdata to confirm the local Python runtime can discover the running QMT.
2. Query holdings through XtQuantTrader, which is the primary supported path.
3. Probe qmttools.get_trade_detail_data as a compatibility fallback and diagnostic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.decode("gb18030", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _safe(v) for k, v in vars(obj).items() if not k.startswith("_")}
    fields: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        fields[name] = _safe(value)
    if fields:
        return fields
    return repr(obj)


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_userdata_path() -> str | None:
    from xtquant import xtdata

    client = xtdata.connect()
    data_dir = Path(client.get_data_dir()).resolve()
    if data_dir.name.lower() == "datadir":
        return str(data_dir.parent)
    return None


def _probe_xtdata() -> dict[str, Any]:
    from xtquant import xtdata

    info: dict[str, Any] = {"ok": False}
    try:
        client = xtdata.connect()
        info.update(
            {
                "ok": True,
                "peer_addr": client.get_peer_addr(),
                "server_tag": client.get_server_tag(),
                "app_dir": client.get_app_dir(),
                "data_dir": client.get_data_dir(),
                "inferred_userdata_path": _infer_userdata_path(),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostic path
        info["error"] = repr(exc)
    return info


def _probe_xttrader(
    account_id: str,
    account_type: str,
    userdata_path: str,
    connect_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xttype import StockAccount

    info: dict[str, Any] = {
        "ok": False,
        "userdata_path": userdata_path,
        "account_id": account_id,
        "account_type": account_type,
    }
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(connect_retries, 1) + 1):
        session_id = int(time.time()) + attempt - 1
        trader = None
        try:
            trader = XtQuantTrader(userdata_path, session_id)
            trader.start()
            connect_ret = trader.connect()
            attempts.append(
                {
                    "attempt": attempt,
                    "session_id": session_id,
                    "connect_ret": connect_ret,
                }
            )
            info["connect_attempts"] = attempts
            if connect_ret != 0:
                if attempt < connect_retries:
                    time.sleep(retry_sleep_seconds)
                continue

            account = StockAccount(account_id, account_type)
            subscribe_ret = trader.subscribe(account)
            asset = trader.query_stock_asset(account)
            positions = trader.query_stock_positions(account)
            info.update(
                {
                    "ok": True,
                    "connect_ret": connect_ret,
                    "subscribe_ret": subscribe_ret,
                    "asset": _safe(asset),
                    "positions": _safe(positions),
                }
            )
            return info
        except Exception as exc:  # pragma: no cover - runtime diagnostic path
            attempts.append(
                {
                    "attempt": attempt,
                    "session_id": session_id,
                    "error": repr(exc),
                }
            )
            info["connect_attempts"] = attempts
            if attempt < connect_retries:
                time.sleep(retry_sleep_seconds)
        finally:
            if trader is not None:
                try:
                    trader.stop()
                except Exception:
                    pass

    info["reason"] = "XtQuantTrader.connect() failed"
    return info


def _probe_qmttools(account_id: str, account_type: str) -> dict[str, Any]:
    from xtquant import xtdata
    from xtquant.qmttools.contextinfo import ContextInfo
    from xtquant.qmttools.functions import get_trade_detail_data

    info: dict[str, Any] = {
        "ok": False,
        "account_id": account_id,
        "account_type": account_type,
    }
    try:
        xtdata.connect()

        def _call() -> list[Any]:
            context = ContextInfo()
            context.request_id = "external_qmt_holdings_probe"
            return get_trade_detail_data(account_id, account_type, "POSITION")

        result = _call()
        info.update({"ok": True, "positions": _safe(result)})
        return info
    except Exception as exc:  # pragma: no cover - runtime diagnostic path
        info["error"] = repr(exc)
        return info


def _extract_trade_payload(report: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    xttrader = report.get("xttrader", {})
    if xttrader.get("ok"):
        return "xttrader", xttrader

    qmttools = report.get("qmttools_trade_detail", {})
    if qmttools.get("ok"):
        return "qmttools_trade_detail", qmttools

    return None, None


def _build_summary(report: dict[str, Any]) -> dict[str, Any]:
    source_name, payload = _extract_trade_payload(report)
    summary: dict[str, Any] = {
        "timestamp": report.get("timestamp"),
        "account_id": report.get("account_id"),
        "account_type": report.get("account_type"),
        "source": source_name,
        "ok": False,
    }
    if not payload:
        summary["error"] = "No available QMT holdings API payload"
        return summary

    asset = payload.get("asset") or {}
    positions = payload.get("positions") or []
    normalized_positions: list[dict[str, Any]] = []
    total_market_value = 0.0
    total_position_profit = 0.0
    for row in positions:
        market_value = _float_or_none(row.get("market_value"))
        position_profit = _float_or_none(row.get("position_profit"))
        normalized = {
            "stock_code": row.get("stock_code"),
            "instrument_name": row.get("instrument_name"),
            "volume": row.get("volume"),
            "can_use_volume": row.get("can_use_volume"),
            "open_price": row.get("open_price"),
            "last_price": row.get("last_price"),
            "market_value": market_value,
            "position_profit": position_profit,
            "profit_rate": row.get("profit_rate"),
        }
        normalized_positions.append(normalized)
        if market_value is not None:
            total_market_value += market_value
        if position_profit is not None:
            total_position_profit += position_profit

    normalized_positions.sort(
        key=lambda row: (
            row["market_value"] is None,
            -(row["market_value"] or 0.0),
            str(row["stock_code"] or ""),
        )
    )
    summary.update(
        {
            "ok": True,
            "asset": {
                "total_asset": asset.get("total_asset"),
                "market_value": asset.get("market_value"),
                "cash": asset.get("cash"),
                "frozen_cash": asset.get("frozen_cash"),
                "fetch_balance": asset.get("fetch_balance"),
            },
            "position_count": len(normalized_positions),
            "positions": normalized_positions,
            "totals": {
                "position_market_value_sum": round(total_market_value, 2),
                "position_profit_sum": round(total_position_profit, 2),
            },
        }
    )
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    if not summary.get("ok"):
        print("QMT 持仓读取失败")
        print(f"账户: {summary.get('account_id')} ({summary.get('account_type')})")
        print(f"原因: {summary.get('error', '未知错误')}")
        return

    asset = summary.get("asset", {})
    print("QMT 当前持仓摘要")
    print(f"时间: {summary.get('timestamp')}")
    print(f"账户: {summary.get('account_id')} ({summary.get('account_type')})")
    print(f"来源: {summary.get('source')}")
    print(
        "资产: "
        f"总资产={asset.get('total_asset')} "
        f"持仓市值={asset.get('market_value')} "
        f"可用资金={asset.get('cash')} "
        f"可取资金={asset.get('fetch_balance')}"
    )
    print(
        "汇总: "
        f"持仓数量={summary.get('position_count')} "
        f"持仓市值合计={summary.get('totals', {}).get('position_market_value_sum')} "
        f"持仓盈亏合计={summary.get('totals', {}).get('position_profit_sum')}"
    )
    print("")
    print("持仓明细:")
    for index, row in enumerate(summary.get("positions", []), start=1):
        print(
            f"{index}. {row.get('stock_code')} {row.get('instrument_name')} "
            f"持仓={row.get('volume')} 可用={row.get('can_use_volume')} "
            f"成本={row.get('open_price')} 现价={row.get('last_price')} "
            f"市值={row.get('market_value')} 盈亏={row.get('position_profit')} "
            f"收益率={row.get('profit_rate')}"
        )


def _print_json(report: dict[str, Any]) -> None:
    print(json.dumps(_safe(report), ensure_ascii=False, indent=2))


def build_report(
    account_id: str,
    account_type: str,
    userdata_path: str | None,
    connect_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "account_id": account_id,
        "account_type": account_type,
    }

    xtdata_info = _probe_xtdata()
    report["xtdata"] = xtdata_info

    resolved_userdata = userdata_path or xtdata_info.get("inferred_userdata_path")
    report["xttrader"] = (
        _probe_xttrader(
            account_id,
            account_type,
            resolved_userdata,
            connect_retries,
            retry_sleep_seconds,
        )
        if resolved_userdata
        else {"ok": False, "reason": "userdata_path unavailable"}
    )
    report["qmttools_trade_detail"] = _probe_qmttools(account_id, account_type)
    report["summary"] = _build_summary(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Read QMT holdings through official APIs.")
    parser.add_argument("--account-id", required=True, help="QMT account id, e.g. 8883530863")
    parser.add_argument(
        "--account-type",
        default="STOCK",
        help="QMT account type. Default: STOCK",
    )
    parser.add_argument(
        "--userdata-path",
        default="",
        help="Optional explicit MiniQMT userdata_mini path.",
    )
    parser.add_argument(
        "--format",
        choices=("summary", "json"),
        default="summary",
        help="Output format. Default: summary",
    )
    parser.add_argument(
        "--connect-retries",
        type=int,
        default=3,
        help="XtQuantTrader connect retry count. Default: 3",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep seconds between XtQuantTrader connect retries. Default: 1.0",
    )
    args = parser.parse_args()

    report = build_report(
        account_id=args.account_id,
        account_type=args.account_type,
        userdata_path=args.userdata_path or None,
        connect_retries=args.connect_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    if args.format == "summary":
        _print_summary(report["summary"])
    else:
        _print_json(report)

    if report.get("xttrader", {}).get("ok") or report.get("qmttools_trade_detail", {}).get("ok"):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

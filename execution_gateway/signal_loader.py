from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .contracts import OrderIntent


def _read_signal_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_latest_signal_batch_payload(
    latest_csv_path: str | Path,
    latest_status_path: str | Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = _read_signal_rows(Path(latest_csv_path))
    status = json.loads(Path(latest_status_path).read_text(encoding="utf-8-sig"))
    if bool(status.get("no_signal")) or str(status.get("signal_semantics") or "") == "no_signal_hold_only":
        if rows:
            raise ValueError("no_signal_hold_only batch must not contain formal action rows")
        return [], status
    if not rows:
        raise ValueError("action signal batch cannot be empty when no_signal_hold_only is false")
    return rows, status


def build_order_intents_from_latest(
    *,
    latest_csv_path: str | Path,
    latest_status_path: str | Path,
    workflow_run_id: str,
    account_id: str,
    strategy_version: str,
    rebalance_version: str = "latest",
) -> tuple[list[OrderIntent], dict[str, Any]]:
    rows, status = load_latest_signal_batch_payload(latest_csv_path, latest_status_path)
    intents: list[OrderIntent] = []
    if not rows:
        return intents, status
    strategy_id = str(status.get("strategy_id") or "").strip()
    signal_date = str(status.get("signal_date") or "").strip()
    buy_date = str(status.get("buy_date") or "").strip()
    for row in rows:
        action = str(row.get("action") or "").strip().upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError(f"unsupported latest signal action for execution gateway: {action}")
        intents.append(
            OrderIntent(
                workflow_run_id=workflow_run_id,
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                signal_date=signal_date,
                buy_date=buy_date,
                account_id=account_id,
                symbol=str(row.get("stock_code") or "").strip().upper(),
                side=action,
                rebalance_version=rebalance_version,
                target_pct=float(row.get("target_pct") or 0.0),
                limit_price=float(row.get("execution_open_raw") or 0.0) or None,
                reason=str(row.get("reason") or "").strip(),
                metadata={
                    "name": row.get("name") or "",
                    "market": row.get("market") or "",
                    "score_rank": row.get("score_rank") or "",
                    "score_denominator": row.get("score_denominator") or "",
                    "status": row.get("status") or "",
                },
            )
        )
    return intents, status

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from build_manual_trade_package import (
    load_holdings_summary_from_json,
    load_holdings_summary_from_qmt,
)
from execution_gateway.contracts import AccountSnapshot, MarketSnapshot
from execution_gateway.kill_switch import load_kill_switch_state
from execution_gateway.policy import load_execution_authorization
from execution_gateway.service import ExecutionGatewayService
from project_paths import resolve_data_path


def _read_market_snapshot(path: Path) -> dict[str, MarketSnapshot]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("ticks") or payload.get("data") or [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    market_by_symbol: dict[str, MarketSnapshot] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        snapshot = MarketSnapshot.from_dict(row)
        if snapshot.symbol:
            market_by_symbol[snapshot.symbol] = snapshot
    return market_by_symbol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a shadow-only execution gateway dry-run.")
    parser.add_argument("--policy", required=True, help="Path to execution authorization JSON.")
    parser.add_argument("--latest-csv", required=True, help="Formal latest signal CSV.")
    parser.add_argument("--latest-status", required=True, help="Formal latest status JSON.")
    parser.add_argument("--market-snapshot-json", required=True, help="Realtime or simulated quote snapshot JSON.")
    parser.add_argument("--workflow-run-id", required=True, help="Workflow run id for idempotency.")
    parser.add_argument("--holdings-json", default="", help="Offline holdings summary JSON.")
    parser.add_argument("--account-id", default="", help="QMT account id for live holdings read.")
    parser.add_argument("--account-type", default="STOCK")
    parser.add_argument("--userdata-path", default="")
    parser.add_argument(
        "--kill-switch",
        default="runtime/execution_gateway/kill_switch_state.json",
        help="Kill switch state JSON path, relative to data_file unless absolute path is provided.",
    )
    parser.add_argument(
        "--output",
        default="runtime/execution_gateway/shadow_runs/latest_shadow_run.json",
        help="Output JSON path, relative to data_file unless absolute path is provided.",
    )
    return parser.parse_args(argv)


def _load_holdings(args: argparse.Namespace) -> dict[str, Any]:
    if args.holdings_json:
        return json.loads(Path(args.holdings_json).read_text(encoding="utf-8-sig"))
    if args.account_id:
        return load_holdings_summary_from_qmt(
            args.account_id,
            account_type=args.account_type,
            userdata_path=args.userdata_path or None,
        )
    raise ValueError("either --holdings-json or --account-id is required")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    authorization = load_execution_authorization(args.policy)
    holdings_summary = _load_holdings(args)
    account = AccountSnapshot.from_holdings_summary(holdings_summary)
    market_by_symbol = _read_market_snapshot(Path(args.market_snapshot_json))
    kill_switch = load_kill_switch_state(resolve_data_path(args.kill_switch))
    service = ExecutionGatewayService(
        authorization=authorization,
        kill_switch_state=kill_switch,
    )
    result = service.evaluate_latest_batch(
        latest_csv_path=args.latest_csv,
        latest_status_path=args.latest_status,
        workflow_run_id=args.workflow_run_id,
        account=account,
        market_by_symbol=market_by_symbol,
    )
    output_path = resolve_data_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "batch_status": result["batch_status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

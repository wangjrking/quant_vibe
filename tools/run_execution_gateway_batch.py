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
from execution_gateway.broker_adapter import MiniQmtBrokerAdapter, PaperBrokerAdapter, ShadowBrokerAdapter
from execution_gateway.contracts import AccountSnapshot, MarketSnapshot
from execution_gateway.executor import PreparedOrder, build_order_request, submit_prepared_orders
from execution_gateway.kill_switch import load_kill_switch_state
from execution_gateway.ledger import ExecutionLedger
from execution_gateway.live_arm import load_live_arm_state, validate_live_arm_state
from execution_gateway.policy import load_execution_authorization
from execution_gateway.reconciliation import reconcile_orders
from execution_gateway.service import ExecutionGatewayService
from execution_gateway.state_store import ExecutionStateStore
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
    parser = argparse.ArgumentParser(description="Evaluate and stage execution-gateway batch submissions.")
    parser.add_argument("--policy", required=True, help="Path to execution authorization JSON.")
    parser.add_argument("--latest-csv", required=True, help="Formal latest signal CSV.")
    parser.add_argument("--latest-status", required=True, help="Formal latest signal status JSON.")
    parser.add_argument("--market-snapshot-json", required=True, help="Realtime or simulated quote snapshot JSON.")
    parser.add_argument("--workflow-run-id", required=True, help="Workflow run id for idempotency.")
    parser.add_argument("--mode", choices=("shadow", "paper", "live"), required=True)
    parser.add_argument("--holdings-json", default="", help="Offline holdings summary JSON.")
    parser.add_argument("--account-id", default="", help="QMT account id for live holdings read.")
    parser.add_argument("--account-type", default="STOCK")
    parser.add_argument("--userdata-path", default="")
    parser.add_argument("--allow-live-submit", action="store_true")
    parser.add_argument(
        "--live-arm",
        default="runtime/execution_gateway/live_arm_state.json",
        help="Short-lived live arm JSON path, relative to data_file unless absolute path is provided.",
    )
    parser.add_argument(
        "--kill-switch",
        default="runtime/execution_gateway/kill_switch_state.json",
        help="Kill switch state JSON path, relative to data_file unless absolute path is provided.",
    )
    parser.add_argument(
        "--ledger",
        default="runtime/execution_gateway/order_ledger.jsonl",
        help="Execution gateway JSONL ledger path, relative to data_file unless absolute path is provided.",
    )
    parser.add_argument(
        "--state-db",
        default="runtime/execution_gateway/execution_state.sqlite3",
        help="SQLite execution state store path, relative to data_file unless absolute path is provided.",
    )
    parser.add_argument(
        "--output",
        default="runtime/execution_gateway/batch_runs/latest_batch_run.json",
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


def _build_adapter(args: argparse.Namespace):
    if args.mode == "shadow":
        return ShadowBrokerAdapter()
    if args.mode == "paper":
        return PaperBrokerAdapter()
    return MiniQmtBrokerAdapter(
        account_id=args.account_id,
        account_type=args.account_type,
        userdata_path=args.userdata_path or None,
        allow_live_submit=args.allow_live_submit,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    authorization = load_execution_authorization(args.policy)
    if authorization.stage != args.mode:
        raise ValueError(f"policy stage {authorization.stage} does not match requested mode {args.mode}")
    if args.mode == "live" and not args.allow_live_submit:
        raise RuntimeError("live mode requires --allow-live-submit and remains subject to owner approval outside this tool")
    live_arm_state = None
    live_arm_validation = {"valid": True, "blockers": []}
    if args.mode == "live":
        live_arm_state = load_live_arm_state(resolve_data_path(args.live_arm))
        valid, blockers = validate_live_arm_state(
            arm_state=live_arm_state,
            authorization=authorization,
        )
        live_arm_validation = {"valid": valid, "blockers": list(blockers), "state": live_arm_state.to_dict()}
        if not valid:
            raise RuntimeError(f"live arm validation failed: {', '.join(blockers)}")

    holdings_summary = _load_holdings(args)
    account = AccountSnapshot.from_holdings_summary(holdings_summary)
    market_by_symbol = _read_market_snapshot(Path(args.market_snapshot_json))
    kill_switch = load_kill_switch_state(resolve_data_path(args.kill_switch))
    service = ExecutionGatewayService(
        authorization=authorization,
        kill_switch_state=kill_switch,
    )
    evaluation = service.evaluate_latest_batch(
        latest_csv_path=args.latest_csv,
        latest_status_path=args.latest_status,
        workflow_run_id=args.workflow_run_id,
        account=account,
        market_by_symbol=market_by_symbol,
    )

    prepared_orders: list[PreparedOrder] = []
    for decision_payload in evaluation["decisions"]:
        if not decision_payload.get("approved"):
            continue
        symbol = str(decision_payload["intent"]["symbol"]).strip().upper()
        market = market_by_symbol.get(symbol)
        if market is None:
            continue
        from execution_gateway.contracts import RiskDecision, OrderIntent  # local import to keep module startup light

        prepared_orders.append(
            build_order_request(
                decision=RiskDecision(
                    intent=OrderIntent.from_dict(decision_payload["intent"]),
                    status=str(decision_payload["status"]),
                    approved=bool(decision_payload["approved"]),
                    blockers=tuple(decision_payload.get("blockers") or ()),
                    notes=tuple(decision_payload.get("notes") or ()),
                    evaluated_at=str(decision_payload.get("evaluated_at") or ""),
                ),
                account=account,
                market=market,
            )
        )

    ledger = ExecutionLedger(resolve_data_path(args.ledger))
    state_store = ExecutionStateStore(resolve_data_path(args.state_db))
    adapter = _build_adapter(args)
    batch_id = args.workflow_run_id
    signal_status = evaluation.get("signal_status") or {}
    state_store.upsert_batch(
        batch_id=batch_id,
        mode=args.mode,
        workflow_run_id=args.workflow_run_id,
        account_id=account.account_id,
        strategy_id=str(signal_status.get("strategy_id") or authorization.strategy_id),
        strategy_version=authorization.strategy_version,
        signal_date=str(signal_status.get("signal_date") or ""),
        buy_date=str(signal_status.get("buy_date") or ""),
        batch_status=str(evaluation.get("batch_status") or "unknown"),
        intent_count=int(evaluation.get("intent_count") or 0),
        approved_count=int(sum(1 for item in prepared_orders)),
        created_at=account.captured_at,
        payload={
            "policy": authorization.to_dict(),
            "evaluation": evaluation,
            "live_arm_validation": live_arm_validation,
        },
    )
    for prepared in prepared_orders:
        state_store.upsert_prepared_order(
            client_order_id=prepared.request.client_order_id,
            batch_id=batch_id,
            symbol=prepared.request.intent.symbol,
            side=prepared.request.intent.side,
            quantity=prepared.request.quantity,
            price_mode=prepared.request.price_mode,
            limit_price=prepared.request.limit_price,
            created_at=prepared.risk_decision.evaluated_at,
            request_payload=prepared.request.to_dict(),
        )
    acknowledgements = submit_prepared_orders(
        prepared_orders=prepared_orders,
        adapter=adapter,
        ledger=ledger,
        batch_id=batch_id,
    )
    for ack in acknowledgements:
        state_store.upsert_broker_ack(
            client_order_id=ack.client_order_id,
            broker_order_id=ack.broker_order_id or ack.state,
            state=ack.state,
            submitted=ack.submitted,
            acknowledged_at=ack.submitted_at,
            ack_payload=ack.to_dict(),
        )
    reconciliation = reconcile_orders(
        prepared_orders=prepared_orders,
        ledger_events=ledger.read_events(),
        broker_orders=adapter.query_orders(account.account_id),
    )

    result = {
        "mode": args.mode,
        "workflow_run_id": args.workflow_run_id,
        "policy": authorization.to_dict(),
        "evaluation": evaluation,
        "live_arm_validation": live_arm_validation,
        "prepared_order_count": len(prepared_orders),
        "prepared_orders": [item.to_dict() for item in prepared_orders],
        "acknowledgements": [item.to_dict() for item in acknowledgements],
        "reconciliation": reconciliation.to_dict(),
        "ledger_path": str(resolve_data_path(args.ledger)),
        "state_db_path": str(resolve_data_path(args.state_db)),
        "state_db_counts": state_store.counts(),
    }
    output_path = resolve_data_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "mode": args.mode,
                "prepared_order_count": len(prepared_orders),
                "acknowledgement_count": len(acknowledgements),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS execution_batches (
  batch_id TEXT PRIMARY KEY,
  mode TEXT NOT NULL,
  workflow_run_id TEXT NOT NULL,
  account_id TEXT NOT NULL,
  strategy_id TEXT NOT NULL,
  strategy_version TEXT NOT NULL,
  signal_date TEXT NOT NULL,
  buy_date TEXT NOT NULL,
  batch_status TEXT NOT NULL,
  intent_count INTEGER NOT NULL,
  approved_count INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prepared_orders (
  client_order_id TEXT PRIMARY KEY,
  batch_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity INTEGER NOT NULL,
  price_mode TEXT NOT NULL,
  limit_price REAL,
  request_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(batch_id) REFERENCES execution_batches(batch_id)
);

CREATE TABLE IF NOT EXISTS broker_acknowledgements (
  client_order_id TEXT NOT NULL,
  broker_order_id TEXT NOT NULL,
  state TEXT NOT NULL,
  submitted INTEGER NOT NULL,
  acknowledged_at TEXT NOT NULL,
  ack_json TEXT NOT NULL,
  PRIMARY KEY (client_order_id, broker_order_id)
);
"""


class ExecutionStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA_SQL)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_batch(
        self,
        *,
        batch_id: str,
        mode: str,
        workflow_run_id: str,
        account_id: str,
        strategy_id: str,
        strategy_version: str,
        signal_date: str,
        buy_date: str,
        batch_status: str,
        intent_count: int,
        approved_count: int,
        created_at: str,
        payload: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO execution_batches (
                  batch_id, mode, workflow_run_id, account_id, strategy_id, strategy_version,
                  signal_date, buy_date, batch_status, intent_count, approved_count, created_at, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                  mode=excluded.mode,
                  workflow_run_id=excluded.workflow_run_id,
                  account_id=excluded.account_id,
                  strategy_id=excluded.strategy_id,
                  strategy_version=excluded.strategy_version,
                  signal_date=excluded.signal_date,
                  buy_date=excluded.buy_date,
                  batch_status=excluded.batch_status,
                  intent_count=excluded.intent_count,
                  approved_count=excluded.approved_count,
                  created_at=excluded.created_at,
                  payload_json=excluded.payload_json
                """,
                (
                    batch_id,
                    mode,
                    workflow_run_id,
                    account_id,
                    strategy_id,
                    strategy_version,
                    signal_date,
                    buy_date,
                    batch_status,
                    intent_count,
                    approved_count,
                    created_at,
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )

    def upsert_prepared_order(
        self,
        *,
        client_order_id: str,
        batch_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price_mode: str,
        limit_price: float | None,
        created_at: str,
        request_payload: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO prepared_orders (
                  client_order_id, batch_id, symbol, side, quantity, price_mode, limit_price, request_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                  batch_id=excluded.batch_id,
                  symbol=excluded.symbol,
                  side=excluded.side,
                  quantity=excluded.quantity,
                  price_mode=excluded.price_mode,
                  limit_price=excluded.limit_price,
                  request_json=excluded.request_json,
                  created_at=excluded.created_at
                """,
                (
                    client_order_id,
                    batch_id,
                    symbol,
                    side,
                    quantity,
                    price_mode,
                    limit_price,
                    json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")),
                    created_at,
                ),
            )

    def upsert_broker_ack(
        self,
        *,
        client_order_id: str,
        broker_order_id: str,
        state: str,
        submitted: bool,
        acknowledged_at: str,
        ack_payload: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO broker_acknowledgements (
                  client_order_id, broker_order_id, state, submitted, acknowledged_at, ack_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id, broker_order_id) DO UPDATE SET
                  state=excluded.state,
                  submitted=excluded.submitted,
                  acknowledged_at=excluded.acknowledged_at,
                  ack_json=excluded.ack_json
                """,
                (
                    client_order_id,
                    broker_order_id,
                    state,
                    1 if submitted else 0,
                    acknowledged_at,
                    json.dumps(ack_payload, ensure_ascii=False, separators=(",", ":")),
                ),
            )

    def batch_summary(self, batch_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM execution_batches WHERE batch_id = ?
                """,
                (batch_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            batches = conn.execute("SELECT COUNT(*) FROM execution_batches").fetchone()[0]
            prepared = conn.execute("SELECT COUNT(*) FROM prepared_orders").fetchone()[0]
            acks = conn.execute("SELECT COUNT(*) FROM broker_acknowledgements").fetchone()[0]
        return {
            "execution_batches": int(batches),
            "prepared_orders": int(prepared),
            "broker_acknowledgements": int(acks),
        }

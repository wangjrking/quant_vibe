from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .contracts import OrderIntent


VALID_PRICE_MODES = {"LIMIT", "LATEST", "MARKET_BEST"}


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("gb18030", errors="replace")
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _safe(item) for key, item in vars(value).items() if not key.startswith("_")}
    fields: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if callable(item):
            continue
        fields[name] = _safe(item)
    return fields or repr(value)


@dataclass(frozen=True)
class BrokerOrderRequest:
    intent: OrderIntent
    client_order_id: str
    quantity: int
    price_mode: str = "LIMIT"
    limit_price: float | None = None
    strategy_name: str = "execution_gateway"
    order_remark: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.client_order_id.strip():
            raise ValueError("broker order request missing client_order_id")
        if self.quantity <= 0:
            raise ValueError("broker order request quantity must be positive")
        if self.price_mode not in VALID_PRICE_MODES:
            raise ValueError(f"unsupported price_mode: {self.price_mode}")
        if self.price_mode == "LIMIT" and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("limit_price must be positive when price_mode=LIMIT")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "client_order_id": self.client_order_id,
            "quantity": self.quantity,
            "price_mode": self.price_mode,
            "limit_price": self.limit_price,
            "strategy_name": self.strategy_name,
            "order_remark": self.order_remark,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BrokerOrderAck:
    client_order_id: str
    account_id: str
    symbol: str
    side: str
    broker_order_id: str
    submitted: bool
    state: str
    submitted_at: str = field(default_factory=_utc_iso_now)
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "side": self.side,
            "broker_order_id": self.broker_order_id,
            "submitted": self.submitted,
            "state": self.state,
            "submitted_at": self.submitted_at,
            "message": self.message,
            "raw": self.raw,
        }


@dataclass(frozen=True)
class BrokerOrderUpdate:
    broker_order_id: str
    account_id: str
    symbol: str
    side: str
    state: str
    client_order_id: str = ""
    filled_quantity: int = 0
    remaining_quantity: int = 0
    average_price: float = 0.0
    updated_at: str = field(default_factory=_utc_iso_now)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "broker_order_id": self.broker_order_id,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "side": self.side,
            "state": self.state,
            "client_order_id": self.client_order_id,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "updated_at": self.updated_at,
            "raw": self.raw,
        }


@dataclass(frozen=True)
class BrokerTradeFill:
    broker_order_id: str
    trade_id: str
    account_id: str
    symbol: str
    side: str
    filled_quantity: int
    filled_price: float
    traded_at: str = field(default_factory=_utc_iso_now)
    client_order_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "broker_order_id": self.broker_order_id,
            "trade_id": self.trade_id,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "side": self.side,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "traded_at": self.traded_at,
            "client_order_id": self.client_order_id,
            "raw": self.raw,
        }


@runtime_checkable
class BrokerAdapter(Protocol):
    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrderAck:
        ...

    def cancel_order(self, account_id: str, broker_order_id: str) -> BrokerOrderUpdate:
        ...

    def query_orders(self, account_id: str) -> list[BrokerOrderUpdate]:
        ...

    def query_trades(self, account_id: str) -> list[BrokerTradeFill]:
        ...


class ShadowBrokerAdapter:
    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrderAck:
        return BrokerOrderAck(
            client_order_id=request.client_order_id,
            account_id=request.intent.account_id,
            symbol=request.intent.symbol,
            side=request.intent.side,
            broker_order_id="",
            submitted=False,
            state="shadow_only",
            message="shadow stage: no broker submission performed",
            raw={"request": request.to_dict()},
        )

    def cancel_order(self, account_id: str, broker_order_id: str) -> BrokerOrderUpdate:
        return BrokerOrderUpdate(
            broker_order_id=broker_order_id,
            account_id=account_id,
            symbol="",
            side="",
            state="shadow_only",
            raw={"message": "shadow stage: no cancel performed"},
        )

    def query_orders(self, account_id: str) -> list[BrokerOrderUpdate]:
        return []

    def query_trades(self, account_id: str) -> list[BrokerTradeFill]:
        return []


class PaperBrokerAdapter:
    def __init__(self) -> None:
        self._orders: dict[str, BrokerOrderUpdate] = {}
        self._trades: list[BrokerTradeFill] = []

    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrderAck:
        broker_order_id = "PAPER-" + sha256(request.client_order_id.encode("utf-8")).hexdigest()[:16].upper()
        update = BrokerOrderUpdate(
            broker_order_id=broker_order_id,
            account_id=request.intent.account_id,
            symbol=request.intent.symbol,
            side=request.intent.side,
            state="acknowledged",
            client_order_id=request.client_order_id,
            filled_quantity=0,
            remaining_quantity=request.quantity,
            average_price=0.0,
            raw={"request": request.to_dict()},
        )
        self._orders[broker_order_id] = update
        return BrokerOrderAck(
            client_order_id=request.client_order_id,
            account_id=request.intent.account_id,
            symbol=request.intent.symbol,
            side=request.intent.side,
            broker_order_id=broker_order_id,
            submitted=True,
            state="acknowledged",
            message="paper adapter acknowledged order locally",
            raw={"request": request.to_dict()},
        )

    def cancel_order(self, account_id: str, broker_order_id: str) -> BrokerOrderUpdate:
        order = self._orders.get(broker_order_id)
        if order is None:
            return BrokerOrderUpdate(
                broker_order_id=broker_order_id,
                account_id=account_id,
                symbol="",
                side="",
                state="reconcile_exception",
                raw={"message": "paper order not found"},
            )
        cancelled = BrokerOrderUpdate(
            broker_order_id=broker_order_id,
            account_id=order.account_id,
            symbol=order.symbol,
            side=order.side,
            state="cancelled",
            client_order_id=order.client_order_id,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            average_price=order.average_price,
            raw=order.raw,
        )
        self._orders[broker_order_id] = cancelled
        return cancelled

    def query_orders(self, account_id: str) -> list[BrokerOrderUpdate]:
        return [item for item in self._orders.values() if item.account_id == account_id]

    def query_trades(self, account_id: str) -> list[BrokerTradeFill]:
        return [item for item in self._trades if item.account_id == account_id]


class MiniQmtBrokerAdapter:
    def __init__(
        self,
        *,
        account_id: str,
        account_type: str = "STOCK",
        userdata_path: str | None = None,
        strategy_name: str = "execution_gateway",
        allow_live_submit: bool = False,
    ) -> None:
        self.account_id = account_id
        self.account_type = account_type
        self.userdata_path = userdata_path
        self.strategy_name = strategy_name
        self.allow_live_submit = allow_live_submit

    def _resolve_userdata_path(self) -> str:
        if self.userdata_path:
            return self.userdata_path
        from xtquant import xtdata

        client = xtdata.connect()
        data_dir = Path(client.get_data_dir()).resolve()
        if data_dir.name.lower() == "datadir":
            return str(data_dir.parent)
        raise RuntimeError("unable to infer MiniQMT userdata path")

    def _map_order_type(self, side: str) -> int:
        import xtquant.xtconstant as xtconstant

        if side == "BUY":
            return xtconstant.STOCK_BUY
        if side == "SELL":
            return xtconstant.STOCK_SELL
        raise ValueError(f"unsupported order side for MiniQMT adapter: {side}")

    def _map_price_type(self, price_mode: str) -> int:
        import xtquant.xtconstant as xtconstant

        mapping = {
            "LIMIT": xtconstant.FIX_PRICE,
            "LATEST": xtconstant.LATEST_PRICE,
            "MARKET_BEST": xtconstant.MARKET_PEER_PRICE_FIRST,
        }
        if price_mode not in mapping:
            raise ValueError(f"unsupported price mode for MiniQMT adapter: {price_mode}")
        return mapping[price_mode]

    def submit_order(self, request: BrokerOrderRequest) -> BrokerOrderAck:
        if not self.allow_live_submit:
            raise RuntimeError(
                "MiniQmtBrokerAdapter live submission is disabled by default; "
                "enable only after owner approval, audit, and explicit live arm."
            )

        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount

        userdata_path = self._resolve_userdata_path()
        session_id = int(datetime.now(timezone.utc).timestamp())
        trader = XtQuantTrader(userdata_path, session_id)
        try:
            trader.start()
            connect_ret = trader.connect()
            if connect_ret != 0:
                raise RuntimeError(f"XtQuantTrader.connect() failed with code {connect_ret}")
            account = StockAccount(self.account_id, self.account_type)
            subscribe_ret = trader.subscribe(account)
            if subscribe_ret != 0:
                raise RuntimeError(f"XtQuantTrader.subscribe() failed with code {subscribe_ret}")
            order_id = trader.order_stock(
                account,
                request.intent.symbol,
                self._map_order_type(request.intent.side),
                request.quantity,
                self._map_price_type(request.price_mode),
                float(request.limit_price or 0.0),
                strategy_name=request.strategy_name or self.strategy_name,
                order_remark=request.order_remark or request.client_order_id,
            )
            submitted = bool(isinstance(order_id, int) and order_id > 0)
            return BrokerOrderAck(
                client_order_id=request.client_order_id,
                account_id=request.intent.account_id,
                symbol=request.intent.symbol,
                side=request.intent.side,
                broker_order_id=str(order_id) if order_id is not None else "",
                submitted=submitted,
                state="submitting" if submitted else "rejected",
                message="" if submitted else "MiniQMT rejected order request",
                raw={
                    "connect_ret": connect_ret,
                    "subscribe_ret": subscribe_ret,
                    "request": request.to_dict(),
                },
            )
        finally:
            try:
                trader.stop()
            except Exception:
                pass

    def cancel_order(self, account_id: str, broker_order_id: str) -> BrokerOrderUpdate:
        if not self.allow_live_submit:
            raise RuntimeError("MiniQmtBrokerAdapter cancellation is disabled while live submit is disabled")

        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount

        userdata_path = self._resolve_userdata_path()
        session_id = int(datetime.now(timezone.utc).timestamp())
        trader = XtQuantTrader(userdata_path, session_id)
        try:
            trader.start()
            connect_ret = trader.connect()
            if connect_ret != 0:
                raise RuntimeError(f"XtQuantTrader.connect() failed with code {connect_ret}")
            account = StockAccount(self.account_id, self.account_type)
            subscribe_ret = trader.subscribe(account)
            if subscribe_ret != 0:
                raise RuntimeError(f"XtQuantTrader.subscribe() failed with code {subscribe_ret}")
            cancel_ret = trader.cancel_order_stock(account, int(broker_order_id))
            return BrokerOrderUpdate(
                broker_order_id=broker_order_id,
                account_id=account_id,
                symbol="",
                side="",
                state="cancel_pending" if cancel_ret == 0 else "reconcile_exception",
                raw={"connect_ret": connect_ret, "subscribe_ret": subscribe_ret, "cancel_ret": cancel_ret},
            )
        finally:
            try:
                trader.stop()
            except Exception:
                pass

    def query_orders(self, account_id: str) -> list[BrokerOrderUpdate]:
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount

        userdata_path = self._resolve_userdata_path()
        session_id = int(datetime.now(timezone.utc).timestamp())
        trader = XtQuantTrader(userdata_path, session_id)
        try:
            trader.start()
            connect_ret = trader.connect()
            if connect_ret != 0:
                raise RuntimeError(f"XtQuantTrader.connect() failed with code {connect_ret}")
            account = StockAccount(self.account_id, self.account_type)
            subscribe_ret = trader.subscribe(account)
            if subscribe_ret != 0:
                raise RuntimeError(f"XtQuantTrader.subscribe() failed with code {subscribe_ret}")
            rows = trader.query_stock_orders(account)
            results: list[BrokerOrderUpdate] = []
            for row in rows or []:
                raw = _safe(row)
                results.append(
                    BrokerOrderUpdate(
                        broker_order_id=str(raw.get("order_id") or raw.get("m_nOrderID") or ""),
                        account_id=account_id,
                        symbol=str(raw.get("stock_code") or raw.get("stockCode") or "").strip().upper(),
                        side=str(raw.get("order_type") or raw.get("direction") or ""),
                        state=str(raw.get("order_status") or raw.get("status") or ""),
                        client_order_id=str(raw.get("order_remark") or raw.get("remark") or ""),
                        filled_quantity=int(float(raw.get("traded_volume") or raw.get("filled_qty") or 0)),
                        remaining_quantity=int(float(raw.get("left_volume") or raw.get("remaining_qty") or 0)),
                        average_price=float(raw.get("traded_price") or raw.get("avg_price") or 0.0),
                        raw=raw,
                    )
                )
            return results
        finally:
            try:
                trader.stop()
            except Exception:
                pass

    def query_trades(self, account_id: str) -> list[BrokerTradeFill]:
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount

        userdata_path = self._resolve_userdata_path()
        session_id = int(datetime.now(timezone.utc).timestamp())
        trader = XtQuantTrader(userdata_path, session_id)
        try:
            trader.start()
            connect_ret = trader.connect()
            if connect_ret != 0:
                raise RuntimeError(f"XtQuantTrader.connect() failed with code {connect_ret}")
            account = StockAccount(self.account_id, self.account_type)
            subscribe_ret = trader.subscribe(account)
            if subscribe_ret != 0:
                raise RuntimeError(f"XtQuantTrader.subscribe() failed with code {subscribe_ret}")
            rows = trader.query_stock_trades(account)
            results: list[BrokerTradeFill] = []
            for row in rows or []:
                raw = _safe(row)
                results.append(
                    BrokerTradeFill(
                        broker_order_id=str(raw.get("order_id") or raw.get("m_nOrderID") or ""),
                        trade_id=str(raw.get("traded_id") or raw.get("trade_id") or ""),
                        account_id=account_id,
                        symbol=str(raw.get("stock_code") or raw.get("stockCode") or "").strip().upper(),
                        side=str(raw.get("order_type") or raw.get("direction") or ""),
                        filled_quantity=int(float(raw.get("traded_volume") or raw.get("volume") or 0)),
                        filled_price=float(raw.get("traded_price") or raw.get("price") or 0.0),
                        client_order_id=str(raw.get("order_remark") or raw.get("remark") or ""),
                        raw=raw,
                    )
                )
            return results
        finally:
            try:
                trader.stop()
            except Exception:
                pass

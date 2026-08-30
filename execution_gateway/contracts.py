from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


VALID_EXECUTION_STAGES = {"shadow", "paper", "live"}
VALID_ORDER_SIDES = {"BUY", "SELL"}


def _float_or_none(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExecutionAuthorization:
    authorization_id: str
    account_id: str
    stage: str
    strategy_id: str
    strategy_version: str
    owner_approval_id: str
    policy_hash: str
    enabled: bool = False
    expires_at: str | None = None
    allowed_sides: tuple[str, ...] = ("BUY", "SELL")
    allowed_symbols: tuple[str, ...] = ()
    max_order_notional: float = 0.0
    max_symbol_position_pct: float = 0.0
    max_total_position_pct: float = 0.0
    max_orders_per_batch: int = 0
    quote_max_age_seconds: int = 10
    kill_switch_required: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionAuthorization":
        stage = str(payload.get("stage") or "").strip().lower()
        if stage not in VALID_EXECUTION_STAGES:
            raise ValueError(f"unsupported execution stage: {stage}")
        allowed_sides = tuple(
            str(item).strip().upper()
            for item in (payload.get("allowed_sides") or ["BUY", "SELL"])
            if str(item).strip()
        )
        invalid_sides = sorted(set(allowed_sides) - VALID_ORDER_SIDES)
        if invalid_sides:
            raise ValueError(f"unsupported order sides in authorization: {', '.join(invalid_sides)}")
        symbols = tuple(
            str(item).strip().upper()
            for item in (payload.get("allowed_symbols") or [])
            if str(item).strip()
        )
        return cls(
            authorization_id=str(payload.get("authorization_id") or "").strip(),
            account_id=str(payload.get("account_id") or "").strip(),
            stage=stage,
            strategy_id=str(payload.get("strategy_id") or "").strip(),
            strategy_version=str(payload.get("strategy_version") or "").strip(),
            owner_approval_id=str(payload.get("owner_approval_id") or "").strip(),
            policy_hash=str(payload.get("policy_hash") or "").strip(),
            enabled=bool(payload.get("enabled", False)),
            expires_at=str(payload["expires_at"]).strip() if payload.get("expires_at") else None,
            allowed_sides=allowed_sides,
            allowed_symbols=symbols,
            max_order_notional=float(payload.get("max_order_notional") or 0.0),
            max_symbol_position_pct=float(payload.get("max_symbol_position_pct") or 0.0),
            max_total_position_pct=float(payload.get("max_total_position_pct") or 0.0),
            max_orders_per_batch=int(payload.get("max_orders_per_batch") or 0),
            quote_max_age_seconds=int(payload.get("quote_max_age_seconds") or 10),
            kill_switch_required=bool(payload.get("kill_switch_required", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorization_id": self.authorization_id,
            "account_id": self.account_id,
            "stage": self.stage,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "owner_approval_id": self.owner_approval_id,
            "policy_hash": self.policy_hash,
            "enabled": self.enabled,
            "expires_at": self.expires_at,
            "allowed_sides": list(self.allowed_sides),
            "allowed_symbols": list(self.allowed_symbols),
            "max_order_notional": self.max_order_notional,
            "max_symbol_position_pct": self.max_symbol_position_pct,
            "max_total_position_pct": self.max_total_position_pct,
            "max_orders_per_batch": self.max_orders_per_batch,
            "quote_max_age_seconds": self.quote_max_age_seconds,
            "kill_switch_required": self.kill_switch_required,
        }


@dataclass(frozen=True)
class OrderIntent:
    workflow_run_id: str
    strategy_id: str
    strategy_version: str
    signal_date: str
    buy_date: str
    account_id: str
    symbol: str
    side: str
    rebalance_version: str
    target_pct: float = 0.0
    quantity: int | None = None
    limit_price: float | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OrderIntent":
        side = str(payload.get("side") or "").strip().upper()
        if side not in VALID_ORDER_SIDES:
            raise ValueError(f"unsupported order side: {side}")
        return cls(
            workflow_run_id=str(payload.get("workflow_run_id") or "").strip(),
            strategy_id=str(payload.get("strategy_id") or "").strip(),
            strategy_version=str(payload.get("strategy_version") or "").strip(),
            signal_date=str(payload.get("signal_date") or "").strip(),
            buy_date=str(payload.get("buy_date") or "").strip(),
            account_id=str(payload.get("account_id") or "").strip(),
            symbol=str(payload.get("symbol") or "").strip().upper(),
            side=side,
            rebalance_version=str(payload.get("rebalance_version") or "").strip(),
            target_pct=float(payload.get("target_pct") or 0.0),
            quantity=_int_or_none(payload.get("quantity")),
            limit_price=_float_or_none(payload.get("limit_price")),
            reason=str(payload.get("reason") or "").strip(),
            metadata=dict(payload.get("metadata") or {}),
        )

    @property
    def idempotency_key(self) -> str:
        raw = "|".join(
            [
                self.workflow_run_id,
                self.strategy_id,
                self.strategy_version,
                self.signal_date,
                self.account_id,
                self.symbol,
                self.side,
                self.rebalance_version,
            ]
        )
        return sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_run_id": self.workflow_run_id,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "signal_date": self.signal_date,
            "buy_date": self.buy_date,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "side": self.side,
            "rebalance_version": self.rebalance_version,
            "target_pct": self.target_pct,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "reason": self.reason,
            "metadata": self.metadata,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class AccountPosition:
    symbol: str
    shares: int
    sellable_shares: int
    market_value: float

    @classmethod
    def from_holdings_row(cls, payload: dict[str, Any]) -> "AccountPosition":
        return cls(
            symbol=str(payload.get("stock_code") or "").strip().upper(),
            shares=int(payload.get("volume") or 0),
            sellable_shares=int(payload.get("can_use_volume") or 0),
            market_value=float(payload.get("market_value") or 0.0),
        )


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    cash: float
    total_asset: float
    positions: dict[str, AccountPosition]
    captured_at: str = field(default_factory=_utc_iso_now)

    @classmethod
    def from_holdings_summary(cls, payload: dict[str, Any]) -> "AccountSnapshot":
        asset = payload.get("asset") or {}
        positions = {
            position.symbol: position
            for position in (
                AccountPosition.from_holdings_row(item)
                for item in (payload.get("positions") or [])
            )
            if position.symbol
        }
        return cls(
            account_id=str(payload.get("account_id") or "").strip(),
            cash=float(asset.get("cash") or 0.0),
            total_asset=float(asset.get("total_asset") or 0.0),
            positions=positions,
        )

    def position_pct(self, symbol: str) -> float:
        if self.total_asset <= 0:
            return 0.0
        position = self.positions.get(symbol.strip().upper())
        if not position:
            return 0.0
        return max(position.market_value, 0.0) / self.total_asset


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    quote_ts: str
    last_price: float
    bid_price_1: float | None = None
    ask_price_1: float | None = None
    pre_close: float | None = None
    open_price: float | None = None
    stock_status: str = ""
    is_st: bool = False
    limit_up_price: float | None = None
    limit_down_price: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarketSnapshot":
        bid_prices = payload.get("bidPrice") or payload.get("bid_price") or []
        ask_prices = payload.get("askPrice") or payload.get("ask_price") or []
        return cls(
            symbol=str(
                payload.get("stock_code")
                or payload.get("symbol")
                or payload.get("code")
                or ""
            ).strip().upper(),
            quote_ts=str(
                payload.get("quote_ts")
                or payload.get("quoteTime")
                or payload.get("trade_time")
                or payload.get("tradeTime")
                or _utc_iso_now()
            ).strip(),
            last_price=float(
                payload.get("last_price")
                or payload.get("lastPrice")
                or payload.get("price")
                or 0.0
            ),
            bid_price_1=_float_or_none(bid_prices[0] if bid_prices else payload.get("bidPrice1")),
            ask_price_1=_float_or_none(ask_prices[0] if ask_prices else payload.get("askPrice1")),
            pre_close=_float_or_none(
                payload.get("pre_close")
                or payload.get("lastClose")
                or payload.get("prev_close")
            ),
            open_price=_float_or_none(payload.get("open") or payload.get("openPrice")),
            stock_status=str(payload.get("stockStatus") or payload.get("stock_status") or "").strip(),
            is_st=bool(payload.get("is_st", False)),
            limit_up_price=_float_or_none(payload.get("up_limit") or payload.get("limit_up_price")),
            limit_down_price=_float_or_none(payload.get("down_limit") or payload.get("limit_down_price")),
        )


@dataclass(frozen=True)
class RiskDecision:
    intent: OrderIntent
    status: str
    approved: bool
    blockers: tuple[str, ...]
    evaluated_at: str = field(default_factory=_utc_iso_now)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "status": self.status,
            "approved": self.approved,
            "blockers": list(self.blockers),
            "evaluated_at": self.evaluated_at,
            "notes": list(self.notes),
        }

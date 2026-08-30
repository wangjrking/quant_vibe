from __future__ import annotations

from datetime import datetime, timezone

from .contracts import AccountSnapshot, ExecutionAuthorization, MarketSnapshot, OrderIntent, RiskDecision
from .kill_switch import KillSwitchState


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _estimate_notional(intent: OrderIntent, market: MarketSnapshot) -> float:
    unit_price = intent.limit_price or market.ask_price_1 or market.last_price
    if intent.quantity:
        return max(intent.quantity, 0) * max(unit_price, 0.0)
    return max(intent.target_pct, 0.0) * 0.0


def evaluate_order_intent(
    *,
    intent: OrderIntent,
    authorization: ExecutionAuthorization,
    account: AccountSnapshot,
    market: MarketSnapshot,
    kill_switch_state: KillSwitchState | None = None,
    now: datetime | None = None,
) -> RiskDecision:
    blockers: list[str] = []
    notes: list[str] = []
    now_value = now or datetime.now(timezone.utc)

    if intent.account_id != authorization.account_id:
        blockers.append("account_id_mismatch")
    if intent.strategy_id != authorization.strategy_id:
        blockers.append("strategy_id_mismatch")
    if intent.strategy_version != authorization.strategy_version:
        blockers.append("strategy_version_mismatch")
    if intent.side not in authorization.allowed_sides:
        blockers.append("side_not_authorized")
    if authorization.allowed_symbols and intent.symbol not in authorization.allowed_symbols:
        blockers.append("symbol_not_authorized")
    if authorization.kill_switch_required and kill_switch_state is None:
        blockers.append("kill_switch_state_missing")
    if kill_switch_state is not None:
        blockers.extend(kill_switch_state.blockers_for(intent.account_id, intent.strategy_id, intent.symbol))

    if not authorization.enabled:
        blockers.append("authorization_disabled")

    expiry = _parse_ts(authorization.expires_at)
    if expiry and now_value > expiry:
        blockers.append("authorization_expired")

    if market.symbol != intent.symbol:
        blockers.append("market_symbol_mismatch")
    if market.last_price <= 0:
        blockers.append("market_last_price_missing")
    if market.is_st:
        blockers.append("market_st_restricted")
    if market.stock_status and market.stock_status not in {"NORMAL", "0"}:
        blockers.append(f"market_status_blocked:{market.stock_status}")

    quote_ts = _parse_ts(market.quote_ts)
    if quote_ts is None:
        blockers.append("quote_timestamp_invalid")
    elif authorization.quote_max_age_seconds >= 0:
        age_seconds = (now_value - quote_ts).total_seconds()
        if age_seconds > authorization.quote_max_age_seconds:
            blockers.append("quote_stale")

    position_pct = account.position_pct(intent.symbol)
    if intent.side == "BUY":
        if market.limit_up_price is not None and market.last_price >= market.limit_up_price:
            blockers.append("buy_limit_up")
        est_notional = _estimate_notional(intent, market)
        if intent.quantity is None and intent.target_pct <= 0:
            blockers.append("buy_missing_quantity_or_target_pct")
        if intent.target_pct > 0:
            if position_pct + intent.target_pct > authorization.max_symbol_position_pct + 1e-9:
                blockers.append("symbol_position_limit_exceeded")
            current_total_pct = sum(
                max(item.market_value, 0.0) for item in account.positions.values()
            ) / account.total_asset if account.total_asset > 0 else 0.0
            if current_total_pct + intent.target_pct > authorization.max_total_position_pct + 1e-9:
                blockers.append("total_position_limit_exceeded")
            est_notional = max(est_notional, intent.target_pct * max(account.total_asset, 0.0))
        if est_notional > authorization.max_order_notional + 1e-9:
            blockers.append("order_notional_limit_exceeded")
        if est_notional > account.cash + 1e-9:
            blockers.append("insufficient_cash")
    else:
        position = account.positions.get(intent.symbol)
        if position is None or position.shares <= 0:
            blockers.append("sell_position_missing")
        elif intent.quantity and intent.quantity > position.sellable_shares:
            blockers.append("sellable_shares_insufficient")
        elif not intent.quantity and position.sellable_shares <= 0:
            blockers.append("sellable_shares_zero")
        if market.limit_down_price is not None and market.last_price <= market.limit_down_price:
            notes.append("sell_limit_down_watch")

    if authorization.stage == "shadow" and not blockers:
        return RiskDecision(
            intent=intent,
            status="shadow_only",
            approved=False,
            blockers=(),
            notes=("shadow_stage_no_submission",),
        )

    approved = not blockers
    return RiskDecision(
        intent=intent,
        status="approved" if approved else "risk_rejected",
        approved=approved,
        blockers=tuple(blockers),
        notes=tuple(notes),
    )

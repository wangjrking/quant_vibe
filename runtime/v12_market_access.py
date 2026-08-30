from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd


Purpose = Literal["execution", "trading", "valuation"]


@dataclass(frozen=True)
class AccessRecord:
    stock_code: str
    date: str
    purpose: Purpose
    daily_hit: bool
    exact_fact_hit: bool
    resolution: str
    unresolved_reason: str | None


class MarketAccessResolver:
    """Single exact-key gateway for all V12 market-data access."""

    def __init__(
        self,
        daily_rows: pd.DataFrame,
        execution_unavailable: frozenset[tuple[str, str]],
        valuation_marks: dict[tuple[str, str], float],
        *,
        mode: Literal["trace", "metrics"] = "trace",
        traced_keys: frozenset[tuple[str, str, str]] = frozenset(),
        allow_execution_trace_for_trading: bool = False,
        allow_audited_valuation_domain: bool = False,
    ) -> None:
        self._daily = daily_rows.set_index(
            [daily_rows["trade_date"].astype(str), daily_rows["stock_code"].astype(str)],
            drop=False,
        )
        self._execution_unavailable = execution_unavailable
        self._valuation_marks = valuation_marks
        self._mode = mode
        self._traced_keys = traced_keys
        self._allow_execution_trace_for_trading = allow_execution_trace_for_trading
        self._allow_audited_valuation_domain = allow_audited_valuation_domain
        self.records: dict[tuple[str, str, Purpose], AccessRecord] = {}

    def _record(self, record: AccessRecord) -> None:
        self.records[(record.stock_code, record.date, record.purpose)] = record

    def _guard(self, code: str, date: str, purpose: Purpose) -> None:
        key = (str(code), str(date), purpose)
        if self._mode != "metrics":
            return
        if key in self._traced_keys:
            return
        daily_key = (str(date), str(code))
        exact_key = (str(code), str(date))
        # Authoritative exact-key data is the final completeness contract. The
        # precomputed trace remains useful diagnostics, but it must not reject
        # a real canonical row or an audited exact non-trading/valuation fact.
        if daily_key in self._daily.index:
            return
        if purpose in {"execution", "trading"} and exact_key in self._execution_unavailable:
            return
        if purpose == "valuation" and exact_key in self._valuation_marks:
            return
        raise RuntimeError(f"metrics_access_not_resolvable:{purpose}:{date}:{code}")

    def _daily_row(self, code: str, date: str) -> pd.Series | None:
        key = (str(date), str(code))
        if key not in self._daily.index:
            return None
        rows = self._daily.loc[[key]]
        if len(rows) != 1:
            raise RuntimeError(f"duplicate_daily_row:{date}:{code}")
        return rows.iloc[0]

    def execution(self, code: str, date: str) -> pd.Series | None:
        self._guard(code, date, "execution")
        row = self._daily_row(code, date)
        fact_hit = (str(code), str(date)) in self._execution_unavailable
        if row is not None:
            self._record(AccessRecord(str(code), str(date), "execution", True, fact_hit, "daily_row", None))
            return row
        if fact_hit:
            self._record(AccessRecord(str(code), str(date), "execution", False, True, "legitimate_non_trading", None))
            return None
        self._record(AccessRecord(str(code), str(date), "execution", False, False, "unresolved", "missing_daily_and_execution_fact"))
        return None

    def trading(self, code: str, date: str) -> pd.Series | None:
        self._guard(code, date, "trading")
        row = self._daily_row(code, date)
        fact_hit = (str(code), str(date)) in self._execution_unavailable
        if row is not None:
            self._record(AccessRecord(str(code), str(date), "trading", True, fact_hit, "daily_row", None))
            return row
        if fact_hit:
            self._record(AccessRecord(str(code), str(date), "trading", False, True, "legitimate_non_trading", None))
            return None
        self._record(AccessRecord(str(code), str(date), "trading", False, False, "unresolved", "missing_daily_trading_row"))
        return None

    def valuation(self, code: str, date: str, shares: int) -> tuple[pd.Series | None, float | None]:
        self._guard(code, date, "valuation")
        row = self._daily_row(code, date)
        key = (str(code), str(date))
        fact_hit = key in self._valuation_marks
        if row is not None:
            self._record(AccessRecord(str(code), str(date), "valuation", True, fact_hit, "daily_row", None))
            return row, None
        if fact_hit:
            self._record(AccessRecord(str(code), str(date), "valuation", False, True, "exact_valuation_mark", None))
            return None, int(shares) * float(self._valuation_marks[key])
        self._record(AccessRecord(str(code), str(date), "valuation", False, False, "unresolved", "missing_daily_and_valuation_fact"))
        return None, None

    def trace_rows(self) -> list[dict[str, Any]]:
        return [asdict(self.records[key]) for key in sorted(self.records, key=lambda value: (value[1], value[0], value[2]))]

    def unresolved_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.trace_rows() if row["resolution"] == "unresolved"]

    def traced_key_set(self) -> frozenset[tuple[str, str, str]]:
        return frozenset((row["stock_code"], row["date"], row["purpose"]) for row in self.trace_rows())

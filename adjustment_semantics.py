from __future__ import annotations

from typing import Any


QFQ_MARKER = "qfq"
QFQ_SUFFIX = "_qfq"
RAW_MARKET_PRICE_SUFFIX = "_raw"
MARKET_FIELD_SCHEMA_KIND_BASE = "market_base_table"
MARKET_FIELD_SCHEMA_KIND_STRATEGY = "strategy_output"
NAKED_MARKET_PRICE_COLUMNS = ("open", "high", "low", "close", "pre_close")
FRONT_ADJUSTED_MARKET_PRICE_COLUMNS = tuple(f"{column}{QFQ_SUFFIX}" for column in NAKED_MARKET_PRICE_COLUMNS)
RAW_MARKET_PRICE_COLUMNS = tuple(f"{column}{RAW_MARKET_PRICE_SUFFIX}" for column in NAKED_MARKET_PRICE_COLUMNS)
NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS = (
    "atr",
    "macd",
    "kdj",
    "wr",
    "boll_upper",
    "boll_mid",
    "boll_lower",
    "mfi",
)
NAKED_RAW_DERIVED_MARKET_COLUMNS = (
    "pct_chg",
    "prev_pct_chg",
    "two_day_ret",
)


def default_adjustment_semantics() -> dict[str, Any]:
    return {
        "explicit_front_adjusted_marker_required": True,
        "front_adjusted_marker": QFQ_MARKER,
        "front_adjusted_column_suffix": QFQ_SUFFIX,
        "front_adjusted_indicator_suffix": QFQ_SUFFIX,
        "front_adjusted_factor_marker_required": True,
        "forbidden_implicit_front_adjusted_columns": list(NAKED_MARKET_PRICE_COLUMNS),
        "contract_rule": (
            "front-adjusted values must be explicitly marked with qfq in field or factor names; "
            "naked open/high/low/close/pre_close columns must not be interpreted as front-adjusted "
            "by downstream consumers without a separate explicit schema contract"
        ),
    }


def validate_adjustment_semantics(
    semantics: dict[str, Any] | None,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(semantics, dict):
        raise ValueError(f"{context} missing adjustment_semantics")

    required_true = (
        "explicit_front_adjusted_marker_required",
        "front_adjusted_factor_marker_required",
    )
    for key in required_true:
        if semantics.get(key) is not True:
            raise ValueError(f"{context} adjustment_semantics.{key} must be true")

    if str(semantics.get("front_adjusted_marker") or "") != QFQ_MARKER:
        raise ValueError(f"{context} adjustment_semantics.front_adjusted_marker must be {QFQ_MARKER}")
    if str(semantics.get("front_adjusted_column_suffix") or "") != QFQ_SUFFIX:
        raise ValueError(f"{context} adjustment_semantics.front_adjusted_column_suffix must be {QFQ_SUFFIX}")
    if str(semantics.get("front_adjusted_indicator_suffix") or "") != QFQ_SUFFIX:
        raise ValueError(f"{context} adjustment_semantics.front_adjusted_indicator_suffix must be {QFQ_SUFFIX}")

    forbidden_columns = semantics.get("forbidden_implicit_front_adjusted_columns")
    if not isinstance(forbidden_columns, list) or sorted(str(item) for item in forbidden_columns) != sorted(
        NAKED_MARKET_PRICE_COLUMNS
    ):
        raise ValueError(
            f"{context} adjustment_semantics.forbidden_implicit_front_adjusted_columns must equal "
            f"{list(NAKED_MARKET_PRICE_COLUMNS)}"
        )

    contract_rule = str(semantics.get("contract_rule") or "").strip()
    if not contract_rule:
        raise ValueError(f"{context} adjustment_semantics.contract_rule must be non-empty")

    return {
        "explicit_front_adjusted_marker_required": True,
        "front_adjusted_marker": QFQ_MARKER,
        "front_adjusted_column_suffix": QFQ_SUFFIX,
        "front_adjusted_indicator_suffix": QFQ_SUFFIX,
        "front_adjusted_factor_marker_required": True,
        "forbidden_implicit_front_adjusted_columns": list(NAKED_MARKET_PRICE_COLUMNS),
        "contract_rule": contract_rule,
    }


def same_adjustment_semantics(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return validate_adjustment_semantics(left, context="left") == validate_adjustment_semantics(
        right,
        context="right",
    )


def default_market_field_semantics() -> dict[str, Any]:
    return {
        "schema_kind": MARKET_FIELD_SCHEMA_KIND_STRATEGY,
        "raw_price_output_suffix": RAW_MARKET_PRICE_SUFFIX,
        "raw_market_price_columns": list(NAKED_MARKET_PRICE_COLUMNS),
        "front_adjusted_indicator_suffix": QFQ_SUFFIX,
        "contract_rule": (
            "raw market price fields must remain explicitly marked as *_raw in downstream strategy assets; "
            "front-adjusted indicators and factors must remain explicitly marked as *_qfq"
        ),
    }


def default_base_market_field_semantics() -> dict[str, Any]:
    return {
        "schema_kind": MARKET_FIELD_SCHEMA_KIND_BASE,
        "raw_price_column_style": "bare_raw_columns",
        "raw_market_price_columns": list(NAKED_MARKET_PRICE_COLUMNS),
        "front_adjusted_indicator_suffix": QFQ_SUFFIX,
        "contract_rule": (
            "naked open/high/low/close/pre_close columns in the active market base table are raw market prices only; "
            "front-adjusted indicators and factors must remain explicitly marked as *_qfq"
        ),
    }


def validate_market_field_semantics(
    semantics: dict[str, Any] | None,
    *,
    context: str,
    expected_schema_kind: str | None = None,
) -> dict[str, Any]:
    if not isinstance(semantics, dict):
        raise ValueError(f"{context} missing market_field_semantics")

    raw_columns = semantics.get("raw_market_price_columns")
    if not isinstance(raw_columns, list) or sorted(str(item) for item in raw_columns) != sorted(
        NAKED_MARKET_PRICE_COLUMNS
    ):
        raise ValueError(
            f"{context} market_field_semantics.raw_market_price_columns must equal "
            f"{list(NAKED_MARKET_PRICE_COLUMNS)}"
        )

    schema_kind = str(semantics.get("schema_kind") or "").strip()
    if not schema_kind:
        if "raw_price_output_suffix" in semantics:
            schema_kind = MARKET_FIELD_SCHEMA_KIND_STRATEGY
        elif "raw_price_column_style" in semantics:
            schema_kind = MARKET_FIELD_SCHEMA_KIND_BASE
        else:
            raise ValueError(
                f"{context} market_field_semantics.schema_kind must be "
                f"{MARKET_FIELD_SCHEMA_KIND_BASE} or {MARKET_FIELD_SCHEMA_KIND_STRATEGY}"
            )

    if schema_kind == MARKET_FIELD_SCHEMA_KIND_STRATEGY:
        if str(semantics.get("raw_price_output_suffix") or "") != RAW_MARKET_PRICE_SUFFIX:
            raise ValueError(
                f"{context} market_field_semantics.raw_price_output_suffix must be {RAW_MARKET_PRICE_SUFFIX}"
            )
        normalized = {
            "schema_kind": MARKET_FIELD_SCHEMA_KIND_STRATEGY,
            "raw_price_output_suffix": RAW_MARKET_PRICE_SUFFIX,
            "raw_market_price_columns": list(NAKED_MARKET_PRICE_COLUMNS),
            "front_adjusted_indicator_suffix": QFQ_SUFFIX,
        }
    elif schema_kind == MARKET_FIELD_SCHEMA_KIND_BASE:
        if str(semantics.get("raw_price_column_style") or "") != "bare_raw_columns":
            raise ValueError(
                f"{context} market_field_semantics.raw_price_column_style must be bare_raw_columns"
            )
        normalized = {
            "schema_kind": MARKET_FIELD_SCHEMA_KIND_BASE,
            "raw_price_column_style": "bare_raw_columns",
            "raw_market_price_columns": list(NAKED_MARKET_PRICE_COLUMNS),
            "front_adjusted_indicator_suffix": QFQ_SUFFIX,
        }
    else:
        raise ValueError(
            f"{context} market_field_semantics.schema_kind must be "
            f"{MARKET_FIELD_SCHEMA_KIND_BASE} or {MARKET_FIELD_SCHEMA_KIND_STRATEGY}"
        )

    if expected_schema_kind and schema_kind != expected_schema_kind:
        raise ValueError(
            f"{context} market_field_semantics.schema_kind must be {expected_schema_kind}"
        )

    if str(semantics.get("front_adjusted_indicator_suffix") or "") != QFQ_SUFFIX:
        raise ValueError(
            f"{context} market_field_semantics.front_adjusted_indicator_suffix must be {QFQ_SUFFIX}"
        )
    contract_rule = str(semantics.get("contract_rule") or "").strip()
    if not contract_rule:
        raise ValueError(f"{context} market_field_semantics.contract_rule must be non-empty")
    normalized["contract_rule"] = contract_rule
    return normalized


def same_market_field_semantics(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    expected_schema_kind: str | None = None,
) -> bool:
    return validate_market_field_semantics(
        left,
        context="left",
        expected_schema_kind=expected_schema_kind,
    ) == validate_market_field_semantics(
        right,
        context="right",
        expected_schema_kind=expected_schema_kind,
    )


def explicit_qfq_column_name(column: str) -> str:
    return f"{column}{QFQ_SUFFIX}"


def require_front_adjusted_column(
    columns: list[str] | tuple[str, ...] | set[str],
    base_name: str,
    *,
    context: str,
) -> str:
    qfq_name = explicit_qfq_column_name(base_name)
    available = {str(column) for column in columns}
    if qfq_name not in available:
        raise KeyError(f"{context} requires explicit front-adjusted column: {qfq_name}")
    return qfq_name


def prefer_explicit_qfq_columns(columns: list[str] | tuple[str, ...]) -> list[str]:
    available = {str(column) for column in columns}
    selected: list[str] = []
    for column in columns:
        name = str(column)
        if not name.endswith(QFQ_SUFFIX):
            qfq_name = explicit_qfq_column_name(name)
            if qfq_name in available:
                continue
        if name not in selected:
            selected.append(name)
    return selected


def validate_strategy_output_field_names(
    field_names: list[str] | tuple[str, ...] | set[str],
    *,
    context: str,
) -> list[str]:
    available = [str(name) for name in field_names]
    forbidden = [name for name in NAKED_MARKET_PRICE_COLUMNS if name in available]
    if forbidden:
        raise ValueError(
            f"{context} contains naked market price fields that are ambiguous for downstream strategy use: "
            f"{forbidden}. Use {list(RAW_MARKET_PRICE_COLUMNS)} for raw prices and *_qfq for front-adjusted fields."
        )
    forbidden_indicator_fields = [
        name for name in NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS if name in available
    ]
    if forbidden_indicator_fields:
        raise ValueError(
            f"{context} contains naked front-adjusted indicator or factor fields that are ambiguous for downstream "
            f"strategy use: {forbidden_indicator_fields}. Use explicit *_qfq names for front-adjusted indicators "
            f"and factors."
        )
    forbidden_raw_derived_fields = [
        name for name in NAKED_RAW_DERIVED_MARKET_COLUMNS if name in available
    ]
    if forbidden_raw_derived_fields:
        raise ValueError(
            f"{context} contains naked raw-derived market fields that are ambiguous for downstream strategy use: "
            f"{forbidden_raw_derived_fields}. Use explicit *_raw names for raw-return and raw-change fields."
        )
    return available


def require_explicit_qfq_market_columns(
    columns: list[str] | tuple[str, ...],
    *,
    context: str,
) -> None:
    available = {str(column) for column in columns}
    missing = [
        explicit_qfq_column_name(column)
        for column in NAKED_MARKET_PRICE_COLUMNS
        if column in available and explicit_qfq_column_name(column) not in available
    ]
    if missing:
        raise ValueError(
            f"{context} missing explicit qfq columns for front-adjusted market prices: {missing}"
        )


def preferred_front_adjusted_column(columns: list[str] | tuple[str, ...] | set[str], base_name: str) -> str | None:
    qfq_name = explicit_qfq_column_name(base_name)
    available = {str(column) for column in columns}
    if qfq_name in available:
        return qfq_name
    if base_name in available:
        return base_name
    return None

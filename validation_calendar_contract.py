from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any


class CalendarContractError(ValueError):
    """Raised when a validation calendar or key domain is ambiguous."""


def normalize_trade_date(value: Any) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, bool):
        raise CalendarContractError("boolean is not a trade date")
    if isinstance(value, int):
        value = str(value)
    elif isinstance(value, float):
        if not value.is_integer():
            raise CalendarContractError("fractional trade date is invalid")
        value = str(int(value))
    elif isinstance(value, str):
        value = value.strip().replace("-", "")
    else:
        raise CalendarContractError(
            f"unsupported trade date type: {type(value).__name__}"
        )
    if len(value) != 8 or not value.isdigit():
        raise CalendarContractError(f"invalid YYYYMMDD trade date: {value!r}")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise CalendarContractError(f"invalid calendar date: {value!r}") from exc
    return value


def canonicalize_calendar(
    dates: Iterable[Any],
    *,
    start: Any,
    end: Any,
    source_name: str,
) -> list[str]:
    start_date = normalize_trade_date(start)
    end_date = normalize_trade_date(end)
    if start_date > end_date:
        raise CalendarContractError("calendar start is after end")

    normalized = [normalize_trade_date(value) for value in dates]
    duplicates = sorted(
        value for value, count in Counter(normalized).items() if count > 1
    )
    if duplicates:
        raise CalendarContractError(
            f"{source_name} contains duplicate dates: {duplicates[:10]}"
        )
    out_of_window = sorted(
        value for value in normalized if value < start_date or value > end_date
    )
    if out_of_window:
        raise CalendarContractError(
            f"{source_name} contains out-of-window dates: {out_of_window[:10]}"
        )
    if not normalized:
        raise CalendarContractError(f"{source_name} calendar is empty")
    return sorted(normalized)


def audit_asset_calendars(
    authoritative_dates: Iterable[Any],
    asset_dates: Mapping[str, Iterable[Any]],
    *,
    start: Any,
    end: Any,
    minimum_dates: int = 1,
) -> dict[str, Any]:
    calendar = canonicalize_calendar(
        authoritative_dates,
        start=start,
        end=end,
        source_name="authoritative_calendar",
    )
    if len(calendar) < minimum_dates:
        raise CalendarContractError(
            f"authoritative calendar has {len(calendar)} dates; "
            f"minimum is {minimum_dates}"
        )

    expected = set(calendar)
    assets: dict[str, dict[str, Any]] = {}
    matched_total = 0
    for name, values in asset_dates.items():
        actual_list = canonicalize_calendar(
            values,
            start=start,
            end=end,
            source_name=name,
        )
        actual = set(actual_list)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        matched = len(expected & actual)
        matched_total += matched
        assets[name] = {
            "date_count": len(actual_list),
            "matched_date_count": matched,
            "missing_dates": missing,
            "extra_dates": extra,
            "exact_match": not missing and not extra,
        }

    denominator = len(calendar) * len(assets)
    closure_ratio = 1.0 if denominator == 0 else matched_total / denominator
    ready = bool(assets) and all(item["exact_match"] for item in assets.values())
    return {
        "status": "ready" if ready else "blocked",
        "authoritative_calendar": {
            "start": calendar[0],
            "end": calendar[-1],
            "date_count": len(calendar),
            "dates": calendar,
        },
        "assets": assets,
        "closure_ratio": closure_ratio,
        "ready": ready,
        "calendar_authority_count": 1,
        "intersection_used_as_calendar": False,
    }


def compare_sorted_key_rows(
    reference_rows: Iterable[tuple[Any, ...]],
    candidate_rows: Iterable[tuple[Any, ...]],
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    reference = iter(reference_rows)
    candidate = iter(candidate_rows)
    ref_value = next(reference, None)
    candidate_value = next(candidate, None)
    missing_count = 0
    extra_count = 0
    duplicate_reference_count = 0
    duplicate_candidate_count = 0
    missing_sample: list[list[Any]] = []
    extra_sample: list[list[Any]] = []
    previous_reference = None
    previous_candidate = None

    while ref_value is not None or candidate_value is not None:
        if (
            ref_value is not None
            and previous_reference is not None
            and ref_value < previous_reference
        ):
            raise CalendarContractError("reference key rows are not sorted")
        if (
            candidate_value is not None
            and previous_candidate is not None
            and candidate_value < previous_candidate
        ):
            raise CalendarContractError("candidate key rows are not sorted")
        if ref_value is not None and ref_value == previous_reference:
            duplicate_reference_count += 1
            ref_value = next(reference, None)
            continue
        if candidate_value is not None and candidate_value == previous_candidate:
            duplicate_candidate_count += 1
            candidate_value = next(candidate, None)
            continue
        if candidate_value is None or (
            ref_value is not None and ref_value < candidate_value
        ):
            missing_count += 1
            if len(missing_sample) < sample_limit:
                missing_sample.append(list(ref_value))
            previous_reference = ref_value
            ref_value = next(reference, None)
            continue
        if ref_value is None or candidate_value < ref_value:
            extra_count += 1
            if len(extra_sample) < sample_limit:
                extra_sample.append(list(candidate_value))
            previous_candidate = candidate_value
            candidate_value = next(candidate, None)
            continue
        previous_reference = ref_value
        previous_candidate = candidate_value
        ref_value = next(reference, None)
        candidate_value = next(candidate, None)

    exact_match = not any(
        (
            missing_count,
            extra_count,
            duplicate_reference_count,
            duplicate_candidate_count,
        )
    )
    return {
        "missing_key_count": missing_count,
        "extra_key_count": extra_count,
        "duplicate_reference_key_count": duplicate_reference_count,
        "duplicate_candidate_key_count": duplicate_candidate_count,
        "missing_key_sample": missing_sample,
        "extra_key_sample": extra_sample,
        "exact_match": exact_match,
    }


def audit_sorted_key_rows(
    rows: Iterable[tuple[Any, ...]], *, sample_limit: int = 20
) -> dict[str, Any]:
    previous = None
    row_count = 0
    unique_key_count = 0
    duplicate_key_count = 0
    duplicate_key_sample: list[list[Any]] = []
    for value in rows:
        row_count += 1
        if previous is not None and value < previous:
            raise CalendarContractError("key rows are not sorted")
        if value == previous:
            duplicate_key_count += 1
            if len(duplicate_key_sample) < sample_limit:
                duplicate_key_sample.append(list(value))
        else:
            unique_key_count += 1
        previous = value
    return {
        "row_count": row_count,
        "unique_key_count": unique_key_count,
        "duplicate_key_count": duplicate_key_count,
        "duplicate_key_sample": duplicate_key_sample,
        "unique": duplicate_key_count == 0,
    }


def build_t1_map(authoritative_dates: Iterable[Any]) -> dict[str, str | None]:
    dates = [normalize_trade_date(value) for value in authoritative_dates]
    if dates != sorted(set(dates)):
        raise CalendarContractError(
            "T+1 map requires unique, strictly increasing authoritative dates"
        )
    return {
        value: dates[index + 1] if index + 1 < len(dates) else None
        for index, value in enumerate(dates)
    }

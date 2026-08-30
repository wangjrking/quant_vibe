from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path
from typing import Any

import duckdb

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from validation_calendar_contract import (  # noqa: E402
    CalendarContractError,
    audit_asset_calendars,
    audit_sorted_key_rows,
    compare_sorted_key_rows,
)


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def resolve_path(descriptor_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = descriptor_path.parent / path
    return path.resolve()


def normalized_date_sql(column: str) -> str:
    return (
        "replace(substr(cast(" + quote_ident(column) + " as varchar), 1, 10), '-', '')"
    )


def where_clause(spec: dict[str, Any], start: str, end: str) -> tuple[str, list[Any]]:
    date_column = normalized_date_sql(str(spec["date_column"]))
    clauses = [f"{date_column} BETWEEN ? AND ?"]
    params: list[Any] = [start, end]
    for column, value in sorted(spec.get("filter_equals", {}).items()):
        clauses.append(f"cast({quote_ident(str(column))} as varchar) = ?")
        params.append(str(value))
    code_column = spec.get("code_column")
    for suffix in spec.get("exclude_code_suffixes", []):
        if not code_column:
            raise CalendarContractError(
                "exclude_code_suffixes requires code_column"
            )
        clauses.append(f"not ends_with({quote_ident(str(code_column))}, ?)")
        params.append(str(suffix))
    return " AND ".join(clauses), params


def source_parts(spec: dict[str, Any]) -> list[dict[str, Any]]:
    declared = spec.get("parts")
    if not declared:
        return [spec]
    common = {key: value for key, value in spec.items() if key != "parts"}
    return [{**common, **part} for part in declared]


def read_dates_from_part(
    descriptor_path: Path,
    spec: dict[str, Any],
    start: str,
    end: str,
) -> list[Any]:
    path = resolve_path(descriptor_path, str(spec["path"]))
    if spec["source_type"] == "json_dates":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        values: Any = payload
        for key in str(spec.get("dates_field", "open_dates")).split("."):
            values = values[key]
        return list(values)
    if spec["source_type"] != "duckdb_table":
        raise CalendarContractError(
            f"unsupported source_type: {spec['source_type']}"
        )
    where, params = where_clause(spec, start, end)
    connection = duckdb.connect(str(path), read_only=True)
    try:
        rows = connection.execute(
            f"SELECT DISTINCT {quote_ident(str(spec['date_column']))} "
            f"FROM {quote_ident(str(spec['table']))} WHERE {where} "
            f"ORDER BY 1",
            params,
        ).fetchall()
    finally:
        connection.close()
    return [row[0] for row in rows]


def read_dates(
    descriptor_path: Path,
    spec: dict[str, Any],
    start: str,
    end: str,
) -> list[Any]:
    values: list[Any] = []
    for part in source_parts(spec):
        values.extend(read_dates_from_part(descriptor_path, part, start, end))
    return values


def iter_keys_from_part(
    descriptor_path: Path,
    spec: dict[str, Any],
    start: str,
    end: str,
):
    path = resolve_path(descriptor_path, str(spec["path"]))
    where, params = where_clause(spec, start, end)
    key_columns = [str(value) for value in spec.get("key_columns", [])]
    if not key_columns:
        raise CalendarContractError(f"{spec['name']} has no key_columns")
    select_expressions = [
        normalized_date_sql(value)
        if value == str(spec["date_column"])
        else f"cast({quote_ident(value)} as varchar)"
        for value in key_columns
    ]
    select = ", ".join(select_expressions)
    connection = duckdb.connect(str(path), read_only=True)
    try:
        cursor = connection.execute(
            f"SELECT {select} FROM {quote_ident(str(spec['table']))} "
            f"WHERE {where} ORDER BY {select}",
            params,
        )
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            for row in rows:
                yield tuple(str(value) for value in row)
    finally:
        connection.close()


def iter_keys(
    descriptor_path: Path,
    spec: dict[str, Any],
    start: str,
    end: str,
):
    streams = [
        iter_keys_from_part(descriptor_path, part, start, end)
        for part in source_parts(spec)
    ]
    yield from heapq.merge(*streams)


def run(descriptor_path: Path) -> dict[str, Any]:
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8-sig"))
    if descriptor.get("schema_version") != 1:
        raise CalendarContractError("schema_version must be 1")
    window = descriptor["window"]
    start = str(window["start"])
    end = str(window["end"])
    authoritative = read_dates(
        descriptor_path, descriptor["authoritative_calendar"], start, end
    )
    assets = descriptor.get("assets", [])
    asset_dates = {
        str(spec["name"]): read_dates(descriptor_path, spec, start, end)
        for spec in assets
    }
    report = audit_asset_calendars(
        authoritative,
        asset_dates,
        start=start,
        end=end,
        minimum_dates=int(window.get("minimum_dates", 1)),
    )

    key_groups: dict[str, list[dict[str, Any]]] = {}
    key_uniqueness: dict[str, Any] = {}
    for spec in assets:
        if spec.get("key_columns"):
            key_uniqueness[str(spec["name"])] = audit_sorted_key_rows(
                iter_keys(descriptor_path, spec, start, end)
            )
        group = spec.get("key_group")
        if group:
            key_groups.setdefault(str(group), []).append(spec)
    key_reports: dict[str, Any] = {}
    for group, members in sorted(key_groups.items()):
        if len(members) < 2:
            raise CalendarContractError(
                f"key_group {group!r} requires at least two assets"
            )
        reference = members[0]
        comparisons = {}
        for candidate in members[1:]:
            comparisons[str(candidate["name"])] = compare_sorted_key_rows(
                iter_keys(descriptor_path, reference, start, end),
                iter_keys(descriptor_path, candidate, start, end),
            )
        key_reports[group] = {
            "reference": str(reference["name"]),
            "comparisons": comparisons,
            "exact_match": all(
                item["exact_match"] for item in comparisons.values()
            ),
        }
    report["key_groups"] = key_reports
    report["key_uniqueness"] = key_uniqueness
    report["ready"] = report["ready"] and all(
        item["exact_match"] for item in key_reports.values()
    ) and all(
        item["unique"] for item in key_uniqueness.values()
    )
    report["status"] = "ready" if report["ready"] else "blocked"
    report["business_metrics_calculated"] = False
    report["constraints"] = [
        "development_and_validation_are_disjoint",
        "pit_inputs_only_and_no_data_guessing",
        "candidate_and_production_are_frozen",
        "production_assets_and_trading_are_untouched",
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one authoritative calendar against all validation assets."
    )
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.descriptor.resolve())
    raw = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(raw, encoding="utf-8")
    else:
        print(raw, end="")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

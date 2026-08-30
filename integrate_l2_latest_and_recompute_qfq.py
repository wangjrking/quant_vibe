from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import psutil

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from database_module import sql as legacy_stock_daily_sql
from l1_raw_data_route import resolve_l1_raw_duckdb_path
from project_paths import resolve_data_dir
from stock_daily_data_route import resolve_stock_daily_duckdb_path


RAW_TABLES = {
    "daily_data": "daily_data",
    "daily_index_data": "daily_index_data",
    "STOCK_BASIC_DATA": "stock_basic_data",
    "FINAN_DATA_SEASON": "finan_data_season",
    "FINAN_DATA_YEAR": "finan_data_year",
    "limit_list_data": "limit_list_data",
    "adj_factor": "adj_factor",
    "moneyflow": "moneyflow",
    "stk_factor": "stk_factor",
    "top_list": "top_list",
    "THS_HOT": "ths_hot",
    "CYQ_PERF": "cyq_perf",
    "stock_st": "stock_st",
    "index_daily": "index_daily",
}

QFQ_PRICE_COLUMNS = {
    "open_qfq": "open",
    "high_qfq": "high",
    "low_qfq": "low",
    "close_qfq": "close",
    "pre_close_qfq": "pre_close",
}

PRICE_QFQ_COLUMNS = set(QFQ_PRICE_COLUMNS)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_VENV = PROJECT_ROOT / ".venv"
PROJECT_VENV_PYTHON = PROJECT_VENV / "Scripts" / "python.exe"


def normalized_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(value)))


def process_identity(pid: int) -> dict[str, object]:
    try:
        process = psutil.Process(pid)
        lineage: list[dict[str, object]] = []
        parent = process.parent()
        for _ in range(12):
            if parent is None:
                break
            lineage.append(
                {
                    "pid": parent.pid,
                    "create_time": parent.create_time(),
                    "name": parent.name(),
                    "command_line": parent.cmdline(),
                }
            )
            parent = parent.parent()
        return {
            "pid": process.pid,
            "create_time": process.create_time(),
            "name": process.name(),
            "command_line": process.cmdline(),
            "parent_lineage": lineage,
            "alive": process.is_running(),
        }
    except (psutil.Error, OSError) as exc:
        return {
            "pid": pid,
            "alive": False,
            "lookup_error": f"{type(exc).__name__}: {exc}",
        }


def command_text(identity: dict[str, object]) -> str:
    parts: list[str] = []
    parts.extend(str(value) for value in identity.get("command_line", []) or [])
    for parent in identity.get("parent_lineage", []) or []:
        parts.extend(str(value) for value in parent.get("command_line", []) or [])
    return " ".join(parts).lower()


def classify_lock_process(
    identity: dict[str, object],
    *,
    current_pid: int | None = None,
    current_create_time: float | None = None,
) -> dict[str, object]:
    pid = int(identity.get("pid") or 0)
    create_time = identity.get("create_time")
    text = command_text(identity)
    project_venv_marker = normalized_path(PROJECT_VENV).lower()
    uses_project_venv = project_venv_marker in normalized_path(text).lower()
    if current_pid is None:
        current_pid = os.getpid()
    if current_create_time is None:
        current_create_time = psutil.Process(current_pid).create_time()

    same_identity = pid == current_pid and (
        create_time is None or abs(float(create_time) - float(current_create_time)) < 0.001
    )
    if same_identity:
        role = "current_entry_preflight"
    elif "capture_l2_metrics.py" in text and uses_project_venv:
        role = "project_venv_readonly_collector"
    elif any(
        marker in text
        for marker in (
            "integrate_l2_latest_and_recompute_qfq.py",
            "switch_l2_incremental_staging.py",
            "rebuild_l2_stock_daily_duckdb_mainline.py",
        )
    ) and uses_project_venv:
        role = "project_venv_business_writer"
    else:
        role = "unknown_writer"
    return {
        "role": role,
        "pid": pid,
        "create_time": create_time,
        "uses_project_venv_lineage": uses_project_venv,
        "fail_closed": role != "current_entry_preflight",
        "identity": identity,
    }


def require_project_venv_runtime() -> dict[str, object]:
    executable = normalized_path(sys.executable)
    expected = normalized_path(PROJECT_VENV_PYTHON)
    prefix = normalized_path(sys.prefix)
    expected_prefix = normalized_path(PROJECT_VENV)
    duckdb_module = normalized_path(Path(duckdb.__file__).resolve())
    duckdb_from_project_venv = duckdb_module.startswith(expected_prefix + os.sep)
    passed = (
        executable == expected
        and prefix == expected_prefix
        and duckdb_from_project_venv
    )
    result = {
        "passed": passed,
        "sys_executable": sys.executable,
        "expected_executable": str(PROJECT_VENV_PYTHON),
        "sys_prefix": sys.prefix,
        "sys_base_executable": getattr(sys, "_base_executable", None),
        "duckdb_version": duckdb.__version__,
        "duckdb_module": str(Path(duckdb.__file__).resolve()),
        "duckdb_from_project_venv": duckdb_from_project_venv,
        "pid": os.getpid(),
        "create_time": psutil.Process(os.getpid()).create_time(),
    }
    if not passed:
        raise RuntimeError(
            "project .venv runtime gate failed: "
            f"sys.executable={sys.executable}, sys.prefix={sys.prefix}, "
            f"duckdb={duckdb.__file__}"
        )
    return result


def iter_year_ranges(min_trade_date: str, max_trade_date: str) -> list[tuple[str, str]]:
    return [
        (max(f"{year}0101", min_trade_date), min(f"{year}1231", max_trade_date))
        for year in range(int(min_trade_date[:4]), int(max_trade_date[:4]) + 1)
    ]


def probe_duckdb_write_lock(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    started = time.perf_counter()
    try:
        with duckdb.connect(str(resolved)) as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as exc:
        error = str(exc)
        match = re.search(r"PID\s+(\d+)", error)
        lock_pid = int(match.group(1)) if match else None
        process_gate = None
        if lock_pid is not None:
            process_gate = classify_lock_process(process_identity(lock_pid))
        return {
            "passed": False,
            "path": str(resolved),
            "error": error,
            "lock_pid": lock_pid,
            "process_gate": process_gate,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    return {
        "passed": True,
        "path": str(resolved),
        "error": None,
        "lock_pid": None,
        "process_gate": None,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def require_duckdb_write_lock(path: str | Path) -> dict[str, object]:
    result = probe_duckdb_write_lock(path)
    if not result["passed"]:
        raise RuntimeError(
            "active L2 write-lock gate failed before processing; "
            f"release the reader/writer first: {result['error']}"
        )
    return result


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def attach_l1_sources(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> dict[str, str]:
    attached: dict[str, str] = {}
    for view_name, table_name in RAW_TABLES.items():
        path = resolve_l1_raw_duckdb_path(table_name, data_dir=data_dir, require_exists=True)
        alias = f"src_{table_name}"
        conn.execute(f"ATTACH {quote_literal(str(path))} AS {quote_ident(alias)} (READ_ONLY)")
        conn.execute(
            f"CREATE OR REPLACE TEMP VIEW {quote_ident(view_name)} AS "
            f"SELECT * FROM {quote_ident(alias)}.{quote_ident(table_name)}"
        )
        attached[view_name] = str(path)
    return attached


def build_target_date_select_sql(target_trade_date: str) -> str:
    marker = "CREATE TABLE STOCK_DAILY_DATA as"
    _, select_part = legacy_stock_daily_sql.split(marker, 1)
    select_sql = select_part.strip().rstrip(";")
    pattern = re.compile(r"FROM\s+daily_data\s+AS\s+T1", re.IGNORECASE)
    return pattern.sub(
        "FROM (SELECT * FROM daily_data "
        f"WHERE trade_date = {quote_literal(target_trade_date)} "
        "AND ts_code NOT LIKE '%.BJ') AS T1",
        select_sql,
        count=1,
    )


def split_select_columns(select_sql: str) -> tuple[list[str], str]:
    lower_sql = select_sql.lower()
    select_index = lower_sql.find("select")
    from_index = lower_sql.find("from")
    if select_index < 0 or from_index < 0 or from_index <= select_index:
        raise ValueError("unable to split target-date SELECT SQL")
    select_part = select_sql[select_index + len("select") : from_index]
    from_part = select_sql[from_index:]

    columns: list[str] = []
    buffer: list[str] = []
    depth = 0
    in_quote = False
    for char in select_part:
        if char == "'":
            in_quote = not in_quote
        elif not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                column = "".join(buffer).strip()
                if column:
                    columns.append(column)
                buffer = []
                continue
        buffer.append(char)
    column = "".join(buffer).strip()
    if column:
        columns.append(column)
    return columns, from_part


def create_materialized_target_aliases(
    conn: duckdb.DuckDBPyConnection,
    target_trade_date: str,
) -> str:
    temp_tables = [
        "__l2_t1",
        "__l2_t2",
        "__l2_t3",
        "__l2_t4",
        "__l2_t5",
        "__l2_t6",
        "__l2_t7",
        "__l2_t8",
        "__l2_t9",
        "__l2_t10",
        "__l2_t13",
        "__l2_t14",
        "__l2_t15",
        "__l2_t16",
        "__l2_t17",
        "__l2_t18",
    ]
    for table in temp_tables:
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(table)}")

    conn.execute(
        """
        CREATE TEMP TABLE __l2_t1 AS
        SELECT *
        FROM daily_data
        WHERE trade_date = ?
          AND ts_code NOT LIKE '%.BJ'
        """,
        [target_trade_date],
    )
    conn.execute(
        """
        CREATE TEMP TABLE __l2_t2 AS
        SELECT s.*
        FROM daily_index_data AS s
        JOIN __l2_t1 AS t
          ON s.ts_code = t.ts_code
         AND s.trade_date = t.trade_date
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE __l2_t3 AS
        SELECT s.*
        FROM STOCK_BASIC_DATA AS s
        JOIN (SELECT DISTINCT ts_code FROM __l2_t1) AS t
          ON s.ts_code = t.ts_code
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE __l2_t4 AS
        SELECT w.*
        FROM (
            SELECT
                *,
                LEAD(ann_date, 1, '30001231') OVER (
                    PARTITION BY ts_code ORDER BY ann_date
                ) AS next_date
            FROM FINAN_DATA_SEASON
        ) AS w
        JOIN __l2_t1 AS t
          ON w.ts_code = t.ts_code
         AND t.trade_date >= w.ann_date
         AND t.trade_date < w.next_date
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE __l2_t5 AS
        SELECT w.*
        FROM (
            SELECT
                *,
                LEAD(ann_date, 1, '30001231') OVER (
                    PARTITION BY ts_code ORDER BY ann_date
                ) AS next_date
            FROM FINAN_DATA_YEAR
        ) AS w
        JOIN __l2_t1 AS t
          ON w.ts_code = t.ts_code
         AND t.trade_date >= w.ann_date
         AND t.trade_date < w.next_date
        """
    )
    keyed_sources = {
        "__l2_t6": "limit_list_data",
        "__l2_t7": "adj_factor",
        "__l2_t9": "moneyflow",
        "__l2_t10": "stk_factor",
        "__l2_t13": "top_list",
        "__l2_t14": "THS_HOT",
        "__l2_t15": "THS_HOT",
        "__l2_t16": "CYQ_PERF",
        "__l2_t17": "stock_st",
    }
    for temp_table, source_table in keyed_sources.items():
        conn.execute(
            f"""
            CREATE TEMP TABLE {quote_ident(temp_table)} AS
            SELECT s.*
            FROM {quote_ident(source_table)} AS s
            JOIN __l2_t1 AS t
              ON s.ts_code = t.ts_code
             AND s.trade_date = t.trade_date
            """
        )
    conn.execute(
        """
        CREATE TEMP TABLE __l2_t8 AS
        SELECT ts_code, adj_factor
        FROM (
            SELECT
                ts_code,
                adj_factor,
                ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) AS rn
            FROM adj_factor
        )
        WHERE rn = 1
        """
    )
    conn.execute(
        """
        CREATE TEMP TABLE __l2_t18 AS
        SELECT s.*
        FROM index_daily AS s
        JOIN (SELECT DISTINCT trade_date FROM __l2_t1) AS t
          ON s.trade_date = t.trade_date
        WHERE s.ts_code = '932000.CSI'
        """
    )
    return """
        FROM __l2_t1 AS T1
        LEFT JOIN __l2_t2 AS T2
          ON T1.ts_code = T2.ts_code
         AND T1.trade_date = T2.trade_date
        LEFT JOIN __l2_t3 AS T3
          ON T1.ts_code = T3.ts_code
        LEFT JOIN __l2_t4 AS T4
          ON T1.ts_code = T4.ts_code
        LEFT JOIN __l2_t5 AS T5
          ON T1.ts_code = T5.ts_code
        LEFT JOIN __l2_t6 AS T6
          ON T1.ts_code = T6.ts_code
         AND T1.trade_date = T6.trade_date
        LEFT JOIN __l2_t7 AS T7
          ON T1.ts_code = T7.ts_code
         AND T1.trade_date = T7.trade_date
        LEFT JOIN __l2_t8 AS T8
          ON T1.ts_code = T8.ts_code
        LEFT JOIN __l2_t9 AS T9
          ON T1.ts_code = T9.ts_code
         AND T1.trade_date = T9.trade_date
        LEFT JOIN __l2_t10 AS T10
          ON T1.ts_code = T10.ts_code
         AND T1.trade_date = T10.trade_date
        LEFT JOIN __l2_t13 AS T13
          ON T1.ts_code = T13.ts_code
         AND T1.trade_date = T13.trade_date
        LEFT JOIN __l2_t14 AS T14
          ON T1.ts_code = T14.ts_code
         AND T1.trade_date = T14.trade_date
        LEFT JOIN __l2_t15 AS T15
          ON T1.ts_code = T15.ts_code
         AND T1.trade_date = T15.trade_date
        LEFT JOIN __l2_t16 AS T16
          ON T1.ts_code = T16.ts_code
         AND T1.trade_date = T16.trade_date
        LEFT JOIN __l2_t17 AS T17
          ON T1.ts_code = T17.ts_code
         AND T1.trade_date = T17.trade_date
        LEFT JOIN __l2_t18 AS T18
          ON T1.trade_date = T18.trade_date
    """


def create_target_day_table_chunked(
    conn: duckdb.DuckDBPyConnection,
    *,
    select_sql: str,
    target_columns: list[str],
    from_part_override: str | None = None,
    chunk_size: int = 10,
) -> None:
    select_columns, from_part = split_select_columns(select_sql)
    if from_part_override is not None:
        from_part = from_part_override
    if len(select_columns) != len(target_columns):
        raise RuntimeError(
            f"L2 target-date SELECT column mismatch: target={len(target_columns)} "
            f"select={len(select_columns)}"
        )

    conn.execute("DROP TABLE IF EXISTS __l2_target_day_new")
    key_cursor = conn.execute(f"SELECT {select_columns[0]}, {select_columns[1]} {from_part}")
    target_df = pd.DataFrame(key_cursor.fetchall(), columns=[column[0] for column in key_cursor.description])
    target_df = target_df.set_index(["stock_code", "trade_date"], drop=False)

    data_column_indexes = list(range(2, len(target_columns)))
    for offset in range(0, len(data_column_indexes), chunk_size):
        indexes = data_column_indexes[offset : offset + chunk_size]
        print(
            f"[L2] materializing target columns {indexes[0]}-{indexes[-1]}",
            flush=True,
        )
        chunk_columns = [select_columns[0], select_columns[1], *[select_columns[index] for index in indexes]]
        chunk_cursor = conn.execute(f"SELECT {', '.join(chunk_columns)} {from_part}")
        chunk_df = pd.DataFrame(chunk_cursor.fetchall(), columns=[column[0] for column in chunk_cursor.description])
        chunk_df = chunk_df.set_index(["stock_code", "trade_date"], drop=False)
        for index in indexes:
            column = target_columns[index]
            target_df[column] = chunk_df[column].reindex(target_df.index)

    target_df = target_df.reset_index(drop=True)
    target_df = target_df[target_columns]
    conn.register("__l2_target_day_df", target_df)
    conn.execute("CREATE TEMP TABLE __l2_target_day_new AS SELECT * FROM __l2_target_day_df")
    conn.unregister("__l2_target_day_df")


def collect_metrics(conn: duckdb.DuckDBPyConnection, target_trade_date: str) -> dict[str, int | str]:
    row = conn.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT stock_code),
            MIN(trade_date),
            MAX(trade_date),
            SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END)
        FROM STOCK_DAILY_DATA
        """
    ).fetchone()
    target = conn.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT stock_code),
            SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END)
        FROM STOCK_DAILY_DATA
        WHERE trade_date = ?
        """,
        [target_trade_date],
    ).fetchone()
    duplicates = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT stock_code, trade_date, COUNT(*) AS c
            FROM STOCK_DAILY_DATA
            GROUP BY stock_code, trade_date
            HAVING c > 1
        )
        """
    ).fetchone()[0]
    qfq_nulls = conn.execute(
        """
        SELECT
            SUM(CASE WHEN open_qfq IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN high_qfq IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN low_qfq IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN close_qfq IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN pre_close_qfq IS NULL THEN 1 ELSE 0 END)
        FROM STOCK_DAILY_DATA
        WHERE trade_date = ?
        """,
        [target_trade_date],
    ).fetchone()
    return {
        "rows": int(row[0] or 0),
        "stock_count": int(row[1] or 0),
        "min_trade_date": str(row[2] or ""),
        "max_trade_date": str(row[3] or ""),
        "bj_rows": int(row[4] or 0),
        "target_rows": int(target[0] or 0),
        "target_stock_count": int(target[1] or 0),
        "target_bj_rows": int(target[2] or 0),
        "duplicate_key_groups": int(duplicates or 0),
        "target_open_qfq_nulls": int(qfq_nulls[0] or 0),
        "target_high_qfq_nulls": int(qfq_nulls[1] or 0),
        "target_low_qfq_nulls": int(qfq_nulls[2] or 0),
        "target_close_qfq_nulls": int(qfq_nulls[3] or 0),
        "target_pre_close_qfq_nulls": int(qfq_nulls[4] or 0),
    }


def collect_qfq_columns(conn: duckdb.DuckDBPyConnection) -> tuple[list[str], list[str]]:
    target_columns = [row[1] for row in conn.execute("PRAGMA table_info(STOCK_DAILY_DATA)").fetchall()]
    stk_factor_columns = [row[1] for row in conn.execute("PRAGMA table_info(stk_factor)").fetchall()]
    target_qfq_columns = [column for column in target_columns if "qfq" in column.lower()]
    stk_factor_qfq_columns = [
        column
        for column in target_qfq_columns
        if column in stk_factor_columns and column not in PRICE_QFQ_COLUMNS
    ]
    return target_qfq_columns, stk_factor_qfq_columns


def collect_qfq_null_metrics(
    conn: duckdb.DuckDBPyConnection,
    target_trade_date: str,
    columns: list[str],
) -> dict[str, int]:
    if not columns:
        return {}
    expressions = [
        f"SUM(CASE WHEN {quote_ident(column)} IS NULL THEN 1 ELSE 0 END) AS {quote_ident(column)}"
        for column in columns
    ]
    row = conn.execute(
        f"SELECT {', '.join(expressions)} FROM STOCK_DAILY_DATA WHERE trade_date = ?",
        [target_trade_date],
    ).fetchone()
    return {column: int(row[index] or 0) for index, column in enumerate(columns)}


def recompute_qfq_prices_by_period(
    conn: duckdb.DuckDBPyConnection,
    periods: list[tuple[str, str]],
) -> int:
    for start_date, end_date in periods:
        print(f"[L2] recomputing qfq prices for {start_date}-{end_date}", flush=True)
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            """
            UPDATE STOCK_DAILY_DATA AS s
            SET
                open_qfq = s.open * d.adj_factor / l.latest_adj_factor,
                high_qfq = s.high * d.adj_factor / l.latest_adj_factor,
                low_qfq = s.low * d.adj_factor / l.latest_adj_factor,
                close_qfq = s.close * d.adj_factor / l.latest_adj_factor,
                pre_close_qfq = s.pre_close * d.adj_factor / l.latest_adj_factor
            FROM __adj_day AS d
            JOIN __adj_latest AS l
              ON d.stock_code = l.stock_code
            WHERE s.stock_code = d.stock_code
              AND s.trade_date = d.trade_date
              AND s.trade_date >= ?
              AND s.trade_date <= ?
              AND d.adj_factor IS NOT NULL
              AND l.latest_adj_factor IS NOT NULL
              AND l.latest_adj_factor <> 0
            """,
            [start_date, end_date],
        )
        conn.execute("COMMIT")
    return len(periods)


def sync_qfq_indicators_by_period(
    conn: duckdb.DuckDBPyConnection,
    columns: list[str],
    periods: list[tuple[str, str]],
) -> int:
    if not columns:
        return 0
    set_clause = ",\n                    ".join(
        f"{quote_ident(column)} = f.{quote_ident(column)}" for column in columns
    )
    for start_date, end_date in periods:
        print(
            f"[L2] syncing {len(columns)} qfq indicator columns for "
            f"{start_date}-{end_date}",
            flush=True,
        )
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            f"""
            UPDATE STOCK_DAILY_DATA AS s
            SET
                {set_clause}
            FROM stk_factor AS f
            WHERE s.stock_code = f.ts_code
              AND s.trade_date = f.trade_date
              AND s.stock_code NOT LIKE '%.BJ'
              AND s.trade_date >= ?
              AND s.trade_date <= ?
            """,
            [start_date, end_date],
        )
        conn.execute("COMMIT")
    return len(periods)


def integrate_target_date_incremental(
    *,
    target_trade_date: str,
    data_dir: str | Path | None = None,
    target_path: str | Path | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    resolved_data_dir = resolve_data_dir(data_dir)
    resolved_target = Path(target_path) if target_path else resolve_stock_daily_duckdb_path(
        data_dir=resolved_data_dir,
        require_exists=True,
    )

    with duckdb.connect(str(resolved_target)) as conn:
        phase_started = time.perf_counter()
        attached = attach_l1_sources(conn, resolved_data_dir)
        before = collect_metrics(conn, target_trade_date)
        phase_seconds["attach_and_before_metrics"] = round(
            time.perf_counter() - phase_started, 6
        )

        phase_started = time.perf_counter()
        select_sql = build_target_date_select_sql(target_trade_date)
        target_columns = [row[0] for row in conn.execute("DESCRIBE STOCK_DAILY_DATA").fetchall()]
        materialized_from_part = create_materialized_target_aliases(conn, target_trade_date)
        create_target_day_table_chunked(
            conn,
            select_sql=select_sql,
            target_columns=target_columns,
            from_part_override=materialized_from_part,
        )
        new_columns = [row[0] for row in conn.execute("DESCRIBE __l2_target_day_new").fetchall()]
        if target_columns != new_columns:
            raise RuntimeError(
                f"L2 target-day schema mismatch: target={len(target_columns)} new={len(new_columns)}"
            )

        generated = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT stock_code), "
            "SUM(CASE WHEN stock_code LIKE '%.BJ' THEN 1 ELSE 0 END) "
            "FROM __l2_target_day_new"
        ).fetchone()
        phase_seconds["materialize_target_day"] = round(
            time.perf_counter() - phase_started, 6
        )

        phase_started = time.perf_counter()
        print(f"[L2] replacing target date {target_trade_date}", flush=True)
        conn.execute("BEGIN TRANSACTION")
        conn.execute("DELETE FROM STOCK_DAILY_DATA WHERE trade_date = ?", [target_trade_date])
        selected_cols = ", ".join(quote_ident(column) for column in target_columns)
        conn.execute(
            f"INSERT INTO STOCK_DAILY_DATA ({selected_cols}) "
            f"SELECT {selected_cols} FROM __l2_target_day_new"
        )
        conn.execute("COMMIT")
        phase_seconds["replace_target_day"] = round(
            time.perf_counter() - phase_started, 6
        )

        after = collect_metrics(conn, target_trade_date)
        target_qfq_columns, stk_factor_qfq_columns = collect_qfq_columns(conn)
        target_qfq_nulls = collect_qfq_null_metrics(
            conn,
            target_trade_date,
            target_qfq_columns,
        )

    return {
        "status": "completed",
        "target_trade_date": target_trade_date,
        "target_path": str(resolved_target),
        "table": "STOCK_DAILY_DATA",
        "operation": "target_day_all_fields_incremental_replace",
        "processing_scope": "target_trade_date_only",
        "qfq_price_scope": "target_trade_date_only",
        "qfq_price_columns_integrated": list(QFQ_PRICE_COLUMNS),
        "qfq_indicator_scope": "target_trade_date_only",
        "qfq_indicator_columns_integrated": stk_factor_qfq_columns,
        "qfq_column_count": len(target_qfq_columns),
        "qfq_indicator_column_count": len(stk_factor_qfq_columns),
        "performance_plan": {
            "historical_scan": False,
            "historical_qfq_price_transactions": 0,
            "historical_qfq_indicator_transactions": 0,
            "phase_seconds": phase_seconds,
            "total_seconds": round(time.perf_counter() - started, 6),
        },
        "attached_l1_sources": attached,
        "before": before,
        "generated_target_day": {
            "rows": int(generated[0] or 0),
            "stock_count": int(generated[1] or 0),
            "bj_rows": int(generated[2] or 0),
        },
        "after": after,
        "target_qfq_nulls": target_qfq_nulls,
        "generated_at": datetime.now().astimezone().isoformat(),
        "boundary": {
            "uses_l1_active_duckdb_table_files": True,
            "uses_sqlite": False,
            "touches_l3_or_downstream": False,
            "driver_table": "daily_data",
            "no_bj_universe": True,
        },
    }


# Compatibility alias for callers that imported the previous function name. The
# default workflow is target-day incremental only; no historical qfq update runs.
integrate_target_date_and_recompute_qfq = integrate_target_date_incremental


def sync_qfq_indicators_full_history(
    *,
    target_trade_date: str,
    data_dir: str | Path | None = None,
    target_path: str | Path | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    resolved_data_dir = resolve_data_dir(data_dir)
    resolved_target = Path(target_path) if target_path else resolve_stock_daily_duckdb_path(
        data_dir=resolved_data_dir,
        require_exists=True,
    )
    stk_factor_path = resolve_l1_raw_duckdb_path(
        "stk_factor",
        data_dir=resolved_data_dir,
        require_exists=True,
    )

    with duckdb.connect(str(resolved_target)) as conn:
        before = collect_metrics(conn, target_trade_date)
        conn.execute(
            f"ATTACH {quote_literal(str(stk_factor_path))} AS src_stk_factor (READ_ONLY)"
        )
        conn.execute(
            "CREATE OR REPLACE TEMP VIEW stk_factor AS "
            "SELECT * FROM src_stk_factor.stk_factor"
        )
        target_qfq_columns, stk_factor_qfq_columns = collect_qfq_columns(conn)
        min_trade_date, max_trade_date = conn.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM STOCK_DAILY_DATA"
        ).fetchone()
        periods = iter_year_ranges(str(min_trade_date), str(max_trade_date))
        periods = [
            (start, end)
            for start, end in periods
            if (start_year is None or int(start[:4]) >= start_year)
            and (end_year is None or int(end[:4]) <= end_year)
        ]

        transaction_count = sync_qfq_indicators_by_period(
            conn,
            stk_factor_qfq_columns,
            periods,
        )
        if periods:
            conn.execute("CHECKPOINT")

        after = collect_metrics(conn, target_trade_date)
        target_qfq_nulls = collect_qfq_null_metrics(
            conn,
            target_trade_date,
            target_qfq_columns,
        )

    return {
        "status": "completed",
        "target_trade_date": target_trade_date,
        "target_path": str(resolved_target),
        "table": "STOCK_DAILY_DATA",
        "operation": "full_history_qfq_indicator_sync_only",
        "qfq_indicator_sync_scope": "all matching STOCK_DAILY_DATA rows from active L1 stk_factor",
        "qfq_indicator_columns_synced": stk_factor_qfq_columns,
        "qfq_column_count": len(target_qfq_columns),
        "qfq_indicator_column_count": len(stk_factor_qfq_columns),
        "qfq_indicator_sync_periods": [
            {"start": start, "end": end} for start, end in periods
        ],
        "performance_plan": {
            "period_granularity": "year",
            "period_count": len(periods),
            "transaction_count": transaction_count,
            "columns_per_transaction": len(stk_factor_qfq_columns),
            "total_seconds": round(time.perf_counter() - started, 6),
        },
        "active_l1_stk_factor": str(stk_factor_path),
        "before": before,
        "after": after,
        "target_qfq_nulls_after_indicator_sync": target_qfq_nulls,
        "generated_at": datetime.now().astimezone().isoformat(),
        "boundary": {
            "uses_l1_active_duckdb_table_files": True,
            "uses_sqlite": False,
            "touches_non_qfq_columns": False,
            "touches_l3_or_downstream": False,
            "no_bj_universe": True,
        },
    }


def validate_qfq_indicators_by_year(
    *,
    target_trade_date: str,
    data_dir: str | Path | None = None,
    target_path: str | Path | None = None,
) -> dict[str, object]:
    resolved_data_dir = resolve_data_dir(data_dir)
    resolved_target = Path(target_path) if target_path else resolve_stock_daily_duckdb_path(
        data_dir=resolved_data_dir,
        require_exists=True,
    )
    stk_factor_path = resolve_l1_raw_duckdb_path(
        "stk_factor",
        data_dir=resolved_data_dir,
        require_exists=True,
    )

    with duckdb.connect(str(resolved_target), read_only=True) as conn:
        conn.execute(
            f"ATTACH {quote_literal(str(stk_factor_path))} AS src_stk_factor (READ_ONLY)"
        )
        conn.execute(
            "CREATE OR REPLACE TEMP VIEW stk_factor AS "
            "SELECT * FROM src_stk_factor.stk_factor"
        )
        target_qfq_columns, stk_factor_qfq_columns = collect_qfq_columns(conn)
        mismatch_predicate = " OR ".join(
            f"s.{quote_ident(column)} IS DISTINCT FROM f.{quote_ident(column)}"
            for column in stk_factor_qfq_columns
        )
        mismatch_by_year = conn.execute(
            f"""
            SELECT SUBSTR(s.trade_date, 1, 4) AS trade_year, COUNT(*) AS mismatched_rows
            FROM STOCK_DAILY_DATA AS s
            JOIN stk_factor AS f
              ON s.stock_code = f.ts_code
             AND s.trade_date = f.trade_date
            WHERE {mismatch_predicate}
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchall()
        target_qfq_nulls = collect_qfq_null_metrics(
            conn,
            target_trade_date,
            target_qfq_columns,
        )

    return {
        "status": "passed" if not mismatch_by_year else "failed",
        "target_trade_date": target_trade_date,
        "target_path": str(resolved_target),
        "active_l1_stk_factor": str(stk_factor_path),
        "qfq_indicator_column_count": len(stk_factor_qfq_columns),
        "mismatch_by_year": [
            {"trade_year": str(year), "mismatched_rows": int(rows)}
            for year, rows in mismatch_by_year
        ],
        "target_qfq_nulls": target_qfq_nulls,
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally replace all L2 fields for one target trade date."
    )
    parser.add_argument("--target-trade-date", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--target-path")
    parser.add_argument(
        "--active-lock-path",
        help="Active L2 DuckDB path checked for a write lock before any mutating mode.",
    )
    parser.add_argument(
        "--clone-active-to-target-after-lock",
        action="store_true",
        help=(
            "After the active write-lock gate passes, clone active L2 to a fresh "
            "target path before applying the target-date increment."
        ),
    )
    parser.add_argument("--summary-json")
    parser.add_argument("--qfq-indicators-only", action="store_true")
    parser.add_argument("--validate-qfq-indicators-only", action="store_true")
    parser.add_argument("--qfq-start-year", type=int)
    parser.add_argument("--qfq-end-year", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runtime_gate: dict[str, object] | None = None
    try:
        runtime_gate = require_project_venv_runtime()
        lock_gate: dict[str, object] | None = None
        staging_clone_gate: dict[str, object] | None = None
        if not args.validate_qfq_indicators_only:
            data_dir = resolve_data_dir(args.data_dir)
            active_lock_path = Path(args.active_lock_path) if args.active_lock_path else (
                resolve_stock_daily_duckdb_path(data_dir=data_dir, require_exists=True)
            )
            lock_gate = require_duckdb_write_lock(active_lock_path)

            if args.clone_active_to_target_after_lock:
                if args.qfq_indicators_only:
                    raise RuntimeError(
                        "active-to-target staging clone is only valid for target-date incremental mode"
                    )
                if not args.target_path:
                    raise RuntimeError(
                        "--target-path is required with --clone-active-to-target-after-lock"
                    )
                source_path = Path(active_lock_path).resolve()
                target_path = Path(args.target_path).resolve()
                if source_path == target_path:
                    raise RuntimeError("staging target must differ from active L2")
                if target_path.exists():
                    raise RuntimeError(
                        f"fresh staging target already exists: {target_path}"
                    )
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                staging_clone_gate = {
                    "passed": True,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "size_bytes": target_path.stat().st_size,
                    "created_after_active_write_lock_gate": True,
                }

        if args.validate_qfq_indicators_only:
            result = validate_qfq_indicators_by_year(
                target_trade_date=args.target_trade_date,
                data_dir=args.data_dir,
                target_path=args.target_path,
            )
        elif args.qfq_indicators_only:
            result = sync_qfq_indicators_full_history(
                target_trade_date=args.target_trade_date,
                data_dir=args.data_dir,
                target_path=args.target_path,
                start_year=args.qfq_start_year,
                end_year=args.qfq_end_year,
            )
        else:
            result = integrate_target_date_incremental(
                target_trade_date=args.target_trade_date,
                data_dir=args.data_dir,
                target_path=args.target_path,
            )
        if lock_gate is not None:
            result["active_write_lock_gate"] = lock_gate
        if staging_clone_gate is not None:
            result["staging_clone_gate"] = staging_clone_gate
        if runtime_gate is not None:
            result["runtime_gate"] = runtime_gate
    except Exception as exc:
        result = {
            "status": "blocked",
            "target_trade_date": args.target_trade_date,
            "target_path": args.target_path,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "active_written": False,
            "runtime_gate": runtime_gate,
            "generated_at": datetime.now().astimezone().isoformat(),
        }
        if args.summary_json:
            Path(args.summary_json).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        raise

    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

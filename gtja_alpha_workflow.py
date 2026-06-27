from __future__ import annotations

import argparse
import json
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stock_daily_data_route import connect_stock_daily_readonly, resolve_stock_daily_db_path

from data_process_module import get_factor_data
from gtja_alpha_official_core import append_official_gtja_alpha, required_official_gtja_raw_columns


GTJA_ALPHA_COLUMNS = [f"gtja_alpha{i:03d}" for i in range(1, 192)]
GTJA_RANK_SENTINEL_COLUMNS = [
    "gtja_alpha101",
    "gtja_alpha102",
    "gtja_alpha103",
    "gtja_alpha104",
    "gtja_alpha105",
    "gtja_alpha106",
    "gtja_alpha107",
    "gtja_alpha108",
    "gtja_alpha109",
    "gtja_alpha110",
]

RAW_FACTOR_INPUT_COLUMNS = [
    "stock_code",
    "trade_date",
    "name",
    "industry",
    "act_ent_type",
    "open",
    "close",
    "high",
    "low",
    "pre_close",
    "vol",
    "amount",
    "total_mv",
    "pb",
    "index_2000_open",
    "index_2000_close",
]


def audit_gtja_rank_scope(
    frame: pd.DataFrame,
    alpha_columns: Iterable[str] | None = None,
    *,
    min_stocks_per_day: int = 500,
    min_unique_ratio: float = 0.75,
) -> dict:
    """Audit whether GTJA cross-sectional ranks look like full-market ranks.

    This is a guardrail for the batch factor workflow. Time-series features can
    be built stock-by-stock, but GTJA RANK() terms must be ranked across the
    whole trade-date universe. Batch-sized rank distributions are suspicious.
    """
    columns = [col for col in (alpha_columns or GTJA_RANK_SENTINEL_COLUMNS) if col in frame.columns]
    issues: list[dict] = []
    day_rows: list[dict] = []

    if "trade_date" not in frame.columns or "stock_code" not in frame.columns:
        return {
            "passed": False,
            "issues": [{"issue": "missing_required_columns"}],
            "checked_columns": columns,
            "day_count": 0,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    if not columns:
        return {
            "passed": False,
            "issues": [{"issue": "missing_gtja_columns"}],
            "checked_columns": [],
            "day_count": 0,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    for trade_date, day in frame.groupby(frame["trade_date"].astype(str), sort=True):
        stock_count = int(day["stock_code"].nunique())
        if stock_count < min_stocks_per_day:
            issues.append(
                {
                    "issue": "stock_count_too_low",
                    "trade_date": trade_date,
                    "stock_count": stock_count,
                    "min_stocks_per_day": min_stocks_per_day,
                }
            )

        for col in columns:
            series = pd.to_numeric(day[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            non_null_count = int(series.shape[0])
            if non_null_count < min_stocks_per_day:
                continue
            unique_count = int(series.nunique())
            unique_ratio = unique_count / non_null_count if non_null_count else 0.0
            day_rows.append(
                {
                    "trade_date": trade_date,
                    "column": col,
                    "stock_count": stock_count,
                    "non_null_count": non_null_count,
                    "unique_count": unique_count,
                    "unique_ratio": unique_ratio,
                }
            )
            if unique_ratio < min_unique_ratio:
                issues.append(
                    {
                        "issue": "rank_unique_count_too_low",
                        "trade_date": trade_date,
                        "column": col,
                        "non_null_count": non_null_count,
                        "unique_count": unique_count,
                        "unique_ratio": unique_ratio,
                        "min_unique_ratio": min_unique_ratio,
                    }
                )

    return {
        "passed": len(issues) == 0,
        "issues": issues[:200],
        "issue_count": len(issues),
        "checked_columns": columns,
        "day_count": int(frame["trade_date"].astype(str).nunique()),
        "sample_stats": day_rows[:200],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_gtja_rank_audit(
    frame: pd.DataFrame,
    output_path: Path,
    alpha_columns: Iterable[str] | None = None,
    *,
    min_stocks_per_day: int = 500,
    min_unique_ratio: float = 0.75,
) -> dict:
    audit = audit_gtja_rank_scope(
        frame,
        alpha_columns,
        min_stocks_per_day=min_stocks_per_day,
        min_unique_ratio=min_unique_ratio,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _read_raw_factor_input(data_dir: Path, source_parquet: Path | None, end_date: str | None = None) -> pd.DataFrame:
    if source_parquet and source_parquet.exists():
        schema_names = set(pq.ParquetFile(source_parquet).schema.names)
        missing = [col for col in RAW_FACTOR_INPUT_COLUMNS if col not in schema_names]
        if missing:
            raise ValueError(f"source parquet missing raw factor input columns: {missing}")
        filters = [("trade_date", "<=", end_date)] if end_date else None
        print(f"global_gtja_read_source=parquet path={source_parquet}", flush=True)
        frame = pd.read_parquet(source_parquet, columns=RAW_FACTOR_INPUT_COLUMNS, filters=filters)
    else:
        db_path = resolve_stock_daily_db_path(data_dir=data_dir)
        print(f"global_gtja_read_source=sqlite path={db_path}", flush=True)
        with closing(connect_stock_daily_readonly(db_path=db_path)) as conn:
            if end_date:
                frame = pd.read_sql(
                    "SELECT * FROM STOCK_DAILY_DATA WHERE trade_date <= ?",
                    conn,
                    params=[end_date],
                )
            else:
                frame = pd.read_sql("SELECT * FROM STOCK_DAILY_DATA", conn)
        frame.columns = frame.columns.str.lower()
        frame = frame[[col for col in RAW_FACTOR_INPUT_COLUMNS if col in frame.columns]]
    frame.columns = frame.columns.str.lower()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame.sort_values(["stock_code", "trade_date"], inplace=True)
    return frame


def rebuild_full_market_gtja(
    data_dir: Path,
    output_path: Path | None = None,
    end_date: str | None = None,
    source_parquet: Path | None = None,
) -> Path:
    """Recompute factor data in one full-market pass so GTJA ranks are global."""
    final_path = output_path or data_dir / "stock_factor_data.parquet"
    integ_data = _read_raw_factor_input(data_dir, source_parquet or final_path, end_date=end_date)
    print(f"global_gtja_raw_loaded rows={integ_data.shape[0]} cols={integ_data.shape[1]}", flush=True)
    factor_data = get_factor_data(integ_data)
    print(f"global_gtja_factor_done rows={factor_data.shape[0]} cols={factor_data.shape[1]}", flush=True)
    factor_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    temp_path = final_path.with_suffix(".tmp.parquet")
    print(f"global_gtja_write_start path={temp_path}", flush=True)
    factor_data.to_parquet(temp_path, index=False)
    temp_path.replace(final_path)
    print(f"global_gtja_write_done path={final_path}", flush=True)
    return final_path


def read_gtja_audit_frame(parquet_path: Path) -> pd.DataFrame:
    schema_names = set(pq.ParquetFile(parquet_path).schema.names)
    columns = ["trade_date", "stock_code", *[col for col in GTJA_ALPHA_COLUMNS if col in schema_names]]
    return pd.read_parquet(parquet_path, columns=columns)


def required_gtja_raw_columns() -> list[str]:
    """Columns needed to recompute official GTJA Alpha formulas from raw factor parts."""
    return required_official_gtja_raw_columns()


def compute_gtja_alpha_from_raw_factor(
    raw_factor: pd.DataFrame,
    *,
    encode: bool = True,
    drop_ts: bool = True,
    cross_sectional_rank_mode: str = "rank",
) -> pd.DataFrame:
    """Compute official GTJA Alpha columns from raw factor data."""
    if cross_sectional_rank_mode not in {"rank", "identity"}:
        raise ValueError("cross_sectional_rank_mode must be 'rank' or 'identity'")
    factor_data = append_official_gtja_alpha(
        raw_factor,
        cross_sectional_rank_mode=cross_sectional_rank_mode,
        benchmark_prefix="index_2000",
    )

    if drop_ts:
        ts_cols = [col for col in factor_data.columns if col.startswith("_ts_")]
        factor_data = factor_data.drop(columns=ts_cols)

    if encode:
        if "stock_code" in factor_data.columns:
            factor_data["stock_encode"] = pd.factorize(factor_data["stock_code"].astype(str), sort=True)[0]
        if "industry" in factor_data.columns:
            factor_data["industry"] = factor_data["industry"].fillna("")
            factor_data["industry_encode"] = pd.factorize(factor_data["industry"].astype(str), sort=True)[0]
        if "act_ent_type" in factor_data.columns:
            factor_data["act_ent_type_encode"] = pd.factorize(factor_data["act_ent_type"].astype(str), sort=True)[0]

    factor_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return factor_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit or rebuild GTJA Alpha full-market rank workflow.")
    parser.add_argument("--data-dir", default="data_file")
    parser.add_argument("--parquet-path")
    parser.add_argument("--source-parquet")
    parser.add_argument("--audit-path")
    parser.add_argument("--rebuild-full-market", action="store_true")
    parser.add_argument("--end-date")
    parser.add_argument("--min-stocks-per-day", type=int, default=500)
    parser.add_argument("--min-unique-ratio", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    parquet_path = Path(args.parquet_path) if args.parquet_path else data_dir / "stock_factor_data.parquet"

    if args.rebuild_full_market:
        parquet_path = rebuild_full_market_gtja(
            data_dir,
            parquet_path,
            end_date=args.end_date,
            source_parquet=Path(args.source_parquet) if args.source_parquet else None,
        )

    frame = read_gtja_audit_frame(parquet_path)
    audit_path = Path(args.audit_path) if args.audit_path else data_dir / "reports" / "gtja_alpha_rank_audit.json"
    audit = write_gtja_rank_audit(
        frame,
        audit_path,
        min_stocks_per_day=args.min_stocks_per_day,
        min_unique_ratio=args.min_unique_ratio,
    )
    print(json.dumps({"passed": audit["passed"], "issue_count": audit.get("issue_count", 0), "audit_path": str(audit_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

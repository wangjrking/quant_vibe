from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gtja_alpha_workflow import GTJA_ALPHA_COLUMNS, compute_gtja_alpha_from_raw_factor, required_gtja_raw_columns


@dataclass(frozen=True)
class DateChunk:
    chunk_index: int
    output_dates: list[str]
    read_start: str
    read_end: str


def build_date_chunks(
    dates: list[str],
    *,
    chunk_days: int,
    past_overlap_days: int,
    future_overlap_days: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[DateChunk]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    ordered = sorted({str(date) for date in dates})
    if start_date:
        ordered = [date for date in ordered if date >= str(start_date)]
    if end_date:
        ordered = [date for date in ordered if date <= str(end_date)]
    all_dates = sorted({str(date) for date in dates})
    date_to_pos = {date: idx for idx, date in enumerate(all_dates)}

    chunks: list[DateChunk] = []
    for chunk_index, start in enumerate(range(0, len(ordered), chunk_days)):
        output_dates = ordered[start : start + chunk_days]
        first_pos = date_to_pos[output_dates[0]]
        last_pos = date_to_pos[output_dates[-1]]
        read_start = all_dates[max(0, first_pos - past_overlap_days)]
        read_end = all_dates[min(len(all_dates) - 1, last_pos + future_overlap_days)]
        chunks.append(
            DateChunk(
                chunk_index=chunk_index,
                output_dates=output_dates,
                read_start=read_start,
                read_end=read_end,
            )
        )
    return chunks


def read_trade_dates(parquet_path: Path) -> list[str]:
    frame = pd.read_parquet(parquet_path, columns=["trade_date"])
    return sorted(frame["trade_date"].astype(str).unique().tolist())


def _read_raw_parts_for_window(
    raw_parts_dir: Path,
    *,
    read_start: str,
    read_end: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for part_path in sorted(raw_parts_dir.glob("raw_part_*.parquet")):
        frame = pd.read_parquet(
            part_path,
            columns=columns,
            filters=[("trade_date", ">=", read_start), ("trade_date", "<=", read_end)],
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns or [])
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = out["trade_date"].astype(str)
    out.sort_values(["trade_date", "stock_code"], inplace=True)
    return out


def add_cross_sectional_standard_features(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    add_rank: bool = True,
    add_zscore: bool = True,
) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby("trade_date", sort=False)
    new_columns: dict[str, pd.Series] = {}
    for col in columns:
        if col not in result.columns:
            continue
        values = pd.to_numeric(result[col], errors="coerce")
        if add_rank:
            new_columns[f"{col}_rank"] = grouped[col].rank(pct=True)
        if add_zscore:
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std").replace(0, np.nan)
            new_columns[f"{col}_zscore"] = (values - mean) / std
    if new_columns:
        result = pd.concat([result, pd.DataFrame(new_columns, index=result.index)], axis=1)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result


def standardize_date_chunk(
    raw_parts_dir: Path,
    output_dir: Path,
    chunk: DateChunk,
    *,
    rank_columns: list[str] | None = None,
    add_rank: bool = False,
    add_zscore: bool = False,
    resume: bool = True,
    gtja_only: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"factor_standard_part_{chunk.chunk_index:04d}.parquet"
    if resume and output_path.exists():
        print(f"standard_chunk_skip index={chunk.chunk_index} path={output_path}", flush=True)
        return output_path

    print(
        f"standard_chunk_start index={chunk.chunk_index} output={chunk.output_dates[0]}..{chunk.output_dates[-1]} "
        f"read={chunk.read_start}..{chunk.read_end}",
        flush=True,
    )
    read_columns = required_gtja_raw_columns() if gtja_only else None
    frame = _read_raw_parts_for_window(
        raw_parts_dir,
        read_start=chunk.read_start,
        read_end=chunk.read_end,
        columns=read_columns,
    )
    if frame.empty:
        raise RuntimeError(f"no raw factor rows found for {chunk.read_start}..{chunk.read_end}")
    frame = compute_gtja_alpha_from_raw_factor(frame, encode=True, drop_ts=True)
    frame = frame[frame["trade_date"].isin(chunk.output_dates)].copy()
    if gtja_only:
        keep_columns = [
            "trade_date",
            "stock_code",
            "industry",
            "act_ent_type",
            "stock_encode",
            "industry_encode",
            "act_ent_type_encode",
            *[col for col in GTJA_ALPHA_COLUMNS if col in frame.columns],
        ]
        frame = frame[[col for col in keep_columns if col in frame.columns]]
    if rank_columns and (add_rank or add_zscore):
        frame = add_cross_sectional_standard_features(frame, rank_columns, add_rank=add_rank, add_zscore=add_zscore)
    frame.to_parquet(output_path, index=False)
    print(f"standard_chunk_done index={chunk.chunk_index} rows={frame.shape[0]} cols={frame.shape[1]} path={output_path}", flush=True)
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build date chunks for factor standardization.")
    parser.add_argument("--factor-path", default="../data_file/stock_factor_data.parquet")
    parser.add_argument("--raw-parts-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--chunk-days", type=int, default=60)
    parser.add_argument("--past-overlap-days", type=int, default=260)
    parser.add_argument("--future-overlap-days", type=int, default=25)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--start-chunk", type=int)
    parser.add_argument("--end-chunk", type=int)
    parser.add_argument("--rank-columns", default="")
    parser.add_argument("--add-rank", action="store_true")
    parser.add_argument("--add-zscore", action="store_true")
    parser.add_argument("--gtja-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    chunks = build_date_chunks(
        read_trade_dates(Path(args.factor_path)),
        chunk_days=args.chunk_days,
        past_overlap_days=args.past_overlap_days,
        future_overlap_days=args.future_overlap_days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if args.raw_parts_dir and args.output_dir:
        rank_columns = [col.strip() for col in args.rank_columns.split(",") if col.strip()]
        for chunk in chunks:
            if args.start_chunk is not None and chunk.chunk_index < args.start_chunk:
                continue
            if args.end_chunk is not None and chunk.chunk_index > args.end_chunk:
                continue
            standardize_date_chunk(
                Path(args.raw_parts_dir),
                Path(args.output_dir),
                chunk,
                rank_columns=rank_columns,
                add_rank=args.add_rank,
                add_zscore=args.add_zscore,
                resume=not args.no_resume,
                gtja_only=args.gtja_only,
            )
        return

    for chunk in chunks:
        print(
            f"chunk={chunk.chunk_index} output={chunk.output_dates[0]}..{chunk.output_dates[-1]} "
            f"read={chunk.read_start}..{chunk.read_end} days={len(chunk.output_dates)}",
            flush=True,
        )


if __name__ == "__main__":
    main()

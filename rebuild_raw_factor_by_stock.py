from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import pandas as pd

from data_process_module import group_factor_eng
from rebuild_factor_data_batched import _chunks, _load_batch, _normalize_types, _stock_codes
def _try_acquire_lock(lock_path: Path, *, stale_seconds: int | None = None) -> bool:
    if stale_seconds is not None and lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age > stale_seconds:
            lock_path.unlink(missing_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()} time={time.time()}\n")
    return True


def rebuild_raw_factor_parts(
    data_dir: Path,
    *,
    batch_size: int,
    resume: bool,
    start_batch: int | None = None,
    end_batch: int | None = None,
    output_dir: Path | None = None,
    use_locks: bool = False,
    lock_stale_minutes: int = 360,
) -> Path:
    part_dir = output_dir or data_dir / "raw_factor_by_stock_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    codes = _stock_codes(data_dir=data_dir)
    print(f"raw_factor_start stocks={len(codes)} batch_size={batch_size} output_dir={part_dir}", flush=True)

    for batch_idx, batch_codes in _chunks(codes, batch_size):
        if start_batch is not None and batch_idx < start_batch:
            continue
        if end_batch is not None and batch_idx > end_batch:
            continue

        part_path = part_dir / f"raw_part_{batch_idx:04d}.parquet"
        if resume and part_path.exists():
            print(f"raw_batch_skip index={batch_idx} path={part_path}", flush=True)
            continue
        lock_path = part_path.with_suffix(part_path.suffix + ".lock")
        lock_acquired = False
        if use_locks:
            lock_acquired = _try_acquire_lock(lock_path, stale_seconds=lock_stale_minutes * 60)
            if not lock_acquired:
                print(f"raw_batch_locked_skip index={batch_idx} lock={lock_path}", flush=True)
                continue
            if resume and part_path.exists():
                print(f"raw_batch_skip_after_lock index={batch_idx} path={part_path}", flush=True)
                lock_path.unlink(missing_ok=True)
                continue

        try:
            print(
                f"raw_batch_start index={batch_idx} stocks={len(batch_codes)} first={batch_codes[0]} last={batch_codes[-1]}",
                flush=True,
            )
            integ = _load_batch(codes=batch_codes, data_dir=data_dir)
            frames = []
            for _, group in integ.groupby("stock_code", sort=False):
                frames.append(group_factor_eng(group))
            raw_factor = pd.concat(frames, ignore_index=True)
            raw_factor = _normalize_types(raw_factor)
            raw_factor.to_parquet(part_path, index=False)
            print(f"raw_batch_done index={batch_idx} rows={raw_factor.shape[0]} cols={raw_factor.shape[1]} path={part_path}", flush=True)
            del integ, frames, raw_factor
            gc.collect()
        finally:
            if use_locks:
                lock_path.unlink(missing_ok=True)

    return part_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build raw factor data by stock batches.")
    parser.add_argument("--data-dir", default="../data_file")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-batch", type=int)
    parser.add_argument("--end-batch", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--use-locks", action="store_true")
    parser.add_argument("--lock-stale-minutes", type=int, default=360)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rebuild_raw_factor_parts(
        Path(args.data_dir),
        batch_size=args.batch_size,
        resume=args.resume,
        start_batch=args.start_batch,
        end_batch=args.end_batch,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        use_locks=args.use_locks,
        lock_stale_minutes=args.lock_stale_minutes,
    )


if __name__ == "__main__":
    main()

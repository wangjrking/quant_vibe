# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_liquidity_risk_v59_20260721"
CONTRACT = OUT / "cache_contract.json"
BASE_CACHE = ROOT / "quant/data_file/reports/strategy_agent_active_l4_rank_rotation_preregistration_20260721/active_formal_rank_arrays_v1.npz"
L2 = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
OUTPUT = OUT / "active_formal_rank_liquidity_arrays_v2.npz"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["status"] != "frozen_cache_build" or digest(Path(__file__)) != contract["code_sha256"]:
        raise RuntimeError("cache build contract or code mismatch")
    if digest(BASE_CACHE) != contract["inputs"]["base_cache_sha256"] or digest(L2) != contract["inputs"]["l2_sha256"]:
        raise RuntimeError("cache input hash mismatch")
    with np.load(BASE_CACHE, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    preserved_before = {key: array_digest(arrays[key]) for key in ("rank_1d", "rank_3d", "rank_5d", "rank_10d", "dates", "stocks")}
    dates, stocks = arrays["dates"].astype(str), arrays["stocks"].astype(str)
    shape = (len(dates), len(stocks))
    arrays["atr_qfq"] = np.full(shape, np.nan, dtype=np.float32)
    arrays["close_qfq"] = np.full(shape, np.nan, dtype=np.float32)
    arrays["turnover_rate"] = np.full(shape, np.nan, dtype=np.float32)
    date_map = pd.DataFrame({"trade_date": dates, "d_idx": np.arange(len(dates), dtype=np.int32)})
    stock_map = pd.DataFrame({"stock_code": stocks, "s_idx": np.arange(len(stocks), dtype=np.int32)})
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{L2.as_posix()}' AS l2 (READ_ONLY)")
        con.register("date_map", date_map)
        con.register("stock_map", stock_map)
        reader = con.execute("""
            SELECT d.d_idx, s.s_idx, m.atr_qfq, m.close_qfq, m.turnover_rate
            FROM l2.STOCK_DAILY_DATA m
            JOIN date_map d USING (trade_date)
            JOIN stock_map s USING (stock_code)
            ORDER BY d.d_idx, s.s_idx
        """).fetch_record_batch(rows_per_batch=250000)
        matched = 0
        for batch in reader:
            frame = batch.to_pandas()
            d = frame["d_idx"].to_numpy(dtype=np.intp, copy=False)
            s = frame["s_idx"].to_numpy(dtype=np.intp, copy=False)
            for column in ("atr_qfq", "close_qfq", "turnover_rate"):
                arrays[column][d, s] = frame[column].to_numpy(dtype=np.float32, copy=False)
            matched += len(frame)
    finally:
        con.close()
    np.savez_compressed(OUTPUT, **arrays)
    summary = {
        "status": "research_cache_built",
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": digest(OUTPUT),
        "dates": len(dates),
        "stocks": len(stocks),
        "matched_rows": matched,
        "atr_qfq_nonnull": int(np.isfinite(arrays["atr_qfq"]).sum()),
        "close_qfq_nonnull": int(np.isfinite(arrays["close_qfq"]).sum()),
        "turnover_rate_nonnull": int(np.isfinite(arrays["turnover_rate"]).sum()),
        "base_arrays_preserved": all(array_digest(arrays[key]) == preserved_before[key] for key in preserved_before),
    }
    (OUT / "cache_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import production_v260_active_l4_10d_smoothing_refine_v95_20260722 as v95
from . import production_v260_active_l4_adaptive_exit_v162_20260722 as v162
from . import production_v260_active_l4_high_score_sizing_v212_20260723 as v212
from . import production_v260_active_l4_score_deterioration_v174_20260722 as v174
from . import production_v260_active_l4_strong_trend_concentration_v153_20260722 as v153
from . import production_v260_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[7]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_v175_latest_selection_20260723"
V175 = ROOT / "quant/data_file/reports/strategy_agent_active_l4_score_deterioration_refine_v175_20260722"
BASE_PROTOCOL = ROOT / "quant/data_file/reports/strategy_agent_active_l4_rank_rotation_preregistration_20260721/preregistered_protocol.json"
L2 = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
TEMP_CACHE = OUT / "runtime_base_cache.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_liquidity_arrays(arrays: dict[str, np.ndarray]) -> None:
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    shape = (len(dates), len(stocks))
    for column in ("atr_qfq", "close_qfq", "turnover_rate"):
        arrays[column] = np.full(shape, np.nan, dtype=np.float32)
    date_map = pd.DataFrame({"trade_date": dates, "d_idx": np.arange(len(dates), dtype=np.int32)})
    stock_map = pd.DataFrame({"stock_code": stocks, "s_idx": np.arange(len(stocks), dtype=np.int32)})
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{L2.as_posix()}' AS l2 (READ_ONLY)")
        con.register("date_map", date_map)
        con.register("stock_map", stock_map)
        reader = con.execute(
            """
            SELECT d.d_idx, s.s_idx, m.atr_qfq, m.close_qfq, m.turnover_rate
            FROM l2.STOCK_DAILY_DATA m
            JOIN date_map d USING (trade_date)
            JOIN stock_map s USING (stock_code)
            ORDER BY d.d_idx, s.s_idx
            """
        ).fetch_record_batch(rows_per_batch=250000)
        for batch in reader:
            frame = batch.to_pandas()
            d = frame["d_idx"].to_numpy(dtype=np.intp, copy=False)
            s = frame["s_idx"].to_numpy(dtype=np.intp, copy=False)
            for column in ("atr_qfq", "close_qfq", "turnover_rate"):
                arrays[column][d, s] = frame[column].to_numpy(dtype=np.float32, copy=False)
    finally:
        con.close()


def populate_latest_signal_row(arrays: dict[str, np.ndarray]) -> None:
    latest = str(arrays["dates"][-1])
    stock_map = {stock: idx for idx, stock in enumerate(arrays["stocks"].astype(str))}
    resolved = {label: core.resolve_manifest(path) for label, path in core.MANIFESTS.items()}
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{L2.as_posix()}' AS l2 (READ_ONLY)")
        for label, (path, _) in resolved.items():
            con.execute(f"ATTACH '{path.as_posix()}' AS p{label} (READ_ONLY)")
        query = f"""
        WITH scores AS (
          SELECT p10.stock_code,
                 percent_rank() OVER (ORDER BY p1.pred_prob, p10.stock_code) AS rank_1d,
                 percent_rank() OVER (ORDER BY p3.pred_prob, p10.stock_code) AS rank_3d,
                 percent_rank() OVER (ORDER BY p5.pred_prob, p10.stock_code) AS rank_5d,
                 percent_rank() OVER (ORDER BY p10.pred_prob, p10.stock_code) AS rank_10d
          FROM p10d."{resolved['10d'][1]}" p10
          JOIN p5d."{resolved['5d'][1]}" p5 USING (trade_date, stock_code)
          JOIN p3d."{resolved['3d'][1]}" p3 USING (trade_date, stock_code)
          JOIN p1d."{resolved['1d'][1]}" p1 USING (trade_date, stock_code)
          WHERE p10.trade_date = ? AND p10.stock_code NOT LIKE '%.BJ'
        )
        SELECT s.*, m.amount, m.total_mv,
               greatest(0, date_diff('day', try_strptime(m.list_date, '%Y%m%d'), try_strptime(m.trade_date, '%Y%m%d'))) AS listed_days,
               ({core.clean_sql('m')}) AS signal_clean
        FROM scores s
        JOIN l2.STOCK_DAILY_DATA m ON m.trade_date = ? AND m.stock_code = s.stock_code
        ORDER BY s.stock_code
        """
        frame = con.execute(query, [latest, latest]).fetchdf()
    finally:
        con.close()
    t = len(arrays["dates"]) - 1
    indices = np.asarray([stock_map[stock] for stock in frame["stock_code"].astype(str)], dtype=np.intp)
    for column in ("rank_1d", "rank_3d", "rank_5d", "rank_10d", "amount", "total_mv"):
        arrays[column][t, indices] = frame[column].to_numpy(dtype=np.float32, copy=False)
    arrays["listed_days"][t, indices] = frame["listed_days"].fillna(-1).to_numpy(dtype=np.int16)
    arrays["signal_clean"][t, indices] = frame["signal_clean"].fillna(False).to_numpy(dtype=np.bool_)


def current_holdings(actions: pd.DataFrame, stock_index: dict[str, int], date_index: dict[str, int]) -> dict[int, int]:
    holdings: dict[int, int] = {}
    for row in actions.itertuples(index=False):
        idx = stock_index[str(row.stock_code)]
        if row.action == "BUY":
            holdings[idx] = date_index[str(row.signal_date)]
        else:
            holdings.pop(idx, None)
    return holdings


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((V175 / "preregistered_protocol.json").read_text(encoding="utf-8"))
    cache_protocol = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    frozen = json.loads((V175 / "frozen_candidates_before_known_2026.json").read_text(encoding="utf-8"))
    item = next(
        candidate
        for candidate in frozen["candidates"]
        if abs(float(candidate["definition"]["max_rank_deterioration"]) - 0.10) < 1e-12
    )

    original_cache = core.CACHE_PATH
    try:
        core.CACHE_PATH = TEMP_CACHE
        arrays = core.build_cache(cache_protocol)
    finally:
        core.CACHE_PATH = original_cache
        TEMP_CACHE.unlink(missing_ok=True)
    populate_latest_signal_row(arrays)
    add_liquidity_arrays(arrays)

    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    definition = item["definition"]
    latest = str(arrays["dates"][-1])
    daily, actions = v174.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        latest,
        record_actions=True,
    )
    daily.to_csv(OUT / "replay_daily_through_latest.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(OUT / "replay_actions_through_latest.csv", index=False, encoding="utf-8-sig")
    daily_15, actions_15 = v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        latest,
        record_actions=True,
        selection_mask_override=v174.selection_mask(arrays, definition["max_rank_deterioration"]),
        max_positions_override=15,
    )
    daily_15.to_csv(OUT / "replay_daily_max_positions15_through_latest.csv", index=False, encoding="utf-8-sig")
    actions_15.to_csv(OUT / "replay_actions_max_positions15_through_latest.csv", index=False, encoding="utf-8-sig")
    daily_16, actions_16 = v162.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        latest,
        record_actions=True,
        selection_mask_override=v174.selection_mask(arrays, definition["max_rank_deterioration"]),
        max_positions_override=16,
    )
    daily_16.to_csv(OUT / "replay_daily_max_positions16_through_latest.csv", index=False, encoding="utf-8-sig")
    actions_16.to_csv(OUT / "replay_actions_max_positions16_through_latest.csv", index=False, encoding="utf-8-sig")
    definition_sell_086 = {**definition, "sell_score_below": 0.86}
    daily_16_sell_086, actions_16_sell_086 = v162.run_case(
        arrays,
        score,
        order,
        definition_sell_086,
        protocol,
        latest,
        record_actions=True,
        selection_mask_override=v174.selection_mask(arrays, definition["max_rank_deterioration"]),
        max_positions_override=16,
    )
    daily_16_sell_086.to_csv(
        OUT / "replay_daily_max_positions16_sell086_through_latest.csv", index=False, encoding="utf-8-sig"
    )
    actions_16_sell_086.to_csv(
        OUT / "replay_actions_max_positions16_sell086_through_latest.csv", index=False, encoding="utf-8-sig"
    )
    definition_exposure_110 = {
        **definition,
        "normal_target_pct": 0.275,
        "strong_target_pct": 0.4125,
    }
    daily_16_exposure_110, actions_16_exposure_110 = v162.run_case(
        arrays,
        score,
        order,
        definition_exposure_110,
        protocol,
        latest,
        record_actions=True,
        selection_mask_override=v174.selection_mask(arrays, definition["max_rank_deterioration"]),
        max_positions_override=16,
    )
    daily_16_exposure_110.to_csv(
        OUT / "replay_daily_max_positions16_exposure110_through_latest.csv", index=False, encoding="utf-8-sig"
    )
    actions_16_exposure_110.to_csv(
        OUT / "replay_actions_max_positions16_exposure110_through_latest.csv", index=False, encoding="utf-8-sig"
    )
    definition_weak_040 = {**definition, "weak_target_pct": 0.04}
    daily_16_weak_040, actions_16_weak_040 = v162.run_case(
        arrays,
        score,
        order,
        definition_weak_040,
        protocol,
        latest,
        record_actions=True,
        selection_mask_override=v174.selection_mask(arrays, definition["max_rank_deterioration"]),
        max_positions_override=16,
    )
    daily_16_weak_040.to_csv(
        OUT / "replay_daily_max_positions16_weak040_through_latest.csv", index=False, encoding="utf-8-sig"
    )
    actions_16_weak_040.to_csv(
        OUT / "replay_actions_max_positions16_weak040_through_latest.csv", index=False, encoding="utf-8-sig"
    )
    definition_highscore_995_mult110 = {
        **definition,
        "high_score_threshold": 0.995,
        "high_score_multiplier": 1.10,
        "max_positions": 16,
    }
    daily_16_highscore_995_mult110, actions_16_highscore_995_mult110 = v162.run_case(
        arrays,
        score,
        order,
        definition_highscore_995_mult110,
        protocol,
        latest,
        record_actions=True,
        selection_mask_override=v174.selection_mask(arrays, definition["max_rank_deterioration"]),
        candidate_target_multiplier_override=v212.score_multiplier(
            score, definition_highscore_995_mult110
        ),
        max_positions_override=16,
    )
    daily_16_highscore_995_mult110.to_csv(
        OUT / "replay_daily_max_positions16_highscore995_mult110_through_latest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    actions_16_highscore_995_mult110.to_csv(
        OUT / "replay_actions_max_positions16_highscore995_mult110_through_latest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stocks = arrays["stocks"].astype(str)
    stock_index = {stock: idx for idx, stock in enumerate(stocks)}
    date_index = {str(date): idx for idx, date in enumerate(arrays["dates"].astype(str))}
    holdings = current_holdings(actions_16, stock_index, date_index)
    t = len(arrays["dates"]) - 1

    selection = v174.selection_mask(arrays, 0.10)[t]
    clean = arrays["signal_clean"][t] & np.isfinite(score[t]) & selection
    clean &= arrays["listed_days"][t] >= int(protocol["fixed_universe"]["listed_days_min"])
    clean &= np.isfinite(arrays["amount"][t]) & (arrays["amount"][t] >= int(protocol["fixed_universe"]["amount_min"]))
    clean &= np.isfinite(arrays["total_mv"][t]) & (arrays["total_mv"][t] >= int(protocol["fixed_universe"]["mv_min"]))
    clean &= np.isfinite(arrays["turnover_rate"][t]) & (arrays["turnover_rate"][t] >= 0)
    clean &= arrays["turnover_rate"][t] <= float(protocol["fixed_universe"]["turnover_max"])
    ranked = [int(idx) for idx in order[t] if clean[int(idx)]]
    unheld = [idx for idx in ranked if idx not in holdings]

    momentum = v153.v134.market_momentum(arrays, int(definition["market_lookback"]))
    target = v153.schedules(arrays, definition)[1]
    min_hold = v162.min_hold_schedule(arrays, definition["min_hold_policy"])
    best_unheld = max((float(score[t, idx]) for idx in unheld), default=-np.inf)
    theoretical_sells = []
    for idx, entry_t in sorted(holdings.items(), key=lambda pair: stocks[pair[0]]):
        age = t - entry_t
        current = float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
        score_exit = (
            age >= int(min_hold[t])
            and current < float(definition["sell_score_below"])
            and best_unheld - current >= float(definition["replacement_advantage"])
        )
        if age >= int(definition["max_hold_days"]) or score_exit:
            theoretical_sells.append(
                {"stock_code": stocks[idx], "score": current, "age": age, "reason": "max_hold" if age >= int(definition["max_hold_days"]) else "score_replacement"}
            )

    con = duckdb.connect(L2.as_posix(), read_only=True)
    try:
        names = con.execute(
            "SELECT stock_code, name, market FROM STOCK_DAILY_DATA WHERE trade_date = ?",
            [latest],
        ).fetchdf().set_index("stock_code")
    finally:
        con.close()

    candidates = []
    raw = arrays["rank_10d"][t]
    smooth = (score[t] - 0.1 * raw) / 0.9
    for rank_position, idx in enumerate(unheld[:10], start=1):
        stock = stocks[idx]
        candidates.append(
            {
                "rank_position": rank_position,
                "stock_code": stock,
                "name": str(names.loc[stock, "name"]) if stock in names.index else "",
                "market": str(names.loc[stock, "market"]) if stock in names.index else "",
                "strategy_score": float(score[t, idx]),
                "rank_10d": float(raw[idx]),
                "rank_10d_7d_mean": float(smooth[idx]),
                "rank_deterioration": float(raw[idx] - smooth[idx]),
                "amount_thousand": float(arrays["amount"][t, idx]),
                "turnover_rate": float(arrays["turnover_rate"][t, idx]),
                "theoretical_target_pct": float(target[t]),
                "status": "pending_buy_day_hard_gate",
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame.to_csv(OUT / "latest_research_candidates_max_positions16.csv", index=False, encoding="utf-8-sig")

    holding_rows = []
    for idx, entry_t in sorted(holdings.items(), key=lambda pair: float(score[t, pair[0]]), reverse=True):
        stock = stocks[idx]
        holding_rows.append(
            {
                "stock_code": stock,
                "name": str(names.loc[stock, "name"]) if stock in names.index else "",
                "strategy_score": float(score[t, idx]) if np.isfinite(score[t, idx]) else None,
                "entry_signal_date": str(arrays["dates"][entry_t]),
                "age": int(t - entry_t),
            }
        )

    result = {
        "status": "research_only_pending_buy_day_hard_gate",
        "strategy_id": "v195_c635e77b350d28d5",
        "base_strategy_id": item["case_id"],
        "max_positions": 16,
        "signal_date": latest,
        "buy_date": "20260723",
        "buy_date_source": "next_weekday_estimate_l2_buy_day_market_not_available",
        "input_manifest": "quant/main/config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json",
        "input_manifest_sha256": sha256(core.MANIFESTS["10d"]),
        "market_momentum_10d": float(momentum[t]),
        "theoretical_target_pct": float(target[t]),
        "holdings_before_latest_signal": len(holdings),
        "theoretical_holdings": holding_rows,
        "theoretical_sell_count": len(theoretical_sells),
        "theoretical_sells": theoretical_sells,
        "candidate_count": len(candidates),
        "primary_candidate": candidates[0] if candidates else None,
        "l7_execution_allowed": False,
        "production_asset_written": False,
        "notes": "买入日不复权开盘价、涨停、停牌、ST/风险警示等硬门槛尚未验证；不得作为正式交易指令。",
    }
    (OUT / "latest_research_selection_max_positions16.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

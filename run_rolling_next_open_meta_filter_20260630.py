from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import duckdb
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "timing1d_scores.duckdb"
SCORE_TABLE = "score_timing_55_30_10_05_g1p4_top5_dyn"

OUT_SIGNAL_DIR = REPORT_DIR / "rolling_next_open_meta_filter_signals"
OUT_LOG_DIR = REPORT_DIR / "rolling_next_open_meta_filter_logs"
OUT_CSV = REPORT_DIR / "rolling_next_open_meta_filter_20260630.csv"
OUT_JSON = REPORT_DIR / "rolling_next_open_meta_filter_20260630.json"


BASES: dict[str, dict[str, Any]] = {
    "top3_full": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top3_s1000_cap99_h2m3_ddoff.csv",
        "topn": 3,
        "max_positions": 3,
    },
    "top5_full": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top5_s1000_cap99_h2m3_ddoff.csv",
        "topn": 5,
        "max_positions": 5,
    },
}


CASES: list[dict[str, Any]] = [
    {
        "name": "rnom_top3_soft_h2m3",
        "base": "top3_full",
        "mode": "soft",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top3_hard_h2m3",
        "base": "top3_full",
        "mode": "hard",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top3_daysoft_h2m3",
        "base": "top3_full",
        "mode": "daysoft",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top3_fast_h1m2",
        "base": "top3_full",
        "mode": "soft",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 3,
    },
    {
        "name": "rnom_top5_soft_h2m3",
        "base": "top5_full",
        "mode": "soft",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top5_hard_h2m3",
        "base": "top5_full",
        "mode": "hard",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top5_daysoft_h2m3",
        "base": "top5_full",
        "mode": "daysoft",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top5_fast_h1m2",
        "base": "top5_full",
        "mode": "soft",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 5,
    },
    {
        "name": "rnom_top5_select3_soft_h2m3",
        "base": "top5_full",
        "mode": "soft",
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top5_select3_hard_h2m3",
        "base": "top5_full",
        "mode": "hard",
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top5_select2_soft_h2m3",
        "base": "top5_full",
        "mode": "soft",
        "select_topn": 2,
        "max_positions": 2,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "rnom_top5_select3_fast_h1m2",
        "base": "top5_full",
        "mode": "soft",
        "select_topn": 3,
        "max_positions": 3,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 3,
    },
]


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _extract_indicator(log_file: Path) -> dict[str, Any] | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "GM_BACKTEST_INDICATOR:" not in line:
            continue
        payload = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
    return None


def _trade_calendar() -> list[str]:
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        return [
            row[0]
            for row in con.execute(
                """
                select distinct trade_date
                from STOCK_DAILY_DATA
                where trade_date between '20220601' and '20260630'
                order by trade_date
                """
            ).fetchall()
        ]
    finally:
        con.close()


def _market_cache(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, str | None]]:
    trade_days = _trade_calendar()
    next_map = {day: trade_days[i + 1] if i + 1 < len(trade_days) else None for i, day in enumerate(trade_days)}
    keys: set[tuple[str, str]] = set()
    for row in rows:
        stock_code = str(row.get("stock_code") or "")
        signal_date = str(row.get("signal_date") or "")
        buy_date = str(row.get("buy_date") or "")
        if stock_code and signal_date:
            keys.add((stock_code, signal_date))
        if stock_code and buy_date:
            keys.add((stock_code, buy_date))
            next_date = next_map.get(buy_date)
            if next_date:
                keys.add((stock_code, next_date))
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        out: dict[tuple[str, str], dict[str, float]] = {}
        for stock_code, trade_date in sorted(keys):
            row = con.execute(
                """
                select open, close, pre_close, pct_chg, amount, turnover_rate, total_mv, atr_qfq, list_date
                from STOCK_DAILY_DATA
                where stock_code=? and trade_date=?
                """,
                (stock_code, trade_date),
            ).fetchone()
            if not row:
                continue
            open_price, close_price, pre_close, pct_chg, amount, turnover_rate, total_mv, atr_qfq, list_date = row
            item: dict[str, float] = {}
            for key, value in [
                ("open", open_price),
                ("close", close_price),
                ("pre_close", pre_close),
                ("pct_chg", pct_chg),
                ("amount", amount),
                ("turnover_rate", turnover_rate),
                ("total_mv", total_mv),
                ("atr_qfq", atr_qfq),
            ]:
                if value is not None:
                    item[key] = float(value)
            if list_date:
                try:
                    signal_dt = datetime_module.datetime.strptime(trade_date, "%Y%m%d")
                    list_dt = datetime_module.datetime.strptime(str(list_date), "%Y%m%d")
                    item["list_age_days"] = float((signal_dt - list_dt).days)
                except ValueError:
                    pass
            out[(stock_code, trade_date)] = item
        return out, next_map
    finally:
        con.close()


def _bucket(value: float | None, bounds: list[float], labels: list[str]) -> str:
    if value is None:
        return "missing"
    for bound, label in zip(bounds, labels):
        if value < bound:
            return label
    return labels[-1]


def _feature_keys(row: dict[str, Any]) -> list[str]:
    rank = _float(row.get("rank"))
    pred = _float(row.get("pred_prob"))
    rank_1d = _float(row.get("rank_1d"))
    rank_3d = _float(row.get("rank_3d"))
    rank_5d = _float(row.get("rank_5d"))
    rank_10d = _float(row.get("rank_10d"))
    amount = _float(row.get("amount"))
    turnover = _float(row.get("turnover_rate"))
    total_mv = _float(row.get("total_mv"))
    atr = _float(row.get("atr_qfq"))
    pct = _float(row.get("pct_chg"))
    prev_pct = _float(row.get("prev_pct_chg"))
    two_day = _float(row.get("two_day_ret"))
    target = _float(row.get("target_pct"))
    list_age = _float(row.get("list_age_days"))
    keys = [
        f"rank:{int(rank) if rank is not None else 'missing'}",
        f"pred:{_bucket(pred, [0.90, 0.94, 0.97, 0.985], ['lt90', '90_94', '94_97', '97_985', 'ge985'])}",
        f"rank1d:{_bucket(rank_1d, [0.20, 0.50, 0.80, 0.95], ['lt20', '20_50', '50_80', '80_95', 'ge95'])}",
        f"rank3d:{_bucket(rank_3d, [0.50, 0.80, 0.95, 0.985], ['lt50', '50_80', '80_95', '95_985', 'ge985'])}",
        f"rank5d:{_bucket(rank_5d, [0.50, 0.80, 0.95, 0.985], ['lt50', '50_80', '80_95', '95_985', 'ge985'])}",
        f"rank10d:{_bucket(rank_10d, [0.50, 0.80, 0.95, 0.985], ['lt50', '50_80', '80_95', '95_985', 'ge985'])}",
        f"amount:{_bucket(amount, [120000, 300000, 800000, 2000000], ['lt12w', '12_30w', '30_80w', '80_200w', 'ge200w'])}",
        f"turnover:{_bucket(turnover, [2, 6, 15, 40], ['lt2', '2_6', '6_15', '15_40', 'ge40'])}",
        f"total_mv:{_bucket(total_mv, [300000, 800000, 3000000, 10000000], ['lt30e', '30_80e', '80_300e', '300_1000e', 'ge1000e'])}",
        f"atr:{_bucket(atr, [0.03, 0.06, 0.10, 0.18], ['lt3p', '3_6p', '6_10p', '10_18p', 'ge18p'])}",
        f"pct:{_bucket(pct, [-6, -2, 2, 6], ['le_m6', 'm6_m2', 'm2_p2', 'p2_p6', 'ge_p6'])}",
        f"prev_pct:{_bucket(prev_pct, [-6, -2, 2, 6], ['le_m6', 'm6_m2', 'm2_p2', 'p2_p6', 'ge_p6'])}",
        f"two_day:{_bucket(two_day, [-0.08, -0.03, 0.03, 0.08], ['le_m8p', 'm8_m3p', 'm3_p3p', 'p3_p8p', 'ge_p8p'])}",
        f"target:{_bucket(target, [0.20, 0.40, 0.70, 0.95], ['lt20', '20_40', '40_70', '70_95', 'ge95'])}",
        f"list_age:{_bucket(list_age, [60, 180, 720, 1800], ['lt60', '60_180', '180_720', '720_1800', 'ge1800'])}",
    ]
    return keys


def _forward_next_open_ret(
    row: dict[str, Any],
    market: dict[tuple[str, str], dict[str, float]],
    next_map: dict[str, str | None],
) -> float | None:
    stock_code = str(row.get("stock_code") or "")
    buy_date = str(row.get("buy_date") or "")
    next_date = next_map.get(buy_date)
    if not stock_code or not buy_date or not next_date:
        return None
    buy = market.get((stock_code, buy_date), {})
    nxt = market.get((stock_code, next_date), {})
    buy_open = buy.get("open")
    next_open = nxt.get("open")
    if buy_open is None or next_open is None or buy_open <= 0:
        return None
    return next_open / buy_open - 1.0


def _rolling_expected(
    keys: list[str],
    stats: dict[str, list[float]],
    global_stats: list[float],
    min_count: int = 40,
    shrink: float = 80.0,
) -> tuple[float, int, int]:
    if not global_stats[1]:
        return 0.0, 0, 0
    global_mean = global_stats[0] / global_stats[1]
    values: list[float] = []
    used_count = 0
    for key in keys:
        total, count = stats.get(key, [0.0, 0.0])
        if count < min_count:
            continue
        mean = total / count
        weight = count / (count + shrink)
        values.append(mean * weight + global_mean * (1.0 - weight))
        used_count += int(count)
    if not values:
        return global_mean, 0, int(global_stats[1])
    return sum(values) / len(values), len(values), used_count


def _scale_from_expected(expected: float, mode: str, day_expected: float | None = None) -> tuple[float, str]:
    scale = 1.0
    tags: list[str] = []
    if mode in {"soft", "daysoft"}:
        if expected <= -0.004:
            scale *= 0.45
            tags.append("meta_le_m40bp")
        elif expected <= -0.002:
            scale *= 0.65
            tags.append("meta_le_m20bp")
        elif expected <= -0.001:
            scale *= 0.80
            tags.append("meta_le_m10bp")
        elif expected >= 0.003:
            scale *= 1.15
            tags.append("meta_ge_30bp")
        elif expected >= 0.0015:
            scale *= 1.08
            tags.append("meta_ge_15bp")
    elif mode == "hard":
        if expected <= -0.005:
            return 0.0, "skip_meta_le_m50bp"
        if expected <= -0.003:
            scale *= 0.35
            tags.append("meta_le_m30bp")
        elif expected <= -0.0015:
            scale *= 0.60
            tags.append("meta_le_m15bp")
        elif expected >= 0.003:
            scale *= 1.18
            tags.append("meta_ge_30bp")
    if mode == "daysoft" and day_expected is not None:
        if day_expected <= -0.003:
            scale *= 0.55
            tags.append("day_meta_le_m30bp")
        elif day_expected <= -0.0015:
            scale *= 0.75
            tags.append("day_meta_le_m15bp")
        elif day_expected >= 0.0025:
            scale *= 1.10
            tags.append("day_meta_ge_25bp")
    return max(0.0, min(scale, 1.25)), "|".join(tags) if tags else "base"


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = BASES[str(case["base"])]
    rows = list(csv.DictReader(base["signal_file"].open("r", encoding="utf-8-sig", newline="")))
    market, next_map = _market_cache(rows)
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row.get("signal_date") or "")].append(row)

    feature_stats: dict[str, list[float]] = {}
    global_stats = [0.0, 0.0]
    output_rows: list[dict[str, Any]] = []
    skipped = 0
    deselected = 0
    tag_counter: Counter[str] = Counter()
    expected_values: list[float] = []
    used_feature_counts: list[int] = []

    for signal_date in sorted(by_date):
        current = by_date[signal_date]
        row_expectations: list[tuple[dict[str, Any], float, int, int]] = []
        for row in current:
            keys = _feature_keys(row)
            expected, used_features, used_rows = _rolling_expected(keys, feature_stats, global_stats)
            row_expectations.append((row, expected, used_features, used_rows))
        day_expected = sum(item[1] for item in row_expectations) / len(row_expectations) if row_expectations else None

        day_candidates: list[tuple[float, int, str, dict[str, Any]]] = []
        for row, expected, used_features, used_rows in row_expectations:
            scale, tag = _scale_from_expected(expected, str(case["mode"]), day_expected)
            tag_counter[tag] += 1
            expected_values.append(expected)
            used_feature_counts.append(used_features)
            if scale <= 0:
                skipped += 1
                continue
            item = dict(row)
            base_target = _float(row.get("target_pct")) or 0.0
            original_rank = int(float(row.get("rank") or 999999))
            item["original_rank"] = str(original_rank)
            item["target_pct"] = f"{min(0.99, max(0.0, base_target * scale)):.5f}"
            item["holding_days"] = str(int(case.get("holding_days", 2)))
            item["max_holding_days"] = str(int(case.get("max_holding_days", 3)))
            item["score_exit_entry_ratio"] = f"{float(case.get('score_exit', 0.98)):.5f}"
            item["score_continue_entry_ratio"] = f"{float(case.get('score_continue', 0.99)):.5f}"
            item["min_holding_days_before_score_exit"] = "1"
            item["strategy_variant"] = str(case["name"])
            item["filter_name"] = str(case["name"])
            item["dynamic_hold_name"] = str(case["name"])
            item["next_open_meta_expected"] = f"{expected:.8f}"
            item["next_open_meta_day_expected"] = "" if day_expected is None else f"{day_expected:.8f}"
            item["next_open_meta_used_features"] = str(used_features)
            item["next_open_meta_used_rows"] = str(used_rows)
            item["next_open_meta_scale"] = f"{scale:.6f}"
            item["next_open_meta_tag"] = tag
            day_candidates.append((expected, original_rank, str(row.get("stock_code") or ""), item))

        if case.get("select_topn"):
            keep = int(case["select_topn"])
            selected = sorted(day_candidates, key=lambda value: (-value[0], value[1], value[2]))[:keep]
            deselected += max(0, len(day_candidates) - len(selected))
            for idx, (_, _, _, item) in enumerate(selected, start=1):
                item["rank"] = str(idx)
                item["next_open_meta_selected_rank"] = str(idx)
                output_rows.append(item)
        else:
            for _, _, _, item in day_candidates:
                output_rows.append(item)

        for row in current:
            outcome = _forward_next_open_ret(row, market, next_map)
            if outcome is None:
                continue
            global_stats[0] += outcome
            global_stats[1] += 1.0
            for key in _feature_keys(row):
                slot = feature_stats.setdefault(key, [0.0, 0.0])
                slot[0] += outcome
                slot[1] += 1.0

    output_rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, output_rows)
    counts = Counter(row["signal_date"] for row in output_rows)
    avg_expected = sum(expected_values) / len(expected_values) if expected_values else None
    avg_used_features = sum(used_feature_counts) / len(used_feature_counts) if used_feature_counts else None
    target_names_per_day = int(case.get("select_topn") or base["topn"])
    return output, {
        "signal_rows": len(output_rows),
        "signal_days": len(counts),
        "target_names_per_day": target_names_per_day,
        "days_below_target": sum(1 for value in counts.values() if value < target_names_per_day),
        "skipped_rows": skipped,
        "deselected_rows": deselected,
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "risk_tag_top": dict(tag_counter.most_common(8)),
        "avg_next_open_meta_expected": avg_expected,
        "avg_next_open_meta_used_features": avg_used_features,
        "rolling_target_note": "uses only prior signal-date historical buy-open-to-next-open returns; no current/future return is used for current signal scaling",
    }


def _run(case: dict[str, Any]) -> dict[str, Any]:
    base = BASES[str(case["base"])]
    signal_file, signal_meta = _make_signal(case)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_MAX_DAILY_SELLS": str(int(case.get("max_daily_sells", 1))),
                "GM_LIGHT_STOP_LOSS_PCT": "none",
                "GM_LOG_EXPOSURE": "1",
                "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
                "GM_SYNC_POSITIONS": "1",
                "GM_CASH_BUFFER": "0.99",
                "GM_VERBOSE_TRADES": "1",
                "GM_FORCE_SELL_MARKET_ORDER": "0",
                "GM_FORCE_BUY_MARKET_ORDER": "0",
                "GM_INTRADAY_RISK_MODE": "0",
                "GM_INTRADAY_REPLACE_BUY": "0",
                "GM_EQUITY_DD_RISK_MODE": "0",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
                "GM_SCORE_EXIT_ENTRY_RATIO": str(float(case.get("score_exit", 0.98))),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(float(case.get("score_continue", 0.99))),
                "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.995",
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
                "GM_RESIZE_HELD_ON_SIGNAL": "0",
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
                "GM_INDEX_RISK_EXIT_MODE": "0",
                "GM_BREADTH_RISK_EXIT_MODE": "0",
            }
        )
        cmd = [
            str(JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(int(case.get("max_positions", base["max_positions"]))),
            "--holding-days",
            str(int(case.get("holding_days", 2))),
            "--max-holding-days",
            str(int(case.get("max_holding_days", 3))),
            "--target-position-pct",
            "0.99",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            SCORE_TABLE,
            "--market-db",
            str(MARKET_DB),
            "--backtest-start",
            "2022-06-07 09:00:00",
            "--backtest-end",
            "2026-06-29 15:30:00",
            "--backtest-adjust",
            "none",
            "--backtest-initial-cash",
            "600000",
            "--backtest-slippage-ratio",
            "0",
            "--stop-loss-pct",
            "0.05",
            "--take-profit-pct",
            "0.08",
        ]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    annual = indicator.get("pnl_ratio_annual") if indicator else None
    sharpe = indicator.get("sharp_ratio") if indicator else None
    max_drawdown = indicator.get("max_drawdown") if indicator else None
    return {
        **case,
        **signal_meta,
        "returncode": returncode,
        "annual": annual,
        "annual_return": annual,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "target_hit_500_sharpe4_mdd40": bool(
            annual is not None and sharpe is not None and max_drawdown is not None and annual >= 5.0 and sharpe >= 4.0 and max_drawdown <= 0.40
        ),
        "signal_file": str(signal_file),
        "score_db": str(SCORE_DB),
        "score_table": SCORE_TABLE,
        "log_file": str(log_file),
        "note": "research-only rolling next-open meta filter; production unchanged",
    }


def main() -> None:
    for base in BASES.values():
        if not base["signal_file"].exists():
            raise FileNotFoundError(base["signal_file"])
    results: list[dict[str, Any]] = []
    for case in CASES:
        row = _run(case)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

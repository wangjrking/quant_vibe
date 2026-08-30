from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import importlib.util
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_top1_open_gap_refill_20260701"
BASE_SCRIPT = MAIN / "run_adaptive70w_latest_top3_frequency_grid_20260630.py"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("latest_l4_grid_base_top1_gap", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()


BUY_RULE = {"name": "cool2d20", "two_day_cap": 0.20, "combo_cap": None, "turnover_floor": None}
SELL_RULE = {
    "name": "h2m3_e097_c098_min1",
    "holding_days": 2,
    "max_holding_days": 3,
    "score_exit_ratio": 0.97,
    "score_continue_ratio": 0.98,
    "min_score_exit_days": 1,
    "day_drop_ratio": None,
}

BASE_TOP_CASES = [
    {"name": "top1_pos37", "topn": 1, "max_positions": 1, "target": 0.37},
    {"name": "top1_pos40", "topn": 1, "max_positions": 1, "target": 0.40},
]

CASES: list[dict[str, Any]] = [
    {"name": "baseline_pool5_pos37", "top_case": "top1_pos37", "pool_depth": 5},
    {"name": "refill_up015_pool5_pos37", "top_case": "top1_pos37", "pool_depth": 5, "skip_up_cut": 0.015},
    {"name": "refill_up020_pool5_pos37", "top_case": "top1_pos37", "pool_depth": 5, "skip_up_cut": 0.020},
    {"name": "refill_up030_pool5_pos37", "top_case": "top1_pos37", "pool_depth": 5, "skip_up_cut": 0.030},
    {"name": "refill_up015_pool10_pos37", "top_case": "top1_pos37", "pool_depth": 10, "skip_up_cut": 0.015},
    {"name": "refill_up020_pool10_pos37", "top_case": "top1_pos37", "pool_depth": 10, "skip_up_cut": 0.020},
    {"name": "refill_up030_pool10_pos37", "top_case": "top1_pos37", "pool_depth": 10, "skip_up_cut": 0.030},
    {"name": "baseline_pool5_pos40", "top_case": "top1_pos40", "pool_depth": 5},
    {"name": "refill_up015_pool5_pos40", "top_case": "top1_pos40", "pool_depth": 5, "skip_up_cut": 0.015},
    {"name": "refill_up020_pool5_pos40", "top_case": "top1_pos40", "pool_depth": 5, "skip_up_cut": 0.020},
    {"name": "refill_up030_pool5_pos40", "top_case": "top1_pos40", "pool_depth": 5, "skip_up_cut": 0.030},
    {"name": "refill_up015_pool10_pos40", "top_case": "top1_pos40", "pool_depth": 10, "skip_up_cut": 0.015},
    {"name": "refill_up020_pool10_pos40", "top_case": "top1_pos40", "pool_depth": 10, "skip_up_cut": 0.020},
    {"name": "refill_up030_pool10_pos40", "top_case": "top1_pos40", "pool_depth": 10, "skip_up_cut": 0.030},
]

FULL_SLICE = ("full", "2022-06-07 09:00:00", "2026-06-29 15:30:00")
CHECK_SLICES = [
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-29 15:30:00"),
    ("recent60", "2026-04-01 09:00:00", "2026-06-29 15:30:00"),
]


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


def _top_case(case_name: str) -> dict[str, Any]:
    return next(item for item in BASE_TOP_CASES if item["name"] == case_name)


def _cooldown_condition() -> str:
    return f"coalesce(two_day_ret, -999.0) < {float(BUY_RULE['two_day_cap'])}"


def _candidate_pool(case: dict[str, Any]) -> list[dict[str, Any]]:
    top_case = _top_case(case["top_case"])
    score_table = base.BASE_SCORE_TABLE
    pool_depth = int(case["pool_depth"])
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{base.MARKET_DUCKDB.as_posix()}' AS marketdb (READ_ONLY)")
        con.execute(f"ATTACH '{base.SCORE_DB.as_posix()}' AS scoredb (READ_ONLY)")
        sql = f"""
        WITH score_base AS (
            SELECT trade_date, stock_code, pred_prob
            FROM scoredb."{score_table}"
        ),
        market_with_prev AS (
            SELECT
                trade_date,
                stock_code,
                name,
                amount,
                turnover_rate,
                total_mv,
                atr_qfq,
                close,
                open,
                pre_close,
                pct_chg,
                ST_TYPE,
                ST_TYPE_name,
                limit_times,
                lag(pct_chg, 1) OVER (PARTITION BY stock_code ORDER BY trade_date) AS prev_pct_chg
            FROM marketdb.STOCK_DAILY_DATA
        ),
        market_dates AS (
            SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA
        ),
        next_dates AS (
            SELECT
                trade_date AS signal_date,
                lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                max(trade_date) OVER () AS latest_market_date
            FROM market_dates
        ),
        scored AS (
            SELECT
                sb.trade_date AS signal_date,
                nd.buy_date,
                nd.latest_market_date,
                sb.stock_code,
                md.name,
                md.amount,
                md.turnover_rate,
                md.total_mv,
                md.atr_qfq,
                md.close,
                md.pct_chg,
                md.prev_pct_chg,
                ((1 + try_cast(md.pct_chg AS DOUBLE) / 100.0)
                 * (1 + try_cast(md.prev_pct_chg AS DOUBLE) / 100.0) - 1.0) AS two_day_ret,
                sb.pred_prob AS entry_score
            FROM score_base sb
            LEFT JOIN market_with_prev md
              ON sb.trade_date = md.trade_date
             AND sb.stock_code = md.stock_code
            LEFT JOIN next_dates nd
              ON sb.trade_date = nd.signal_date
            WHERE NOT (sb.stock_code LIKE '%.BJ' OR substr(sb.stock_code, 1, 1) IN ('4', '8'))
              AND NOT (
                upper(coalesce(md.name, '')) LIKE 'ST%%'
                OR upper(coalesce(md.name, '')) LIKE '*ST%%'
                OR coalesce(md.ST_TYPE_name, '') LIKE '%风险%'
                OR (try_cast(md.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(md.ST_TYPE AS DOUBLE) <> 0)
              )
              AND NOT (
                instr(coalesce(md.name, ''), '退市') > 0
                OR coalesce(md.name, '') LIKE '退%%'
                OR coalesce(md.name, '') LIKE '%%退'
              )
              AND coalesce(try_cast(md.limit_times AS DOUBLE), 0.0) = 0.0
              AND try_cast(md.amount AS DOUBLE) >= {float(base.ENTRY['amount_min'])}
              AND try_cast(md.total_mv AS DOUBLE) >= {float(base.ENTRY['total_mv_min'])}
              AND try_cast(md.close AS DOUBLE) <= {float(base.ENTRY['price_cap'])}
              AND {_cooldown_condition()}
        ),
        with_buy_checks AS (
            SELECT
                s.*,
                bm.open AS buy_open,
                bm.close AS buy_close,
                bm.pre_close AS buy_pre_close,
                bm.name AS buy_name,
                CASE
                    WHEN bm.stock_code IS NULL THEN 0
                    WHEN bm.stock_code LIKE '%.BJ' THEN 0
                    WHEN upper(coalesce(bm.name, '')) LIKE 'ST%%' THEN 0
                    WHEN upper(coalesce(bm.name, '')) LIKE '*ST%%' THEN 0
                    WHEN coalesce(bm.ST_TYPE_name, '') LIKE '%风险%' THEN 0
                    WHEN try_cast(bm.ST_TYPE AS DOUBLE) IS NOT NULL AND try_cast(bm.ST_TYPE AS DOUBLE) <> 0 THEN 0
                    WHEN instr(coalesce(bm.name, ''), '退市') > 0 THEN 0
                    WHEN coalesce(try_cast(bm.limit_times AS DOUBLE), 0.0) > 0.0 THEN 0
                    WHEN try_cast(bm.pre_close AS DOUBLE) IS NULL OR try_cast(bm.open AS DOUBLE) IS NULL THEN 0
                    WHEN try_cast(bm.pre_close AS DOUBLE) <= 0 OR try_cast(bm.open AS DOUBLE) <= 0 THEN 0
                    WHEN try_cast(bm.open AS DOUBLE) >= try_cast(bm.pre_close AS DOUBLE) * (
                        1.0 + CASE WHEN substr(s.stock_code, 1, 3) IN ('300', '301', '688') THEN 0.20 ELSE 0.10 END
                    ) * 0.995 THEN 0
                    WHEN try_cast(bm.open AS DOUBLE) > {float(base.ENTRY['price_cap'])} THEN 0
                    ELSE 1
                END AS buy_day_ok
            FROM scored s
            LEFT JOIN marketdb.STOCK_DAILY_DATA bm
              ON s.buy_date = bm.trade_date
             AND s.stock_code = bm.stock_code
        )
        SELECT
            signal_date,
            buy_date,
            stock_code,
            name,
            entry_score,
            amount,
            turnover_rate,
            total_mv,
            atr_qfq,
            close,
            pct_chg,
            prev_pct_chg,
            two_day_ret,
            buy_open,
            buy_close,
            buy_pre_close,
            buy_day_ok,
            latest_market_date,
            row_number() OVER (PARTITION BY signal_date ORDER BY entry_score DESC, stock_code ASC) AS pool_rank
        FROM with_buy_checks
        WHERE buy_day_ok = 1
        """
        cur = con.execute(sql)
        cols = [item[0] for item in cur.description]
        rows = [dict(zip(cols, record)) for record in cur.fetchall()]
    finally:
        con.close()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["signal_date"])].append(row)
    out: list[dict[str, Any]] = []
    for signal_date, day_rows in grouped.items():
        day_rows.sort(key=lambda row: (-float(row["entry_score"]), str(row["stock_code"])))
        out.extend(day_rows[:pool_depth])
    out.sort(key=lambda row: (str(row["signal_date"]), int(row["pool_rank"]), str(row["stock_code"])))
    return out


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    top_case = _top_case(case["top_case"])
    pool_rows = _candidate_pool(case)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        grouped[str(row["signal_date"])].append(row)
    out_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    shortage_examples: list[str] = []
    for signal_date, day_rows in sorted(grouped.items()):
        kept: list[dict[str, Any]] = []
        for row in sorted(day_rows, key=lambda item: (-float(item["entry_score"]), str(item["stock_code"]))):
            signal_close = _float(row.get("close"))
            buy_open = _float(row.get("buy_open"))
            gap = None
            if signal_close not in (None, 0) and buy_open is not None:
                gap = buy_open / signal_close - 1.0
            skip_up = _float(case.get("skip_up_cut"))
            if skip_up is not None and gap is not None and gap >= skip_up:
                reason_counts["skip_up_gap"] += 1
                continue
            kept.append(row)
            reason_counts["kept"] += 1
            if len(kept) >= int(top_case["topn"]):
                break
        if len(kept) < int(top_case["topn"]):
            shortage_examples.append(f"{signal_date}:{len(kept)}")
        for idx, row in enumerate(kept, start=1):
            stock_code = str(row["stock_code"])
            symbol = f"SHSE.{stock_code[:6]}" if stock_code.endswith(".SH") else f"SZSE.{stock_code[:6]}"
            signal_close = _float(row.get("close"))
            buy_open = _float(row.get("buy_open"))
            gap = None
            if signal_close not in (None, 0) and buy_open is not None:
                gap = buy_open / signal_close - 1.0
            out_rows.append(
                {
                    "signal_date": signal_date,
                    "buy_date": str(row["buy_date"]),
                    "symbol": symbol,
                    "stock_code": stock_code,
                    "name": row.get("name", ""),
                    "rank": idx,
                    "pred_prob": f"{float(row['entry_score']):.10f}",
                    "entry_score": f"{float(row['entry_score']):.10f}",
                    "amount": row.get("amount", ""),
                    "turnover_rate": row.get("turnover_rate", ""),
                    "total_mv": row.get("total_mv", ""),
                    "atr_qfq": row.get("atr_qfq", ""),
                    "pct_chg": row.get("pct_chg", ""),
                    "prev_pct_chg": row.get("prev_pct_chg", ""),
                    "two_day_ret": row.get("two_day_ret", ""),
                    "buy_open_gap": "" if gap is None else f"{gap:.8f}",
                    "target_pct": f"{float(top_case['target']):.5f}",
                    "holding_days": str(int(SELL_RULE["holding_days"])),
                    "max_holding_days": str(int(SELL_RULE["max_holding_days"])),
                    "score_exit_entry_ratio": f"{float(SELL_RULE['score_exit_ratio']):.5f}",
                    "min_holding_days_before_score_exit": str(int(SELL_RULE["min_score_exit_days"])),
                    "score_continue_entry_ratio": f"{float(SELL_RULE['score_continue_ratio']):.5f}",
                    "signal_stop_loss_pct": "0.05000",
                    "signal_take_profit_pct": "0.08000",
                    "strategy_variant": case["name"],
                    "filter_name": BUY_RULE["name"],
                    "entry_weight_name": base.ENTRY["name"],
                    "dynamic_hold_name": SELL_RULE["name"],
                    "buy_day_market_available": "True",
                    "buy_day_hard_gate_complete": "True",
                    "buy_day_st_rejected": "False",
                    "buy_day_open_limit_up_rejected": "False",
                    "latest_market_date": row.get("latest_market_date", ""),
                }
            )
    output = REPORT_DIR / "signals" / f"{case['name']}.csv"
    _write_rows(output, out_rows)
    return output, {
        "signal_rows": len(out_rows),
        "signal_days": len({row["signal_date"] for row in out_rows}),
        "below_target_days": sum(1 for signal_date in grouped if sum(1 for row in out_rows if row["signal_date"] == signal_date) < int(top_case["topn"])),
        "reason_counts": dict(reason_counts),
        "shortage_examples": shortage_examples[:20],
    }


def _compare_with_base_signal(top_case: dict[str, Any], signal_file: Path) -> dict[str, Any]:
    meta = base._ensure_score_and_signal(top_case, BUY_RULE, SELL_RULE)
    orig_rows = list(csv.DictReader(Path(meta["signal_file"]).open("r", encoding="utf-8-sig")))
    new_rows = list(csv.DictReader(signal_file.open("r", encoding="utf-8-sig")))
    orig_keys = {(row["signal_date"], row["buy_date"], row["stock_code"], row["rank"]) for row in orig_rows}
    new_keys = {(row["signal_date"], row["buy_date"], row["stock_code"], row["rank"]) for row in new_rows}
    return {
        "orig_rows": len(orig_rows),
        "new_rows": len(new_rows),
        "common": len(orig_keys & new_keys),
        "orig_only": len(orig_keys - new_keys),
        "new_only": len(new_keys - orig_keys),
    }


def _run(case: dict[str, Any], signal_file: Path, tag: str, start: str, end: str) -> dict[str, Any]:
    top_case = _top_case(case["top_case"])
    log_file = REPORT_DIR / "logs" / f"{case['name']}_{tag}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(
            {
                "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
                "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
                "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.25",
                "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
                "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
                "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
                "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
                "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": f"{700000.0 * float(top_case['target']):.2f}",
                "GM_BREADTH_RISK_EXIT_MODE": "0",
                "GM_CASH_BUFFER": "0.99",
                "GM_DEFER_EXITS_WITHOUT_BUY_SIGNAL": "0",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
                "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
                "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
                "GM_EQUITY_DD_SOFT_SCALE": "0.80",
                "GM_EQUITY_DD_HARD_SCALE": "0.60",
                "GM_FORCE_BUY_MARKET_ORDER": "0",
                "GM_FORCE_SELL_MARKET_ORDER": "0",
                "GM_INDEX_RISK_EXIT_MODE": "0",
                "GM_INTRADAY_REPLACE_BUY": "0",
                "GM_INTRADAY_RISK_MODE": "0",
                "GM_LIGHT_STOP_LOSS_PCT": "none",
                "GM_LOG_EXPOSURE": "1",
                "GM_MAX_DAILY_SELLS": "0",
                "GM_OPEN_DAILY_SCORE_EXIT": "1",
                "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
                "GM_STOP_LOSS_PCT": "0.05",
                "GM_SYNC_POSITIONS": "1",
                "GM_TAKE_PROFIT_PCT": "0.08",
                "GM_VERBOSE_TRADES": "1",
            }
        )
        cmd = [
            str(base.JUEJIN_PYTHON),
            str(MAIN / "run_juejin_signal_backtest.py"),
            "--strategy-dir",
            str(base.STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(int(top_case["topn"])),
            "--holding-days",
            str(int(SELL_RULE["holding_days"])),
            "--max-holding-days",
            str(int(SELL_RULE["max_holding_days"])),
            "--target-position-pct",
            str(float(top_case["target"])),
            "--score-db",
            str(base.SCORE_DB),
            "--score-table",
            str(base.BASE_SCORE_TABLE),
            "--market-db",
            str(base.MARKET_DB),
            "--score-exit-entry-ratio",
            str(float(SELL_RULE["score_exit_ratio"])),
            "--score-continue-entry-ratio",
            str(float(SELL_RULE["score_continue_ratio"])),
            "--min-holding-days-before-score-exit",
            str(int(SELL_RULE["min_score_exit_days"])),
            "--backtest-start",
            start,
            "--backtest-end",
            end,
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
    return {
        "slice": tag,
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def _summary_row(case: dict[str, Any], signal_meta: dict[str, Any], compare_meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_case = _top_case(case["top_case"])
    out = {
        "name": case["name"],
        "top_case": case["top_case"],
        "target_pct_each": top_case["target"],
        "pool_depth": case["pool_depth"],
        "skip_up_cut": case.get("skip_up_cut"),
        "signal_rows": signal_meta["signal_rows"],
        "signal_days": signal_meta["signal_days"],
        "below_target_days": signal_meta["below_target_days"],
        "reason_counts": json.dumps(signal_meta["reason_counts"], ensure_ascii=False, sort_keys=True),
        "shortage_examples": json.dumps(signal_meta["shortage_examples"], ensure_ascii=False),
        "baseline_common": compare_meta["common"],
        "baseline_orig_only": compare_meta["orig_only"],
        "baseline_new_only": compare_meta["new_only"],
    }
    by_slice = {row["slice"]: row for row in rows}
    for tag in [FULL_SLICE[0], *[item[0] for item in CHECK_SLICES]]:
        item = by_slice.get(tag, {})
        out[f"{tag}_annual"] = item.get("annual")
        out[f"{tag}_sharpe"] = item.get("sharpe")
        out[f"{tag}_max_drawdown"] = item.get("max_drawdown")
        out[f"{tag}_win_ratio"] = item.get("win_ratio")
        out[f"{tag}_open_count"] = item.get("open_count")
    return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for case in CASES:
        signal_file, signal_meta = _make_signal(case)
        compare_meta = _compare_with_base_signal(_top_case(case["top_case"]), signal_file)
        rows = [_run(case, signal_file, *FULL_SLICE)]
        rows.extend(_run(case, signal_file, tag, start, end) for tag, start, end in CHECK_SLICES)
        detail_rows.extend([{**row, **signal_meta, **compare_meta, "signal_file": str(signal_file)} for row in rows])
        summary_rows.append(_summary_row(case, signal_meta, compare_meta, rows))
        summary_rows.sort(
            key=lambda row: (
                float(row["full_annual"]) if row["full_annual"] not in (None, "") else -999.0,
                float(row["full_sharpe"]) if row["full_sharpe"] not in (None, "") else -999.0,
            ),
            reverse=True,
        )
        _write_rows(REPORT_DIR / "summary.csv", summary_rows)
        _write_rows(REPORT_DIR / "detail.csv", detail_rows)
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "full_annual": rows[0].get("annual"),
                    "full_sharpe": rows[0].get("sharpe"),
                    "baseline_common": compare_meta["common"],
                    "baseline_orig_only": compare_meta["orig_only"],
                    "baseline_new_only": compare_meta["new_only"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_dir": str(REPORT_DIR), "case_count": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

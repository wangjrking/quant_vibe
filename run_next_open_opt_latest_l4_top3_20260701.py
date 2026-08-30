from __future__ import annotations

import ast
import csv
import datetime as dt
import json
import math
import os
import duckdb
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
SOURCE_REPORT = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
REPORT_DIR = DATA / "reports" / "strategy_agent_next_open_opt_latest_l4_top3_20260701"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
BASE_SIGNAL = SOURCE_REPORT / "signals" / "w72_23_05_amt150_mv30_top3_pos25_cool2d20_h3m5_e096_c097_min1.csv"
SCORE_DB = SOURCE_REPORT / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"

OUT_SIGNAL_DIR = REPORT_DIR / "signals"
OUT_LOG_DIR = REPORT_DIR / "logs"
OUT_CSV = REPORT_DIR / "summary.csv"
OUT_JSON = REPORT_DIR / "summary.json"
OUT_DETAIL = REPORT_DIR / "detail.csv"
OUT_REPORT = REPORT_DIR / "report.md"

FULL_SLICE = ("full", "2022-06-07 09:00:00", "2026-06-29 15:30:00")
CHECK_SLICES = [
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-29 15:30:00"),
    ("recent60", "2026-04-01 09:00:00", "2026-06-29 15:30:00"),
]
ADAPTIVE_TOTAL_CAPITAL = 700000.0


BASE = {
    "name": "latest_l4_top3_pos25_cool2d20_h3m5",
    "path": BASE_SIGNAL,
    "topn": 3,
    "target_cap": 0.25,
    "holding_days": 3,
    "max_holding_days": 5,
    "score_exit": 0.96,
    "score_continue": 0.97,
    "min_hold_before_score_exit": 1,
    "max_daily_sells": 0,
}


CASES: list[dict[str, Any]] = [
    {
        "name": "baseline_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "max_daily_sells": 0,
    },
    {
        "name": "baseline_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "min_hold_before_score_exit": 1,
        "max_daily_sells": 0,
    },
    {
        "name": "skip_u10_d9_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "max_daily_sells": 0,
        "skip_up_cut": 0.010,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "skip_u9_d9_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "max_daily_sells": 0,
        "skip_up_cut": 0.009,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "skip_u12_d9_h3m5",
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.96,
        "score_continue": 0.97,
        "min_hold_before_score_exit": 1,
        "max_daily_sells": 0,
        "skip_up_cut": 0.012,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "skip_u10_d9_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "min_hold_before_score_exit": 1,
        "max_daily_sells": 0,
        "skip_up_cut": 0.010,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "skip_u9_d9_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "min_hold_before_score_exit": 1,
        "max_daily_sells": 0,
        "skip_up_cut": 0.009,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "skip_u12_d9_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "min_hold_before_score_exit": 1,
        "max_daily_sells": 0,
        "skip_up_cut": 0.012,
        "skip_deep_down_cut": -0.09,
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
            try:
                return eval(payload, {"__builtins__": {}}, {"datetime": dt})
            except Exception:
                return None
    return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_market(keys: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    if not keys:
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    con = duckdb.connect(str(MARKET_DB), read_only=True)
    try:
        for stock_code, trade_date in keys:
            row = con.execute(
                """
                select open, close, pre_close
                from STOCK_DAILY_DATA
                where stock_code=? and trade_date=?
                """,
                (stock_code, trade_date),
            ).fetchone()
            if row is None:
                continue
            out[(stock_code, trade_date)] = {
                "open": float(row[0]) if row[0] is not None else math.nan,
                "close": float(row[1]) if row[1] is not None else math.nan,
                "pre_close": float(row[2]) if row[2] is not None else math.nan,
            }
    finally:
        con.close()
    return out


def _enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        stock_code = str(row.get("stock_code") or "")
        signal_date = str(row.get("signal_date") or "")
        buy_date = str(row.get("buy_date") or "")
        if stock_code and signal_date:
            keys.add((stock_code, signal_date))
        if stock_code and buy_date:
            keys.add((stock_code, buy_date))
    market = _load_market(keys)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        stock_code = str(item.get("stock_code") or "")
        signal_date = str(item.get("signal_date") or "")
        buy_date = str(item.get("buy_date") or "")
        signal_row = market.get((stock_code, signal_date))
        buy_row = market.get((stock_code, buy_date))
        gap = None
        if signal_row and buy_row:
            signal_close = signal_row.get("close")
            buy_open = buy_row.get("open")
            if signal_close not in (None, 0) and buy_open is not None and math.isfinite(signal_close) and math.isfinite(buy_open):
                gap = buy_open / signal_close - 1.0
        item["buy_open_gap"] = "" if gap is None else f"{gap:.8f}"
        enriched.append(item)
    return enriched


def _adjust_target(case: dict[str, Any], row: dict[str, Any], base_cap: float) -> tuple[bool, float, str]:
    target = _float(row.get("target_pct")) or base_cap
    gap = _float(row.get("buy_open_gap"))
    if gap is not None:
        skip_up = _float(case.get("skip_up_cut"))
        if skip_up is not None and gap >= skip_up:
            return False, 0.0, "skip_up_gap"
        skip_down = _float(case.get("skip_deep_down_cut"))
        if skip_down is not None and gap <= skip_down:
            return False, 0.0, "skip_deep_down_gap"
    return True, target, "kept"


def _gap_bucket(gap: float | None) -> str:
    if gap is None:
        return "missing_gap"
    if gap <= -0.09:
        return "deep_down9"
    if gap <= -0.02:
        return "mid_down_band"
    if gap >= 0.015:
        return "up2"
    if gap >= 0.008:
        return "up1"
    return "other"


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    rows = _enrich_rows(_load_rows(BASE_SIGNAL))
    out_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    gap_bucket_counts: Counter[str] = Counter()
    for row in rows:
        gap = _float(row.get("buy_open_gap"))
        gap_bucket_counts[_gap_bucket(gap)] += 1
        keep, target, reason = _adjust_target(case, row, float(BASE["target_cap"]))
        reason_counts[reason] += 1
        if not keep:
            continue
        item = dict(row)
        item["target_pct"] = f"{target:.5f}"
        item["holding_days"] = str(int(case["holding_days"]))
        item["max_holding_days"] = str(int(case["max_holding_days"]))
        item["score_exit_entry_ratio"] = f"{float(case['score_exit']):.5f}"
        item["score_continue_entry_ratio"] = f"{float(case['score_continue']):.5f}"
        item["min_holding_days_before_score_exit"] = str(int(case["min_hold_before_score_exit"]))
        item["research_case"] = str(case["name"])
        out_rows.append(item)

    out_rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, out_rows)
    day_counts = Counter(row["signal_date"] for row in out_rows)
    below_target_days = sum(1 for count in day_counts.values() if count < int(BASE["topn"]))
    return output, {
        "signal_rows": len(out_rows),
        "signal_days": len(day_counts),
        "below_target_days": below_target_days,
        "reason_counts": dict(reason_counts),
        "gap_bucket_counts": dict(gap_bucket_counts),
    }


def _run_backtest(case: dict[str, Any], signal_file: Path, tag: str, start: str, end: str) -> dict[str, Any]:
    log_file = OUT_LOG_DIR / f"{case['name']}_{tag}.log"
    order_value = ADAPTIVE_TOTAL_CAPITAL * float(BASE["target_cap"])
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
            "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": f"{order_value:.2f}",
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
            "GM_MAX_DAILY_SELLS": str(int(case["max_daily_sells"])),
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_STOP_LOSS_PCT": "0.05",
            "GM_SYNC_POSITIONS": "1",
            "GM_TAKE_PROFIT_PCT": "0.08",
            "GM_VERBOSE_TRADES": "1",
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
        str(int(BASE["topn"])),
        "--holding-days",
        str(int(case["holding_days"])),
        "--max-holding-days",
        str(int(case["max_holding_days"])),
        "--target-position-pct",
        str(float(BASE["target_cap"])),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        str(float(case["score_exit"])),
        "--score-continue-entry-ratio",
        str(float(case["score_continue"])),
        "--min-holding-days-before-score-exit",
        str(int(case["min_hold_before_score_exit"])),
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
    indicator = _extract_indicator(log_file)
    return {
        "case_name": case["name"],
        "slice": tag,
        "returncode": proc.returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "log_file": str(log_file),
    }


def _summary_row(case: dict[str, Any], signal_meta: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_slice = {row["slice"]: row for row in rows}
    out = {
        "name": case["name"],
        "holding_days": case["holding_days"],
        "max_holding_days": case["max_holding_days"],
        "score_exit": case["score_exit"],
        "score_continue": case["score_continue"],
        "min_hold_before_score_exit": case["min_hold_before_score_exit"],
        "signal_rows": signal_meta["signal_rows"],
        "signal_days": signal_meta["signal_days"],
        "below_target_days": signal_meta["below_target_days"],
        "reason_counts": json.dumps(signal_meta["reason_counts"], ensure_ascii=False, sort_keys=True),
        "gap_bucket_counts": json.dumps(signal_meta["gap_bucket_counts"], ensure_ascii=False, sort_keys=True),
    }
    for tag in [FULL_SLICE[0], *[item[0] for item in CHECK_SLICES]]:
        item = by_slice.get(tag, {})
        out[f"{tag}_annual"] = item.get("annual")
        out[f"{tag}_sharpe"] = item.get("sharpe")
        out[f"{tag}_max_drawdown"] = item.get("max_drawdown")
        out[f"{tag}_win_ratio"] = item.get("win_ratio")
        out[f"{tag}_open_count"] = item.get("open_count")
    return out


def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    def f(key: str, default: float = -9999.0) -> float:
        try:
            value = row.get(key)
            return float(value) if value not in (None, "") else default
        except Exception:
            return default

    return (f("full_annual"), f("full_sharpe"), -f("full_max_drawdown", 9999.0))


def _write_report(summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 鏈€鏂?formal L4 Top3 娆℃棩寮€鐩樹紭鍖栫爺绌?",
        "",
        "## 璇存槑",
        "",
        "- 鏈疆鍙仛 research-only 鍥炴祴锛屼笉淇敼浠讳綍鐢熶骇绛栫暐鍙傛暟銆?",
        "- 杈撳叆淇″彿鏉ヨ嚜 `strategy_agent_latest_l4_top3_frequency_grid_20260630` 宸茶惤鍦扮殑 latest formal L4 鍊欓€変俊鍙枫€?",
        "- 鑷€傚簲婊戠偣璁㈠崟棰濇敼涓?`70W 鎬昏祫浜?* 25% 鍗曠エ浠撲綅 = 17.5W`銆?",
        "- 涓嶅惎鐢ㄧ洏涓鎺э紝鍙瘮杈?open_only 鍙ｅ緞涓嬬殑娆℃棩寮€鐩?gap 璺宠繃瑙勫垯涓庢寔鏈夋湡棰戠巼銆?",
        "",
        "## Top 缁撴灉",
        "",
        "| name | full annual | full sharpe | full mdd | ytd2026 annual | recent60 annual | signal_rows | below_target_days |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows[:8]:
        lines.append(
            "| {name} | {fa} | {fs} | {fm} | {ya} | {ra} | {sr} | {bd} |".format(
                name=row["name"],
                fa="" if row.get("full_annual") is None else f"{float(row['full_annual']):.6f}",
                fs="" if row.get("full_sharpe") is None else f"{float(row['full_sharpe']):.6f}",
                fm="" if row.get("full_max_drawdown") is None else f"{float(row['full_max_drawdown']):.6f}",
                ya="" if row.get("ytd2026_annual") is None else f"{float(row['ytd2026_annual']):.6f}",
                ra="" if row.get("recent60_annual") is None else f"{float(row['recent60_annual']):.6f}",
                sr=row.get("signal_rows", ""),
                bd=row.get("below_target_days", ""),
            )
        )
    lines.extend(
        [
            "",
            "## 缁撹鍙ｅ緞",
            "",
            "- 褰撳墠鐮旂┒鍙瘮杈冨悓涓€鏉?latest formal L4 -> Top3 -> open_only 鍙璺戦摼璺笂鐨勭浉瀵瑰彉鍖栥€?",
            "- 鑻ユ鏃ュ紑鐩樿拷楂樿烦杩囪兘鍚屾椂鎶崌骞村寲涓?Sharpe锛屼笖鍥炴挙涓嶆伓鍖栵紝鍒欒涓烘湁鏁堝€欓€夈€?",
            "- 鑻ョ粨鏋滀粛鏄庢樉浣庝簬鐩爣锛岃鏄庣摱棰堝凡缁忎粠鎵ц棰戠巼杞洖鍒板垎鏁版í鎴潰鏈韩銆?",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not BASE_SIGNAL.exists():
        raise FileNotFoundError(BASE_SIGNAL)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for case in CASES:
        signal_file, signal_meta = _make_signal(case)
        rows = [_run_backtest(case, signal_file, *FULL_SLICE)]
        rows.extend(_run_backtest(case, signal_file, tag, start, end) for tag, start, end in CHECK_SLICES)
        detail_rows.extend([{**row, **signal_meta, "signal_file": str(signal_file)} for row in rows])
        summary_rows.append(_summary_row(case, signal_meta, rows))
        summary_rows.sort(key=_sort_key, reverse=True)
        _write_rows(OUT_CSV, summary_rows)
        _write_rows(OUT_DETAIL, detail_rows)
        _write_json(OUT_JSON, summary_rows)
        print(
            json.dumps(
                {"case": case["name"], "full_annual": rows[0].get("annual"), "full_sharpe": rows[0].get("sharpe")},
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary_rows.sort(key=_sort_key, reverse=True)
    _write_rows(OUT_CSV, summary_rows)
    _write_rows(OUT_DETAIL, detail_rows)
    _write_json(OUT_JSON, summary_rows)
    _write_report(summary_rows)
    print(json.dumps({"report_dir": str(REPORT_DIR), "case_count": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

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
REPORT_DIR = DATA / "reports" / "strategy_agent_next_open_exec_focus_20260701"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
BASE_SIGNAL = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_dyn_mild_09_12_15_v20260630"
    / "signals"
    / "full_history_dyn_mild_09_12_15.csv"
)
SCORE_DB = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630" / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"

OUT_SIGNAL_DIR = REPORT_DIR / "signals"
OUT_LOG_DIR = REPORT_DIR / "logs"
OUT_CSV = REPORT_DIR / "summary.csv"
OUT_JSON = REPORT_DIR / "summary.json"
OUT_DETAIL = REPORT_DIR / "detail.csv"
OUT_REPORT = REPORT_DIR / "report.md"

FULL_SLICE = ("full", "2022-06-07 09:00:00", "2026-06-30 15:30:00")
CHECK_SLICES = [
    ("ytd2026", "2026-01-05 09:00:00", "2026-06-30 15:30:00"),
    ("recent60", "2026-04-01 09:00:00", "2026-06-30 15:30:00"),
]
MAX_BUY_DATE = "20260630"


BASE = {
    "name": "dyn_mild_prod_current",
    "path": BASE_SIGNAL,
    "topn": 5,
    "target_cap": 0.15,
    "open_daily_score_exit": 1,
    "max_daily_sells": 1,
}


CASES: list[dict[str, Any]] = [
    {
        "name": "baseline_current_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
    },
    {
        "name": "gap_skip_u15_d9_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "skip_up_cut": 0.015,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "gap_skip_u20_d9_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "skip_up_cut": 0.02,
        "skip_deep_down_cut": -0.09,
    },
    {
        "name": "gap_scale_u15_d9_band_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "up_cut": 0.015,
        "up_scale": 0.45,
        "skip_deep_down_cut": -0.09,
        "mid_down_low": -0.08,
        "mid_down_high": -0.025,
        "mid_down_scale": 1.12,
    },
    {
        "name": "gap_combo_r5d973_turn63_h2m3",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.97,
        "score_continue": 0.98,
        "max_daily_sells": 1,
        "up_cut": 0.015,
        "up_scale": 0.45,
        "skip_deep_down_cut": -0.09,
        "mid_down_low": -0.08,
        "mid_down_high": -0.025,
        "mid_down_scale": 1.12,
        "high_rank5d_cut": 0.973,
        "high_turnover_cut": 6.3,
        "combo_nonneg_gap_cut": 0.0,
        "high_rank_scale": 0.60,
    },
    {
        "name": "gap_combo_r5d973_turn63_fast",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.985,
        "score_continue": 0.995,
        "max_daily_sells": 3,
        "up_cut": 0.015,
        "up_scale": 0.40,
        "skip_deep_down_cut": -0.09,
        "mid_down_low": -0.08,
        "mid_down_high": -0.025,
        "mid_down_scale": 1.08,
        "high_rank5d_cut": 0.973,
        "high_turnover_cut": 6.3,
        "combo_nonneg_gap_cut": -0.002,
        "high_rank_scale": 0.55,
    },
    {
        "name": "gap_combo_r5d985_turn80_fast",
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.99,
        "score_continue": 1.0,
        "max_daily_sells": 3,
        "up_cut": 0.012,
        "up_scale": 0.35,
        "skip_deep_down_cut": -0.085,
        "mid_down_low": -0.07,
        "mid_down_high": -0.02,
        "mid_down_scale": 1.05,
        "high_rank5d_cut": 0.985,
        "high_turnover_cut": 8.0,
        "combo_nonneg_gap_cut": -0.005,
        "high_rank_scale": 0.45,
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
            return eval(payload, {"__builtins__": {}}, {"datetime": dt})
    return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if str(row.get("buy_date") or "") <= MAX_BUY_DATE]


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
        keys.add((stock_code, str(row.get("signal_date") or "")))
        keys.add((stock_code, str(row.get("buy_date") or "")))
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
    rank_5d = _float(row.get("rank_5d"))
    turnover = _float(row.get("turnover_rate"))

    if gap is not None:
        skip_up = _float(case.get("skip_up_cut"))
        if skip_up is not None and gap >= skip_up:
            return False, 0.0, "skip_up_gap"

        skip_down = _float(case.get("skip_deep_down_cut"))
        if skip_down is not None and gap <= skip_down:
            return False, 0.0, "skip_deep_down_gap"

        up_cut = _float(case.get("up_cut"))
        up_scale = _float(case.get("up_scale"))
        if up_cut is not None and up_scale is not None and gap >= up_cut:
            target *= up_scale

        mid_low = _float(case.get("mid_down_low"))
        mid_high = _float(case.get("mid_down_high"))
        mid_scale = _float(case.get("mid_down_scale"))
        if mid_low is not None and mid_high is not None and mid_scale is not None and mid_low <= gap <= mid_high:
            target *= mid_scale

    high_rank5d_cut = _float(case.get("high_rank5d_cut"))
    high_turnover_cut = _float(case.get("high_turnover_cut"))
    combo_nonneg_gap_cut = _float(case.get("combo_nonneg_gap_cut"))
    high_rank_scale = _float(case.get("high_rank_scale"))
    if (
        high_rank5d_cut is not None
        and high_turnover_cut is not None
        and high_rank_scale is not None
        and rank_5d is not None
        and turnover is not None
        and rank_5d >= high_rank5d_cut
        and turnover >= high_turnover_cut
    ):
        if gap is not None and combo_nonneg_gap_cut is not None and gap >= combo_nonneg_gap_cut:
            return False, 0.0, "skip_high_rank_turn_combo"
        target *= high_rank_scale

    target = max(0.0, min(float(base_cap), target))
    if target <= 0:
        return False, 0.0, "skip_zero_target"
    return True, target, "kept"


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base_cap = float(BASE["target_cap"])
    rows = _enrich_rows(_load_rows(Path(BASE["path"])))
    out_rows: list[dict[str, Any]] = []
    reason_counts = Counter()
    gap_bucket_counts = Counter()
    for row in rows:
        gap = _float(row.get("buy_open_gap"))
        if gap is None:
            gap_bucket_counts["missing_gap"] += 1
        elif gap >= 0.02:
            gap_bucket_counts["up2"] += 1
        elif gap >= 0.01:
            gap_bucket_counts["up1"] += 1
        elif gap <= -0.09:
            gap_bucket_counts["deep_down9"] += 1
        elif gap <= -0.025:
            gap_bucket_counts["mid_down_band"] += 1
        else:
            gap_bucket_counts["other"] += 1

        keep, target, reason = _adjust_target(case, row, base_cap)
        reason_counts[reason] += 1
        if not keep:
            continue
        item = dict(row)
        item["target_pct"] = f"{target:.5f}"
        item["holding_days"] = str(int(case["holding_days"]))
        item["max_holding_days"] = str(int(case["max_holding_days"]))
        item["score_exit_entry_ratio"] = f"{float(case['score_exit']):.5f}"
        item["score_continue_entry_ratio"] = f"{float(case['score_continue']):.5f}"
        item["strategy_variant"] = str(case["name"])
        item["filter_name"] = str(case["name"])
        item["dynamic_hold_name"] = str(case["name"])
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
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": "1",
            "GM_MAX_DAILY_SELLS": str(int(case["max_daily_sells"])),
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
            "GM_EQUITY_DD_RISK_MODE": "1",
            "GM_EQUITY_DD_RESIZE_EXISTING": "0",
            "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
            "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
            "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
            "GM_EQUITY_DD_SOFT_SCALE": "0.80",
            "GM_EQUITY_DD_HARD_SCALE": "0.60",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
            "GM_SCORE_EXIT_ENTRY_RATIO": str(case["score_exit"]),
            "GM_SCORE_CONTINUE_ENTRY_RATIO": str(case["score_continue"]),
            "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.99",
            "GM_ADAPTIVE_LIQUIDITY_SLIPPAGE": "1",
            "GM_ADAPTIVE_SLIPPAGE_ORDER_VALUE": "700000",
            "GM_ADAPTIVE_SLIPPAGE_BASE": "0.0015",
            "GM_ADAPTIVE_SLIPPAGE_IMPACT_COEF": "0.0200",
            "GM_ADAPTIVE_SLIPPAGE_CAP": "0.0065",
            "GM_ADAPTIVE_SLIPPAGE_AMOUNT_UNIT": "1000",
            "GM_ADAPTIVE_BUY_SLIPPAGE_MULT": "1.0",
            "GM_ADAPTIVE_SELL_SLIPPAGE_MULT": "0.0",
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
        "max_daily_sells": case["max_daily_sells"],
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
        "# 娆℃棩寮€鐩樻墽琛屼紭鍖栬仛鐒﹀洖娴?",
        "",
        "## 璇存槑",
        "",
        "- 鏈疆鍙仛 research-only 鍥炴祴锛屼笉淇敼浠讳綍鐢熶骇绛栫暐鍙傛暟銆?",
        "- 杈撳叆淇″彿鍩轰簬褰撳墠鐢熶骇绛栫暐 `prod_dyn_mild_09_12_15_v20260630` 鍏ㄥ巻鍙蹭俊鍙凤紝鍓旈櫎 `buy_date>20260630` 鐨勬湭瀹屾垚鏍锋湰銆?",
        "- 鐩爣鏄洿缁曚拱鍏ユ棩寮€鐩?gap 涓庢洿蹇崠鍑洪鐜囷紝娴嬭瘯鏄惁鑳藉湪淇濇寔楂樺勾鍖栫殑鍚屾椂鏀瑰杽 Sharpe 鍜屽洖鎾ゃ€?",
        "",
        "## Top 缁撴灉",
        "",
        "| name | full annual | full sharpe | full mdd | ytd2026 annual | recent60 annual | signal_rows | below_target_days |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows[:6]:
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
            "- 鑻ユ煇鐗堟湰浠呮彁鍗?Sharpe 浣嗘妸骞村寲鏄庢樉鍘嬪埌 `5.0` 浠ヤ笅锛屽垯涓嶇畻婊¤冻鏈疆涓荤洰鏍囥€?",
            "- 鑻ユ煇鐗堟湰閫氳繃璺宠繃杩囧淇″彿鎹㈠彇鎸囨爣鏀瑰杽锛岄渶瑕佸悓鏃剁湅 `below_target_days` 鍜屼俊鍙疯鐩栦笅闄嶅箙搴︺€?",
            "- 鏈姤鍛婂彧浣跨敤涔板叆鏃ュ彲瑙傚療鐨勫紑鐩?gap锛屼互鍙婁俊鍙锋棩宸茬煡鐨?`rank_5d` / `turnover_rate` 鍋氱爺绌讹紝涓嶄娇鐢ㄦ湭鏉ユ敹鐩婄洿鎺ヤ笅瑙勫垯銆?",
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
        print(json.dumps({"case": case["name"], "full_annual": rows[0].get("annual"), "full_sharpe": rows[0].get("sharpe")}, ensure_ascii=False), flush=True)

    summary_rows.sort(key=_sort_key, reverse=True)
    _write_rows(OUT_CSV, summary_rows)
    _write_rows(OUT_DETAIL, detail_rows)
    _write_json(OUT_JSON, summary_rows)
    _write_report(summary_rows)
    print(json.dumps({"report_dir": str(REPORT_DIR), "case_count": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

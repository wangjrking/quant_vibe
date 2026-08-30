from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import subprocess
from collections import Counter, deque
from pathlib import Path
from typing import Any

from run_rolling_next_open_meta_filter_20260630 import (
    _float,
    _forward_next_open_ret,
    _market_cache,
    _write_rows,
)


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
STRATEGY_DIR = DATA / "reports" / "strategy_agent_dynamic_slippage_70w_20260628" / "code_snapshot_dynamic_slippage"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = REPORT_DIR / "scores" / "timing1d_scores.duckdb"
SCORE_TABLE = "score_timing_55_30_10_05_g1p4_top5_dyn"

OUT_SIGNAL_DIR = REPORT_DIR / "next_open_regime_state_signals"
OUT_LOG_DIR = REPORT_DIR / "next_open_regime_state_logs"
OUT_CSV = REPORT_DIR / "next_open_regime_state_probe_20260630.csv"
OUT_JSON = REPORT_DIR / "next_open_regime_state_probe_20260630.json"
OUT_MD = REPORT_DIR / "next_open_regime_state_probe_summary_20260630.md"


BASES: dict[str, dict[str, Any]] = {
    "top3_full": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top3_s1000_cap99_h2m3_ddoff.csv",
        "topn": 3,
        "max_positions": 3,
        "target_position_pct": 0.99,
    },
    "top5_full": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top5_s1000_cap99_h2m3_ddoff.csv",
        "topn": 5,
        "max_positions": 5,
        "target_position_pct": 0.99,
    },
}


CASES: list[dict[str, Any]] = [
    {
        "name": "nors_top3_w20_soft_h2m3",
        "base": "top3_full",
        "window": 20,
        "bad_mean": -0.003,
        "bad_win": 0.42,
        "bad_scale": 0.45,
        "warn_mean": 0.000,
        "warn_win": 0.48,
        "warn_scale": 0.70,
        "good_mean": 0.004,
        "good_win": 0.56,
        "good_scale": 1.05,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "nors_top3_w40_soft_h2m3",
        "base": "top3_full",
        "window": 40,
        "bad_mean": -0.002,
        "bad_win": 0.44,
        "bad_scale": 0.45,
        "warn_mean": 0.001,
        "warn_win": 0.50,
        "warn_scale": 0.75,
        "good_mean": 0.004,
        "good_win": 0.56,
        "good_scale": 1.05,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "nors_top3_w40_hard_h2m3",
        "base": "top3_full",
        "window": 40,
        "bad_mean": -0.002,
        "bad_win": 0.44,
        "bad_scale": 0.25,
        "warn_mean": 0.001,
        "warn_win": 0.50,
        "warn_scale": 0.60,
        "good_mean": 0.005,
        "good_win": 0.58,
        "good_scale": 1.05,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "nors_top3_w60_fast_h1m2",
        "base": "top3_full",
        "window": 60,
        "bad_mean": -0.001,
        "bad_win": 0.46,
        "bad_scale": 0.35,
        "warn_mean": 0.001,
        "warn_win": 0.50,
        "warn_scale": 0.70,
        "good_mean": 0.004,
        "good_win": 0.56,
        "good_scale": 1.05,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 3,
    },
    {
        "name": "nors_top5_w20_soft_h2m3",
        "base": "top5_full",
        "window": 20,
        "bad_mean": -0.003,
        "bad_win": 0.42,
        "bad_scale": 0.45,
        "warn_mean": 0.000,
        "warn_win": 0.48,
        "warn_scale": 0.70,
        "good_mean": 0.004,
        "good_win": 0.56,
        "good_scale": 1.05,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "nors_top5_w40_soft_h2m3",
        "base": "top5_full",
        "window": 40,
        "bad_mean": -0.002,
        "bad_win": 0.44,
        "bad_scale": 0.45,
        "warn_mean": 0.001,
        "warn_win": 0.50,
        "warn_scale": 0.75,
        "good_mean": 0.004,
        "good_win": 0.56,
        "good_scale": 1.05,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "nors_top5_w40_hard_h2m3",
        "base": "top5_full",
        "window": 40,
        "bad_mean": -0.002,
        "bad_win": 0.44,
        "bad_scale": 0.25,
        "warn_mean": 0.001,
        "warn_win": 0.50,
        "warn_scale": 0.60,
        "good_mean": 0.005,
        "good_win": 0.58,
        "good_scale": 1.05,
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
    },
    {
        "name": "nors_top5_w60_fast_h1m2",
        "base": "top5_full",
        "window": 60,
        "bad_mean": -0.001,
        "bad_win": 0.46,
        "bad_scale": 0.35,
        "warn_mean": 0.001,
        "warn_win": 0.50,
        "warn_scale": 0.70,
        "good_mean": 0.004,
        "good_win": 0.56,
        "good_scale": 1.05,
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 5,
    },
]


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


def _regime_scale(history: deque[float], case: dict[str, Any]) -> tuple[float, str, float | None, float | None]:
    if len(history) < max(10, int(case["window"]) // 2):
        return 1.0, "warmup", None, None
    values = list(history)
    mean = sum(values) / len(values)
    win = sum(1 for value in values if value > 0) / len(values)
    if mean <= float(case["bad_mean"]) or win <= float(case["bad_win"]):
        return float(case["bad_scale"]), "bad", mean, win
    if mean <= float(case["warn_mean"]) or win <= float(case["warn_win"]):
        return float(case["warn_scale"]), "warn", mean, win
    if mean >= float(case["good_mean"]) and win >= float(case["good_win"]):
        return float(case["good_scale"]), "good", mean, win
    return 1.0, "neutral", mean, win


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = BASES[str(case["base"])]
    rows = list(csv.DictReader(base["signal_file"].open("r", encoding="utf-8-sig", newline="")))
    market, next_map = _market_cache(rows)
    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        signal_date = str(row.get("signal_date") or "")
        by_date.setdefault(signal_date, []).append(row)

    history: deque[float] = deque(maxlen=int(case["window"]))
    output_rows: list[dict[str, Any]] = []
    tag_counter: Counter[str] = Counter()
    scale_values: list[float] = []
    mean_values: list[float] = []
    win_values: list[float] = []
    target_values: list[float] = []

    for signal_date in sorted(by_date):
        scale, tag, mean, win = _regime_scale(history, case)
        tag_counter[tag] += 1
        scale_values.append(scale)
        if mean is not None:
            mean_values.append(mean)
        if win is not None:
            win_values.append(win)

        selected = sorted(
            by_date[signal_date],
            key=lambda row: (int(float(row.get("rank") or 999999)), str(row.get("stock_code") or "")),
        )[: int(base["topn"])]
        day_output: list[dict[str, Any]] = []
        for idx, row in enumerate(selected, start=1):
            base_target = _float(row.get("target_pct")) or float(base["target_position_pct"])
            target_pct = min(float(base["target_position_pct"]), max(0.0, base_target * scale))
            item = dict(row)
            item["rank"] = str(idx)
            item["target_pct"] = f"{target_pct:.5f}"
            item["holding_days"] = str(int(case["holding_days"]))
            item["max_holding_days"] = str(int(case["max_holding_days"]))
            item["score_exit_entry_ratio"] = f"{float(case['score_exit']):.5f}"
            item["score_continue_entry_ratio"] = f"{float(case['score_continue']):.5f}"
            item["min_holding_days_before_score_exit"] = "1"
            item["strategy_variant"] = str(case["name"])
            item["filter_name"] = str(case["name"])
            item["dynamic_hold_name"] = str(case["name"])
            item["next_open_regime_tag"] = tag
            item["next_open_regime_scale"] = f"{scale:.6f}"
            item["next_open_regime_mean"] = "" if mean is None else f"{mean:.8f}"
            item["next_open_regime_win_rate"] = "" if win is None else f"{win:.6f}"
            item["next_open_regime_window"] = str(int(case["window"]))
            day_output.append(item)
            target_values.append(target_pct)
        output_rows.extend(day_output)

        for row in day_output:
            outcome = _forward_next_open_ret(row, market, next_map)
            if outcome is not None and math.isfinite(outcome):
                history.append(outcome)

    output_rows.sort(key=lambda row: (row["signal_date"], int(float(row["rank"])), row["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, output_rows)
    counts = Counter(row["signal_date"] for row in output_rows)
    return output, {
        "signal_rows": len(output_rows),
        "signal_days": len(counts),
        "target_names_per_day": int(base["topn"]),
        "days_below_target": sum(1 for value in counts.values() if value < int(base["topn"])),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "regime_tag_counts": dict(tag_counter),
        "avg_regime_scale": sum(scale_values) / len(scale_values) if scale_values else None,
        "avg_regime_mean": sum(mean_values) / len(mean_values) if mean_values else None,
        "avg_regime_win_rate": sum(win_values) / len(win_values) if win_values else None,
        "avg_target_pct": sum(target_values) / len(target_values) if target_values else None,
    }


def _run(case: dict[str, Any]) -> dict[str, Any]:
    base = BASES[str(case["base"])]
    signal_file, meta = _make_signal(case)
    log_file = OUT_LOG_DIR / f"{case['name']}.log"
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
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
                "GM_EQUITY_DD_RISK_MODE": "0",
                "GM_EQUITY_DD_RESIZE_EXISTING": "0",
                "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
                "GM_SCORE_EXIT_ENTRY_RATIO": str(float(case["score_exit"])),
                "GM_SCORE_CONTINUE_ENTRY_RATIO": str(float(case["score_continue"])),
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
            str(int(base["max_positions"])),
            "--holding-days",
            str(int(case["holding_days"])),
            "--max-holding-days",
            str(int(case["max_holding_days"])),
            "--target-position-pct",
            str(float(base["target_position_pct"])),
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
        **meta,
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
            annual is not None
            and sharpe is not None
            and max_drawdown is not None
            and annual >= 5.0
            and sharpe >= 4.0
            and max_drawdown <= 0.40
        ),
        "signal_file": str(signal_file),
        "score_db": str(SCORE_DB),
        "score_table": SCORE_TABLE,
        "log_file": str(log_file),
        "note": "research-only rolling next-open regime state scaling; production unchanged",
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def _write_report(results: list[dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda row: float(row.get("annual_return") or -999), reverse=True)
    target_hits = [row for row in ordered if row.get("target_hit_500_sharpe4_mdd40")]
    mdd_ok = [row for row in ordered if row.get("max_drawdown") is not None and float(row["max_drawdown"]) <= 0.40]
    lines = [
        "# 婊氬姩娆℃棩寮€鐩樼哗鏁堢姸鎬佹満澶嶆祴 20260630",
        "",
        "## 杈圭晫",
        "",
        "鏈姤鍛婁负 L5/L6 research-only 楠岃瘉锛屼笉淇敼鐢熶骇绛栫暐鍙傛暟锛屼笉鐢熸垚姝ｅ紡鐢熶骇淇″彿锛屼笉瑙﹀彂浜ゆ槗銆?",
        "鐘舵€佹満鍙娇鐢ㄥ綋鍓嶆棩鏈熶箣鍓嶅凡缁忓彂鐢熺殑涔板叆寮€鐩樺埌娆℃棩寮€鐩樻敹鐩婏紝涓嶄娇鐢ㄥ綋鍓嶆垨鏈潵鐪熷疄鏀剁泭绛涢€夊綋鍓嶄俊鍙枫€?",
        "",
        "## 瀹為獙鍙ｅ緞",
        "",
        "- 杈撳叆锛氭渶鏂?formal L4 琛嶇敓鐨勯珮鏀剁泭 Top3/Top5 鐮旂┒淇″彿銆?",
        "- 鍥炴祴鍏ュ彛锛歚D:/work/quant/quant_mcp/quant/main/run_juejin_signal_backtest.py`銆?",
        "- 鍥炴祴绐楀彛锛歚2022-06-07 09:00:00` 鍒?`2026-06-29 15:30:00`銆?",
        "- 婊戠偣锛?0 涓囪鍗曢鑷€傚簲涔板叆婊戠偣锛屽崠鍑烘粦鐐硅繎浼间负 0銆?",
        "- 鏂拌鍒欙細鎸夎繃鍘?20/40/60 绗斿凡瀹炵幇娆℃棩寮€鐩樻敹鐩婂潎鍊煎拰鑳滅巼锛屽姩鎬佺缉鏀惧悗缁拱鍏ヤ粨浣嶃€?",
        "",
        "## 缁撴灉鎽樿",
        "",
        f"- 鏂板鍊欓€夋暟閲忥細`{len(results)}`",
        f"- 鍛戒腑 `骞村寲 >= 500% / Sharpe >= 4 / 鏈€澶у洖鎾?<= 40%`锛歚{len(target_hits)}`",
    ]
    if ordered:
        best = ordered[0]
        lines.append(
            f"- best annual candidate: `{best['name']}`, annual `{_pct(best.get('annual_return'))}`, Sharpe `{float(best.get('sharpe') or 0):.3f}`, max drawdown `{_pct(best.get('max_drawdown'))}`."
        )
    if mdd_ok:
        best_ok = sorted(mdd_ok, key=lambda row: float(row.get("annual_return") or -999), reverse=True)[0]
        lines.append(
            f"- best candidate with max drawdown <= 40%: `{best_ok['name']}`, annual `{_pct(best_ok.get('annual_return'))}`, Sharpe `{float(best_ok.get('sharpe') or 0):.3f}`, max drawdown `{_pct(best_ok.get('max_drawdown'))}`."
        )
    else:
        lines.append("- no candidate met the max drawdown <= 40% filter.")
    lines.extend(
        [
            "",
            "## Top 8",
            "",
            "| 鍊欓€?| 骞村寲 | Sharpe | 鏈€澶у洖鎾?| 骞冲潎鐘舵€佺缉鏀?| 骞冲潎鐩爣浠撲綅 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ordered[:8]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('name')}`",
                    _pct(row.get("annual_return")),
                    f"{float(row.get('sharpe') or 0):.3f}",
                    _pct(row.get("max_drawdown")),
                    f"{float(row.get('avg_regime_scale') or 0):.3f}",
                    _pct(row.get("avg_target_pct")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 鍒ゆ柇",
            "",
            "婊氬姩娆℃棩寮€鐩樼哗鏁堢姸鎬佹満鑻ヨ兘闄嶄綆鍥炴挙浣嗘棤娉曞悓姝ユ彁鍗?Sharpe锛岃鏄庡巻鍙茬煭绐楁鏃ュ紑鐩樿〃鐜板彧鑳藉仛椋庨櫓鎻愮ず锛屼笉鑳芥妸褰撳墠 L4 鎺掑簭鏀归€犳垚绋冲畾楂樺鏅瓥鐣ャ€?",
            "",
            "## 璇佹嵁璺緞",
            "",
            f"- `{OUT_CSV}`",
            f"- `{OUT_JSON}`",
            f"- `{OUT_SIGNAL_DIR}`",
            f"- `{OUT_LOG_DIR}`",
            f"- `{Path(__file__)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        _write_report(results)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


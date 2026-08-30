from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import subprocess
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

OUT_LOG_DIR = REPORT_DIR / "next_open_regime_dd_resize_logs"
OUT_CSV = REPORT_DIR / "next_open_regime_dd_resize_probe_20260630.csv"
OUT_JSON = REPORT_DIR / "next_open_regime_dd_resize_probe_20260630.json"
OUT_MD = REPORT_DIR / "next_open_regime_dd_resize_probe_summary_20260630.md"


CASES: list[dict[str, Any]] = [
    {
        "name": "nors_top3_w20_soft_dd_08_14_resize",
        "signal_file": REPORT_DIR / "next_open_regime_state_signals" / "nors_top3_w20_soft_h2m3.csv",
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "target_position_pct": 0.99,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "soft_trigger": 0.08,
        "hard_trigger": 0.14,
        "recover_trigger": 0.04,
        "soft_scale": 0.65,
        "hard_scale": 0.35,
    },
    {
        "name": "nors_top3_w20_soft_dd_06_10_resize",
        "signal_file": REPORT_DIR / "next_open_regime_state_signals" / "nors_top3_w20_soft_h2m3.csv",
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "target_position_pct": 0.99,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "soft_trigger": 0.06,
        "hard_trigger": 0.10,
        "recover_trigger": 0.03,
        "soft_scale": 0.55,
        "hard_scale": 0.25,
    },
    {
        "name": "nors_top3_w20_soft_dd_04_08_resize",
        "signal_file": REPORT_DIR / "next_open_regime_state_signals" / "nors_top3_w20_soft_h2m3.csv",
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "target_position_pct": 0.99,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "soft_trigger": 0.04,
        "hard_trigger": 0.08,
        "recover_trigger": 0.02,
        "soft_scale": 0.45,
        "hard_scale": 0.20,
    },
    {
        "name": "nors_top3_w40_hard_dd_06_10_resize",
        "signal_file": REPORT_DIR / "next_open_regime_state_signals" / "nors_top3_w40_hard_h2m3.csv",
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "target_position_pct": 0.99,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "soft_trigger": 0.06,
        "hard_trigger": 0.10,
        "recover_trigger": 0.03,
        "soft_scale": 0.55,
        "hard_scale": 0.25,
    },
    {
        "name": "nors_top3_w20_soft_dd_06_10_fast_sell",
        "signal_file": REPORT_DIR / "next_open_regime_state_signals" / "nors_top3_w20_soft_h2m3.csv",
        "max_positions": 3,
        "holding_days": 1,
        "max_holding_days": 2,
        "target_position_pct": 0.99,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 3,
        "soft_trigger": 0.06,
        "hard_trigger": 0.10,
        "recover_trigger": 0.03,
        "soft_scale": 0.55,
        "hard_scale": 0.25,
    },
    {
        "name": "orig_top3_dd_06_10_resize",
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top3_s1000_cap99_h2m3_ddoff.csv",
        "max_positions": 3,
        "holding_days": 2,
        "max_holding_days": 3,
        "target_position_pct": 0.99,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "soft_trigger": 0.06,
        "hard_trigger": 0.10,
        "recover_trigger": 0.03,
        "soft_scale": 0.55,
        "hard_scale": 0.25,
    },
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


def _run(case: dict[str, Any]) -> dict[str, Any]:
    if not Path(case["signal_file"]).exists():
        raise FileNotFoundError(case["signal_file"])
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
                "GM_EQUITY_DD_RISK_MODE": "1",
                "GM_EQUITY_DD_RESIZE_EXISTING": "1",
                "GM_EQUITY_DD_SOFT_TRIGGER": str(float(case["soft_trigger"])),
                "GM_EQUITY_DD_HARD_TRIGGER": str(float(case["hard_trigger"])),
                "GM_EQUITY_DD_RECOVER_TRIGGER": str(float(case["recover_trigger"])),
                "GM_EQUITY_DD_SOFT_SCALE": str(float(case["soft_scale"])),
                "GM_EQUITY_DD_HARD_SCALE": str(float(case["hard_scale"])),
                "GM_EQUITY_DD_STRICT_WHEN_DRAWDOWN": "0",
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
            str(Path(case["signal_file"])),
            "--log-file",
            str(log_file),
            "--max-positions",
            str(int(case["max_positions"])),
            "--holding-days",
            str(int(case["holding_days"])),
            "--max-holding-days",
            str(int(case["max_holding_days"])),
            "--target-position-pct",
            str(float(case["target_position_pct"])),
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
        "signal_file": str(case["signal_file"]),
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
        "log_file": str(log_file),
        "note": "research-only next-open state plus daily equity drawdown resize; production unchanged",
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
        "# 娆℃棩寮€鐩樼姸鎬佹満鍙犲姞鏃ョ骇鍥炴挙缂╂斁澶嶆祴 20260630",
        "",
        "## 杈圭晫",
        "",
        "鏈姤鍛婁负 L5/L6 research-only 楠岃瘉锛屼笉淇敼鐢熶骇绛栫暐鍙傛暟锛屼笉鐢熸垚姝ｅ紡鐢熶骇淇″彿锛屼笉瑙﹀彂浜ゆ槗銆?",
        "鏈疆娴嬭瘯鐨勬槸鏃ョ骇璐︽埛鍥炴挙缂╂斁鍜屽凡鏈夋寔浠撳悓姝ュ噺浠擄紝闈炵洏涓珮棰戦鎺с€?",
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
            "## 鍊欓€夋槑缁?",
            "",
            "| 鍊欓€?| 骞村寲 | Sharpe | 鏈€澶у洖鎾?| 寮€浠?| 骞充粨 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('name')}`",
                    _pct(row.get("annual_return")),
                    f"{float(row.get('sharpe') or 0):.3f}",
                    _pct(row.get("max_drawdown")),
                    str(row.get("open_count") or ""),
                    str(row.get("close_count") or ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 鍒ゆ柇",
            "",
            "鑻ユ棩绾у洖鎾ょ缉鏀捐兘鏄捐憲闄嶄綆鏈€澶у洖鎾や絾骞村寲鍜?Sharpe 鍚屾鍧嶅锛屽垯璇存槑褰撳墠楂樻敹鐩婅矾寰勪粛涓昏渚濊禆楂樻尝鍔ㄦ毚闇诧紱濡傛灉鍥炴挙浠嶉珮浜?40%锛屽垯璇存槑寮€鐩樼骇椋庨櫓宸茬粡鏃犳硶闈犳棩绾т粨浣嶇姸鎬佸畬鍏ㄦ帶鍒躲€?",
            "",
            "## 璇佹嵁璺緞",
            "",
            f"- `{OUT_CSV}`",
            f"- `{OUT_JSON}`",
            f"- `{OUT_LOG_DIR}`",
            f"- `{Path(__file__)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
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


from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
SOURCE_REPORT_DIR = DATA / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
REPORT_DIR = DATA / "reports" / "strategy_agent_repro_scale_neighborhood_20260630"
SIGNAL_DIR = REPORT_DIR / "signals"
LOG_DIR = REPORT_DIR / "juejin_logs"
WRAPPER_DIR = REPORT_DIR / "wrapper_stdout"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_dyn_mild_09_12_15_v20260630" / "code_snapshot"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SCORE_DB = SOURCE_REPORT_DIR / "scores" / "grid_scores.duckdb"
SCORE_TABLE = "score_w72_23_05_amt150_mv30_latest_l4_full"
BASE_SIGNAL = SOURCE_REPORT_DIR / "open_gap_frequency_signals" / "hp_new70_turn80_top5_s170_cap27_h2m3__gap_rerank_penalty_1p5.csv"
OUT_CSV = REPORT_DIR / "repro_scale_neighborhood_summary_20260630.csv"
OUT_JSON = REPORT_DIR / "repro_scale_neighborhood_summary_20260630.json"
OUT_MANIFEST = REPORT_DIR / "repro_scale_neighborhood_manifest_20260630.json"
OUT_REPORT = REPORT_DIR / "repro_scale_neighborhood_report_20260630.md"


BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
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
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.08",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.14",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.04",
    "GM_EQUITY_DD_SOFT_SCALE": "0.80",
    "GM_EQUITY_DD_HARD_SCALE": "0.60",
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


SELL_RULES = {
    "base": {
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "day_drop_ratio": 0.995,
        "max_daily_sells": 1,
    },
    "looser_exit": {
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.975,
        "score_continue": 0.985,
        "day_drop_ratio": 0.995,
        "max_daily_sells": 1,
    },
    "stricter_exit": {
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.985,
        "score_continue": 0.995,
        "day_drop_ratio": 0.997,
        "max_daily_sells": 2,
    },
    "hold1_exit": {
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.985,
        "score_continue": 0.995,
        "day_drop_ratio": 0.997,
        "max_daily_sells": 2,
    },
}


CASES: list[dict[str, Any]] = [
    {"name": "ref_scale130_cap35_a", "topn": 5, "target_scale": 1.30, "target_cap": 0.35, "sell_rule": "base"},
    {"name": "ref_scale130_cap35_b", "topn": 5, "target_scale": 1.30, "target_cap": 0.35, "sell_rule": "base"},
    {"name": "scale125_cap34", "topn": 5, "target_scale": 1.25, "target_cap": 0.34, "sell_rule": "base"},
    {"name": "scale132_cap35", "topn": 5, "target_scale": 1.32, "target_cap": 0.35, "sell_rule": "base"},
    {"name": "scale133_cap36", "topn": 5, "target_scale": 1.33, "target_cap": 0.36, "sell_rule": "base"},
    {"name": "scale134_cap36", "topn": 5, "target_scale": 1.34, "target_cap": 0.36, "sell_rule": "base"},
    {"name": "scale135_cap36", "topn": 5, "target_scale": 1.35, "target_cap": 0.36, "sell_rule": "base"},
    {"name": "scale135_cap35", "topn": 5, "target_scale": 1.35, "target_cap": 0.35, "sell_rule": "base"},
    {"name": "scale135_cap37", "topn": 5, "target_scale": 1.35, "target_cap": 0.37, "sell_rule": "base"},
    {"name": "scale136_cap36", "topn": 5, "target_scale": 1.36, "target_cap": 0.36, "sell_rule": "base"},
    {"name": "scale137_cap36", "topn": 5, "target_scale": 1.37, "target_cap": 0.36, "sell_rule": "base"},
    {"name": "scale138_cap37", "topn": 5, "target_scale": 1.38, "target_cap": 0.37, "sell_rule": "base"},
    {"name": "ref_scale138_cap37_b", "topn": 5, "target_scale": 1.38, "target_cap": 0.37, "sell_rule": "base"},
    {"name": "scale140_cap37", "topn": 5, "target_scale": 1.40, "target_cap": 0.37, "sell_rule": "base"},
    {"name": "scale145_cap38", "topn": 5, "target_scale": 1.45, "target_cap": 0.38, "sell_rule": "base"},
    {"name": "top4_scale135_cap36", "topn": 4, "target_scale": 1.35, "target_cap": 0.36, "sell_rule": "base"},
    {"name": "gap_nochase_scale150_cap40", "topn": 5, "target_scale": 1.50, "target_cap": 0.40, "gap_nochase": True, "sell_rule": "base"},
    {"name": "scale130_cap35_looser_exit", "topn": 5, "target_scale": 1.30, "target_cap": 0.35, "sell_rule": "looser_exit"},
    {"name": "scale135_cap36_looser_exit", "topn": 5, "target_scale": 1.35, "target_cap": 0.36, "sell_rule": "looser_exit"},
    {"name": "scale130_cap35_stricter_exit", "topn": 5, "target_scale": 1.30, "target_cap": 0.35, "sell_rule": "stricter_exit"},
    {"name": "scale135_cap36_stricter_exit", "topn": 5, "target_scale": 1.35, "target_cap": 0.36, "sell_rule": "stricter_exit"},
    {"name": "scale130_cap35_hold1_exit", "topn": 5, "target_scale": 1.30, "target_cap": 0.35, "sell_rule": "hold1_exit"},
    {"name": "top3_equal32", "topn": 3, "fixed_target": 0.32, "target_cap": 0.32, "sell_rule": "base"},
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _target_for_case(row: dict[str, Any], case: dict[str, Any]) -> float:
    if "fixed_target" in case:
        target = float(case["fixed_target"])
    else:
        target = (_float(row.get("target_pct")) or 0.0) * float(case.get("target_scale", 1.0))
    gap = _float(row.get("buy_open_gap"))
    if case.get("gap_nochase") and gap is not None:
        if gap >= 0.02:
            target *= 0.50
        elif gap <= -0.08:
            target *= 0.70
    return max(0.0, min(target, float(case["target_cap"])))


def _make_signal(case: dict[str, Any], base_rows: list[dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    sell_rule = SELL_RULES[str(case["sell_rule"])]
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        rank = int(float(row["rank"]))
        if rank > int(case["topn"]):
            continue
        item = dict(row)
        target = _target_for_case(row, case)
        if target <= 0:
            continue
        item["target_pct"] = f"{target:.5f}"
        item["holding_days"] = str(sell_rule["holding_days"])
        item["max_holding_days"] = str(sell_rule["max_holding_days"])
        item["score_exit_entry_ratio"] = f"{sell_rule['score_exit']:.5f}"
        item["score_continue_entry_ratio"] = f"{sell_rule['score_continue']:.5f}"
        item["min_holding_days_before_score_exit"] = "1"
        item["strategy_variant"] = str(case["name"])
        item["filter_name"] = str(case["name"])
        item["dynamic_hold_name"] = (
            f"h{sell_rule['holding_days']}m{sell_rule['max_holding_days']}"
            f"_e{str(sell_rule['score_exit']).replace('.', '')}"
            f"_c{str(sell_rule['score_continue']).replace('.', '')}"
            f"_daydrop{str(sell_rule['day_drop_ratio']).replace('.', '')}"
            f"_ms{sell_rule['max_daily_sells']}"
        )
        rows.append(item)
    rows.sort(key=lambda item: (item["signal_date"], int(float(item["rank"])), item["stock_code"]))
    output = SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, rows)
    counts = Counter(row["signal_date"] for row in rows)
    target_topn = int(case["topn"])
    return output, {
        "signal_rows": len(rows),
        "signal_days": len(counts),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "days_below_target": sum(1 for count in counts.values() if count < target_topn),
        "signal_sha256": _sha256(output),
    }


def _env_for(case: dict[str, Any]) -> dict[str, str]:
    sell_rule = SELL_RULES[str(case["sell_rule"])]
    env = dict(BASE_ENV)
    env["GM_MAX_DAILY_SELLS"] = str(sell_rule["max_daily_sells"])
    env["GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT"] = "1"
    env["GM_SCORE_EXIT_ENTRY_RATIO"] = str(sell_rule["score_exit"])
    env["GM_SCORE_CONTINUE_ENTRY_RATIO"] = str(sell_rule["score_continue"])
    env["GM_SCORE_STOP_LOSS_DAY_DROP_RATIO"] = str(sell_rule["day_drop_ratio"])
    return env


def _command(case: dict[str, Any], signal_file: Path, log_file: Path) -> list[str]:
    sell_rule = SELL_RULES[str(case["sell_rule"])]
    return [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        "5",
        "--holding-days",
        str(sell_rule["holding_days"]),
        "--max-holding-days",
        str(sell_rule["max_holding_days"]),
        "--target-position-pct",
        str(case["target_cap"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--score-exit-entry-ratio",
        str(sell_rule["score_exit"]),
        "--score-continue-entry-ratio",
        str(sell_rule["score_continue"]),
        "--min-holding-days-before-score-exit",
        "1",
        "--score-stop-loss-day-drop-ratio",
        str(sell_rule["day_drop_ratio"]),
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


def _run_case(case: dict[str, Any], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    signal_file, signal_meta = _make_signal(case, base_rows)
    log_file = LOG_DIR / f"{case['name']}.log"
    wrapper_file = WRAPPER_DIR / f"{case['name']}.json"
    cmd = _command(case, signal_file, log_file)
    env_delta = _env_for(case)
    indicator = _extract_indicator(log_file)
    returncode = 0
    if indicator is None:
        env = os.environ.copy()
        env.update(env_delta)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        wrapper_file.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, cwd=str(MAIN), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wrapper_file.write_text(proc.stdout or "", encoding="utf-8")
        returncode = proc.returncode
        indicator = _extract_indicator(log_file)
    row = {
        "name": case["name"],
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "topn": case["topn"],
        "target_scale": case.get("target_scale"),
        "fixed_target": case.get("fixed_target"),
        "target_cap": case["target_cap"],
        "gap_nochase": bool(case.get("gap_nochase", False)),
        "sell_rule": case["sell_rule"],
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "wrapper_stdout_file": str(wrapper_file),
        "command": " ".join(cmd),
        "env_delta_json": json.dumps(env_delta, ensure_ascii=False, sort_keys=True),
        **signal_meta,
    }
    return row


def _write_report(rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row.get("returncode") == 0 and row.get("annual") is not None]
    sorted_rows = sorted(valid, key=lambda row: (float(row["annual"]), float(row.get("sharpe") or 0.0)), reverse=True)
    under_mdd40 = [row for row in sorted_rows if float(row.get("max_drawdown") or 9.0) <= 0.40]
    ref_keys = ["annual", "pnl_ratio", "sharpe", "max_drawdown"]
    count_keys = ["open_count", "close_count"]

    def _repeatable(prefix: str) -> bool:
        ref_rows = [row for row in valid if str(row["name"]).startswith(prefix)]
        if len(ref_rows) < 2:
            return False
        first = ref_rows[0]
        for row in ref_rows[1:]:
            for key in ref_keys:
                if abs(float(first.get(key) or 0.0) - float(row.get(key) or 0.0)) > 1e-12:
                    return False
            for key in count_keys:
                if int(first.get(key) or 0) != int(row.get(key) or 0):
                    return False
        return True

    def _pair_repeatable(left_name: str, right_name: str) -> bool:
        found = {str(row["name"]): row for row in valid if row["name"] in {left_name, right_name}}
        if left_name not in found or right_name not in found:
            return False
        left = found[left_name]
        right = found[right_name]
        for key in ref_keys:
            if abs(float(left.get(key) or 0.0) - float(right.get(key) or 0.0)) > 1e-12:
                return False
        for key in count_keys:
            if int(left.get(key) or 0) != int(right.get(key) or 0):
                return False
        return True

    baseline_repeatable = _repeatable("ref_scale130_cap35_")
    best_repeatable = _pair_repeatable("scale138_cap37", "ref_scale138_cap37_b")
    lines = [
        "# 鍙鐜颁粨浣嶉偦鍩熶紭鍖栨姤鍛?",
        "",
        "## 缁撹鍙ｅ緞",
        "",
        "- 鏈疆涓?research-only锛屼笉淇敼 L5 鐢熶骇褰掓。銆佺敓浜у弬鏁版垨浜ゆ槗淇″彿銆?",
        "- 鏃?573% 涓?445% 鏃ュ織涓嶄綔涓烘湰杞熀绾匡紱鏈疆鍙噰鐢ㄥ彲閲嶆柊鎵ц骞朵繚鐣欏畬鏁村懡浠ゅ拰鍝堝笇鐨勭粨鏋溿€?",
        "- 鎺橀噾鏃ュ織涓?wrapper stdout 鍒嗙钀界洏锛岄伩鍏嶅啀娆″嚭鐜版棩蹇椾簰鐩歌鐩栧鑷翠笉鍙鐜般€?",
        f"- baseline repeatability: {'passed' if baseline_repeatable else 'not passed or insufficient samples'}.",
        f"- best candidate repeatability: {'passed' if best_repeatable else 'not passed or insufficient samples'}.",
        "",
        "## 鍏ㄩ儴鍊欓€夌粨鏋?",
        "",
        "| 鍚嶇О | 骞村寲 | Sharpe | 鏈€澶у洖鎾?| 鑳滅巼 | 寮€浠?| TopN | 浠撲綅涓婇檺 | 鍗栧嚭瑙勫垯 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted_rows:
        lines.append(
            "| {name} | {annual:.2%} | {sharpe:.3f} | {mdd:.2%} | {win:.2%} | {open_count} | {topn} | {cap:.0%} | {sell_rule} |".format(
                name=row["name"],
                annual=float(row["annual"]),
                sharpe=float(row.get("sharpe") or 0.0),
                mdd=float(row.get("max_drawdown") or 0.0),
                win=float(row.get("win_ratio") or 0.0),
                open_count=row.get("open_count", ""),
                topn=row.get("topn", ""),
                cap=float(row.get("target_cap") or 0.0),
                sell_rule=row.get("sell_rule", ""),
            )
        )
    if sorted_rows:
        best = sorted_rows[0]
        lines.extend(
            [
                "",
                "## 鏀剁泭鏈€楂樺€欓€?",
                "",
                f"- best candidate: `{best['name']}`",
                f"- annual: {float(best['annual']):.2%}",
                f"- Sharpe: {float(best.get('sharpe') or 0.0):.3f}",
                f"- max drawdown: {float(best.get('max_drawdown') or 0.0):.2%}",
                f"- signal file: `{best['signal_file']}`",
                f"- log file: `{best['log_file']}`",
            ]
        )
    if under_mdd40:
        best_safe = under_mdd40[0]
        lines.extend(
            [
                "",
                "## 鍥炴挙 40% 鍐呮敹鐩婃渶楂樺€欓€?",
                "",
                f"- best safe candidate: `{best_safe['name']}`",
                f"- annual: {float(best_safe['annual']):.2%}",
                f"- Sharpe: {float(best_safe.get('sharpe') or 0.0):.3f}",
                f"- max drawdown: {float(best_safe.get('max_drawdown') or 0.0):.2%}",
                f"- signal file: `{best_safe['signal_file']}`",
                f"- log file: `{best_safe['log_file']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 澶嶇幇璧勪骇",
            "",
            f"- 姹囨€?CSV锛歚{OUT_CSV}`",
            f"- 姹囨€?JSON锛歚{OUT_JSON}`",
            f"- 杩愯 manifest锛歚{OUT_MANIFEST}`",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [JUEJIN_PYTHON, STRATEGY_DIR / "main.py", BASE_SIGNAL, MARKET_DB, SCORE_DB]:
        if not path.exists():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base_rows = list(csv.DictReader(BASE_SIGNAL.open("r", encoding="utf-8-sig", newline="")))
    manifest = {
        "schema_version": 1,
        "generated_at": datetime_module.datetime.now().isoformat(timespec="seconds"),
        "status": "research_only_no_production_change",
        "purpose": "reproducible parameter neighborhood around verified scale1p30_cap35 candidate",
        "base_signal": str(BASE_SIGNAL),
        "base_signal_sha256": _sha256(BASE_SIGNAL),
        "strategy_dir": str(STRATEGY_DIR),
        "strategy_main_sha256": _sha256(STRATEGY_DIR / "main.py"),
        "score_db": str(SCORE_DB),
        "score_db_sha256": _sha256(SCORE_DB),
        "score_table": SCORE_TABLE,
        "market_db": str(MARKET_DB),
        "market_db_sha256": _sha256(MARKET_DB),
        "juejin_python": str(JUEJIN_PYTHON),
        "base_env": BASE_ENV,
        "sell_rules": SELL_RULES,
        "cases": CASES,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    if OUT_JSON.exists():
        try:
            rows = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    done = {row.get("name") for row in rows}
    for case in CASES:
        if case["name"] in done:
            continue
        row = _run_case(case, base_rows)
        rows.append(row)
        _write_rows(OUT_CSV, rows)
        OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_report(rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    _write_report(rows)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


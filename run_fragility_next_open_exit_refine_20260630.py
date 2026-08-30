from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_fragility_refill_optimization_20260630 import _risk_scale
from run_rolling_next_open_meta_filter_20260630 import (
    _feature_keys,
    _float,
    _forward_next_open_ret,
    _market_cache,
    _rolling_expected,
    _scale_from_expected,
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

OUT_SIGNAL_DIR = REPORT_DIR / "fragility_next_open_exit_refine_signals"
OUT_LOG_DIR = REPORT_DIR / "fragility_next_open_exit_refine_logs"
OUT_CSV = REPORT_DIR / "fragility_next_open_exit_refine_20260630.csv"
OUT_JSON = REPORT_DIR / "fragility_next_open_exit_refine_20260630.json"
OUT_MD = REPORT_DIR / "fragility_next_open_exit_refine_summary_20260630.md"

FRAGILE_TOP1 = {"SZSE.301396"}

BASES: dict[str, dict[str, Any]] = {
    "s1000_top5": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top5_s1000_cap99_h2m3_ddoff.csv",
        "target_position_pct": 0.99,
        "base_topn": 5,
    },
    "s500_top5": {
        "signal_file": REPORT_DIR
        / "timing_high_exposure_extension_signals"
        / "thex_t55_top5_s500_cap80_h2m3_ddoff.csv",
        "target_position_pct": 0.80,
        "base_topn": 5,
    },
}

EXIT_PROFILES: dict[str, dict[str, Any]] = {
    "fast_h1m2": {
        "holding_days": 1,
        "max_holding_days": 2,
        "score_exit": 0.995,
        "score_continue": 1.005,
        "max_daily_sells": 5,
    },
    "normal_h2m3": {
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit": 0.985,
        "score_continue": 0.995,
        "max_daily_sells": 2,
    },
    "slow_h3m5": {
        "holding_days": 3,
        "max_holding_days": 5,
        "score_exit": 0.955,
        "score_continue": 0.980,
        "max_daily_sells": 1,
    },
}


def _build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for base in ["s1000_top5"]:
        for select_topn in [5, 3]:
            for risk_mode in ["basic", "strict"]:
                for meta_mode in ["soft", "daysoft"]:
                    for exit_name, exit_profile in EXIT_PROFILES.items():
                        cases.append(
                            {
                                "name": f"fnox_{base}_rm301396_top{select_topn}_{risk_mode}_{meta_mode}_{exit_name}",
                                "base": base,
                                "remove_symbols": sorted(FRAGILE_TOP1),
                                "select_topn": select_topn,
                                "max_positions": select_topn,
                                "risk_mode": risk_mode,
                                "meta_mode": meta_mode,
                                "rank_by_meta": False,
                                **exit_profile,
                            }
                        )
    for exit_name, exit_profile in EXIT_PROFILES.items():
        cases.append(
            {
                "name": f"fnox_s500_top5_rm301396_top5_strict_soft_{exit_name}",
                "base": "s500_top5",
                "remove_symbols": sorted(FRAGILE_TOP1),
                "select_topn": 5,
                "max_positions": 5,
                "risk_mode": "strict",
                "meta_mode": "soft",
                "rank_by_meta": False,
                **exit_profile,
            }
        )
    return cases


CASES = _build_cases()


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


def _make_signal(case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    base = BASES[str(case["base"])]
    rows = list(csv.DictReader(base["signal_file"].open("r", encoding="utf-8-sig", newline="")))
    market, next_map = _market_cache(rows)
    remove_symbols = set(case.get("remove_symbols") or [])

    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    removed = 0
    for row in rows:
        if str(row.get("symbol") or "") in remove_symbols:
            removed += 1
            continue
        by_date[str(row.get("signal_date") or "")].append(row)

    feature_stats: dict[str, list[float]] = {}
    global_stats = [0.0, 0.0]
    output_rows: list[dict[str, Any]] = []
    skipped = 0
    deselected = 0
    tags: Counter[str] = Counter()
    meta_values: list[float] = []
    risk_values: list[float] = []
    target_values: list[float] = []

    for signal_date in sorted(by_date):
        current = sorted(
            by_date[signal_date],
            key=lambda row: (int(float(row.get("rank") or 999999)), str(row.get("stock_code") or "")),
        )
        row_expectations: list[tuple[dict[str, Any], float, int, int]] = []
        for row in current:
            expected, used_features, used_rows = _rolling_expected(
                _feature_keys(row),
                feature_stats,
                global_stats,
                min_count=35,
                shrink=100.0,
            )
            row_expectations.append((row, expected, used_features, used_rows))
        day_expected = sum(item[1] for item in row_expectations) / len(row_expectations) if row_expectations else None

        candidates: list[tuple[float, int, str, dict[str, Any]]] = []
        for row, expected, used_features, used_rows in row_expectations:
            meta_scale, meta_tag = _scale_from_expected(expected, str(case["meta_mode"]), day_expected)
            risk_scale, risk_tag = _risk_scale(row, market, str(case["risk_mode"]))
            combined_scale = max(0.0, min(1.10, meta_scale * risk_scale))
            tags[f"meta:{meta_tag}"] += 1
            tags[f"risk:{risk_tag}"] += 1
            meta_values.append(expected)
            risk_values.append(risk_scale)
            if combined_scale <= 0:
                skipped += 1
                continue
            item = dict(row)
            original_rank = int(float(row.get("rank") or 999999))
            base_target = _float(row.get("target_pct")) or float(base["target_position_pct"])
            target_pct = min(float(base["target_position_pct"]), max(0.0, base_target * combined_scale))
            item["original_rank"] = str(original_rank)
            item["target_pct"] = f"{target_pct:.5f}"
            item["holding_days"] = str(int(case["holding_days"]))
            item["max_holding_days"] = str(int(case["max_holding_days"]))
            item["score_exit_entry_ratio"] = f"{float(case['score_exit']):.5f}"
            item["score_continue_entry_ratio"] = f"{float(case['score_continue']):.5f}"
            item["min_holding_days_before_score_exit"] = "1"
            item["strategy_variant"] = str(case["name"])
            item["filter_name"] = str(case["name"])
            item["dynamic_hold_name"] = str(case["name"])
            item["fragility_removed_symbols"] = ",".join(sorted(remove_symbols))
            item["next_open_meta_expected"] = f"{expected:.8f}"
            item["next_open_meta_day_expected"] = "" if day_expected is None else f"{day_expected:.8f}"
            item["next_open_meta_used_features"] = str(used_features)
            item["next_open_meta_used_rows"] = str(used_rows)
            item["next_open_meta_scale"] = f"{meta_scale:.6f}"
            item["risk_scale"] = f"{risk_scale:.6f}"
            item["combined_scale"] = f"{combined_scale:.6f}"
            item["risk_tags"] = risk_tag
            item["next_open_meta_tag"] = meta_tag
            target_values.append(target_pct)
            sort_score = expected if case.get("rank_by_meta") else -float(original_rank)
            candidates.append((sort_score, original_rank, str(row.get("stock_code") or ""), item))

        if case.get("rank_by_meta"):
            selected = sorted(candidates, key=lambda value: (-value[0], value[1], value[2]))[: int(case["select_topn"])]
        else:
            selected = sorted(candidates, key=lambda value: (value[1], value[2]))[: int(case["select_topn"])]
        deselected += max(0, len(candidates) - len(selected))
        for idx, (_, _, _, item) in enumerate(selected, start=1):
            item["rank"] = str(idx)
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

    output_rows.sort(key=lambda row: (row["signal_date"], int(float(row["rank"])), row["stock_code"]))
    output = OUT_SIGNAL_DIR / f"{case['name']}.csv"
    _write_rows(output, output_rows)
    counts = Counter(row["signal_date"] for row in output_rows)
    target = int(case["select_topn"])
    return output, {
        "signal_rows": len(output_rows),
        "signal_days": len(counts),
        "removed_rows": removed,
        "skipped_rows": skipped,
        "deselected_rows": deselected,
        "target_names_per_day": target,
        "days_below_target": sum(1 for value in counts.values() if value < target),
        "min_per_day": min(counts.values()) if counts else 0,
        "max_per_day": max(counts.values()) if counts else 0,
        "avg_target_pct": sum(target_values) / len(target_values) if target_values else None,
        "avg_meta_expected": sum(meta_values) / len(meta_values) if meta_values else None,
        "avg_risk_scale": sum(risk_values) / len(risk_values) if risk_values else None,
        "tag_top": dict(tags.most_common(10)),
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
            str(int(case["max_positions"])),
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
        "note": "research-only fragility + next-open meta + exit-frequency refine; production unchanged",
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
    best_mdd_ok = mdd_ok[0] if mdd_ok else None
    lines = [
        "# 鑴嗗急楂樻敹鐩婂€欓€夌殑娆℃棩寮€鐩樹笌鍗栧嚭棰戠巼鑱斿悎澶嶆祴 20260630",
        "",
        "## 杈圭晫",
        "",
        "鏈姤鍛婁负 L5/L6 research-only 楠岃瘉锛屼笉淇敼鐢熶骇绛栫暐鍙傛暟锛屼笉鐢熸垚姝ｅ紡鐢熶骇淇″彿锛屼笉瑙﹀彂浜ゆ槗銆?",
        "鏈疆鍙娇鐢ㄦ棦鏈夋渶鏂?formal L4 琛嶇敓鐮旂┒淇″彿姹狅紝骞跺宸茬粡纭鐨勬渶澶ц础鐚偂 `SZSE.301396` 鍋氬墧闄ゅ悗琛ヤ綅銆?",
        "",
        "## 瀹為獙鍙ｅ緞",
        "",
        "- 鍥炴祴鍏ュ彛锛歚D:/work/quant/quant_mcp/quant/main/run_juejin_signal_backtest.py`",
        "- 鍥炴祴绐楀彛锛歚2022-06-07 09:00:00` 鍒?`2026-06-29 15:30:00`",
        "- 婊戠偣锛?0 涓囪鍗曢鑷€傚簲涔板叆婊戠偣锛屽崠鍑烘粦鐐硅繎浼间负 0銆?",
        "- 椋庢帶锛氫笉鍚敤鐩樹腑椋庢帶锛涗繚鐣欎笉涔版定鍋溿€佸悓姝ユ寔浠撱€乀+1 鏈€灏忔寔鏈夋棩绾︽潫銆?",
        "- 鏂板鍙橀噺锛氭粴鍔ㄥ巻鍙蹭拱鍏ュ紑鐩樺埌娆℃棩寮€鐩樻敹鐩婅川閲忕缉鏀俱€佸熀纭€/涓ユ牸椋庨櫓闄嶄粨銆佸揩/涓?鎱笁妗ｅ崠鍑洪鐜囥€?",
        "",
        "## 缁撴灉鎽樿",
        "",
        f"- 鏂板鍊欓€夋暟閲忥細`{len(results)}`",
        f"- 鍛戒腑 `骞村寲 >= 500% / Sharpe >= 4 / 鏈€澶у洖鎾?<= 40%`锛歚{len(target_hits)}`",
    ]
    if ordered:
        best = ordered[0]
        lines.extend(
            [
                f"- best annual candidate: `{best['name']}`, annual `{_pct(best.get('annual_return'))}`, Sharpe `{float(best.get('sharpe') or 0):.3f}`, max drawdown `{_pct(best.get('max_drawdown'))}`.",
            ]
        )
    if best_mdd_ok:
        lines.append(
            f"- best candidate with max drawdown <= 40%: `{best_mdd_ok['name']}`, annual `{_pct(best_mdd_ok.get('annual_return'))}`, Sharpe `{float(best_mdd_ok.get('sharpe') or 0):.3f}`, max drawdown `{_pct(best_mdd_ok.get('max_drawdown'))}`."
        )
    else:
        lines.append("- no candidate met the max drawdown <= 40% filter.")
    lines.extend(
        [
            "",
            "## Top 10",
            "",
            "| 鍊欓€?| 骞村寲 | Sharpe | 鏈€澶у洖鎾?| 淇″彿琛?| 涔板叆鏃?| 骞冲潎鐩爣浠撲綅 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ordered[:10]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('name')}`",
                    _pct(row.get("annual_return")),
                    f"{float(row.get('sharpe') or 0):.3f}",
                    _pct(row.get("max_drawdown")),
                    str(row.get("signal_rows") or ""),
                    str(row.get("signal_days") or ""),
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
            "鏈疆鎶娾€滃墧闄ゆ渶澶ц础鐚偂鍚庤ˉ浣嶁€濃€滄鏃ュ紑鐩樻粴鍔ㄨ川閲忕缉鏀锯€濆拰鈥滃崠鍑洪鐜囪皟鑺傗€濈粍鍚堝埌涓€璧峰悗锛屼粛鏈懡涓洰鏍囥€傝嫢鏈€楂樺勾鍖栨垨鍥炴挙鏀瑰杽鏈夐檺锛岃鏄庨棶棰樹笉鏄崟涓€閫€鍑洪鐜囧彲瑙ｅ喅锛岃€屾槸鍓旈櫎鏋佺璧㈠鍚庡€欓€夋睜鏈韩鐨勫崟浣嶉闄╂敹鐩婁笉瓒炽€?",
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


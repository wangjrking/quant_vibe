from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("D:/work/quant/quant_mcp")
REPORT_DIR = ROOT / "quant/data_file/reports/strategy_agent_active_l4_prod_repro_grid_20260703"
STATE_FILE = REPORT_DIR / "old500_date_state_features/profile_day_market_state.csv"
RET_FILE = REPORT_DIR / "old500_date_selection_bias/profile_top1_with_ret_h1.csv"
BASE_SIGNAL_FILE = (
    REPORT_DIR
    / "latest_l4_old500_profile_candidates/signals/prof_rank16_pct175_gap15_top1/signals.csv"
)
OUT_DIR = REPORT_DIR / "latest_l4_market_state_date_selector"
SIGNAL_DIR = OUT_DIR / "signals"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_FILE = OUT_DIR / "market_state_selector_summary.csv"
JUEJIN_SUMMARY_FILE = OUT_DIR / "market_state_selector_juejin_summary.csv"

PYTHON = ROOT / ".venv/Scripts/python.exe"
RUNNER = ROOT / "quant/main/run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    ROOT
    / "quant/data_file/reports/strategy_agent_latest_l4_weight_candidates_20260702/postrank_open_filter_candidates/code_snapshot_sell_available_safe_20260702"
)
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = ROOT / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


def max_drawdown(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return float("nan")
    curve = (1.0 + daily_returns.fillna(0.0)).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def perf_stats(daily_returns: pd.Series) -> dict[str, float]:
    daily_returns = daily_returns.dropna()
    n = int(len(daily_returns))
    if n == 0:
        return {
            "days": 0,
            "annual_proxy": float("nan"),
            "pnl_proxy": float("nan"),
            "sharpe_proxy": float("nan"),
            "max_drawdown_proxy": float("nan"),
            "win_day_ratio": float("nan"),
        }
    curve_end = float((1.0 + daily_returns).prod())
    annual = curve_end ** (252.0 / n) - 1.0 if curve_end > 0 else -1.0
    std = float(daily_returns.std(ddof=1))
    sharpe = float(daily_returns.mean() / std * math.sqrt(252.0)) if std > 0 else float("nan")
    return {
        "days": n,
        "annual_proxy": annual,
        "pnl_proxy": curve_end - 1.0,
        "sharpe_proxy": sharpe,
        "max_drawdown_proxy": abs(max_drawdown(daily_returns)),
        "win_day_ratio": float((daily_returns > 0).mean()),
    }


def make_rule_name(rule: dict[str, float]) -> str:
    parts = []
    for key, value in rule.items():
        token = str(value).replace("-", "m").replace(".", "p")
        parts.append(f"{key}{token}")
    return "ms_" + "_".join(parts)


def apply_rule(state: pd.DataFrame, rule: dict[str, float]) -> pd.Series:
    mask = pd.Series(True, index=state.index)
    for key, value in rule.items():
        if key.endswith("_ge"):
            col = key[: -len("_ge")]
            mask &= state[col] >= value
        elif key.endswith("_le"):
            col = key[: -len("_le")]
            mask &= state[col] <= value
        else:
            raise ValueError(f"Unsupported rule key: {key}")
    return mask


def build_rules() -> list[dict[str, float]]:
    rules: list[dict[str, float]] = []

    # Single broad filters first: these are intentionally coarse and available
    # at signal-day close. They do not use old-signal-day labels.
    for deep in [0.025, 0.035, 0.045, 0.055, 0.065, 0.075]:
        rules.append({"deep_down_ratio_ge": deep})
    for up in [0.40, 0.45, 0.50, 0.55, 0.60]:
        rules.append({"up_ratio_le": up})
    for med in [-1.5, -1.0, -0.5, 0.0, 0.5]:
        rules.append({"mkt_med_pct_chg_le": med})
    for turnover in [1.8, 2.2, 2.6, 3.0, 3.4]:
        rules.append({"med_turnover_ge": turnover})
    for low_open in [0.45, 0.50, 0.55, 0.60]:
        rules.append({"low_open_ratio_ge": low_open})

    # Two- and three-factor combinations. Keep the grid small and interpretable.
    for deep, up in product([0.035, 0.045, 0.055, 0.065], [0.45, 0.50, 0.55]):
        rules.append({"deep_down_ratio_ge": deep, "up_ratio_le": up})
    for deep, med in product([0.035, 0.045, 0.055], [-1.0, -0.5, 0.0]):
        rules.append({"deep_down_ratio_ge": deep, "mkt_med_pct_chg_le": med})
    for med, turnover in product([-1.0, -0.5, 0.0], [2.2, 2.6, 3.0]):
        rules.append({"mkt_med_pct_chg_le": med, "med_turnover_ge": turnover})
    for deep, up, turnover in product([0.035, 0.045, 0.055], [0.45, 0.50, 0.55], [2.2, 2.6]):
        rules.append({"deep_down_ratio_ge": deep, "up_ratio_le": up, "med_turnover_ge": turnover})
    for med, up, low_open in product([-1.0, -0.5, 0.0], [0.45, 0.50, 0.55], [0.50, 0.55]):
        rules.append({"mkt_med_pct_chg_le": med, "up_ratio_le": up, "low_open_ratio_ge": low_open})

    # Deduplicate after overlapping combinations.
    seen = set()
    unique: list[dict[str, float]] = []
    for rule in rules:
        key = tuple(sorted(rule.items()))
        if key not in seen:
            seen.add(key)
            unique.append(rule)
    return unique


def generate_candidates(min_days: int, max_candidates: int) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

    state = pd.read_csv(STATE_FILE)
    ret = pd.read_csv(RET_FILE)
    signals = pd.read_csv(BASE_SIGNAL_FILE)
    state["signal_date"] = state["signal_date"].astype(int)
    ret["signal_date"] = ret["signal_date"].astype(int)
    signals["signal_date"] = signals["signal_date"].astype(int)

    ret_by_day = ret[["signal_date", "ret_h1", "target_pct"]].copy()
    ret_by_day["daily_return_proxy"] = ret_by_day["ret_h1"] * ret_by_day["target_pct"]
    state = state.merge(ret_by_day[["signal_date", "daily_return_proxy"]], on="signal_date", how="inner")

    rows = []
    for rule in build_rules():
        mask = apply_rule(state, rule)
        selected = state.loc[mask].copy()
        if len(selected) < min_days:
            continue
        stats = perf_stats(selected["daily_return_proxy"])
        name = make_rule_name(rule)
        row = {"candidate": name, "rule_json": json.dumps(rule, ensure_ascii=False), **stats}
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
        return summary

    summary = summary.sort_values(
        ["sharpe_proxy", "annual_proxy", "max_drawdown_proxy"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    for _, row in summary.head(max_candidates).iterrows():
        rule = json.loads(row["rule_json"])
        selected_dates = set(state.loc[apply_rule(state, rule), "signal_date"].astype(int))
        out = signals[signals["signal_date"].isin(selected_dates)].copy()
        out["strategy_variant"] = row["candidate"]
        out["filter_name"] = row["candidate"]
        cand_dir = SIGNAL_DIR / row["candidate"]
        cand_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(cand_dir / "signals.csv", index=False, encoding="utf-8-sig")

    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    return summary


def parse_runner_result(stdout: str, runner_json_path: Path) -> dict[str, float | str | None]:
    result: dict[str, float | str | None] = {
        "annual_return": None,
        "pnl_ratio": None,
        "sharpe": None,
        "max_drawdown": None,
        "win_ratio": None,
        "open_count": None,
        "close_count": None,
    }
    if runner_json_path.exists():
        try:
            payload = json.loads(runner_json_path.read_text(encoding="utf-8"))
            indicator = payload.get("indicator") or {}
            for key in result:
                if key in indicator:
                    result[key] = indicator[key]
        except Exception:
            pass

    if result["annual_return"] is not None:
        return result

    text = None
    for line in stdout.splitlines():
        if "GM_BACKTEST_INDICATOR:" in line:
            text = line.split("GM_BACKTEST_INDICATOR:", 1)[1].strip()
            break
    if text is None:
        match = re.search(r"indicator=(\{.*?\})", stdout, flags=re.S)
        if not match:
            return result
        text = match.group(1)
    text = re.sub(r"datetime\.datetime\(.*?\)\)", "'datetime'", text)
    try:
        indicator = ast.literal_eval(text)
    except Exception:
        return result
    key_map = {
        "annual_return": "pnl_ratio_annual",
        "sharpe": "sharp_ratio",
    }
    for key in result:
        source_key = key_map.get(key, key)
        if source_key in indicator:
            result[key] = indicator[source_key]
    return result


def run_juejin(top_n: int) -> pd.DataFrame:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_FILE)
    rows = []
    for _, row in summary.head(top_n).iterrows():
        candidate = row["candidate"]
        signal_file = SIGNAL_DIR / candidate / "signals.csv"
        if not signal_file.exists():
            continue
        log_file = LOG_DIR / f"{candidate}.log"
        cmd = [
            str(PYTHON),
            str(RUNNER),
            "--strategy-dir",
            str(STRATEGY_DIR),
            "--signal-file",
            str(signal_file),
            "--log-file",
            str(log_file),
            "--max-positions",
            "1",
            "--holding-days",
            "1",
            "--max-holding-days",
            "1",
            "--score-exit-entry-ratio",
            "9.99",
            "--min-holding-days-before-score-exit",
            "1",
            "--score-continue-entry-ratio",
            "9.99",
            "--light-stop-loss-pct",
            "0.05",
            "--min-holding-days-before-light-stop",
            "1",
            "--take-profit-pct",
            "0.08",
            "--score-db",
            str(SCORE_DB),
            "--score-table",
            "score",
            "--market-db",
            str(MARKET_DB),
            "--backtest-adjust",
            "none",
            "--backtest-slippage-ratio",
            "0.0015",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")
        runner_json = Path(str(log_file) + ".runner.json")
        parsed = parse_runner_result(proc.stdout + "\n" + proc.stderr, runner_json)
        out_row = row.to_dict()
        out_row.update(
            {
                "returncode": proc.returncode,
                "signal_file": str(signal_file),
                "log_file": str(log_file),
                **{f"juejin_{k}": v for k, v in parsed.items()},
            }
        )
        rows.append(out_row)
    out = pd.DataFrame(rows)
    out.to_csv(JUEJIN_SUMMARY_FILE, index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-days", type=int, default=120)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--run-top", type=int, default=0)
    args = parser.parse_args()

    summary = generate_candidates(args.min_days, args.max_candidates)
    print(f"generated={len(summary)} summary={SUMMARY_FILE}")
    if not summary.empty:
        print(summary.head(10).to_string(index=False))
    if args.run_top > 0 and not summary.empty:
        result = run_juejin(args.run_top)
        print(f"juejin_rows={len(result)} summary={JUEJIN_SUMMARY_FILE}")
        if not result.empty:
            cols = [
                "candidate",
                "days",
                "annual_proxy",
                "sharpe_proxy",
                "max_drawdown_proxy",
                "returncode",
                "juejin_annual_return",
                "juejin_sharpe",
                "juejin_max_drawdown",
                "juejin_open_count",
            ]
            print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()

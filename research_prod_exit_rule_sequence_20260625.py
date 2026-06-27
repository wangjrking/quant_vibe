from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
REPORT_DIR = DATA / "reports" / "strategy_agent_prod_exit_rule_sequence_20260625"
TMP_STRATEGY_DIR = REPORT_DIR / "strategy_abs_exit_patch"

BASE_SIGNAL = DATA / "reports" / "strategy_agent_goal_high_annual_20260623" / "top1_no_delist_rerun_20260624" / "diversification_tune_20260624" / "strict_sync_liquidity_neighborhood_20260624" / "formal_horizon_entry_confirmation_20260624" / "dynamic_h2_m3_sl06_dd0712_pos89_c975_high_pos_boundary_20260625" / "signals" / "pos8975_scale7555.csv"
SOURCE_STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_dynamic_h2m3_pos8975_scale7555_v20260625" / "code_snapshot"
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
SCORE_DB = DATA / "reports" / "strategy_agent_goal_high_annual_20260623" / "top1_no_delist_rerun_20260624" / "diversification_tune_20260624" / "scores_diversification.db"
SCORE_TABLE = "score_div_top1_w90_5d10_h5_e099"

BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-06-23 15:30:00"

BASE_ENV = {
    "GM_OPEN_DAILY_SCORE_EXIT": "1",
    "GM_MAX_DAILY_SELLS": "1",
    "GM_STOP_LOSS_PCT": "0.06",
    "GM_TAKE_PROFIT_PCT": "0.07",
    "GM_LIGHT_STOP_LOSS_PCT": "none",
    "GM_LOG_EXPOSURE": "1",
    "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
    "GM_SYNC_POSITIONS": "1",
    "GM_CASH_BUFFER": "0.99",
    "GM_VERBOSE_TRADES": "0",
    "GM_SCORE_CONTINUE_ENTRY_RATIO": "0.975",
    "GM_EQUITY_DD_RISK_MODE": "1",
    "GM_EQUITY_DD_SOFT_TRIGGER": "0.07",
    "GM_EQUITY_DD_HARD_TRIGGER": "0.12",
    "GM_EQUITY_DD_RECOVER_TRIGGER": "0.03",
    "GM_EQUITY_DD_SOFT_SCALE": "0.75",
    "GM_EQUITY_DD_HARD_SCALE": "0.55",
    "GM_FORCE_SELL_MARKET_ORDER": "1",
    "GM_INTRADAY_RISK_MODE": "1",
    "GM_INTRADAY_REPLACE_BUY": "0",
    "GM_INTRADAY_RISK_TIMES": "10:00:00,11:00:00,14:30:00",
    "GM_EQUITY_DD_RESIZE_EXISTING": "0",
}

BASELINE = {
    "name": "baseline_prod",
    "family": "baseline",
    "label": "生产基线",
    "env": {
        "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
        "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
    },
}

CASES = [
    BASELINE,
    {
        "name": "abs_0998",
        "family": "absolute_score",
        "label": "绝对分数<=0.998",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_EXIT_PRED": "0.998",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "abs_0996",
        "family": "absolute_score",
        "label": "绝对分数<=0.996",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_EXIT_PRED": "0.996",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "abs_0994",
        "family": "absolute_score",
        "label": "绝对分数<=0.994",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_EXIT_PRED": "0.994",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "rank_0995",
        "family": "rank_threshold",
        "label": "排名分位<=0.995",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_EXIT_RANK": "0.995",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "rank_0990",
        "family": "rank_threshold",
        "label": "排名分位<=0.990",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_EXIT_RANK": "0.990",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "rank_0985",
        "family": "rank_threshold",
        "label": "排名分位<=0.985",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_EXIT_RANK": "0.985",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "drop_0995",
        "family": "decay_ratio",
        "label": "单日分数衰退<=0.995",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.995",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "drop_0990",
        "family": "decay_ratio",
        "label": "单日分数衰退<=0.990",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.990",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "drop_0985",
        "family": "decay_ratio",
        "label": "单日分数衰退<=0.985",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "none",
            "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.985",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
]


def _extract_indicator(log_file: Path) -> dict | None:
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


def _exposure_stats(log_file: Path) -> dict:
    pattern = re.compile(r"EXPOSURE .* invested_pct=([0-9.]+).*active_positions=(\d+)")
    values: list[float] = []
    actives: list[int] = []
    if not log_file.exists():
        return {"avg_invested_pct": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        actives.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "max_active_positions": max(actives) if actives else None,
        "exposure_points": len(values),
    }


def _ensure_abs_exit_strategy() -> Path:
    target_main = TMP_STRATEGY_DIR / "main.py"
    if target_main.exists():
        return TMP_STRATEGY_DIR
    if TMP_STRATEGY_DIR.exists():
        shutil.rmtree(TMP_STRATEGY_DIR)
    shutil.copytree(SOURCE_STRATEGY_DIR, TMP_STRATEGY_DIR)
    text = target_main.read_text(encoding="utf-8")
    if "GM_SCORE_EXIT_PRED" not in text:
        text = text.replace(
            'SCORE_EXIT_RANK = _optional_float_env("GM_SCORE_EXIT_RANK")\n',
            'SCORE_EXIT_RANK = _optional_float_env("GM_SCORE_EXIT_RANK")\n'
            'SCORE_EXIT_PRED = _optional_float_env("GM_SCORE_EXIT_PRED")\n',
            1,
        )
        anchor = (
            "    if (\n"
            '        pred_rank is not None\n'
            "        and SCORE_EXIT_RANK is not None\n"
            "        and holding_days_elapsed >= MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT\n"
            "        and pred_rank <= SCORE_EXIT_RANK\n"
            "    ):\n"
            "        return True\n"
        )
        injected = (
            "    if (\n"
            "        pred_score is not None\n"
            "        and SCORE_EXIT_PRED is not None\n"
            "        and holding_days_elapsed >= MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT\n"
            "        and pred_score <= SCORE_EXIT_PRED\n"
            "    ):\n"
            "        return True\n"
        )
        if anchor not in text:
            raise RuntimeError("failed to patch strategy main.py for abs exit")
        text = text.replace(anchor, anchor + injected, 1)
        target_main.write_text(text, encoding="utf-8")
    return TMP_STRATEGY_DIR


def _run_case(case: dict) -> dict:
    strategy_dir = _ensure_abs_exit_strategy()
    log_file = REPORT_DIR / "logs" / f"{case['name']}.log"
    env = os.environ.copy()
    env.update(BASE_ENV)
    env.update(case["env"])
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(strategy_dir),
        "--signal-file",
        str(BASE_SIGNAL),
        "--log-file",
        str(log_file),
        "--max-positions",
        "1",
        "--holding-days",
        "2",
        "--max-holding-days",
        "3",
        "--target-position-pct",
        "0.8975",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(DATA / "STOCK_DAILY_DATA.db"),
        "--backtest-start",
        BACKTEST_START,
        "--backtest-end",
        BACKTEST_END,
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
        "--stop-loss-pct",
        "0.06",
        "--take-profit-pct",
        "0.07",
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    returncode = proc.returncode
    indicator = _extract_indicator(log_file)
    exposure = _exposure_stats(log_file)
    return {
        "name": case["name"],
        "family": case["family"],
        "label": case["label"],
        "returncode": returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "avg_invested_pct": exposure["avg_invested_pct"],
        "max_active_positions": exposure["max_active_positions"],
        "exposure_points": exposure["exposure_points"],
        "log_file": str(log_file),
        "env": json.dumps(case["env"], ensure_ascii=False, sort_keys=True),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "-"


def _num(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _write_report(rows: list[dict]) -> None:
    baseline = next(row for row in rows if row["name"] == "baseline_prod")
    best_by_family: dict[str, dict] = {}
    for row in rows:
        if row["family"] == "baseline" or row["annual"] is None:
            continue
        current = best_by_family.get(row["family"])
        if current is None or float(row["annual"]) > float(current["annual"]):
            best_by_family[row["family"]] = row
    md_lines = [
        "# 生产策略卖出规则顺序实验",
        "",
        "## 当前结论",
        "",
        f"- 基线策略年化 `{_pct(baseline['annual'])}`，Sharpe `{_num(baseline['sharpe'])}`，最大回撤 `{_pct(baseline['max_drawdown'])}`。",
    ]
    for family in ["absolute_score", "rank_threshold", "decay_ratio"]:
        row = best_by_family.get(family)
        if row is None:
            continue
        delta = float(row["annual"]) - float(baseline["annual"])
        md_lines.append(
            f"- `{family}` 最优为 `{row['name']}`：年化 `{_pct(row['annual'])}`，较基线 {'提升' if delta >= 0 else '下降'} `{_pct(abs(delta))}`，Sharpe `{_num(row['sharpe'])}`，回撤 `{_pct(row['max_drawdown'])}`。"
        )
    md_lines.extend(
        [
            "",
            "## 全量结果",
            "",
            "| 方案 | 类型 | 年化 | Sharpe | 最大回撤 | 胜率 | 开/平仓 | 平均仓位 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        md_lines.append(
            f"| {row['name']} | {row['family']} | {_pct(row['annual'])} | {_num(row['sharpe'])} | {_pct(row['max_drawdown'])} | "
            f"{_pct(row['win_ratio'])} | {row['open_count']}/{row['close_count']} | {_pct(row['avg_invested_pct'])} |"
        )
    md_lines.extend(
        [
            "",
            "## 研究边界",
            "",
            "- 本次只复用生产信号文件和现有掘金回测入口做 research-only 对比。",
            "- 绝对分数阈值通过临时复制的回测策略目录增加研究开关 `GM_SCORE_EXIT_PRED`，未修改生产策略目录。",
            "- 未生成正式生产信号，未修改 L5 已发布参数。",
            "",
            "## 证据路径",
            "",
            f"- 汇总 CSV：`{REPORT_DIR / 'summary.csv'}`",
            f"- 研究报告：`{REPORT_DIR / 'report.md'}`",
            f"- 回测日志目录：`{REPORT_DIR / 'logs'}`",
            f"- 临时策略目录：`{TMP_STRATEGY_DIR}`",
        ]
    )
    (REPORT_DIR / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if TMP_STRATEGY_DIR.exists():
        shutil.rmtree(TMP_STRATEGY_DIR)
    rows = [_run_case(case) for case in CASES]
    frame = pd.DataFrame(rows).sort_values(["family", "annual"], ascending=[True, False], na_position="last")
    summary_rows = frame.to_dict("records")
    _write_csv(REPORT_DIR / "summary.csv", summary_rows)
    _write_report(summary_rows)
    print(json.dumps({"report_dir": str(REPORT_DIR), "rows": len(summary_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

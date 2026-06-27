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

import research_prod_exit_rule_sequence_20260625 as base


REPORT_DIR = base.DATA / "reports" / "strategy_agent_prod_exit_combo_refine_20260625"
TMP_STRATEGY_DIR = REPORT_DIR / "strategy_abs_exit_patch"


CASES = [
    {
        "name": "baseline_prod",
        "family": "baseline",
        "label": "生产基线",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_abs0999",
        "family": "absolute_plus_ratio",
        "label": "比例0.97 + 绝对分数0.999",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_EXIT_PRED": "0.999",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_abs09985",
        "family": "absolute_plus_ratio",
        "label": "比例0.97 + 绝对分数0.9985",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_EXIT_PRED": "0.9985",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_abs0998",
        "family": "absolute_plus_ratio",
        "label": "比例0.97 + 绝对分数0.998",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_EXIT_PRED": "0.998",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_rank0988",
        "family": "rank_plus_ratio",
        "label": "比例0.97 + 排名分位0.988",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_EXIT_RANK": "0.988",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_rank0985",
        "family": "rank_plus_ratio",
        "label": "比例0.97 + 排名分位0.985",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_EXIT_RANK": "0.985",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_rank0982",
        "family": "rank_plus_ratio",
        "label": "比例0.97 + 排名分位0.982",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_EXIT_RANK": "0.982",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_drop0988",
        "family": "drop_plus_ratio",
        "label": "比例0.97 + 单日衰退0.988",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.988",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_drop0985",
        "family": "drop_plus_ratio",
        "label": "比例0.97 + 单日衰退0.985",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.985",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_drop0982",
        "family": "drop_plus_ratio",
        "label": "比例0.97 + 单日衰退0.982",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.982",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "1",
        },
    },
    {
        "name": "ratio097_abs09985_mh2",
        "family": "absolute_plus_ratio",
        "label": "比例0.97 + 绝对分数0.9985 + 最少持有2天",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_EXIT_PRED": "0.9985",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
        },
    },
    {
        "name": "ratio097_rank0985_mh2",
        "family": "rank_plus_ratio",
        "label": "比例0.97 + 排名分位0.985 + 最少持有2天",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_EXIT_RANK": "0.985",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
        },
    },
    {
        "name": "ratio097_drop0985_mh2",
        "family": "drop_plus_ratio",
        "label": "比例0.97 + 单日衰退0.985 + 最少持有2天",
        "env": {
            "GM_SCORE_EXIT_ENTRY_RATIO": "0.97",
            "GM_SCORE_STOP_LOSS_DAY_DROP_RATIO": "0.985",
            "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2",
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
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        actives.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(actives) if actives else None,
        "exposure_points": len(values),
    }


def _ensure_abs_exit_strategy() -> Path:
    target_main = TMP_STRATEGY_DIR / "main.py"
    if target_main.exists():
        return TMP_STRATEGY_DIR
    if TMP_STRATEGY_DIR.exists():
        shutil.rmtree(TMP_STRATEGY_DIR)
    shutil.copytree(base.SOURCE_STRATEGY_DIR, TMP_STRATEGY_DIR)
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
    env.update(base.BASE_ENV)
    env.update(case["env"])
    command = [
        str(base.JUEJIN_PYTHON),
        str(base.MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(strategy_dir),
        "--signal-file",
        str(base.BASE_SIGNAL),
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
        str(base.SCORE_DB),
        "--score-table",
        base.SCORE_TABLE,
        "--market-db",
        str(base.DATA / "STOCK_DAILY_DATA.db"),
        "--backtest-start",
        base.BACKTEST_START,
        "--backtest-end",
        base.BACKTEST_END,
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
        proc = subprocess.run(command, cwd=str(base.MAIN), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    indicator = _extract_indicator(log_file)
    exposure = _exposure_stats(log_file)
    return {
        "name": case["name"],
        "family": case["family"],
        "label": case["label"],
        "returncode": proc.returncode,
        "annual": indicator.get("pnl_ratio_annual") if indicator else None,
        "pnl_ratio": indicator.get("pnl_ratio") if indicator else None,
        "sharpe": indicator.get("sharp_ratio") if indicator else None,
        "max_drawdown": indicator.get("max_drawdown") if indicator else None,
        "win_ratio": indicator.get("win_ratio") if indicator else None,
        "open_count": indicator.get("open_count") if indicator else None,
        "close_count": indicator.get("close_count") if indicator else None,
        "avg_invested_pct": exposure["avg_invested_pct"],
        "ge80_ratio": exposure["ge80_ratio"],
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
        "# 生产策略卖出增强组合实验",
        "",
        "## 当前结论",
        "",
        f"- 生产基线：年化 `{_pct(baseline['annual'])}`，Sharpe `{_num(baseline['sharpe'])}`，最大回撤 `{_pct(baseline['max_drawdown'])}`。",
    ]
    for family in ["absolute_plus_ratio", "rank_plus_ratio", "drop_plus_ratio"]:
        row = best_by_family.get(family)
        if row is None:
            continue
        delta = float(row["annual"]) - float(baseline["annual"])
        md_lines.append(
            f"- `{family}` 最优为 `{row['name']}`：年化 `{_pct(row['annual'])}`，较基线"
            f"{'提升' if delta >= 0 else '下降'} `{_pct(abs(delta))}`，Sharpe `{_num(row['sharpe'])}`，最大回撤 `{_pct(row['max_drawdown'])}`。"
        )
    md_lines.extend(
        [
            "",
            "## 全量结果",
            "",
            "| 方案 | 类型 | 年化 | Sharpe | 最大回撤 | 胜率 | 开/平仓 | 平均持仓率 |",
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
            "- 本次只复用当前生产信号、当前生产分数表和掘金标准回测入口做研究筛选。",
            "- 绝对分数阈值通过临时研究策略目录增加 `GM_SCORE_EXIT_PRED` 开关实现，未修改正式生产目录。",
            "- 未生成正式生产信号，未修改 L5 已发布参数。",
            "",
            "## 证据路径",
            "",
            f"- 汇总 CSV：`{REPORT_DIR / 'summary.csv'}`",
            f"- 研究报告：`{REPORT_DIR / 'report.md'}`",
            f"- 掘金日志：`{REPORT_DIR / 'logs'}`",
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

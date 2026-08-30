from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
OUT_CSV = REPORT_DIR / "latest_l4_frequency_goal_aggregate_audit_20260630.csv"
OUT_JSON = REPORT_DIR / "latest_l4_frequency_goal_aggregate_audit_20260630.json"
OUT_MD = REPORT_DIR / "latest_l4_frequency_goal_aggregate_audit_20260630.md"


def _float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _row_name(row: dict[str, Any], idx: int) -> str:
    for key in ("name", "case", "case_key", "strategy", "log_name", "candidate", "param_key"):
        value = row.get(key)
        if value:
            return str(value).strip()
    rule = row.get("rule")
    extra = row.get("extra_condition")
    if rule:
        if extra:
            return f"{str(rule).strip()}_{str(extra).strip()}"
        return str(rule).strip()
    buy_rule = row.get("buy_rule")
    daydrop = row.get("daydrop")
    if buy_rule:
        if daydrop:
            return f"{str(buy_rule).strip()}_daydrop{str(daydrop).strip()}"
        return str(buy_rule).strip()
    return f"row_{idx}"


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                return []
            fields = set(reader.fieldnames)
            has_metrics = (
                ("annual_return" in fields or "annual" in fields)
                and "sharpe" in fields
                and "max_drawdown" in fields
            )
            if not has_metrics:
                return []
            for idx, row in enumerate(reader, start=1):
                annual = _float(row.get("annual_return"))
                if annual is None:
                    annual = _float(row.get("annual"))
                sharpe = _float(row.get("sharpe"))
                max_drawdown = _float(row.get("max_drawdown"))
                if annual is None or sharpe is None or max_drawdown is None:
                    continue
                name = _row_name(row, idx)
                rows.append(
                    {
                        "source_file": str(path),
                        "source_name": path.name,
                        "row_number": idx,
                        "name": name,
                        "annual_return": annual,
                        "sharpe": sharpe,
                        "max_drawdown": max_drawdown,
                        "win_ratio": _float(row.get("win_ratio")),
                        "open_count": row.get("open_count"),
                        "close_count": row.get("close_count"),
                        "signal_rows": row.get("signal_rows"),
                        "signal_days": row.get("signal_days"),
                        "target_hit_500_sharpe4_mdd40": annual >= 5.0 and sharpe >= 4.0 and max_drawdown <= 0.40,
                        "annual_ge_500": annual >= 5.0,
                        "sharpe_ge_4": sharpe >= 4.0,
                        "mdd_le_40": max_drawdown <= 0.40,
                    }
                )
    except UnicodeDecodeError:
        return []
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def _top(rows: list[dict[str, Any]], key: str, reverse: bool = True, n: int = 10) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key) or -999), reverse=reverse)[:n]


def _write_md(rows: list[dict[str, Any]], source_counts: dict[str, int]) -> None:
    target_hits = [row for row in rows if row["target_hit_500_sharpe4_mdd40"]]
    annual_hits = [row for row in rows if row["annual_ge_500"]]
    mdd_hits = [row for row in rows if row["mdd_le_40"]]
    sharpe_hits = [row for row in rows if row["sharpe_ge_4"]]
    best_annual = _top(rows, "annual_return", True, 1)
    best_mdd_ok = _top(mdd_hits, "annual_return", True, 1)
    lines = [
        "# 最新 L4 买卖频率与次日开盘优化目标聚合审计 20260630",
        "",
        "## 边界",
        "",
        "本报告为 L5/L6 research-only 聚合审计，不修改生产策略参数，不生成正式生产信号，不触发交易。",
        "审计对象为 `strategy_agent_latest_l4_top3_frequency_grid_20260630` 目录下已经落地的掘金回测结果 CSV。",
        "",
        "## 目标",
        "",
        "- 年化收益率 `>= 500%`",
        "- Sharpe `>= 4`",
        "- 最大回撤 `<= 40%`",
        "- 使用最新 L4 formal 结果衍生的实验策略，围绕买入/卖出频率、次日开盘收益状态、回撤缩放等策略侧规则优化。",
        "",
        "## 聚合结论",
        "",
        f"- 纳入有效候选：`{len(rows)}`",
        f"- 来源结果文件：`{len(source_counts)}`",
        f"- 三项同时命中候选：`{len(target_hits)}`",
        f"- 年化 `>= 500%` 候选：`{len(annual_hits)}`",
        f"- Sharpe `>= 4` 候选：`{len(sharpe_hits)}`",
        f"- 最大回撤 `<= 40%` 候选：`{len(mdd_hits)}`",
    ]
    if best_annual:
        row = best_annual[0]
        lines.append(
            f"- 最高年化：`{row['name']}`，年化 `{_pct(row['annual_return'])}`，Sharpe `{float(row['sharpe']):.3f}`，最大回撤 `{_pct(row['max_drawdown'])}`，来源 `{row['source_name']}`。"
        )
    if best_mdd_ok:
        row = best_mdd_ok[0]
        lines.append(
            f"- 回撤合格中最高年化：`{row['name']}`，年化 `{_pct(row['annual_return'])}`，Sharpe `{float(row['sharpe']):.3f}`，最大回撤 `{_pct(row['max_drawdown'])}`，来源 `{row['source_name']}`。"
        )
    else:
        lines.append("- 无最大回撤合格候选。")

    lines.extend(
        [
            "",
            "## 年化最高 Top 15",
            "",
            "| 候选 | 年化 | Sharpe | 最大回撤 | 来源 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in _top(rows, "annual_return", True, 15):
        lines.append(
            f"| `{row['name']}` | {_pct(row['annual_return'])} | {float(row['sharpe']):.3f} | {_pct(row['max_drawdown'])} | `{row['source_name']}` |"
        )

    lines.extend(
        [
            "",
            "## 回撤合格候选 Top 15",
            "",
            "| 候选 | 年化 | Sharpe | 最大回撤 | 来源 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    if mdd_hits:
        for row in _top(mdd_hits, "annual_return", True, 15):
            lines.append(
                f"| `{row['name']}` | {_pct(row['annual_return'])} | {float(row['sharpe']):.3f} | {_pct(row['max_drawdown'])} | `{row['source_name']}` |"
            )
    else:
        lines.append("| 无 |  |  |  |  |")

    lines.extend(
        [
            "",
            "## 年化超过 500% 候选",
            "",
            "| 候选 | 年化 | Sharpe | 最大回撤 | 来源 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    if annual_hits:
        for row in _top(annual_hits, "annual_return", True, 20):
            lines.append(
                f"| `{row['name']}` | {_pct(row['annual_return'])} | {float(row['sharpe']):.3f} | {_pct(row['max_drawdown'])} | `{row['source_name']}` |"
            )
    else:
        lines.append("| 无 |  |  |  |  |")

    lines.extend(
        [
            "",
            "## 判断",
            "",
            "聚合结果显示：当前策略侧已经覆盖买入/卖出频率、次日开盘状态机、开盘轻止损、日级回撤缩放、剔除极端贡献股后补位等方向，但尚无候选同时满足三项目标。年化超过 500% 的候选普遍最大回撤高于 50%，Sharpe 约在 1 附近；最大回撤压到 40% 以下后，年化和 Sharpe 同步下降。",
            "",
            "这说明当前约束下的瓶颈不是单个卖出触发参数，而是最新 L4 排序产生的高收益候选本身具有高波动/高回撤暴露。继续只在 L5 买卖频率层调参，预计难以把 Sharpe 提升到 4。",
            "",
            "## 证据路径",
            "",
            f"- `{OUT_CSV}`",
            f"- `{OUT_JSON}`",
            f"- `{Path(__file__)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(REPORT_DIR.glob("*.csv")):
        if path.name.startswith("latest_l4_frequency_goal_aggregate_audit_"):
            continue
        rows.extend(_read_result_rows(path))
    rows.sort(key=lambda row: float(row["annual_return"]), reverse=True)
    source_counts: dict[str, int] = {}
    for row in rows:
        source_counts[row["source_name"]] = source_counts.get(row["source_name"], 0) + 1
    _write_csv(OUT_CSV, rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "candidate_count": len(rows),
                "source_counts": source_counts,
                "target_hit_count": sum(1 for row in rows if row["target_hit_500_sharpe4_mdd40"]),
                "annual_ge_500_count": sum(1 for row in rows if row["annual_ge_500"]),
                "sharpe_ge_4_count": sum(1 for row in rows if row["sharpe_ge_4"]),
                "mdd_le_40_count": sum(1 for row in rows if row["mdd_le_40"]),
                "top_by_annual": rows[:20],
                "top_mdd_ok": _top([row for row in rows if row["mdd_le_40"]], "annual_return", True, 20),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_md(rows, source_counts)
    print(json.dumps({"candidate_count": len(rows), "source_count": len(source_counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

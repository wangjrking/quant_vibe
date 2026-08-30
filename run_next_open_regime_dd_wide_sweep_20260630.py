from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import run_next_open_regime_dd_resize_probe_20260630 as base


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_top3_frequency_grid_20260630"
OUT_LOG_DIR = REPORT_DIR / "next_open_regime_dd_wide_sweep_logs"
OUT_CSV = REPORT_DIR / "next_open_regime_dd_wide_sweep_20260630.csv"
OUT_JSON = REPORT_DIR / "next_open_regime_dd_wide_sweep_20260630.json"
OUT_MD = REPORT_DIR / "next_open_regime_dd_wide_sweep_summary_20260630.md"


SIGNALS = {
    "nors_top3_w20": REPORT_DIR / "next_open_regime_state_signals" / "nors_top3_w20_soft_h2m3.csv",
    "nors_top3_w40hard": REPORT_DIR / "next_open_regime_state_signals" / "nors_top3_w40_hard_h2m3.csv",
    "orig_top3": REPORT_DIR
    / "timing_high_exposure_extension_signals"
    / "thex_t55_top3_s1000_cap99_h2m3_ddoff.csv",
    "orig_top5": REPORT_DIR
    / "timing_high_exposure_extension_signals"
    / "thex_t55_top5_s1000_cap99_h2m3_ddoff.csv",
}


DD_PROFILES = [
    ("dd_10_18_s75_45", 0.10, 0.18, 0.05, 0.75, 0.45),
    ("dd_12_20_s80_55", 0.12, 0.20, 0.06, 0.80, 0.55),
    ("dd_15_25_s85_65", 0.15, 0.25, 0.08, 0.85, 0.65),
    ("dd_20_30_s90_70", 0.20, 0.30, 0.10, 0.90, 0.70),
]


def _case(
    signal_key: str,
    profile: tuple[str, float, float, float, float, float],
    *,
    max_positions: int,
    target_position_pct: float = 0.99,
) -> dict[str, Any]:
    suffix, soft, hard, recover, soft_scale, hard_scale = profile
    return {
        "name": f"{signal_key}_{suffix}",
        "signal_file": SIGNALS[signal_key],
        "max_positions": max_positions,
        "holding_days": 2,
        "max_holding_days": 3,
        "target_position_pct": target_position_pct,
        "score_exit": 0.98,
        "score_continue": 0.99,
        "max_daily_sells": 1,
        "soft_trigger": soft,
        "hard_trigger": hard,
        "recover_trigger": recover,
        "soft_scale": soft_scale,
        "hard_scale": hard_scale,
    }


CASES: list[dict[str, Any]] = []
for profile in DD_PROFILES:
    CASES.append(_case("nors_top3_w20", profile, max_positions=3))
    CASES.append(_case("orig_top3", profile, max_positions=3))
for profile in DD_PROFILES[:3]:
    CASES.append(_case("nors_top3_w40hard", profile, max_positions=3))
    CASES.append(_case("orig_top5", profile, max_positions=5))


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
        "# 次日开盘状态机与宽松日级回撤缩放扫描 20260630",
        "",
        "## 边界",
        "",
        "本报告为 L5/L6 research-only 验证，不修改生产策略参数，不生成正式生产信号，不触发交易。",
        "本轮只改变日级账户回撤缩放触发阈值与缩放强度，用于检查是否存在收益和回撤之间更好的折中点。",
        "",
        "## 结果摘要",
        "",
        f"- 新增候选数量：`{len(results)}`",
        f"- 命中 `年化 >= 500% / Sharpe >= 4 / 最大回撤 <= 40%`：`{len(target_hits)}`",
    ]
    if ordered:
        best = ordered[0]
        lines.append(
            f"- 本轮最高年化：`{best['name']}`，年化 `{_pct(best.get('annual_return'))}`，Sharpe `{float(best.get('sharpe') or 0):.3f}`，最大回撤 `{_pct(best.get('max_drawdown'))}`。"
        )
    if mdd_ok:
        best_ok = sorted(mdd_ok, key=lambda row: float(row.get("annual_return") or -999), reverse=True)[0]
        lines.append(
            f"- 最大回撤不超过 40% 的最高年化：`{best_ok['name']}`，年化 `{_pct(best_ok.get('annual_return'))}`，Sharpe `{float(best_ok.get('sharpe') or 0):.3f}`，最大回撤 `{_pct(best_ok.get('max_drawdown'))}`。"
        )
    else:
        lines.append("- 本轮没有最大回撤不超过 40% 的候选。")
    lines.extend(
        [
            "",
            "## Top 12",
            "",
            "| 候选 | 年化 | Sharpe | 最大回撤 | 开仓 | 平仓 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ordered[:12]:
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
            "## 判断",
            "",
            "如果宽松回撤缩放能把年化拉回 500% 附近但最大回撤仍高于 40%，说明回撤主要来自高波动持仓本身；如果回撤达标但年化低于 500%，说明收益和回撤约束在当前模型排序下仍无法同时满足。",
            "",
            "## 证据路径",
            "",
            f"- `{OUT_CSV}`",
            f"- `{OUT_JSON}`",
            f"- `{OUT_LOG_DIR}`",
            f"- `{Path(__file__)}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base.OUT_LOG_DIR = OUT_LOG_DIR
    results: list[dict[str, Any]] = []
    for case in CASES:
        row = base._run(case)
        results.append(row)
        _write_rows(OUT_CSV, results)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_report(results)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

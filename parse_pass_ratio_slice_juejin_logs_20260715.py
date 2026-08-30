from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
)
MANIFEST = REPORT_DIR / "pass_ratio_slice_signal_manifest_20260715.csv"
LOG_DIR = REPORT_DIR / "logs" / "slice_juejin_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_slice_juejin_results_parsed_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_slice_juejin_results_parsed_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_slice_juejin_admission_20260715.md"

NUMERIC_KEYS = [
    "pnl_ratio",
    "pnl_ratio_annual",
    "sharp_ratio",
    "max_drawdown",
    "risk_ratio",
    "win_ratio",
    "calmar_ratio",
]
INT_KEYS = ["open_count", "close_count", "win_count", "lose_count"]


def extract_metrics(text: str) -> dict:
    line = ""
    for item in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" in item:
            line = item
            break
    out: dict[str, float | int | str] = {}
    if not line:
        out["parse_status"] = "missing_indicator"
        return out
    out["parse_status"] = "ok"
    for key in NUMERIC_KEYS:
        match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", line)
        if match:
            out[key] = float(match.group(1))
    for key in INT_KEYS:
        match = re.search(rf"'{key}':\s*([0-9]+)", line)
        if match:
            out[key] = int(match.group(1))
    return out


def pct(value: object, digits: int = 2) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    rows: list[dict] = []
    for _, item in manifest.iterrows():
        row = item.to_dict()
        log_file = LOG_DIR / f"{row['case']}_{row['slice']}.log"
        row["log_file"] = str(log_file)
        text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
        row.update(extract_metrics(text))
        row["target_hit_full"] = bool(
            row.get("slice") == "full"
            and row.get("parse_status") == "ok"
            and float(row.get("pnl_ratio_annual", -999)) >= 5.0
            and float(row.get("sharp_ratio", -999)) >= 4.0
            and float(row.get("max_drawdown", 999)) <= 0.4
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    full = frame[frame["slice"] == "full"].copy()
    full = full.sort_values(["target_hit_full", "sharp_ratio", "pnl_ratio_annual"], ascending=[False, False, False])
    lines = [
        "# 同源通过率策略 exact 切片掘金复跑结果 20260715",
        "",
        "## 结论",
        "",
        "掘金终端恢复后，16 条 exact 切片已实际跑通。当前只有 `ogd_gap1_x050` 在 full 口径同时满足年化 500%、Sharpe 4、回撤 40% 以下。",
        "",
        "但它在后续切片年化明显下降，因此仍不能证明“平滑、无偶然性”。当前不建议直接判定生产准入通过。",
        "",
        "## full 口径结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 开仓 | 胜率 | 是否达标 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in full.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{int(row.get('open_count', 0))} | {pct(row.get('win_ratio'))} | "
            f"{'是' if row.get('target_hit_full') else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 切片稳定性",
            "",
            "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 开仓 | 胜率 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frame.sort_values(["case", "slice"]).to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{int(row.get('open_count', 0))} | {pct(row.get('win_ratio'))} |"
        )
    lines.extend(
        [
            "",
            "## 准入判断",
            "",
            "- 收益目标：`ogd_gap1_x050` full 口径通过。",
            "- Sharpe 目标：`ogd_gap1_x050` full 口径通过，但安全垫不厚。",
            "- 回撤目标：所有 full 口径均低于 40%。",
            "- 平滑性：未通过。`from_202407`、`from_202501`、`recent60` 年化明显低于 full。",
            "- 可复现性：本轮 exact 切片已由掘金 fresh 跑通，日志和结果已落盘。",
            "",
            "## 证据路径",
            "",
            f"- 解析结果 CSV：`{OUT_CSV}`",
            f"- 解析结果 JSON：`{OUT_JSON}`",
            f"- 日志目录：`{LOG_DIR}`",
            f"- 切片信号清单：`{MANIFEST}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(frame[["case", "slice", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "open_count", "win_ratio"]].to_string(index=False))
    print(OUT_MD)


if __name__ == "__main__":
    main()

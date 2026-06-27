from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
TRIAGE_DIR = DATA_DIR / "reports" / "model_agent_research_triage_v2_20260625"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_stability_v2_20260625"

DAILY_CSV = TRIAGE_DIR / "unified_model_eval_daily_v2.csv"
SUMMARY_CSV = TRIAGE_DIR / "unified_model_eval_summary_v2.csv"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def period_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    d = frame["trade_date"].astype(str)
    if name == "2024H2":
        return (d >= "20240604") & (d <= "20241231")
    if name == "2025H1":
        return (d >= "20250101") & (d <= "20250630")
    if name == "2025H2":
        return (d >= "20250701") & (d <= "20251231")
    if name == "2026YTD":
        return d >= "20260101"
    raise ValueError(name)


def recent_mask(frame: pd.DataFrame, n: int) -> pd.Series:
    dates = sorted(frame["trade_date"].astype(str).unique().tolist())
    keep = set(dates[-n:])
    return frame["trade_date"].astype(str).isin(keep)


def summarize_slice(frame: pd.DataFrame, asset: str, label: str, track: str, period: str, sub: pd.DataFrame) -> dict:
    row = {
        "asset": asset,
        "label": label,
        "track": track,
        "period": period,
        "days": int(sub["trade_date"].nunique()),
        "min_trade_date": str(sub["trade_date"].min()) if len(sub) else None,
        "max_trade_date": str(sub["trade_date"].max()) if len(sub) else None,
    }
    for col in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
        row[col] = float(sub[col].mean()) if len(sub) else np.nan
    return row


def build_period_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset, label, track), group in daily.groupby(["asset", "label", "track"], sort=False):
        for period in ["2024H2", "2025H1", "2025H2", "2026YTD"]:
            sub = group[period_mask(group, period)]
            rows.append(summarize_slice(group, asset, label, track, period, sub))
        for n in [126, 63, 20]:
            sub = group[recent_mask(group, n)]
            rows.append(summarize_slice(group, asset, label, track, f"recent{n}", sub))
    return pd.DataFrame(rows)


def add_deltas(period_summary: pd.DataFrame) -> pd.DataFrame:
    baseline_by_label = {
        "executable_1d_open_return": "formal_1d",
        "executable_3d_open_return": "formal_3d",
        "executable_5d_open_return": "formal_5d",
        "executable_10d_open_return": "formal_10d",
    }
    out = period_summary.copy()
    metric_cols = ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    for idx, row in out.iterrows():
        base_asset = baseline_by_label[row["label"]]
        base = out[(out["asset"] == base_asset) & (out["period"] == row["period"])].iloc[0]
        out.loc[idx, "baseline_asset"] = base_asset
        for col in metric_cols:
            out.loc[idx, f"{col}_delta"] = float(row[col] - base[col])
    return out


def build_stability_score(period_delta: pd.DataFrame) -> pd.DataFrame:
    research = period_delta[period_delta["track"] == "research"].copy()
    rows = []
    for (asset, label), group in research.groupby(["asset", "label"], sort=False):
        full_like = group[group["period"].isin(["2024H2", "2025H1", "2025H2", "2026YTD"])]
        recent = group[group["period"] == "recent63"].iloc[0]
        recent20 = group[group["period"] == "recent20"].iloc[0]
        pos_top5_periods = int((full_like["top5_delta"] > 0).sum())
        pos_top1_periods = int((full_like["top1_delta"] > 0).sum())
        min_top5_delta = float(full_like["top5_delta"].min())
        min_rank_ic_delta = float(full_like["rank_ic_delta"].min())
        avg_period_top5_delta = float(full_like["top5_delta"].mean())
        avg_period_rank_ic_delta = float(full_like["rank_ic_delta"].mean())
        stability_score = (
            3.0 * float(recent["top1_delta"])
            + 2.0 * float(recent["top5_delta"])
            + float(recent["top10_delta"])
            + float(recent20["top5_delta"])
            + 0.5 * avg_period_top5_delta
            + 0.2 * avg_period_rank_ic_delta
            + 0.01 * pos_top5_periods
            - 0.05 * max(0.0, -min_top5_delta)
            - 0.02 * max(0.0, -min_rank_ic_delta)
        )
        rows.append(
            {
                "asset": asset,
                "label": label,
                "recent63_top1_delta": float(recent["top1_delta"]),
                "recent63_top5_delta": float(recent["top5_delta"]),
                "recent63_top10_delta": float(recent["top10_delta"]),
                "recent63_rank_ic_delta": float(recent["rank_ic_delta"]),
                "recent20_top5_delta": float(recent20["top5_delta"]),
                "positive_top5_periods": pos_top5_periods,
                "positive_top1_periods": pos_top1_periods,
                "min_period_top5_delta": min_top5_delta,
                "min_period_rank_ic_delta": min_rank_ic_delta,
                "avg_period_top5_delta": avg_period_top5_delta,
                "avg_period_rank_ic_delta": avg_period_rank_ic_delta,
                "stability_score": stability_score,
            }
        )
    return pd.DataFrame(rows).sort_values("stability_score", ascending=False)


def write_report(period_delta: pd.DataFrame, stability: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    period_path = REPORT_DIR / "research_period_stability_v2.csv"
    stability_path = REPORT_DIR / "research_stability_ranking_v2.csv"
    period_delta.to_csv(period_path, index=False, encoding="utf-8-sig")
    stability.to_csv(stability_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_period_stability_v2",
        "source_daily_csv": str(DAILY_CSV),
        "source_summary_csv": str(SUMMARY_CSV),
        "period_csv": str(period_path),
        "ranking_csv": str(stability_path),
        "top_candidates": stability.head(10).to_dict(orient="records"),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "research_stability_v2_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 模型研究稳定性压力测试 V2（20260625）",
        "",
        "## 当前结论",
        "",
        "本报告在统一评价 V2 的日度指标基础上，按 2024H2、2025H1、2025H2、2026YTD、recent126、recent63、recent20 分段检查研究候选相对同标签 formal baseline 的稳定性。",
        "",
        "稳定性评分不会替代生产审批；它只用于模型侧筛掉过度依赖单一近期窗口的候选。",
        "",
        "## 稳定性排序",
        "",
        "| 候选 | 标签 | 稳定性分 | 近63日Top1增量 | 近63日Top5增量 | 正Top5阶段数 | 最差阶段Top5增量 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in stability.head(10).iterrows():
        lines.append(
            f"| `{row['asset']}` | `{row['label']}` | {row['stability_score']:.6f} | {row['recent63_top1_delta']:.6f} | {row['recent63_top5_delta']:.6f} | {int(row['positive_top5_periods'])} | {row['min_period_top5_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 生产边界",
            "",
            "- 本轮只生成研究稳定性报告。",
            "- 未训练模型。",
            "- 未修改 production/formal manifest。",
            "- 未写入 formal L4 表。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
            "",
            "## 证据路径",
            "",
            f"- 分段稳定性表：`{period_path.as_posix()}`",
            f"- 稳定性排序表：`{stability_path.as_posix()}`",
            f"- JSON 摘要：`{(REPORT_DIR / 'research_stability_v2_summary.json').as_posix()}`",
        ]
    )
    (REPORT_DIR / "research_stability_v2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    daily = pd.read_csv(DAILY_CSV, dtype={"trade_date": str})
    period_summary = build_period_summary(daily)
    period_delta = add_deltas(period_summary)
    stability = build_stability_score(period_delta)
    write_report(period_delta, stability)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "period_rows": int(len(period_delta)),
                "candidate_rows": int(len(stability)),
                "top_candidates": stability.head(5).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

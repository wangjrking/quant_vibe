from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
TRIAGE_DIR = DATA_DIR / "reports" / "model_agent_research_triage_v2_20260625"
STABILITY_DIR = DATA_DIR / "reports" / "model_agent_research_stability_v2_20260625"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_risk_adjusted_20260625"

RANKING_CSV = TRIAGE_DIR / "research_candidate_ranking_v2.csv"
STABILITY_CSV = STABILITY_DIR / "research_stability_ranking_v2.csv"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def minmax_by_label(frame: pd.DataFrame, col: str) -> pd.Series:
    out = pd.Series(index=frame.index, dtype=float)
    for _, idx in frame.groupby("label").groups.items():
        sub = frame.loc[idx, col].astype(float)
        lo = float(sub.min())
        hi = float(sub.max())
        if abs(hi - lo) < 1e-12:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (sub - lo) / (hi - lo)
    return out


def classify(row: pd.Series) -> str:
    if row["risk_adjusted_score"] <= 0:
        return "reject"
    if row["rank_ic_penalty"] > 0.08 or row["period_rank_ic_penalty"] > 0.08:
        return "research_only_rankic_risk"
    if row["risk_adjusted_score"] >= 0.5 and row["rank_ic_penalty"] < 0.03 and row["period_rank_ic_penalty"] < 0.04:
        return "strong_research_candidate"
    return "research_candidate_watch"


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking = pd.read_csv(RANKING_CSV)
    stability = pd.read_csv(STABILITY_CSV)
    research = ranking[ranking["track"] == "research"].merge(
        stability,
        on=["asset", "label"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_stable"),
    )
    if research[["stability_score", "min_period_rank_ic_delta"]].isna().any().any():
        missing = research[research["stability_score"].isna()][["asset", "label"]].to_dict(orient="records")
        raise RuntimeError(f"missing stability rows: {missing}")

    research = research.copy()
    research["objective_norm"] = minmax_by_label(research, "research_objective_delta")
    research["stability_norm"] = minmax_by_label(research, "stability_score")
    research["top_recent_norm"] = minmax_by_label(research, "top1_delta_recent63")
    research["rank_ic_penalty"] = (-research["rank_ic_delta_full"].clip(upper=0.0)).astype(float)
    research["period_rank_ic_penalty"] = (-research["min_period_rank_ic_delta"].clip(upper=0.0)).astype(float)
    research["period_top5_penalty"] = (-research["min_period_top5_delta"].clip(upper=0.0)).astype(float)
    research["risk_adjusted_score"] = (
        0.45 * research["objective_norm"]
        + 0.35 * research["stability_norm"]
        + 0.20 * research["top_recent_norm"]
        - 12.0 * research["rank_ic_penalty"]
        - 6.0 * research["period_rank_ic_penalty"]
        - 4.0 * research["period_top5_penalty"]
    )
    research["risk_adjusted_decision"] = research.apply(classify, axis=1)
    research = research.sort_values(["label", "risk_adjusted_score"], ascending=[True, False]).reset_index(drop=True)
    best = research.groupby("label", sort=True).head(1).reset_index(drop=True)
    return research, best


def write_report(ranking: pd.DataFrame, best: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_path = REPORT_DIR / "risk_adjusted_research_ranking.csv"
    best_path = REPORT_DIR / "risk_adjusted_best_by_label.csv"
    json_path = REPORT_DIR / "risk_adjusted_research_summary.json"
    md_path = REPORT_DIR / "risk_adjusted_research_report.md"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_risk_adjusted_model_ranking",
        "source_ranking_csv": str(RANKING_CSV),
        "source_stability_csv": str(STABILITY_CSV),
        "ranking_csv": str(ranking_path),
        "best_by_label_csv": str(best_path),
        "score_formula": (
            "0.45*objective_norm + 0.35*stability_norm + 0.20*top_recent_norm "
            "- 12*max(0,-rank_ic_delta_full) - 6*max(0,-min_period_rank_ic_delta) "
            "- 4*max(0,-min_period_top5_delta)"
        ),
        "best_by_label": best[
            [
                "label",
                "asset",
                "risk_adjusted_score",
                "risk_adjusted_decision",
                "research_objective_delta",
                "stability_score",
                "rank_ic_delta_full",
                "min_period_rank_ic_delta",
            ]
        ].to_dict(orient="records"),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_prediction_table_write": True,
            "no_production_manifest_change": True,
            "no_formal_l4_write": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 模型研究风险调整排名（20260625）",
        "",
        "## 当前结论",
        "",
        "本报告在统一 Top 指标评价之外加入 RankIC 风险惩罚，用于区分“Top 指标增强”与“更适合作为后续研究主线”的候选。它只服务研究决策，不代表生产发布。",
        "",
        "## 评分公式",
        "",
        "```text",
        payload["score_formula"],
        "```",
        "",
        "## 各标签风险调整后最优",
        "",
        "| 标签 | 候选 | 风险调整分 | 决策 | objective | stability | 全样本RankIC增量 | 最差阶段RankIC增量 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| `{row['label']}` | `{row['asset']}` | {row['risk_adjusted_score']:.6f} | "
            f"`{row['risk_adjusted_decision']}` | {row['research_objective_delta']:.6f} | "
            f"{row['stability_score']:.6f} | {row['rank_ic_delta_full']:.6f} | "
            f"{row['min_period_rank_ic_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 未训练模型。",
            "- 未写入新的预测表。",
            "- 未修改 production/formal manifest。",
            "- 未写入 formal L4 表。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
            "",
            "## 证据路径",
            "",
            f"- 风险调整全量排名：`{ranking_path.as_posix()}`",
            f"- 各标签最优：`{best_path.as_posix()}`",
            f"- JSON 摘要：`{json_path.as_posix()}`",
            f"- 源 Top 排名：`{RANKING_CSV.as_posix()}`",
            f"- 源稳定性排名：`{STABILITY_CSV.as_posix()}`",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ranking, best = build()
    write_report(ranking, best)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "candidate_rows": int(len(ranking)),
                "best_by_label": best[
                    ["label", "asset", "risk_adjusted_score", "risk_adjusted_decision"]
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
TRIAGE_DIR = DATA_DIR / "reports" / "model_agent_research_triage_v2_20260625"
STABILITY_DIR = DATA_DIR / "reports" / "model_agent_research_stability_v2_20260625"
RISK_DIR = DATA_DIR / "reports" / "model_agent_research_risk_adjusted_20260625"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_current_best_research_snapshot_20260625_v4"

RANKING_CSV = TRIAGE_DIR / "research_candidate_ranking_v2.csv"
STABILITY_CSV = STABILITY_DIR / "research_stability_ranking_v2.csv"
RISK_BEST_CSV = RISK_DIR / "risk_adjusted_best_by_label.csv"


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_stats(conn: sqlite3.Connection, table: str) -> dict[str, int | str]:
    row = conn.execute(
        f"""
        select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
               sum(case when pred_prob is null then 1 else 0 end)
        from {quote(table)}
        """
    ).fetchone()
    dup = conn.execute(
        f"""
        select count(*)
        from (
          select trade_date, stock_code, count(*) c
          from {quote(table)}
          group by trade_date, stock_code
          having c > 1
        )
        """
    ).fetchone()[0]
    latest = conn.execute(
        f"""
        select trade_date, count(*), count(distinct stock_code)
        from {quote(table)}
        group by trade_date
        order by trade_date desc
        limit 1
        """
    ).fetchone()
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_trade_date": str(latest[0]),
        "latest_day_rows": int(latest[1]),
        "latest_day_stock_count": int(latest[2]),
    }


def pick_metric(frame: pd.DataFrame, asset: str, label: str) -> dict:
    row = frame[(frame["asset"] == asset) & (frame["label"] == label)]
    if row.empty:
        raise RuntimeError(f"missing metric row for {asset} / {label}")
    return row.iloc[0].to_dict()


def asset_table_map() -> dict[tuple[str, str], str]:
    from build_model_research_triage_v2_20260625 import ASSETS

    return {(str(item["label"]), str(item["asset"])): str(item["table"]) for item in ASSETS}


def risk_note(risk_row: dict, stable_row: dict) -> str:
    decision = str(risk_row["risk_adjusted_decision"])
    rank_ic_delta = float(risk_row["rank_ic_delta_full"])
    min_rank_ic_delta = float(stable_row["min_period_rank_ic_delta"])
    if decision == "strong_research_candidate" and rank_ic_delta >= -0.001 and min_rank_ic_delta >= -0.003:
        return "风险调整后属于强研究候选，RankIC 损失较小，可继续作为研究主线。"
    if decision == "strong_research_candidate":
        return "风险调整后属于强研究候选，Top 指标改善明显，但仍需关注 RankIC 或分段稳定性。"
    if decision == "research_only_rankic_risk":
        return "可保留为研究候选，但 RankIC 风险偏高，不适合直接作为生产替代。"
    return "仅保留为研究观察对象，不作为生产候选。"


def build_snapshot() -> dict:
    ranking = pd.read_csv(RANKING_CSV)
    stability = pd.read_csv(STABILITY_CSV)
    risk_best = pd.read_csv(RISK_BEST_CSV)
    tables = asset_table_map()
    rows = []
    with sqlite3.connect(MODEL_DB) as conn:
        for _, best_row in risk_best.sort_values("label").iterrows():
            label = str(best_row["label"])
            asset = str(best_row["asset"])
            table = tables.get((label, asset))
            if not table:
                raise RuntimeError(f"missing table mapping for {label} / {asset}")
            rank_row = pick_metric(ranking, asset, label)
            stable_row = pick_metric(stability, asset, label)
            stats = table_stats(conn, table)
            rows.append(
                {
                    "label": label,
                    "asset": asset,
                    "table": table,
                    "risk_adjusted_score": float(best_row["risk_adjusted_score"]),
                    "risk_adjusted_decision": str(best_row["risk_adjusted_decision"]),
                    "research_objective_delta": float(rank_row["research_objective_delta"]),
                    "stability_score": float(stable_row["stability_score"]),
                    "top1_delta_recent63": float(rank_row["top1_delta_recent63"]),
                    "top5_delta_recent63": float(rank_row["top5_delta_recent63"]),
                    "top5_delta_recent20": float(rank_row["top5_delta_recent20"]),
                    "top5_delta_full": float(rank_row["top5_delta_full"]),
                    "rank_ic_delta_full": float(rank_row["rank_ic_delta_full"]),
                    "min_period_top5_delta": float(stable_row["min_period_top5_delta"]),
                    "min_period_rank_ic_delta": float(stable_row["min_period_rank_ic_delta"]),
                    "positive_top5_periods": int(stable_row["positive_top5_periods"]),
                    "risk_note": risk_note(best_row.to_dict(), stable_row),
                    **stats,
                }
            )
    return {
        "generated_at": now_iso(),
        "scope": "research_only_current_best_snapshot_v4",
        "selection_basis": "risk_adjusted_best_by_label",
        "source_ranking_csv": str(RANKING_CSV),
        "source_stability_csv": str(STABILITY_CSV),
        "source_risk_best_csv": str(RISK_BEST_CSV),
        "model_db": str(MODEL_DB),
        "current_best": rows,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "research_prediction_tables_may_exist": True,
            "no_new_formal_or_production_prediction_table": True,
            "no_production_manifest_change": True,
            "no_formal_l4_write": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }


def write_outputs(snapshot: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "current_best_research_snapshot_v4.json"
    csv_path = REPORT_DIR / "current_best_research_snapshot_v4.csv"
    md_path = REPORT_DIR / "current_best_research_snapshot_v4.md"
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = pd.DataFrame(snapshot["current_best"])
    rows.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = [
        "# 当前研究最优模型快照 V4（20260625）",
        "",
        "## 当前结论",
        "",
        "本快照按风险调整排序，选出四个预测标签当前最优的 research-only 资产，用作后续模型研究基线。它不代表生产发布，也不会修改 formal L4 / L5 入口。",
        "",
        "## 最优资产",
        "",
        "| 标签 | 当前研究最优 | 风险调整分 | 决策 | 目标增量 | 稳定性分 | 近63日Top1增量 | 近63日Top5增量 | 全样本Top5增量 | 全样本RankIC增量 | 风险备注 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in snapshot["current_best"]:
        lines.append(
            "| `{label}` | `{asset}` | {risk_adjusted_score:.6f} | `{risk_adjusted_decision}` | "
            "{research_objective_delta:.6f} | {stability_score:.6f} | {top1_delta_recent63:.6f} | "
            "{top5_delta_recent63:.6f} | {top5_delta_full:.6f} | {rank_ic_delta_full:.6f} | {risk_note} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 覆盖与质量",
            "",
            "| 标签 | 表 | 日期范围 | 最新日行数 | 空分数 | 重复键 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in snapshot["current_best"]:
        lines.append(
            "| `{label}` | `{table}` | `{min_trade_date}` 到 `{max_trade_date}` | "
            "`{latest_day_rows}` | `{null_pred_prob}` | `{duplicate_key_groups}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 未训练模型。",
            "- 未生成新预测表；这里只汇总已有的 research 资产。",
            "- 未修改 production / formal manifest。",
            "- 未写入 formal L4 表。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
            "",
            "## 证据路径",
            "",
            f"- JSON：`{json_path.as_posix()}`",
            f"- CSV：`{csv_path.as_posix()}`",
            f"- 风险调整最优：`{RISK_BEST_CSV.as_posix()}`",
            f"- 统一排序：`{RANKING_CSV.as_posix()}`",
            f"- 稳定性排序：`{STABILITY_CSV.as_posix()}`",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    snapshot = build_snapshot()
    write_outputs(snapshot)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

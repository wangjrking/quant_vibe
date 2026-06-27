from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_10d_dategate_rerank_refine_20260623"

BASE_TABLE = "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research"
AUX_TABLE = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal"
LABEL_COL = "executable_10d_open_return"

CURRENT_BEST = {
    "start_date": "20260401",
    "rank_threshold": 0.98,
    "alpha": 0.02,
    "candidate_table": (
        "stock_predict_data_model_agent_10d_dategate_rerank_3dformal_20260623_"
        "executable_10d_open_return_research"
    ),
}

GRID = {
    "start_date": ["20260315", "20260401", "20260415"],
    "rank_threshold": [0.98, 0.985, 0.99],
    "alpha": [0.015, 0.02, 0.025],
}

WINDOWS = {
    "full": ("20240604", "20260528"),
    "recent126": ("20251118", "20260528"),
    "recent63": ("20260225", "20260528"),
}


def read_prediction(conn: sqlite3.Connection, table: str, alias: str, *, include_label: bool) -> pd.DataFrame:
    select_cols = f"trade_date, stock_code, pred_prob as {alias}"
    if include_label:
        select_cols += f", [{LABEL_COL}] as {LABEL_COL}"
    frame = pd.read_sql_query(f"select {select_cols} from '{table}'", conn)
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def add_rank(frame: pd.DataFrame, score_col: str, rank_col: str) -> None:
    frame[rank_col] = frame.groupby("trade_date")[score_col].rank(method="average", pct=True)


def evaluate(frame: pd.DataFrame, score_col: str) -> tuple[dict, pd.DataFrame]:
    data = frame[["trade_date", "stock_code", score_col, LABEL_COL]].dropna().copy()
    daily_rows: list[dict] = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(ordered[score_col].corr(ordered[LABEL_COL], method="spearman")),
                "top1": float(ordered.head(1)[LABEL_COL].mean()),
                "top3": float(ordered.head(3)[LABEL_COL].mean()),
                "top5": float(ordered.head(5)[LABEL_COL].mean()),
                "top10": float(ordered.head(10)[LABEL_COL].mean()),
                "top20": float(ordered.head(20)[LABEL_COL].mean()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    summary = {}
    for name, (date_from, date_to) in WINDOWS.items():
        win = daily[(daily["trade_date"] >= date_from) & (daily["trade_date"] <= date_to)].copy()
        summary[name] = {
            "date_from": date_from,
            "date_to": date_to,
            "trade_days": int(win["trade_date"].nunique()),
            "rank_ic": float(win["rank_ic"].mean()),
            "top1": float(win["top1"].mean()),
            "top3": float(win["top3"].mean()),
            "top5": float(win["top5"].mean()),
            "top10": float(win["top10"].mean()),
            "top20": float(win["top20"].mean()),
        }
    return summary, daily


def objective(summary: dict, baseline: dict) -> float:
    return (
        1.0 * (summary["full"]["top1"] - baseline["full"]["top1"])
        + 1.0 * (summary["full"]["top3"] - baseline["full"]["top3"])
        + 2.0 * (summary["recent126"]["top1"] - baseline["recent126"]["top1"])
        + 2.0 * (summary["recent126"]["top3"] - baseline["recent126"]["top3"])
        + 3.0 * (summary["recent63"]["top1"] - baseline["recent63"]["top1"])
        + 3.0 * (summary["recent63"]["top3"] - baseline["recent63"]["top3"])
        + 1.5 * (summary["recent63"]["top5"] - baseline["recent63"]["top5"])
    )


def row_from_summary(name: str, start_date: str, rank_threshold: float, alpha: float, summary: dict, score: float) -> dict:
    return {
        "name": name,
        "start_date": start_date,
        "rank_threshold": rank_threshold,
        "alpha": alpha,
        "objective": score,
        "full_rank_ic": summary["full"]["rank_ic"],
        "full_top1": summary["full"]["top1"],
        "full_top3": summary["full"]["top3"],
        "full_top5": summary["full"]["top5"],
        "full_top10": summary["full"]["top10"],
        "recent126_rank_ic": summary["recent126"]["rank_ic"],
        "recent126_top1": summary["recent126"]["top1"],
        "recent126_top3": summary["recent126"]["top3"],
        "recent126_top5": summary["recent126"]["top5"],
        "recent126_top10": summary["recent126"]["top10"],
        "recent63_rank_ic": summary["recent63"]["rank_ic"],
        "recent63_top1": summary["recent63"]["top1"],
        "recent63_top3": summary["recent63"]["top3"],
        "recent63_top5": summary["recent63"]["top5"],
        "recent63_top10": summary["recent63"]["top10"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        base = read_prediction(conn, BASE_TABLE, "base_pred", include_label=True)
        aux = read_prediction(conn, AUX_TABLE, "aux_pred", include_label=False)
    merged = base.merge(aux, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    add_rank(merged, "base_pred", "base_rank")
    add_rank(merged, "aux_pred", "aux_rank")
    merged["aux_rank"] = merged["aux_rank"].fillna(merged["base_rank"])

    baseline_summary, _ = evaluate(merged, "base_pred")
    rows = [
        row_from_summary(
            name="baseline_formal_10d",
            start_date="none",
            rank_threshold=0.0,
            alpha=0.0,
            summary=baseline_summary,
            score=0.0,
        )
    ]

    best_row = None
    best_summary = None
    for start_date in GRID["start_date"]:
        for rank_threshold in GRID["rank_threshold"]:
            for alpha in GRID["alpha"]:
                candidate = merged.copy()
                candidate["score"] = candidate["base_rank"]
                gate_mask = (candidate["trade_date"] >= start_date) & (candidate["base_rank"] >= rank_threshold)
                candidate.loc[gate_mask, "score"] = (
                    candidate.loc[gate_mask, "base_rank"] + alpha * candidate.loc[gate_mask, "aux_rank"]
                )
                summary, _ = evaluate(candidate, "score")
                score = objective(summary, baseline_summary)
                row = row_from_summary(
                    name="dategate_rerank_3dformal",
                    start_date=start_date,
                    rank_threshold=rank_threshold,
                    alpha=alpha,
                    summary=summary,
                    score=score,
                )
                rows.append(row)
                if best_row is None or row["objective"] > best_row["objective"]:
                    best_row = row
                    best_summary = summary

    grid = pd.DataFrame(rows)
    grid = grid.sort_values(
        ["objective", "recent63_top1", "recent126_top3", "full_top3"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    grid.to_csv(OUT_DIR / "focused_grid.csv", index=False, encoding="utf-8-sig")
    grid.head(10).to_csv(OUT_DIR / "focused_grid_top10.csv", index=False, encoding="utf-8-sig")

    best_is_existing = (
        best_row is not None
        and best_row["start_date"] == CURRENT_BEST["start_date"]
        and abs(float(best_row["rank_threshold"]) - float(CURRENT_BEST["rank_threshold"])) < 1e-12
        and abs(float(best_row["alpha"]) - float(CURRENT_BEST["alpha"])) < 1e-12
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_table": BASE_TABLE,
        "aux_table": AUX_TABLE,
        "baseline_summary": baseline_summary,
        "current_best_reference": CURRENT_BEST,
        "grid_size": int(len(grid) - 1),
        "best_row": best_row,
        "best_summary": best_summary,
        "best_is_existing_candidate": best_is_existing,
        "new_research_asset_needed": False,
        "conclusion": (
            "当前最优仍是既有 20260401 / 0.98 / 0.02 方案，无需新增 10D research 资产。"
            if best_is_existing
            else "邻域内发现更优参数，需考虑新增 research 资产。"
        ),
    }
    (OUT_DIR / "focused_grid_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 10D 日期门控重排局部复核",
        "",
        "## 当前结论",
        "",
        payload["conclusion"],
        "",
        "## 复核范围",
        "",
        f"- 基线 10D 表：`{BASE_TABLE}`",
        f"- 辅助 3D 表：`{AUX_TABLE}`",
        f"- 起始日期：`{', '.join(GRID['start_date'])}`",
        f"- 顶部分位阈值：`{', '.join(str(v) for v in GRID['rank_threshold'])}`",
        f"- 辅助权重：`{', '.join(str(v) for v in GRID['alpha'])}`",
        "",
        "## 最优参数",
        "",
        f"- start_date：`{best_row['start_date']}`",
        f"- rank_threshold：`{best_row['rank_threshold']}`",
        f"- alpha：`{best_row['alpha']}`",
        f"- objective：`{best_row['objective']:.12f}`",
        "",
        "## 与当前 formal 10D 的关键差异",
        "",
        f"- full Top1：`{baseline_summary['full']['top1']:.8f} -> {best_summary['full']['top1']:.8f}`",
        f"- full Top3：`{baseline_summary['full']['top3']:.8f} -> {best_summary['full']['top3']:.8f}`",
        f"- recent126 Top1：`{baseline_summary['recent126']['top1']:.8f} -> {best_summary['recent126']['top1']:.8f}`",
        f"- recent126 Top3：`{baseline_summary['recent126']['top3']:.8f} -> {best_summary['recent126']['top3']:.8f}`",
        f"- recent63 Top1：`{baseline_summary['recent63']['top1']:.8f} -> {best_summary['recent63']['top1']:.8f}`",
        f"- recent63 Top3：`{baseline_summary['recent63']['top3']:.8f} -> {best_summary['recent63']['top3']:.8f}`",
        f"- recent63 Top5：`{baseline_summary['recent63']['top5']:.8f} -> {best_summary['recent63']['top5']:.8f}`",
        "",
        "## 治理说明",
        "",
        "- 本轮只做模型侧评分复核，不训练模型，不生成交易信号，不跑回测。",
        "- 本轮未修改 formal manifest，也未新增 formal 资产。",
        "- 如需把该 10D 候选升格，仍需单独走策略验证和审计流程。",
        "",
        "## 证据路径",
        "",
        f"- 网格全量：`{(OUT_DIR / 'focused_grid.csv').as_posix()}`",
        f"- 网格前十：`{(OUT_DIR / 'focused_grid_top10.csv').as_posix()}`",
        f"- 汇总 JSON：`{(OUT_DIR / 'focused_grid_summary.json').as_posix()}`",
    ]
    (OUT_DIR / "focused_grid_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

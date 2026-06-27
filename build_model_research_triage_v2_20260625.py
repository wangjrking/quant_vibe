from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_triage_v2_20260625"

ASSETS = [
    {
        "asset": "formal_1d",
        "track": "formal",
        "label": "executable_1d_open_return",
        "table": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
        "baseline": "formal_1d",
    },
    {
        "asset": "research_1d_dynamic_guard_broad",
        "track": "research",
        "label": "executable_1d_open_return",
        "table": "stock_predict_data_model_agent_1d_dynamic_guard_20260625_executable_1d_open_return_research",
        "baseline": "formal_1d",
    },
    {
        "asset": "research_1d_dynamic_guard_strict",
        "track": "research",
        "label": "executable_1d_open_return",
        "table": "stock_predict_data_model_agent_1d_dynamic_guard_strict_20260625_executable_1d_open_return_research",
        "baseline": "formal_1d",
    },
    {
        "asset": "research_1d_cross_horizon_guard",
        "track": "research",
        "label": "executable_1d_open_return",
        "table": "stock_predict_data_model_agent_1d_cross_horizon_guard_20260625_executable_1d_open_return_research",
        "baseline": "formal_1d",
    },
    {
        "asset": "research_1d_cross_horizon_guard_v2",
        "track": "research",
        "label": "executable_1d_open_return",
        "table": "stock_predict_data_model_agent_1d_cross_horizon_guard_v2_20260625_executable_1d_open_return_research",
        "baseline": "formal_1d",
    },
    {
        "asset": "research_1d_broad_stability_guard",
        "track": "research",
        "label": "executable_1d_open_return",
        "table": "stock_predict_data_model_agent_1d_broad_stability_guard_20260625_executable_1d_open_return_research",
        "baseline": "formal_1d",
    },
    {
        "asset": "research_1d_new5d_condition_gate_v6",
        "track": "research",
        "label": "executable_1d_open_return",
        "table": "stock_predict_data_model_agent_1d_new5d_condition_gate_v6_20260625_executable_1d_open_return_research",
        "baseline": "formal_1d",
    },
    {
        "asset": "research_1d_risk_balanced_v7",
        "track": "research",
        "label": "executable_1d_open_return",
        "table": "stock_predict_data_model_agent_1d_risk_balanced_v7_20260625_executable_1d_open_return_research",
        "baseline": "formal_1d",
    },
    {
        "asset": "formal_3d",
        "track": "formal",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_practical_std_guard",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_dynamic_practical_std_guard_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_cross_horizon_guard",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_3d_cross_horizon_guard_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_cross_horizon_guard_v2",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_3d_cross_horizon_guard_v2_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_cross_horizon_guard_v3",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_3d_cross_horizon_guard_v3_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_dynamic_blend",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_3d_dynamic_blend_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_recent_enhance_v4",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_3d_recent_enhance_v4_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_rankic_balanced_v5",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_3d_rankic_balanced_v5_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_rankic_recent_focus",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_3d_rankic_recent_focus_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "research_3d_risk_balanced_v7",
        "track": "research",
        "label": "executable_3d_open_return",
        "table": "stock_predict_data_model_agent_3d_risk_balanced_v7_20260625_executable_3d_open_return_research",
        "baseline": "formal_3d",
    },
    {
        "asset": "formal_5d",
        "track": "formal",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_dynamic_guard",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_dynamic_guard_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_dynamic_guard_strict",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_dynamic_guard_strict_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_cross_horizon_guard",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_cross_horizon_guard_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_cross_horizon_guard_v2",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_cross_horizon_guard_v2_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_dynamic_blend",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_dynamic_blend_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_stability_guard_v3",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_stability_guard_v3_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_recent_enhance_v4",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_recent_enhance_v4_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_condition_gate_v6",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_condition_gate_v6_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_condition_gate_v6_shrink",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_condition_gate_v6_shrink_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "research_5d_risk_balanced_v7",
        "track": "research",
        "label": "executable_5d_open_return",
        "table": "stock_predict_data_model_agent_5d_risk_balanced_v7_20260625_executable_5d_open_return_research",
        "baseline": "formal_5d",
    },
    {
        "asset": "formal_10d",
        "track": "formal",
        "label": "executable_10d_open_return",
        "table": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
        "baseline": "formal_10d",
    },
    {
        "asset": "research_10d_dynamic_guard",
        "track": "research",
        "label": "executable_10d_open_return",
        "table": "stock_predict_data_model_agent_10d_dynamic_guard_20260625_executable_10d_open_return_research",
        "baseline": "formal_10d",
    },
    {
        "asset": "research_10d_cross_horizon_guard",
        "track": "research",
        "label": "executable_10d_open_return",
        "table": "stock_predict_data_model_agent_10d_cross_horizon_guard_20260625_executable_10d_open_return_research",
        "baseline": "formal_10d",
    },
    {
        "asset": "research_10d_cross_horizon_guard_v2",
        "track": "research",
        "label": "executable_10d_open_return",
        "table": "stock_predict_data_model_agent_10d_cross_horizon_guard_v2_20260625_executable_10d_open_return_research",
        "baseline": "formal_10d",
    },
    {
        "asset": "research_10d_cross_horizon_guard_v3",
        "track": "research",
        "label": "executable_10d_open_return",
        "table": "stock_predict_data_model_agent_10d_cross_horizon_guard_v3_20260625_executable_10d_open_return_research",
        "baseline": "formal_10d",
    },
    {
        "asset": "research_10d_recent_enhance_v4",
        "track": "research",
        "label": "executable_10d_open_return",
        "table": "stock_predict_data_model_agent_10d_recent_enhance_v4_20260625_executable_10d_open_return_research",
        "baseline": "formal_10d",
    },
    {
        "asset": "research_10d_risk_balanced_v5",
        "track": "research",
        "label": "executable_10d_open_return",
        "table": "stock_predict_data_model_agent_10d_risk_balanced_v5_20260625_executable_10d_open_return_research",
        "baseline": "formal_10d",
    },
    {
        "asset": "research_10d_risk_balanced_v5_shrink",
        "track": "research",
        "label": "executable_10d_open_return",
        "table": "stock_predict_data_model_agent_10d_risk_balanced_v5_shrink_20260625_executable_10d_open_return_research",
        "baseline": "formal_10d",
    },
]

WINDOWS = {
    "full": None,
    "recent126": 126,
    "recent63": 63,
    "recent20": 20,
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def table_stats(conn: sqlite3.Connection, table: str) -> dict:
    row = conn.execute(
        f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(table)}"
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
        f"select trade_date, count(*), count(distinct stock_code) from {quote(table)} group by trade_date order by trade_date desc limit 1"
    ).fetchone()
    return {
        "pred_rows": int(row[0]),
        "pred_min_trade_date": str(row[1]),
        "pred_max_trade_date": str(row[2]),
        "pred_trade_days": int(row[3]),
        "pred_nulls": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_trade_date": str(latest[0]),
        "latest_rows": int(latest[1]),
        "latest_stock_count": int(latest[2]),
    }


def load_labels(labels: set[str], min_date: str, max_date: str) -> pd.DataFrame:
    columns = ["trade_date", "stock_code"] + sorted(labels)
    chunks = []
    for path in LABEL_DIR.glob("*.parquet"):
        part = pd.read_parquet(path, columns=columns)
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def top_mean(group: pd.DataFrame, n: int) -> float:
    if group.empty:
        return np.nan
    return float(group.nlargest(min(n, len(group)), "score_rank")["label_value"].mean())


def bottom_mean(group: pd.DataFrame, n: int) -> float:
    if group.empty:
        return np.nan
    return float(group.nsmallest(min(n, len(group)), "score_rank")["label_value"].mean())


def evaluate_asset(asset: dict, pred: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    label = asset["label"]
    joined = pred.merge(labels[["trade_date", "stock_code", label]], on=["trade_date", "stock_code"], how="inner")
    joined = joined.dropna(subset=[label, "pred_prob"]).rename(columns={label: "label_value"})
    joined["score_rank"] = joined.groupby("trade_date")["pred_prob"].rank(method="average", pct=True)
    rows = []
    for trade_date, group in joined.groupby("trade_date", sort=True):
        label_rank = group["label_value"].rank(method="average", pct=True)
        row = {
            "asset": asset["asset"],
            "label": label,
            "track": asset["track"],
            "trade_date": trade_date,
            "rank_ic": float(group["score_rank"].corr(label_rank)) if len(group) > 1 else np.nan,
            "pearson_ic": float(group["score_rank"].corr(group["label_value"])) if len(group) > 1 else np.nan,
        }
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = top_mean(group, n)
        row["top_bottom"] = top_mean(group, 50) - bottom_mean(group, 50)
        rows.append(row)
    daily = pd.DataFrame(rows)
    eval_stats = {
        "eval_rows": int(len(joined)),
        "eval_min_trade_date": str(joined["trade_date"].min()) if len(joined) else None,
        "eval_max_trade_date": str(joined["trade_date"].max()) if len(joined) else None,
        "eval_trade_days": int(joined["trade_date"].nunique()),
    }
    return daily, eval_stats


def summarize_windows(daily: pd.DataFrame, stats: dict) -> pd.DataFrame:
    rows = []
    for asset, group in daily.groupby("asset", sort=False):
        dates = sorted(group["trade_date"].unique().tolist())
        meta = group.iloc[0][["label", "track"]].to_dict()
        for window, n in WINDOWS.items():
            use_dates = set(dates if n is None else dates[-n:])
            sub = group[group["trade_date"].isin(use_dates)]
            row = {
                "asset": asset,
                "label": meta["label"],
                "track": meta["track"],
                "window": window,
                "window_days": int(sub["trade_date"].nunique()),
            }
            for col in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
                row[col] = float(sub[col].mean())
            row.update(stats[asset])
            rows.append(row)
    return pd.DataFrame(rows)


def add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    baseline_by_label = {
        asset["label"]: asset["baseline"]
        for asset in ASSETS
        if asset["asset"] == asset["baseline"]
    }
    out = summary.copy()
    metric_cols = ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    for idx, row in out.iterrows():
        baseline_asset = baseline_by_label[row["label"]]
        base = out[(out["asset"] == baseline_asset) & (out["window"] == row["window"])].iloc[0]
        out.loc[idx, "baseline_asset"] = baseline_asset
        for col in metric_cols:
            out.loc[idx, f"{col}_delta"] = float(row[col] - base[col])
    out["objective_delta"] = (
        3.0 * out["recent63_top1_delta"] if "recent63_top1_delta" in out.columns else 0
    )
    return out


def build_focus(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot_table(
        index=["asset", "label", "track"],
        columns="window",
        values=["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50", "rank_ic_delta", "top1_delta", "top3_delta", "top5_delta", "top10_delta"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{window}" for metric, window in wide.columns]
    wide = wide.reset_index()
    for col in [
        "top1_delta_recent63",
        "top3_delta_recent63",
        "top5_delta_recent63",
        "top10_delta_recent63",
        "rank_ic_delta_recent63",
        "top5_delta_full",
    ]:
        if col not in wide.columns:
            wide[col] = 0.0
    wide["research_objective_delta"] = (
        3.0 * wide["top1_delta_recent63"]
        + 2.0 * wide["top3_delta_recent63"]
        + 2.0 * wide["top5_delta_recent63"]
        + wide["top10_delta_recent63"]
        + 0.5 * wide["top5_delta_full"]
        + 0.2 * wide["rank_ic_delta_recent63"]
    )
    wide["formal_weakness_score"] = -(
        wide.get("top1_recent63", 0.0)
        + wide.get("top5_recent63", 0.0)
        + wide.get("top10_recent63", 0.0)
    )
    return wide.sort_values(["track", "research_objective_delta"], ascending=[True, False])


def write_report(summary: pd.DataFrame, focus: pd.DataFrame, assets: list[dict], stats: dict, eval_stats: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORT_DIR / "unified_model_eval_summary_v2.csv"
    daily_path = REPORT_DIR / "unified_model_eval_daily_v2.csv"
    focus_path = REPORT_DIR / "research_candidate_ranking_v2.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    focus.to_csv(focus_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": now_iso(),
        "scope": "research_only_unified_model_eval_triage_v2",
        "assets": assets,
        "prediction_stats": stats,
        "eval_stats": eval_stats,
        "summary_csv": str(summary_path),
        "daily_csv": str(daily_path),
        "focus_csv": str(focus_path),
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "unified_model_eval_triage_v2_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 模型研究统一评价 V2（20260625）",
        "",
        "## 当前结论",
        "",
        "本报告将 1D/3D/5D/10D formal 资产与近期 research-only 候选放到同一评价口径下比较。评价只使用模型预测分数与标准标签，不生成交易信号，不运行策略回测。",
        "",
        "## 候选排序",
        "",
        "| 候选 | 标签 | 近63日Top1增量 | 近63日Top5增量 | 近63日RankIC增量 | 研究目标增量 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    candidates = focus[focus["track"] == "research"].sort_values("research_objective_delta", ascending=False).head(10)
    for _, row in candidates.iterrows():
        lines.append(
            f"| `{row['asset']}` | `{row['label']}` | {row['top1_delta_recent63']:.6f} | {row['top5_delta_recent63']:.6f} | {row['rank_ic_delta_recent63']:.6f} | {row['research_objective_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 生产边界",
            "",
            "- 本轮只生成研究评价报告。",
            "- 未训练模型。",
            "- 未修改 production/formal manifest。",
            "- 未写入 formal L4 表。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
            "",
            "## 证据路径",
            "",
            f"- 汇总表：`{summary_path.as_posix()}`",
            f"- 候选排序：`{focus_path.as_posix()}`",
            f"- JSON 摘要：`{(REPORT_DIR / 'unified_model_eval_triage_v2_summary.json').as_posix()}`",
        ]
    )
    (REPORT_DIR / "research_triage_v2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        stats = {asset["asset"]: table_stats(conn, asset["table"]) for asset in ASSETS}
        min_date = min(s["pred_min_trade_date"] for s in stats.values())
        max_date = max(s["pred_max_trade_date"] for s in stats.values())
        labels = load_labels({asset["label"] for asset in ASSETS}, min_date, max_date)
        daily_frames = []
        eval_stats = {}
        for asset in ASSETS:
            pred = read_table(conn, asset["table"])
            daily, one_stats = evaluate_asset(asset, pred, labels)
            daily_frames.append(daily)
            eval_stats[asset["asset"]] = one_stats
    daily_all = pd.concat(daily_frames, ignore_index=True)
    daily_path = REPORT_DIR / "unified_model_eval_daily_v2.csv"
    daily_all.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary = summarize_windows(daily_all, stats)
    summary = add_baseline_deltas(summary)
    focus = build_focus(summary)
    write_report(summary, focus, ASSETS, stats, eval_stats)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "assets": len(ASSETS),
                "daily_rows": int(len(daily_all)),
                "summary_rows": int(len(summary)),
                "top_research": focus[focus["track"] == "research"].head(5).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

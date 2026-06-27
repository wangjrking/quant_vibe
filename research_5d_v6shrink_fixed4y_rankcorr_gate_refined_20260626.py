from __future__ import annotations

import hashlib
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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_v6shrink_fixed4y_rankcorr_gate_refined_20260626"

LABEL = "executable_5d_open_return"
FORMAL_TABLE = "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal"
BASE_TABLE = "stock_predict_data_model_agent_5d_v6shrink_lowstd_guard_20260626_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_v6shrink_fixed4y_rankcorr_gate_refined_20260626_executable_5d_open_return_research"
ALT_PATH = (
    DATA_DIR
    / "reports"
    / "model_agent_research_5d_fixed4y_fold08_fs120_k10_structure_scan_20260626"
    / "d3_l8_a01_lr004_n5000"
    / "fold08"
    / "fold_predictions"
    / "fold08.parquet"
)

FEATURE = "rank_corr"
THRESHOLD = 0.2835926192776411


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_scores(conn: sqlite3.Connection, table: str, score_name: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": score_name})


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        formal = read_scores(conn, FORMAL_TABLE, "formal_score")
        base = read_scores(conn, BASE_TABLE, "base_score")
    alt = pd.read_parquet(ALT_PATH, columns=["trade_date", "stock_code", "pred_prob"])
    alt["trade_date"] = alt["trade_date"].astype(str)
    alt["stock_code"] = alt["stock_code"].astype(str)
    alt = alt.rename(columns={"pred_prob": "alt_score"})

    frame = formal.merge(base, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame = frame.merge(alt, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame["formal_rank"] = frame.groupby("trade_date")["formal_score"].rank(method="average", pct=True)
    frame["base_rank"] = frame.groupby("trade_date")["base_score"].rank(method="average", pct=True)
    alt_mask = frame["alt_score"].notna()
    frame.loc[alt_mask, "alt_rank"] = frame.loc[alt_mask].groupby("trade_date")["alt_score"].rank(method="average", pct=True)
    return frame


def add_gate(scores: pd.DataFrame) -> pd.DataFrame:
    gate_rows = []
    overlap = scores[scores["alt_score"].notna()].copy()
    for trade_date, group in overlap.groupby("trade_date", sort=True):
        gate_rows.append({"trade_date": trade_date, FEATURE: float(group["base_rank"].corr(group["alt_rank"]))})
    gates = pd.DataFrame(gate_rows)
    gates["gate_active"] = gates[FEATURE] >= THRESHOLD
    out = scores.merge(gates, on="trade_date", how="left", validate="many_to_one")
    out["gate_active"] = out["gate_active"].where(out["gate_active"].notna(), False).astype(bool)
    out["pred_prob"] = np.where(out["gate_active"], out["alt_rank"], out["base_rank"])
    out["score_formula"] = f"if {FEATURE} >= {THRESHOLD:.15f}: alt_rank else base_rank"
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.parquet")):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True).dropna(subset=[LABEL])


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean()) if len(group) else np.nan


def daily_metrics(eval_frame: pd.DataFrame, rank_col: str, source: str) -> pd.DataFrame:
    rows = []
    for trade_date, group in eval_frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        row = {
            "trade_date": trade_date,
            "source": source,
            "rank_ic": float(group[rank_col].corr(label_rank)),
            "pearson_ic": float(group[rank_col].corr(group[LABEL])),
        }
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = top_mean(group, rank_col, n)
        row["top_bottom"] = top_mean(group, rank_col, 50) - bottom_mean(group, rank_col, 50)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(daily: pd.DataFrame, dates: list[str], window: int | None) -> dict[str, float]:
    keep = set(dates if window is None else dates[-window:])
    sub = daily[daily["trade_date"].isin(keep)]
    return {
        col: float(sub[col].mean())
        for col in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    }


def period_stats(delta: pd.DataFrame) -> dict[str, float | int]:
    periods = {
        "2024H2": ("20240604", "20241231"),
        "2025H1": ("20250101", "20250630"),
        "2025H2": ("20250701", "20251231"),
        "2026YTD": ("20260101", "99999999"),
    }
    out: dict[str, float | int] = {}
    top5s = []
    rankics = []
    for name, (lo, hi) in periods.items():
        sub = delta[(delta["trade_date"] >= lo) & (delta["trade_date"] <= hi)]
        top5 = float(sub["top5_delta"].mean())
        rank_ic = float(sub["rank_ic_delta"].mean())
        out[f"{name}_top5_delta"] = top5
        out[f"{name}_rank_ic_delta"] = rank_ic
        top5s.append(top5)
        rankics.append(rank_ic)
    out["positive_top5_periods"] = int(sum(v > 0 for v in top5s))
    out["min_period_top5_delta"] = float(min(top5s))
    out["min_period_rank_ic_delta"] = float(min(rankics))
    return out


def add_deltas(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    delta = candidate.merge(baseline, on="trade_date", suffixes=("", "_baseline"))
    for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
        delta[f"{metric}_delta"] = delta[metric] - delta[f"{metric}_baseline"]
    return delta


def objective(summary: dict[str, float]) -> float:
    return (
        1.8 * summary["top1"]
        + 1.2 * summary["top3"]
        + 1.0 * summary["top5"]
        + 0.4 * summary["top10"]
        + 0.45 * summary["rank_ic"]
        + 0.2 * summary["top_bottom"]
    )


def evaluate(scored: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    eval_frame = scored.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    candidate = daily_metrics(eval_frame, "pred_prob", "fixed4y_rankcorr_gate_refined")
    formal = daily_metrics(eval_frame, "formal_rank", "formal")
    base = daily_metrics(eval_frame, "base_rank", "base_v6shrink_lowstd")
    dates = sorted(candidate["trade_date"].unique().tolist())

    delta_formal = add_deltas(candidate, formal)
    delta_base = add_deltas(candidate, base)
    summary: dict[str, object] = {
        "eval_min_trade_date": str(eval_frame["trade_date"].min()),
        "eval_max_trade_date": str(eval_frame["trade_date"].max()),
        "eval_trade_days": int(eval_frame["trade_date"].nunique()),
        "gate_active_days": int(scored.groupby("trade_date")["gate_active"].first().sum()),
        "overlap_days": int(scored.loc[scored["alt_score"].notna(), "trade_date"].nunique()),
        "full_candidate_objective": objective(summarize(candidate, dates, None)),
        "full_base_objective": objective(summarize(base, dates, None)),
        "full_formal_objective": objective(summarize(formal, dates, None)),
    }
    for prefix, cur, baseline in [
        ("full_vs_formal", summarize(candidate, dates, None), summarize(formal, dates, None)),
        ("recent63_vs_formal", summarize(candidate, dates, 63), summarize(formal, dates, 63)),
        ("recent20_vs_formal", summarize(candidate, dates, 20), summarize(formal, dates, 20)),
        ("full_vs_base", summarize(candidate, dates, None), summarize(base, dates, None)),
        ("recent63_vs_base", summarize(candidate, dates, 63), summarize(base, dates, 63)),
        ("recent20_vs_base", summarize(candidate, dates, 20), summarize(base, dates, 20)),
    ]:
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
            summary[f"{prefix}_{metric}_delta"] = float(cur[metric] - baseline[metric])
    for key, value in period_stats(delta_formal).items():
        summary[f"formal_{key}"] = value
    for key, value in period_stats(delta_base).items():
        summary[f"base_{key}"] = value
    metrics = pd.concat([formal, base, candidate], ignore_index=True)
    return metrics, delta_formal, delta_base, summary


def write_table(scored: pd.DataFrame) -> dict[str, object]:
    columns = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "formal_score",
        "base_score",
        "alt_score",
        "formal_rank",
        "base_rank",
        "alt_rank",
        FEATURE,
        "gate_active",
        "score_formula",
    ]
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        scored[columns].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        short = hashlib.sha1(TARGET_TABLE.encode("utf-8")).hexdigest()[:12]
        conn.execute(f"create index if not exists idx_{short}_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_{short}_date_pred on {quote(TARGET_TABLE)}(trade_date, pred_prob desc)")
        conn.commit()
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(TARGET_TABLE)} group by trade_date order by trade_date desc limit 5"
        ).fetchall()
    return {
        "table": TARGET_TABLE,
        "formula": scored["score_formula"].iloc[0],
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
        "gate_active_days": int(scored.groupby("trade_date")["gate_active"].first().sum()),
        "overlap_days": int(scored.loc[scored["alt_score"].notna(), "trade_date"].nunique()),
    }


def write_outputs(
    stats: dict[str, object],
    summary: dict[str, object],
    metrics: pd.DataFrame,
    delta_formal: pd.DataFrame,
    delta_base: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "rankcorr_gate_refined_daily_metrics.csv"
    delta_formal_path = REPORT_DIR / "rankcorr_gate_refined_delta_vs_formal.csv"
    delta_base_path = REPORT_DIR / "rankcorr_gate_refined_delta_vs_base.csv"
    manifest_path = REPORT_DIR / "rankcorr_gate_refined_research_manifest.json"
    report_path = REPORT_DIR / "rankcorr_gate_refined_research_report.md"

    metrics.to_csv(metrics_path, index=False)
    delta_formal.to_csv(delta_formal_path, index=False)
    delta_base.to_csv(delta_base_path, index=False)

    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "source_type": "sqlite_table",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db",
        "table": TARGET_TABLE,
        "label": LABEL,
        "generated_at": now_iso(),
        "candidate_id": "model_agent_5d_v6shrink_fixed4y_rankcorr_gate_refined_20260626",
        "score_formula": stats["formula"],
        "research_hypothesis": "use_fixed4y_candidate_only_on_refined_high_rankcorr_days_with_current_5d_mainline",
        "base_table": BASE_TABLE,
        "alt_source_path": str(ALT_PATH),
        "gate_feature": FEATURE,
        "gate_threshold": THRESHOLD,
        "row_count": stats["row_count"],
        "trade_days": stats["trade_days"],
        "min_trade_date": stats["min_trade_date"],
        "max_trade_date": stats["max_trade_date"],
        "latest_days": stats["latest_days"],
        "null_pred_prob": stats["null_pred_prob"],
        "duplicate_key_groups": stats["duplicate_key_groups"],
        "gate_active_days": stats["gate_active_days"],
        "overlap_days": stats["overlap_days"],
        "evaluation_summary": summary,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True
        }
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 5D v6_shrink 与近四年候选的 refined rank_corr 门控研究报告（20260626）",
        "",
        "## 当前结论",
        "",
        "这是一个 research-only 的 5D 全覆盖候选，不是 formal L4 或 L5 生产资产。",
        f"它只在与近四年候选重叠的 {stats['overlap_days']} 个交易日里，当 `rank_corr >= {THRESHOLD:.15f}` 时切到近四年候选，其余日期保持当前 5D 主线不变。",
        "",
        "## 评分公式",
        "",
        "```text",
        stats["formula"],
        "```",
        "",
        "## 关键评价摘要",
        "",
        f"- 相对当前 5D 主线：full objective delta `{summary['full_candidate_objective'] - summary['full_base_objective']:.12f}`，full RankIC delta `{summary['full_vs_base_rank_ic_delta']:.12f}`，full Top5 delta `{summary['full_vs_base_top5_delta']:.12f}`。",
        f"- 相对当前 5D 主线：recent63 Top1 delta `{summary['recent63_vs_base_top1_delta']:.12f}`，recent63 Top5 delta `{summary['recent63_vs_base_top5_delta']:.12f}`，recent20 Top5 delta `{summary['recent20_vs_base_top5_delta']:.12f}`。",
        f"- 触发交易日数：`{stats['gate_active_days']}` / overlap `{stats['overlap_days']}`。",
        "",
        "## 资产覆盖",
        "",
        f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
        f"- 日期范围：`{stats['min_trade_date']}` 到 `{stats['max_trade_date']}`",
        f"- 总行数：`{stats['row_count']}`，交易日数：`{stats['trade_days']}`",
        f"- 最新日：`{stats['latest_days'][0][0]}`，行数 `{stats['latest_days'][0][1]}`，股票数 `{stats['latest_days'][0][2]}`",
        f"- 空分数：`{stats['null_pred_prob']}`，重复键：`{stats['duplicate_key_groups']}`",
        "",
        "## 当前判断",
        "",
        "- 这条分支是在原有 5D repair 候选上做 full-window refine 后得到的更优版本。",
        "- 它仍然是增量修补候选，不是新的独立训练主线。",
        "",
        "## 边界",
        "",
        "- 未训练模型。",
        "- 未调整训练参数。",
        "- 未修改 formal / production manifest。",
        "- 未生成交易信号。",
        "- 未运行策略回测。"
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    scores = load_scores()
    scored = add_gate(scores)
    labels = load_labels(str(scored["trade_date"].min()), str(scored["trade_date"].max()))
    metrics, delta_formal, delta_base, summary = evaluate(scored, labels)
    stats = write_table(scored)
    write_outputs(stats, summary, metrics, delta_formal, delta_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

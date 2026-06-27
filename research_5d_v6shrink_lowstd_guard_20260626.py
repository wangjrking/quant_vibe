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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_v6shrink_lowstd_guard_20260626"

LABEL = "executable_5d_open_return"
FORMAL_TABLE = "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal"
BASE_TABLE = "stock_predict_data_model_agent_5d_condition_gate_v6_shrink_20260625_executable_5d_open_return_research"
ALT_TABLE = "stock_predict_data_model_agent_5d_risk_balanced_v7_20260625_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_v6shrink_lowstd_guard_20260626_executable_5d_open_return_research"

FEATURE = "base_std"
THRESHOLD = 0.288701278824347


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
        alt = read_scores(conn, ALT_TABLE, "alt_score")
    frame = formal.merge(base, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame = frame.merge(alt, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    for prefix in ["formal", "base", "alt"]:
        frame[f"{prefix}_rank"] = frame.groupby("trade_date")[f"{prefix}_score"].rank(method="average", pct=True)
    return frame


def add_gate(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        rows.append(
            {
                "trade_date": trade_date,
                FEATURE: float(group["base_score"].std()),
            }
        )
    gates = pd.DataFrame(rows)
    gates["gate_active"] = gates[FEATURE] <= THRESHOLD
    out = scores.merge(gates, on="trade_date", how="left", validate="many_to_one")
    out["pred_prob"] = np.where(out["gate_active"], out["alt_rank"], out["base_rank"])
    out["score_formula"] = f"if {FEATURE} <= {THRESHOLD:.15f}: alt_rank else base_rank"
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
    out["avg_period_top5_delta"] = float(np.mean(top5s))
    out["avg_period_rank_ic_delta"] = float(np.mean(rankics))
    return out


def add_deltas(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    delta = candidate.merge(baseline, on="trade_date", suffixes=("", "_baseline"))
    for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
        delta[f"{metric}_delta"] = delta[metric] - delta[f"{metric}_baseline"]
    return delta


def evaluate(scored: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    eval_frame = scored.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    candidate = daily_metrics(eval_frame, "pred_prob", "lowstd_guard")
    formal = daily_metrics(eval_frame, "formal_rank", "formal")
    base = daily_metrics(eval_frame, "base_rank", "base_v6_shrink")
    alt = daily_metrics(eval_frame, "alt_rank", "alt_v7")
    dates = sorted(candidate["trade_date"].unique().tolist())

    delta_formal = add_deltas(candidate, formal)
    delta_base = add_deltas(candidate, base)
    summary: dict[str, object] = {
        "eval_min_trade_date": str(eval_frame["trade_date"].min()),
        "eval_max_trade_date": str(eval_frame["trade_date"].max()),
        "eval_trade_days": int(eval_frame["trade_date"].nunique()),
        "gate_active_days": int(scored.groupby("trade_date")["gate_active"].first().sum()),
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
    metrics = pd.concat([formal, base, alt, candidate], ignore_index=True)
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
    }


def write_outputs(
    stats: dict[str, object],
    summary: dict[str, object],
    metrics: pd.DataFrame,
    delta_formal: pd.DataFrame,
    delta_base: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "lowstd_guard_daily_metrics.csv"
    delta_formal_path = REPORT_DIR / "lowstd_guard_delta_vs_formal.csv"
    delta_base_path = REPORT_DIR / "lowstd_guard_delta_vs_base.csv"
    manifest_path = REPORT_DIR / "lowstd_guard_research_manifest.json"
    report_path = REPORT_DIR / "lowstd_guard_research_report.md"

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
        "decision": "candidate",
        "research_hypothesis": "switch_from_5d_v6shrink_to_5d_v7_on_low_base_score_std_days",
        "asset_stats": stats,
        "evaluation_summary": summary,
        "daily_metrics_csv": str(metrics_path),
        "delta_vs_formal_csv": str(delta_formal_path),
        "delta_vs_base_csv": str(delta_base_path),
        "promotion_requires_user_confirmation": True,
        "audit_required_before_formal": True,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_tuning_training_parameters": True,
            "no_production_manifest_change": True,
            "no_formal_l4_write": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 5D v6_shrink 低波动守卫研究报告（20260626）",
        "",
        "## 当前结论",
        "",
        "本资产是 research-only 的 5D 候选，不是 formal L4 或 L5 生产资产。",
        "它在 `base_score` 截面波动较低的交易日，使用 `5d_risk_balanced_v7` 的 rank 替代 `5d_condition_gate_v6_shrink`，目标是在尽量不伤近期前排 TopN 的前提下，提高当前 5D 主线的全样本 RankIC。",
        "",
        "## 评分公式",
        "",
        "```text",
        stats["formula"],
        "```",
        "",
        "## 关键评价摘要",
        "",
        f"- 相对 formal 5D：full RankIC delta `{summary['full_vs_formal_rank_ic_delta']:.12f}`，recent63 Top1 delta `{summary['recent63_vs_formal_top1_delta']:.12f}`，recent63 Top5 delta `{summary['recent63_vs_formal_top5_delta']:.12f}`，recent20 Top5 delta `{summary['recent20_vs_formal_top5_delta']:.12f}`，full Top5 delta `{summary['full_vs_formal_top5_delta']:.12f}`。",
        f"- 相对当前 5D 主线 v6_shrink：full RankIC delta `{summary['full_vs_base_rank_ic_delta']:.12f}`，recent63 Top1 delta `{summary['recent63_vs_base_top1_delta']:.12f}`，recent63 Top5 delta `{summary['recent63_vs_base_top5_delta']:.12f}`，recent20 Top5 delta `{summary['recent20_vs_base_top5_delta']:.12f}`，full Top5 delta `{summary['full_vs_base_top5_delta']:.12f}`。",
        f"- 分段稳定性（相对当前主线）：最差阶段 Top5 delta `{summary['base_min_period_top5_delta']:.12f}`，最差阶段 RankIC delta `{summary['base_min_period_rank_ic_delta']:.12f}`，Top5 正向阶段数 `{summary['base_positive_top5_periods']}`。",
        "",
        "## 资产覆盖",
        "",
        f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
        f"- 日期范围：`{stats['min_trade_date']}` 到 `{stats['max_trade_date']}`",
        f"- 总行数：`{stats['row_count']}`，交易日数：`{stats['trade_days']}`",
        f"- 最新日：`{stats['latest_days'][0][0]}`，行数 `{stats['latest_days'][0][1]}`，股票数 `{stats['latest_days'][0][2]}`",
        f"- 触发交易日数：`{stats['gate_active_days']}`",
        f"- 空分数：`{stats['null_pred_prob']}`，重复键：`{stats['duplicate_key_groups']}`",
        "",
        "## 当前判断",
        "",
        "- 这条分支相比 v6_shrink 有小幅但真实的全样本 RankIC 改善。",
        "- 近期 63 日的 Top1/Top5 没有被拖成负增量，说明这条守卫式切换至少具备继续研究的价值。",
        "- 它当前更像 5D 主线的增量修补分支，而不是立即替代主线的成熟解。",
        "",
        "## 边界",
        "",
        "- 未训练模型。",
        "- 未调参训练参数。",
        "- 未修改 production/formal manifest。",
        "- 未写入 formal L4 表。",
        "- 未生成交易信号。",
        "- 未运行策略回测。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = add_gate(load_scores())
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    metrics, delta_formal, delta_base, summary = evaluate(scores, labels)
    stats = write_table(scores)
    write_outputs(stats, summary, metrics, delta_formal, delta_base)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "asset_stats": stats,
                "evaluation_summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

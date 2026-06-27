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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_condition_gate_v6_research_20260625"

LABEL = "executable_5d_open_return"
FORMAL_TABLE = "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal"
STABLE_TABLE = "stock_predict_data_model_agent_5d_stability_guard_v3_20260625_executable_5d_open_return_research"
RECENT_TABLE = "stock_predict_data_model_agent_5d_recent_enhance_v4_20260625_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_condition_gate_v6_20260625_executable_5d_open_return_research"
OVERLAP_THRESHOLD = 2


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
    with sqlite3.connect(MODEL_DB) as conn:
        formal = read_scores(conn, FORMAL_TABLE, "formal_score")
        stable = read_scores(conn, STABLE_TABLE, "stable_score")
        recent = read_scores(conn, RECENT_TABLE, "recent_score")
    frame = formal.merge(stable, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame = frame.merge(recent, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    for prefix in ["formal", "stable", "recent"]:
        frame[f"{prefix}_rank"] = frame.groupby("trade_date")[f"{prefix}_score"].rank(method="average", pct=True)
    return frame


def top_set(group: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    return set(group.nlargest(min(n, len(group)), rank_col)["stock_code"].astype(str))


def add_gate(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        stable_top10 = top_set(group, "stable_rank", 10)
        formal_top10 = top_set(group, "formal_rank", 10)
        rows.append(
            {
                "trade_date": trade_date,
                "overlap10_stable_formal": len(stable_top10 & formal_top10),
            }
        )
    gates = pd.DataFrame(rows)
    gates["gate_active"] = gates["overlap10_stable_formal"] <= OVERLAP_THRESHOLD
    out = scores.merge(gates, on="trade_date", how="left", validate="many_to_one")
    out["pred_prob"] = np.where(out["gate_active"], out["recent_rank"], out["stable_rank"])
    out["score_formula"] = f"if overlap10_stable_formal <= {OVERLAP_THRESHOLD}: recent_rank else stable_rank"
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in LABEL_DIR.glob("*.parquet"):
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


def summarize(daily: pd.DataFrame, dates: list[str], window: int | None) -> dict:
    keep = set(dates if window is None else dates[-window:])
    sub = daily[daily["trade_date"].isin(keep)]
    return {
        col: float(sub[col].mean())
        for col in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]
    }


def period_stats(delta: pd.DataFrame) -> dict:
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


def evaluate(scored: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eval_frame = scored.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    candidate = daily_metrics(eval_frame, "pred_prob", "condition_gate_v6")
    stable = daily_metrics(eval_frame, "stable_rank", "stable")
    formal = daily_metrics(eval_frame, "formal_rank", "formal")
    recent = daily_metrics(eval_frame, "recent_rank", "recent")
    dates = sorted(candidate["trade_date"].unique().tolist())
    delta_stable = candidate.merge(stable, on="trade_date", suffixes=("", "_stable"))
    delta_formal = candidate.merge(formal, on="trade_date", suffixes=("", "_formal"))
    for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
        delta_stable[f"{metric}_delta"] = delta_stable[metric] - delta_stable[f"{metric}_stable"]
        delta_formal[f"{metric}_delta"] = delta_formal[metric] - delta_formal[f"{metric}_formal"]
    summary: dict[str, object] = {
        "eval_min_trade_date": str(eval_frame["trade_date"].min()),
        "eval_max_trade_date": str(eval_frame["trade_date"].max()),
        "eval_trade_days": int(eval_frame["trade_date"].nunique()),
        "gate_active_days": int(scored.groupby("trade_date")["gate_active"].first().sum()),
    }
    for prefix, cur, base in [
        ("full_vs_stable", summarize(candidate, dates, None), summarize(stable, dates, None)),
        ("recent63_vs_stable", summarize(candidate, dates, 63), summarize(stable, dates, 63)),
        ("recent20_vs_stable", summarize(candidate, dates, 20), summarize(stable, dates, 20)),
        ("full_vs_formal", summarize(candidate, dates, None), summarize(formal, dates, None)),
        ("recent63_vs_formal", summarize(candidate, dates, 63), summarize(formal, dates, 63)),
    ]:
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
            summary[f"{prefix}_{metric}_delta"] = float(cur[metric] - base[metric])
    for key, value in period_stats(delta_stable).items():
        summary[f"stable_{key}"] = value
    for key, value in period_stats(delta_formal).items():
        summary[f"formal_{key}"] = value
    source_metrics = pd.concat([formal, stable, recent, candidate], ignore_index=True)
    return source_metrics, delta_stable, summary


def write_table(scored: pd.DataFrame) -> dict:
    columns = [
        "trade_date",
        "stock_code",
        "pred_prob",
        "formal_score",
        "stable_score",
        "recent_score",
        "formal_rank",
        "stable_rank",
        "recent_rank",
        "overlap10_stable_formal",
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
    }


def write_outputs(source_metrics: pd.DataFrame, daily_delta: pd.DataFrame, summary: dict, stats: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_path = REPORT_DIR / "condition_gate_source_daily_metrics.csv"
    daily_path = REPORT_DIR / "condition_gate_daily_delta_vs_stable.csv"
    summary_path = REPORT_DIR / "condition_gate_summary.json"
    manifest_path = REPORT_DIR / "condition_gate_research_manifest.json"
    report_path = REPORT_DIR / "condition_gate_research_report.md"
    source_metrics.to_csv(source_path, index=False, encoding="utf-8-sig")
    daily_delta.to_csv(daily_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": now_iso(),
        "asset_stats": stats,
        "evaluation_summary": summary,
        "sources": {
            "formal": FORMAL_TABLE,
            "stable": STABLE_TABLE,
            "recent": RECENT_TABLE,
        },
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
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "score_formula": stats["formula"],
        "asset_stats": stats,
        "evaluation_summary": summary,
        "promotion_requires_user_confirmation": True,
        "audit_required_before_formal": True,
        "boundaries": payload["boundaries"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 5D 条件门控研究报告（20260625）",
        "",
        "## 当前结论",
        "",
        "本轮生成一个 5D research-only 条件门控候选：默认使用 5D 稳定版；当稳定版与当前 formal 5D 的 Top10 重合度不超过 2 时，切换到近期增强版。",
        "",
        "```text",
        stats["formula"],
        "```",
        "",
        f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
        f"- 日期范围：`{stats['min_trade_date']}` 到 `{stats['max_trade_date']}`",
        f"- 总行数：`{stats['row_count']}`",
        f"- 最新日：`{stats['latest_days'][0][0]}`，行数 `{stats['latest_days'][0][1]}`，股票数 `{stats['latest_days'][0][2]}`",
        f"- 门控触发交易日：`{summary['gate_active_days']}`",
        f"- `pred_prob` 空值：`{stats['null_pred_prob']}`",
        f"- 重复键组：`{stats['duplicate_key_groups']}`",
        "",
        "## 相对 5D 稳定版的评价摘要",
        "",
        f"- 近期 63 日 Top1 增量：`{summary['recent63_vs_stable_top1_delta']:.6f}`",
        f"- 近期 63 日 Top5 增量：`{summary['recent63_vs_stable_top5_delta']:.6f}`",
        f"- 近期 63 日 RankIC 增量：`{summary['recent63_vs_stable_rank_ic_delta']:.6f}`",
        f"- 全样本 Top5 增量：`{summary['full_vs_stable_top5_delta']:.6f}`",
        f"- 全样本 RankIC 增量：`{summary['full_vs_stable_rank_ic_delta']:.6f}`",
        f"- 最差阶段 RankIC 增量：`{summary['stable_min_period_rank_ic_delta']:.6f}`",
        "",
        "## 边界说明",
        "",
        "- 未训练模型。",
        "- 未调整训练参数。",
        "- 未修改 production/formal manifest。",
        "- 未写入 formal L4 表。",
        "- 未生成交易信号。",
        "- 未运行策略回测。",
        "",
        "## 证据路径",
        "",
        f"- 汇总 JSON：`{summary_path.as_posix()}`",
        f"- 日度指标：`{source_path.as_posix()}`",
        f"- 日度增量：`{daily_path.as_posix()}`",
        f"- 研究 manifest：`{manifest_path.as_posix()}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    scores = load_scores()
    scored = add_gate(scores)
    labels = load_labels(str(scored["trade_date"].min()), str(scored["trade_date"].max()))
    source_metrics, daily_delta, summary = evaluate(scored, labels)
    stats = write_table(scored)
    write_outputs(source_metrics, daily_delta, summary, stats)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "table": TARGET_TABLE,
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

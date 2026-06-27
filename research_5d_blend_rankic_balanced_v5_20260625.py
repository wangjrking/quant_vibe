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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_5d_blend_rankic_balanced_v5_research_20260625"

LABEL = "executable_5d_open_return"
FORMAL_TABLE = "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal"
STABLE_TABLE = "stock_predict_data_model_agent_5d_stability_guard_v3_20260625_executable_5d_open_return_research"
RECENT_TABLE = "stock_predict_data_model_agent_5d_recent_enhance_v4_20260625_executable_5d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_5d_blend_rankic_balanced_v5_20260625_executable_5d_open_return_research"


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


def load_score_frame() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        formal = read_scores(conn, FORMAL_TABLE, "formal_score")
        stable = read_scores(conn, STABLE_TABLE, "stable_score")
        recent = read_scores(conn, RECENT_TABLE, "recent_score")
    frame = formal.merge(stable, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    frame = frame.merge(recent, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    for score_col in ["formal_score", "stable_score", "recent_score"]:
        rank_col = score_col.replace("_score", "_rank")
        frame[rank_col] = frame.groupby("trade_date")[score_col].rank(method="average", pct=True)
    return frame


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
    if group.empty:
        return np.nan
    return float(group.nlargest(min(n, len(group)), rank_col)[LABEL].mean())


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    if group.empty:
        return np.nan
    return float(group.nsmallest(min(n, len(group)), rank_col)[LABEL].mean())


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


def add_deltas(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    merged = candidate.merge(baseline, on="trade_date", suffixes=("", "_base"))
    for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
        merged[f"{metric}_delta"] = merged[metric] - merged[f"{metric}_base"]
    return merged


def period_stats(delta: pd.DataFrame) -> dict:
    periods = {
        "2024H2": ("20240604", "20241231"),
        "2025H1": ("20250101", "20250630"),
        "2025H2": ("20250701", "20251231"),
        "2026YTD": ("20260101", "99999999"),
    }
    out: dict[str, float | int] = {}
    top5s: list[float] = []
    rankics: list[float] = []
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


def scan_alphas(scores: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    eval_frame = eval_frame.dropna(subset=[LABEL])
    dates = sorted(eval_frame["trade_date"].unique().tolist())

    stable_daily = daily_metrics(eval_frame, "stable_rank", "stable")
    recent_daily = daily_metrics(eval_frame, "recent_rank", "recent")
    formal_daily = daily_metrics(eval_frame, "formal_rank", "formal")

    stable_by_date = stable_daily.set_index("trade_date")
    formal_by_date = formal_daily.set_index("trade_date")
    source_metrics = pd.concat([formal_daily, stable_daily, recent_daily], ignore_index=True)

    rows = []
    daily_rows = []
    for alpha in [round(x, 2) for x in np.arange(0.05, 1.00, 0.05)]:
        rank_col = f"blend_rank_a{int(alpha * 100):02d}"
        eval_frame[rank_col] = (1.0 - alpha) * eval_frame["stable_rank"] + alpha * eval_frame["recent_rank"]
        cand_daily = daily_metrics(eval_frame, rank_col, f"blend_alpha_{alpha:.2f}")
        delta_stable = add_deltas(cand_daily, stable_by_date.reset_index())
        delta_formal = add_deltas(cand_daily, formal_by_date.reset_index())
        full = summarize(cand_daily, dates, None)
        stable_full = summarize(stable_daily, dates, None)
        r63 = summarize(cand_daily, dates, 63)
        stable63 = summarize(stable_daily, dates, 63)
        r20 = summarize(cand_daily, dates, 20)
        stable20 = summarize(stable_daily, dates, 20)
        row = {
            "alpha_recent": alpha,
            "source": f"blend_alpha_{alpha:.2f}",
        }
        for prefix, cur, base in [("full", full, stable_full), ("recent63", r63, stable63), ("recent20", r20, stable20)]:
            for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20", "top50"]:
                row[f"{prefix}_{metric}"] = cur[metric]
                row[f"{prefix}_{metric}_delta_vs_stable"] = cur[metric] - base[metric]
        row.update(period_stats(delta_stable))
        formal_period = period_stats(delta_formal)
        for key, value in formal_period.items():
            row[f"{key}_vs_formal"] = value
        row["rank_ic_penalty_vs_stable"] = max(0.0, -float(row["full_rank_ic_delta_vs_stable"]))
        row["period_rank_ic_penalty_vs_stable"] = max(0.0, -float(row["min_period_rank_ic_delta"]))
        row["pass_balanced"] = bool(
            row["recent63_top1_delta_vs_stable"] >= -0.001
            and row["recent63_top5_delta_vs_stable"] >= 0.0008
            and row["recent20_top5_delta_vs_stable"] >= -0.0005
            and row["full_top5_delta_vs_stable"] >= 0.0003
            and row["full_rank_ic_delta_vs_stable"] >= -0.0022
            and row["min_period_rank_ic_delta"] >= -0.006
            and row["positive_top5_periods"] >= 3
        )
        row["objective"] = (
            1.2 * row["recent63_top1_delta_vs_stable"]
            + 2.5 * row["recent63_top5_delta_vs_stable"]
            + 1.2 * row["recent20_top5_delta_vs_stable"]
            + 1.0 * row["full_top5_delta_vs_stable"]
            + 0.8 * row["min_period_top5_delta"]
            - 8.0 * row["rank_ic_penalty_vs_stable"]
            - 4.0 * row["period_rank_ic_penalty_vs_stable"]
        )
        rows.append(row)
        daily_part = delta_stable[["trade_date", "rank_ic_delta", "top1_delta", "top5_delta", "top10_delta"]].copy()
        daily_part["alpha_recent"] = alpha
        daily_rows.append(daily_part)
    scan = pd.DataFrame(rows).sort_values(["pass_balanced", "objective"], ascending=[False, False]).reset_index(drop=True)
    daily = pd.concat(daily_rows, ignore_index=True)
    return scan, daily, source_metrics


def write_candidate(scores: pd.DataFrame, alpha: float) -> dict:
    out = scores[["trade_date", "stock_code", "formal_score", "stable_score", "recent_score", "formal_rank", "stable_rank", "recent_rank"]].copy()
    out["alpha_recent"] = alpha
    out["pred_prob"] = (1.0 - alpha) * out["stable_rank"] + alpha * out["recent_rank"]
    out["score_formula"] = f"pred_prob = {1.0 - alpha:.2f} * stable_rank + {alpha:.2f} * recent_rank"
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
        "alpha_recent",
        "score_formula",
    ]
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[columns].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
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
        "formula": out["score_formula"].iloc[0],
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def write_outputs(scan: pd.DataFrame, daily: pd.DataFrame, source_metrics: pd.DataFrame, stats: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scan_path = REPORT_DIR / "blend_alpha_scan_results.csv"
    daily_path = REPORT_DIR / "blend_alpha_daily_deltas.csv"
    source_path = REPORT_DIR / "blend_source_daily_metrics.csv"
    manifest_path = REPORT_DIR / "blend_rankic_balanced_research_manifest.json"
    report_path = REPORT_DIR / "blend_rankic_balanced_research_report.md"
    scan.to_csv(scan_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    source_metrics.to_csv(source_path, index=False, encoding="utf-8-sig")
    best = scan.iloc[0].to_dict() if not scan.empty else None
    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset" if stats else "l4_research_rejected_scan_artifact",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5" if stats else "research_rejected_no_improvement",
        "source_type": "sqlite_table" if stats else "scan_report_only",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db" if stats else None,
        "table": TARGET_TABLE if stats else None,
        "label": LABEL,
        "generated_at": now_iso(),
        "decision": "candidate" if stats else "rejected",
        "baseline_for_scan": "research_5d_stability_guard_v3",
        "blend_sources": {
            "stable": STABLE_TABLE,
            "recent": RECENT_TABLE,
            "formal_reference": FORMAL_TABLE,
        },
        "best_scan": best,
        "asset_stats": stats,
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
        "# 5D RankIC 平衡融合研究报告（20260625）",
        "",
        "## 当前结论",
        "",
    ]
    if stats:
        lines.extend(
            [
                "本轮找到一个 research-only 5D 融合候选，用稳定版作为主体，按固定权重加入近期增强版，以改善 Top5 表现并控制 RankIC 损失。",
                "",
                "候选规则：",
                "",
                "```text",
                stats["formula"],
                "```",
                "",
                f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
                f"- 日期范围：`{stats['min_trade_date']}` 到 `{stats['max_trade_date']}`",
                f"- 总行数：`{stats['row_count']}`",
                f"- 最新日：`{stats['latest_days'][0][0]}`，行数 `{stats['latest_days'][0][1]}`，股票数 `{stats['latest_days'][0][2]}`",
                f"- `pred_prob` 空值：`{stats['null_pred_prob']}`",
                f"- 重复键组：`{stats['duplicate_key_groups']}`",
                "",
                "## 相对 5D 稳定版的评价摘要",
                "",
                f"- alpha_recent：`{best['alpha_recent']:.2f}`",
                f"- 近期 63 日 Top1 增量：`{best['recent63_top1_delta_vs_stable']:.6f}`",
                f"- 近期 63 日 Top5 增量：`{best['recent63_top5_delta_vs_stable']:.6f}`",
                f"- 近期 20 日 Top5 增量：`{best['recent20_top5_delta_vs_stable']:.6f}`",
                f"- 全样本 Top5 增量：`{best['full_top5_delta_vs_stable']:.6f}`",
                f"- 全样本 RankIC 增量：`{best['full_rank_ic_delta_vs_stable']:.6f}`",
                f"- 最差阶段 RankIC 增量：`{best['min_period_rank_ic_delta']:.6f}`",
                f"- 正 Top5 阶段数：`{int(best['positive_top5_periods'])}/4`",
            ]
        )
    else:
        lines.append("本轮未找到通过 RankIC 平衡门槛的 5D 融合候选，仅保留扫描证据。")
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 扫描明细：`{scan_path.as_posix()}`",
            f"- 日度增量：`{daily_path.as_posix()}`",
            f"- 来源日度指标：`{source_path.as_posix()}`",
            f"- 研究 manifest：`{manifest_path.as_posix()}`",
            "",
            "## 边界说明",
            "",
            "- 未训练模型。",
            "- 未调整训练参数。",
            "- 未修改 production/formal manifest。",
            "- 未写入 formal L4 表。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    scores = load_score_frame()
    labels = load_labels(str(scores["trade_date"].min()), str(scores["trade_date"].max()))
    scan, daily, source_metrics = scan_alphas(scores, labels)
    stats = None
    if not scan.empty and bool(scan.iloc[0]["pass_balanced"]):
        stats = write_candidate(scores, float(scan.iloc[0]["alpha_recent"]))
    write_outputs(scan, daily, source_metrics, stats)
    print(
        json.dumps(
            {
                "report_dir": str(REPORT_DIR),
                "scan_rows": int(len(scan)),
                "daily_rows": int(len(daily)),
                "best": scan.iloc[0].to_dict() if not scan.empty else None,
                "asset_stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

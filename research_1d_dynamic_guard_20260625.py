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
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_dynamic_guard_research_20260625"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
LABEL_DIR = DATA_DIR / "prediction_label_parts"
TARGET_TABLE = "stock_predict_data_model_agent_1d_dynamic_guard_20260625_executable_1d_open_return_research"
LABEL = "executable_1d_open_return"

SOURCES = {
    "formal1": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
    "formal3": "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal",
    "formal5": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
    "formal10": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
    "dyn5": "stock_predict_data_model_agent_5d_dynamic_guard_20260625_executable_5d_open_return_research",
    "dyn10": "stock_predict_data_model_agent_10d_dynamic_guard_20260625_executable_10d_open_return_research",
    "prior1d_daygate": "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_ext_20260624_executable_1d_open_return_research",
    "prior1d_narrow": "stock_predict_data_model_agent_1d_narrow_balanced_grid_20260624_executable_1d_open_return_research",
    "prior1d_multi": "stock_predict_data_model_agent_1d_multihorizon_max_3d5d10d_20260623_executable_1d_open_return_research",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_source(conn: sqlite3.Connection, alias: str, table: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": f"{alias}_score"})


def load_scores() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        base = read_source(conn, "formal1", SOURCES["formal1"])
        out = base
        for alias, table in SOURCES.items():
            if alias == "formal1":
                continue
            src = read_source(conn, alias, table)
            out = out.merge(src, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    for alias in SOURCES:
        score_col = f"{alias}_score"
        if score_col not in out.columns:
            continue
        out[score_col] = out[score_col].fillna(out["formal1_score"])
        out[f"{alias}_rank"] = out.groupby("trade_date")[score_col].rank(method="average", pct=True)
    return out


def load_labels(min_date: str, max_date: str) -> pd.DataFrame:
    chunks = []
    for path in LABEL_DIR.glob("*.parquet"):
        part = pd.read_parquet(path, columns=["trade_date", "stock_code", LABEL])
        part["trade_date"] = part["trade_date"].astype(str)
        part["stock_code"] = part["stock_code"].astype(str)
        part = part[(part["trade_date"] >= min_date) & (part["trade_date"] <= max_date)]
        chunks.append(part)
    label = pd.concat(chunks, ignore_index=True)
    label = label.dropna(subset=[LABEL])
    return label


def top_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    if group.empty:
        return float("nan")
    take = min(n, len(group))
    return float(group.nlargest(take, rank_col)[LABEL].mean())


def bottom_mean(group: pd.DataFrame, rank_col: str, n: int) -> float:
    if group.empty:
        return float("nan")
    take = min(n, len(group))
    return float(group.nsmallest(take, rank_col)[LABEL].mean())


def daily_metrics_for_source(frame: pd.DataFrame, alias: str) -> pd.DataFrame:
    rank_col = f"{alias}_rank"
    rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        label_rank = group[LABEL].rank(method="average", pct=True)
        rank_ic = group[rank_col].corr(label_rank, method="pearson")
        pearson_ic = group[rank_col].corr(group[LABEL], method="pearson")
        row = {
            "trade_date": trade_date,
            "source": alias,
            "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
            "pearson_ic": float(pearson_ic) if pd.notna(pearson_ic) else np.nan,
        }
        for n in [1, 3, 5, 10, 20, 50]:
            row[f"top{n}"] = top_mean(group, rank_col, n)
        row["top_bottom"] = top_mean(group, rank_col, 50) - bottom_mean(group, rank_col, 50)
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_source_metrics(eval_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [daily_metrics_for_source(eval_frame, alias) for alias in SOURCES],
        ignore_index=True,
    )


def top_set(group: pd.DataFrame, rank_col: str, n: int) -> set[str]:
    return set(group.nlargest(min(n, len(group)), rank_col)["stock_code"].astype(str))


def daily_features(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in scores.groupby("trade_date", sort=True):
        base10 = top_set(group, "formal1_rank", 10)
        base50 = top_set(group, "formal1_rank", 50)
        row: dict[str, float | str] = {"trade_date": trade_date}
        for alias in SOURCES:
            score_col = f"{alias}_score"
            rank_col = f"{alias}_rank"
            row[f"{alias}_score_std"] = float(group[score_col].std())
            row[f"{alias}_top10_score_mean"] = float(group.nlargest(min(10, len(group)), rank_col)[score_col].mean())
            row[f"{alias}_top50_score_mean"] = float(group.nlargest(min(50, len(group)), rank_col)[score_col].mean())
            if alias != "formal1":
                row[f"corr_formal1_{alias}"] = float(group["formal1_rank"].corr(group[rank_col]))
                row[f"overlap10_formal1_{alias}"] = len(base10 & top_set(group, rank_col, 10))
                row[f"overlap50_formal1_{alias}"] = len(base50 & top_set(group, rank_col, 50))
        rows.append(row)
    return pd.DataFrame(rows)


def window_dates(dates: list[str], n: int | None) -> set[str]:
    if n is None:
        return set(dates)
    return set(dates[-n:])


def summarize_daily(metric_by_source: pd.DataFrame, source: str, dates: set[str]) -> dict[str, float]:
    sub = metric_by_source[(metric_by_source["source"] == source) & (metric_by_source["trade_date"].isin(dates))]
    return {col: float(sub[col].mean()) for col in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]}


def scan_rules(metric_by_source: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    mature_dates = sorted(metric_by_source["trade_date"].unique().tolist())
    date_sets = {
        "full": window_dates(mature_dates, None),
        "recent126": window_dates(mature_dates, 126),
        "recent63": window_dates(mature_dates, 63),
        "recent20": window_dates(mature_dates, 20),
    }
    base_daily = metric_by_source[metric_by_source["source"] == "formal1"].set_index("trade_date")
    source_daily = {
        alias: metric_by_source[metric_by_source["source"] == alias].set_index("trade_date")
        for alias in SOURCES
        if alias != "formal1"
    }
    base_summary = {win: summarize_daily(metric_by_source, "formal1", dates) for win, dates in date_sets.items()}
    feature_cols = [c for c in features.columns if c != "trade_date"]
    feature_frame = features.set_index("trade_date")
    rows = []
    for alt, alt_daily in source_daily.items():
        candidate_features = [c for c in feature_cols if c.startswith(f"{alt}_") or c.endswith(f"_{alt}")]
        for feature in candidate_features:
            values = feature_frame[feature].replace([np.inf, -np.inf], np.nan).dropna()
            if values.nunique() < 3:
                continue
            thresholds = sorted(set(float(values.quantile(q)) for q in [0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.85, 0.9, 0.95]))
            for threshold in thresholds:
                for op in ["<=", ">="]:
                    active = feature_frame[feature] <= threshold if op == "<=" else feature_frame[feature] >= threshold
                    active_dates = set(active[active].index.astype(str))
                    if len(active_dates) < 5:
                        continue
                    row = {
                        "alt_source": alt,
                        "feature": feature,
                        "op": op,
                        "threshold": threshold,
                        "active_days": len(active_dates),
                        "active_ratio": len(active_dates) / len(feature_frame),
                    }
                    for win, dates in date_sets.items():
                        common_dates = sorted(dates & set(base_daily.index) & set(alt_daily.index))
                        win_active = set(common_dates) & active_dates
                        row[f"{win}_active_days"] = len(win_active)
                        for metric in ["rank_ic", "pearson_ic", "top_bottom", "top1", "top3", "top5", "top10", "top20", "top50"]:
                            vals = []
                            for d in common_dates:
                                vals.append(float(alt_daily.loc[d, metric]) if d in active_dates else float(base_daily.loc[d, metric]))
                            mean_val = float(np.nanmean(vals)) if vals else np.nan
                            row[f"{win}_{metric}"] = mean_val
                            row[f"{win}_{metric}_delta"] = mean_val - base_summary[win][metric]
                    row["pass_basic"] = bool(
                        row["recent63_active_days"] >= 2
                        and row["recent63_top1_delta"] > 0
                        and row["recent63_top3_delta"] > 0
                        and row["recent63_top5_delta"] > 0
                        and row["recent63_rank_ic_delta"] >= -0.002
                        and row["recent20_top5_delta"] >= 0
                        and row["full_top5_delta"] >= -0.003
                        and row["full_rank_ic_delta"] >= -0.0015
                    )
                    row["objective"] = (
                        3.0 * row["recent63_top1_delta"]
                        + 2.0 * row["recent63_top3_delta"]
                        + 2.0 * row["recent63_top5_delta"]
                        + row["recent20_top5_delta"]
                        + 0.5 * row["full_top5_delta"]
                        + 0.2 * row["recent63_rank_ic_delta"]
                    )
                    rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["pass_basic", "objective"], ascending=[False, False]).reset_index(drop=True)


def write_prediction_table(scores: pd.DataFrame, features: pd.DataFrame, best: pd.Series) -> dict:
    feature = str(best["feature"])
    op = str(best["op"])
    threshold = float(best["threshold"])
    alt = str(best["alt_source"])
    feature_map = features.set_index("trade_date")[feature]
    active = feature_map <= threshold if op == "<=" else feature_map >= threshold
    active_dates = set(active[active].index.astype(str))

    out = scores[["trade_date", "stock_code", "formal1_score", "formal1_rank", f"{alt}_score", f"{alt}_rank"]].copy()
    out["guard_active"] = out["trade_date"].isin(active_dates)
    out["pred_prob"] = np.where(out["guard_active"], out[f"{alt}_rank"], out["formal1_rank"])
    out["score_formula"] = f"if {feature} {op} {threshold:.12g} then {alt}_rank else formal1_rank"
    out = out.rename(columns={"formal1_score": "formal1", f"{alt}_score": alt})
    keep = ["trade_date", "stock_code", "pred_prob", "formal1", alt, "formal1_rank", f"{alt}_rank", "guard_active", "score_formula"]

    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out[keep].to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        short = hashlib.sha1(TARGET_TABLE.encode("utf-8")).hexdigest()[:12]
        conn.execute(f"create index if not exists idx_{short}_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_{short}_date_pred on {quote(TARGET_TABLE)}(trade_date, pred_prob desc)")
        conn.commit()
        stats = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c>1)"
        ).fetchone()[0]
        latest = conn.execute(
            f"select trade_date, count(*), count(distinct stock_code) from {quote(TARGET_TABLE)} group by trade_date order by trade_date desc limit 10"
        ).fetchall()
    return {
        "table": TARGET_TABLE,
        "formula": out["score_formula"].iloc[0],
        "rule_feature": feature,
        "op": op,
        "threshold": threshold,
        "alt_source": alt,
        "active_days_full": len(active_dates),
        "row_count": int(stats[0]),
        "min_trade_date": str(stats[1]),
        "max_trade_date": str(stats[2]),
        "trade_days": int(stats[3]),
        "null_pred_prob": int(stats[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def write_reports(dataset_summary: dict, source_summary: pd.DataFrame, scan: pd.DataFrame, asset_stats: dict | None, best: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_summary.to_csv(REPORT_DIR / "dynamic_guard_source_summary.csv", index=False, encoding="utf-8-sig")
    scan.to_csv(REPORT_DIR / "dynamic_source_guard_scan_results.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "dynamic_guard_dataset_summary.json").write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    decision = "candidate" if asset_stats else "rejected"
    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset" if asset_stats else "l4_research_rejected_scan_artifact",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5" if asset_stats else "research_rejected_no_improvement",
        "source_type": "sqlite_table" if asset_stats else "scan_report_only",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db" if asset_stats else None,
        "table": TARGET_TABLE if asset_stats else None,
        "label": LABEL,
        "generated_at": now_iso(),
        "decision": decision,
        "best_scan": best,
        "asset_stats": asset_stats,
        "promotion_requires_user_confirmation": True,
        "audit_required_before_formal": True,
        "boundaries": {
            "no_training": True,
            "no_tuning_training_parameters": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "dynamic_guard_research_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 1D 动态守卫研究报告（20260625）",
        "",
        "## 当前结论",
        "",
    ]
    if asset_stats:
        lines.extend(
            [
                "本轮在 research-only 范围内找到一个通过基础模型评价门槛的 1D 动态守卫候选。",
                "",
                "候选规则：",
                "",
                "```text",
                asset_stats["formula"],
                "```",
                "",
                "该规则只使用同日模型分数及其截面统计，不使用未来标签；但规则阈值来自研究扫描，因此不能直接发布到生产。",
                "",
                "## 候选资产",
                "",
                f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
                "- 状态：`research_only_not_approved_for_l4_or_l5`",
                f"- 日期范围：`{asset_stats['min_trade_date']}` 到 `{asset_stats['max_trade_date']}`",
                f"- 总行数：`{asset_stats['row_count']}`",
                f"- 交易日数：`{asset_stats['trade_days']}`",
                f"- 最新日：`{asset_stats['latest_days'][0][0]}`，行数 `{asset_stats['latest_days'][0][1]}`，股票数 `{asset_stats['latest_days'][0][2]}`",
                f"- `pred_prob` 空值：`{asset_stats['null_pred_prob']}`",
                f"- `(trade_date, stock_code)` 重复键组：`{asset_stats['duplicate_key_groups']}`",
                f"- 触发交易日数：`{asset_stats['active_days_full']}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "本轮未找到通过基础模型评价门槛的 1D 动态守卫候选，因此不写入新的研究预测表。",
                "扫描结果仅作为 rejected research artifact 保留。",
                "",
            ]
        )
    if best:
        lines.extend(
            [
                "## 最优扫描结果",
                "",
                f"- 替代源：`{best['alt_source']}`",
                f"- 规则：`{best['feature']} {best['op']} {best['threshold']}`",
                f"- `recent63_rank_ic_delta`：`{best['recent63_rank_ic_delta']:.6f}`",
                f"- `recent63_top1_delta`：`{best['recent63_top1_delta']:.6f}`",
                f"- `recent63_top3_delta`：`{best['recent63_top3_delta']:.6f}`",
                f"- `recent63_top5_delta`：`{best['recent63_top5_delta']:.6f}`",
                f"- `recent63_top10_delta`：`{best['recent63_top10_delta']:.6f}`",
                f"- `full_rank_ic_delta`：`{best['full_rank_ic_delta']:.6f}`",
                f"- `full_top5_delta`：`{best['full_top5_delta']:.6f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 证据路径",
            "",
            f"- 数据摘要：`{(REPORT_DIR / 'dynamic_guard_dataset_summary.json').as_posix()}`",
            f"- 评分源摘要：`{(REPORT_DIR / 'dynamic_guard_source_summary.csv').as_posix()}`",
            f"- 扫描明细：`{(REPORT_DIR / 'dynamic_source_guard_scan_results.csv').as_posix()}`",
            f"- 研究 manifest：`{(REPORT_DIR / 'dynamic_guard_research_manifest.json').as_posix()}`",
            "",
            "## 边界说明",
            "",
            "- 未训练模型。",
            "- 未调参训练参数。",
            "- 未修改 production manifest。",
            "- 未写入 formal L4 表。",
            "- 未修改 `production_tasks.json`。",
            "- 未生成交易信号。",
            "- 未运行策略回测。",
        ]
    )
    (REPORT_DIR / "dynamic_guard_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    min_date = str(scores["trade_date"].min())
    max_date = str(scores["trade_date"].max())
    labels = load_labels(min_date, max_date)
    eval_frame = scores.merge(labels, on=["trade_date", "stock_code"], how="inner", validate="one_to_one")
    metric_by_source = build_daily_source_metrics(eval_frame)
    features = daily_features(scores)
    source_summary_rows = []
    mature_dates = sorted(metric_by_source["trade_date"].unique().tolist())
    for source in SOURCES:
        for win, dates in {
            "full": window_dates(mature_dates, None),
            "recent126": window_dates(mature_dates, 126),
            "recent63": window_dates(mature_dates, 63),
            "recent20": window_dates(mature_dates, 20),
        }.items():
            row = {"source": source, "window": win}
            row.update(summarize_daily(metric_by_source, source, dates))
            source_summary_rows.append(row)
    source_summary = pd.DataFrame(source_summary_rows)
    scan = scan_rules(metric_by_source, features)
    if scan.empty:
        best = None
        asset_stats = None
    else:
        best_row = scan.iloc[0]
        best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best_row.to_dict().items()}
        asset_stats = write_prediction_table(scores, features, best_row) if bool(best_row["pass_basic"]) else None
    dataset_summary = {
        "generated_at": now_iso(),
        "score_rows": int(len(scores)),
        "score_min_trade_date": min_date,
        "score_max_trade_date": max_date,
        "score_trade_days": int(scores["trade_date"].nunique()),
        "label_rows": int(len(labels)),
        "label_min_trade_date": str(labels["trade_date"].min()),
        "label_max_trade_date": str(labels["trade_date"].max()),
        "label_trade_days": int(labels["trade_date"].nunique()),
        "eval_rows": int(len(eval_frame)),
        "eval_trade_days": int(eval_frame["trade_date"].nunique()),
        "sources": SOURCES,
    }
    write_reports(dataset_summary, source_summary, scan, asset_stats, best)
    print(json.dumps({"dataset_summary": dataset_summary, "best": best, "asset_stats": asset_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
SOURCE_REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_dynamic_guard_research_20260625"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_1d_dynamic_guard_strict_research_20260625"
TARGET_TABLE = "stock_predict_data_model_agent_1d_dynamic_guard_strict_20260625_executable_1d_open_return_research"

sys.path.insert(0, str(MAIN_DIR))
import research_1d_dynamic_guard_20260625 as broad  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def choose_strict_rule(scan: pd.DataFrame) -> pd.Series:
    sub = scan[
        (scan["active_ratio"] <= 0.25)
        & (scan["recent63_active_days"] >= 5)
        & (scan["recent63_top1_delta"] > 0)
        & (scan["recent63_top3_delta"] > 0)
        & (scan["recent63_top5_delta"] > 0)
        & (scan["recent63_top10_delta"] > 0)
        & (scan["recent63_rank_ic_delta"] >= -0.002)
        & (scan["recent20_top5_delta"] >= 0)
        & (scan["full_top5_delta"] >= -0.002)
        & (scan["full_rank_ic_delta"] >= -0.0015)
    ].copy()
    if sub.empty:
        raise RuntimeError("no strict 1D dynamic guard candidate passed the research gates")
    sub["strict_objective"] = (
        3.0 * sub["recent63_top1_delta"]
        + 2.0 * sub["recent63_top3_delta"]
        + 2.0 * sub["recent63_top5_delta"]
        + sub["recent20_top5_delta"]
        + 0.5 * sub["full_top5_delta"]
        + 0.2 * sub["recent63_rank_ic_delta"]
        - 0.01 * sub["active_ratio"]
    )
    return sub.sort_values("strict_objective", ascending=False).iloc[0]


def write_strict_table(scores: pd.DataFrame, features: pd.DataFrame, best: pd.Series) -> dict:
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
        row = conn.execute(
            f"select count(*), min(trade_date), max(trade_date), count(distinct trade_date), sum(case when pred_prob is null then 1 else 0 end) from {quote(TARGET_TABLE)}"
        ).fetchone()
        dup = conn.execute(
            f"select count(*) from (select trade_date, stock_code, count(*) c from {quote(TARGET_TABLE)} group by trade_date, stock_code having c > 1)"
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
        "active_days_full": int(out.groupby("trade_date")["guard_active"].first().sum()),
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def write_outputs(best: pd.Series, asset_stats: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best_dict = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()}
    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "source_type": "sqlite_table",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db",
        "table": TARGET_TABLE,
        "label": broad.LABEL,
        "generated_at": now_iso(),
        "decision": "strict_candidate",
        "formula": asset_stats["formula"],
        "source_scan": str((SOURCE_REPORT_DIR / "dynamic_source_guard_scan_results.csv").as_posix()),
        "best_scan": best_dict,
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
    (REPORT_DIR / "strict_dynamic_guard_research_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([best_dict]).to_csv(REPORT_DIR / "strict_dynamic_guard_selected_rule.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 1D 严格动态守卫研究报告（20260625）",
        "",
        "## 当前结论",
        "",
        "本轮从上一版 1D 动态守卫扫描结果中筛出一个更严格的 research-only 候选。它避免了上一版 `overlap50 >= 1` 过宽的问题，只在约 20% 交易日切换评分源。",
        "",
        "候选规则：",
        "",
        "```text",
        asset_stats["formula"],
        "```",
        "",
        "该规则只使用同日分数分布特征，不使用未来标签；但阈值来自研究扫描，因此不能直接作为生产模型或 L5 输入。",
        "",
        "## 候选资产",
        "",
        f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
        "- 状态：`research_only_not_approved_for_l4_or_l5`",
        f"- 日期范围：`{asset_stats['min_trade_date']}` 到 `{asset_stats['max_trade_date']}`",
        f"- 总行数：`{asset_stats['row_count']}`",
        f"- 交易日数：`{asset_stats['trade_days']}`",
        f"- 最新日：`{asset_stats['latest_days'][0][0]}`，行数 `{asset_stats['latest_days'][0][1]}`，股票数 `{asset_stats['latest_days'][0][2]}`",
        f"- 触发交易日数：`{asset_stats['active_days_full']}`",
        f"- `pred_prob` 空值：`{asset_stats['null_pred_prob']}`",
        f"- `(trade_date, stock_code)` 重复键组：`{asset_stats['duplicate_key_groups']}`",
        "",
        "## 相对 formal 1D 的评价增量",
        "",
        f"- 全样本 RankIC：`{best_dict['full_rank_ic_delta']:.6f}`",
        f"- 全样本 Top1：`{best_dict['full_top1_delta']:.6f}`",
        f"- 全样本 Top5：`{best_dict['full_top5_delta']:.6f}`",
        f"- 近 63 日 RankIC：`{best_dict['recent63_rank_ic_delta']:.6f}`",
        f"- 近 63 日 Top1：`{best_dict['recent63_top1_delta']:.6f}`",
        f"- 近 63 日 Top3：`{best_dict['recent63_top3_delta']:.6f}`",
        f"- 近 63 日 Top5：`{best_dict['recent63_top5_delta']:.6f}`",
        f"- 近 63 日 Top10：`{best_dict['recent63_top10_delta']:.6f}`",
        "",
        "## 证据路径",
        "",
        f"- 研究 manifest：`{(REPORT_DIR / 'strict_dynamic_guard_research_manifest.json').as_posix()}`",
        f"- 选中规则：`{(REPORT_DIR / 'strict_dynamic_guard_selected_rule.csv').as_posix()}`",
        f"- 原扫描明细：`{(SOURCE_REPORT_DIR / 'dynamic_source_guard_scan_results.csv').as_posix()}`",
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
    (REPORT_DIR / "strict_dynamic_guard_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    scan = pd.read_csv(SOURCE_REPORT_DIR / "dynamic_source_guard_scan_results.csv")
    best = choose_strict_rule(scan)
    scores = broad.load_scores()
    features = broad.daily_features(scores)
    asset_stats = write_strict_table(scores, features, best)
    write_outputs(best, asset_stats)
    print(json.dumps({"best": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()}, "asset_stats": asset_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

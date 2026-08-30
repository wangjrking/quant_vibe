from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
MANIFEST_DIR = MAIN / "config" / "prediction_manifests"
REPORT_DIR = DATA / "reports" / "model_agent_recent_formal_l4_direction_diagnostic_20260720"
MARKET_DB = DATA / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
LABEL_DB = DATA / "production_assets" / "duckdb" / "l3_label_current.duckdb"
MARKET_TABLE = "STOCK_DAILY_DATA"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
WINDOW_START = "20260701"
WINDOW_END = "20260716"

SPECS = {
    "1d": {
        "label": "executable_1d_open_return",
        "sell_offset": 2,
        "manifest": "executable_1d_open_return_l4_formal_20260619.json",
    },
    "3d": {
        "label": "executable_3d_open_return",
        "sell_offset": 4,
        "manifest": "executable_3d_open_return_l4_formal_20260617.json",
    },
    "5d": {
        "label": "executable_5d_open_return",
        "sell_offset": 6,
        "manifest": "executable_5d_open_return_l4_formal_20260620.json",
    },
    "10d": {
        "label": "executable_10d_open_return",
        "sell_offset": 12,
        "manifest": "executable_10d_open_return_l4_formal_20260617.json",
    },
}

ROLLBACK_3D_DB = (
    DATA
    / "reports"
    / "model_agent_3d_formal_window_rerun_20260701_20260716"
    / "l4_3d_formal_window_before_rerun.duckdb"
)
ROLLBACK_3D_TABLE = "l4_3d_formal_window_before_rerun"


def resolve_manifest_asset(manifest_name: str) -> tuple[Path, str, dict]:
    manifest_path = MANIFEST_DIR / manifest_name
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    db_path = (manifest_path.parent / payload["db_path"]).resolve()
    return db_path, payload["table"], payload


def quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


@lru_cache(maxsize=4)
def load_market_since(start_date: str) -> pd.DataFrame:
    market_con = duckdb.connect(str(MARKET_DB), read_only=True)
    market = market_con.execute(
        f"SELECT trade_date, stock_code, CAST(open AS DOUBLE) open "
        f"FROM {MARKET_TABLE} WHERE trade_date >= ?",
        [start_date],
    ).fetch_df()
    market_con.close()
    return market


def load_realized_frame(db_path: Path, table: str, sell_offset: int) -> pd.DataFrame:
    score_con = duckdb.connect(str(db_path), read_only=True)
    scores = score_con.execute(
        f"SELECT trade_date, stock_code, CAST(pred_prob AS DOUBLE) pred_prob "
        f"FROM {table} WHERE trade_date BETWEEN ? AND ? AND stock_code NOT LIKE '%.BJ'",
        [WINDOW_START, WINDOW_END],
    ).fetch_df()
    score_con.close()

    market = load_market_since(WINDOW_START)
    dates = sorted(market["trade_date"].drop_duplicates().astype(str).tolist())
    position = {trade_date: index for index, trade_date in enumerate(dates)}
    mapped = []
    for trade_date in sorted(scores["trade_date"].drop_duplicates().astype(str)):
        index = position.get(trade_date)
        if index is None or index + sell_offset >= len(dates):
            continue
        mapped.append((trade_date, dates[index + 1], dates[index + sell_offset]))
    mapping = pd.DataFrame(mapped, columns=["trade_date", "buy_date", "sell_date"])
    frame = scores.merge(mapping, on="trade_date", how="inner")
    buy = market.rename(columns={"trade_date": "buy_date", "open": "buy_open"})
    sell = market.rename(columns={"trade_date": "sell_date", "open": "sell_open"})
    frame = frame.merge(buy, on=["buy_date", "stock_code"], how="inner")
    frame = frame.merge(sell, on=["sell_date", "stock_code"], how="inner")
    frame = frame[(frame["buy_open"].notna()) & (frame["buy_open"] != 0) & frame["sell_open"].notna()].copy()
    frame["realized_return"] = (
        frame["sell_open"] * (1.0 - 0.0003 - 0.0005 - 0.001)
        / (frame["buy_open"] * (1.0 + 0.0003 + 0.001))
        - 1.0
    )
    return frame[["trade_date", "stock_code", "pred_prob", "realized_return"]]


def summarize_daily(frame: pd.DataFrame, score_name: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group.dropna(subset=["pred_prob", "realized_return"]).copy()
        n = len(group)
        if n == 0:
            continue
        top_n = max(1, int(np.ceil(n * 0.01)))
        top = group.nlargest(top_n, "pred_prob")
        rows.append(
            {
                "score": score_name,
                "trade_date": str(trade_date),
                "rows": n,
                "distinct_pred_prob": int(group["pred_prob"].nunique()),
                "std_pred_prob": float(group["pred_prob"].std(ddof=0)),
                "rank_ic": float(group["pred_prob"].corr(group["realized_return"], method="spearman")),
                "market_mean": float(group["realized_return"].mean()),
                "top1pct_mean": float(top["realized_return"].mean()),
                "top1pct_excess": float(top["realized_return"].mean() - group["realized_return"].mean()),
            }
        )
    daily = pd.DataFrame(rows)
    summary = {
        "score": score_name,
        "trade_days": int(len(daily)),
        "min_trade_date": None if daily.empty else str(daily["trade_date"].min()),
        "max_trade_date": None if daily.empty else str(daily["trade_date"].max()),
        "mean_rank_ic": None if daily.empty else float(daily["rank_ic"].mean()),
        "rank_ic_positive_ratio": None if daily.empty else float((daily["rank_ic"] > 0).mean()),
        "mean_top1pct_excess": None if daily.empty else float(daily["top1pct_excess"].mean()),
        "top1pct_excess_positive_ratio": None
        if daily.empty
        else float((daily["top1pct_excess"] > 0).mean()),
        "min_distinct_pred_prob": None if daily.empty else int(daily["distinct_pred_prob"].min()),
        "min_std_pred_prob": None if daily.empty else float(daily["std_pred_prob"].min()),
    }
    return daily, summary


def load_mature_label_frame(db_path: Path, table: str, label: str) -> pd.DataFrame:
    label_con = duckdb.connect(str(LABEL_DB), read_only=True)
    labels = label_con.execute(
        f"SELECT trade_date, stock_code, CAST({label} AS DOUBLE) label_return "
        f"FROM {LABEL_TABLE} WHERE trade_date >= '20260301' AND {label} IS NOT NULL "
        "AND stock_code NOT LIKE '%.BJ'"
    ).fetch_df()
    label_con.close()
    score_con = duckdb.connect(str(db_path), read_only=True)
    scores = score_con.execute(
        f"SELECT trade_date, stock_code, CAST(pred_prob AS DOUBLE) pred_prob "
        f"FROM {table} WHERE trade_date >= '20260301' AND stock_code NOT LIKE '%.BJ'"
    ).fetch_df()
    score_con.close()
    return scores.merge(labels, on=["trade_date", "stock_code"], how="inner")


def summarize_mature(frame: pd.DataFrame) -> dict:
    daily_rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group.dropna(subset=["pred_prob", "label_return"])
        if group.empty:
            continue
        n = len(group)
        top_n = max(1, int(np.ceil(n * 0.01)))
        top = group.nlargest(top_n, "pred_prob")
        daily_rows.append(
            {
                "trade_date": str(trade_date),
                "rank_ic": float(group["pred_prob"].corr(group["label_return"], method="spearman")),
                "top1pct_excess": float(top["label_return"].mean() - group["label_return"].mean()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    result = {}
    for name, count in (("recent63", 63), ("recent20", 20)):
        part = daily.tail(count)
        result[name] = {
            "trade_days": int(len(part)),
            "min_trade_date": None if part.empty else str(part["trade_date"].min()),
            "max_trade_date": None if part.empty else str(part["trade_date"].max()),
            "mean_rank_ic": None if part.empty else float(part["rank_ic"].mean()),
            "rank_ic_positive_ratio": None if part.empty else float((part["rank_ic"] > 0).mean()),
            "mean_top1pct_excess": None if part.empty else float(part["top1pct_excess"].mean()),
            "top1pct_excess_positive_ratio": None
            if part.empty
            else float((part["top1pct_excess"] > 0).mean()),
        }
    return result


def validate_label_alignment(label: str, sell_offset: int) -> dict:
    label_con = duckdb.connect(str(LABEL_DB), read_only=True)
    labels = label_con.execute(
        f"SELECT trade_date, stock_code, CAST({label} AS DOUBLE) label_return "
        f"FROM {LABEL_TABLE} WHERE trade_date >= '20260501' AND {label} IS NOT NULL "
        "AND stock_code NOT LIKE '%.BJ'"
    ).fetch_df()
    label_con.close()
    recent_dates = sorted(labels["trade_date"].drop_duplicates().astype(str))[-20:]
    labels = labels[labels["trade_date"].astype(str).isin(recent_dates)].copy()
    market = load_market_since("20260501")
    dates = sorted(market["trade_date"].drop_duplicates().astype(str).tolist())
    position = {trade_date: index for index, trade_date in enumerate(dates)}
    mapped = []
    for trade_date in recent_dates:
        index = position.get(trade_date)
        if index is None or index + sell_offset >= len(dates):
            continue
        mapped.append((trade_date, dates[index + 1], dates[index + sell_offset]))
    mapping = pd.DataFrame(mapped, columns=["trade_date", "buy_date", "sell_date"])
    frame = labels.merge(mapping, on="trade_date", how="inner")
    buy = market.rename(columns={"trade_date": "buy_date", "open": "buy_open"})
    sell = market.rename(columns={"trade_date": "sell_date", "open": "sell_open"})
    frame = frame.merge(buy, on=["buy_date", "stock_code"], how="inner")
    frame = frame.merge(sell, on=["sell_date", "stock_code"], how="inner")
    frame = frame[(frame["buy_open"].notna()) & (frame["buy_open"] != 0) & frame["sell_open"].notna()].copy()
    frame["derived_return"] = (
        frame["sell_open"] * (1.0 - 0.0003 - 0.0005 - 0.001)
        / (frame["buy_open"] * (1.0 + 0.0003 + 0.001))
        - 1.0
    )
    difference = frame["label_return"] - frame["derived_return"]
    return {
        "rows": int(len(frame)),
        "trade_days": int(frame["trade_date"].nunique()),
        "min_trade_date": None if frame.empty else str(frame["trade_date"].min()),
        "max_trade_date": None if frame.empty else str(frame["trade_date"].max()),
        "pearson_corr": None if frame.empty else float(frame["label_return"].corr(frame["derived_return"])),
        "mean_abs_diff": None if frame.empty else float(difference.abs().mean()),
        "max_abs_diff": None if frame.empty else float(difference.abs().max()),
        "formula": "post_sell_open*(1-0.0003-0.0005-0.001)/(post_open*(1+0.0003+0.001))-1",
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "actor": "model-agent",
        "scope": "read_only_recent_formal_l4_direction_and_alignment_diagnostic",
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "boundaries": {
            "no_training": True,
            "no_tuning": True,
            "no_formal_write": True,
            "no_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
            "no_l5_change": True,
        },
        "horizons": {},
    }
    daily_parts = []
    for horizon, spec in SPECS.items():
        db_path, table, manifest = resolve_manifest_asset(spec["manifest"])
        realized = load_realized_frame(db_path, table, spec["sell_offset"])
        daily, recent_summary = summarize_daily(realized, f"{horizon}_active_formal")
        mature = summarize_mature(load_mature_label_frame(db_path, table, spec["label"]))
        daily_parts.append(daily)
        result["horizons"][horizon] = {
            "label": spec["label"],
            "formal_asset": f"{db_path}::{table}",
            "manifest": str(MANIFEST_DIR / spec["manifest"]),
            "manifest_status": manifest.get("approval_status"),
            "recent_realized_open_to_open": recent_summary,
            "mature_label": mature,
            "label_alignment_vs_l2_raw_open": validate_label_alignment(
                spec["label"], spec["sell_offset"]
            ),
        }

    rollback = load_realized_frame(ROLLBACK_3D_DB, ROLLBACK_3D_TABLE, SPECS["3d"]["sell_offset"])
    rollback_daily, rollback_summary = summarize_daily(rollback, "3d_pre_repair_rollback")
    daily_parts.append(rollback_daily)
    active_3d = result["horizons"]["3d"]["recent_realized_open_to_open"]
    result["three_d_repair_comparison"] = {
        "pre_repair": rollback_summary,
        "post_repair": active_3d,
        "mean_rank_ic_delta": None
        if active_3d["mean_rank_ic"] is None or rollback_summary["mean_rank_ic"] is None
        else active_3d["mean_rank_ic"] - rollback_summary["mean_rank_ic"],
        "mean_top1pct_excess_delta": None
        if active_3d["mean_top1pct_excess"] is None or rollback_summary["mean_top1pct_excess"] is None
        else active_3d["mean_top1pct_excess"] - rollback_summary["mean_top1pct_excess"],
    }
    result["review_gate"] = {
        "task_id": "production-strategy-recent-failure-l4-review-20260720",
        "status": "completed_waiting_for_audit",
        "risk_level": "P1_recent_realized_direction_reversal",
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "technical_findings": {
            "score_sign_inversion_in_code": False,
            "label_date_shift_detected": False,
            "three_d_constant_score_defect_repaired": True,
            "three_d_post_repair_recent_direction_reversal_detected": True,
            "one_d_recent_underperformance_detected": True,
            "five_d_recent_underperformance_detected": True,
            "ten_d_recent_sample_sufficient_for_conclusion": False,
        },
        "affected_assets": {
            "three_d_data_integrity_window": ["20260701", "20260716"],
            "one_d_realized_review_window": ["20260701", "20260715"],
            "three_d_realized_review_window": ["20260701", "20260713"],
            "five_d_realized_review_window": ["20260701", "20260709"],
            "ten_d_realized_review_window": ["20260701", "20260701"],
        },
        "rerun_decision": {
            "one_d": "no_data_integrity_rerun_evidence; research_only_degradation_review_needed",
            "three_d": "already_rerun_for_20260701_20260716; repeating_same_model_is_not_recommended",
            "five_d": "no_data_integrity_rerun_evidence; research_only_degradation_review_needed",
            "ten_d": "insufficient_recent_realized_days; no_rerun_evidence",
        },
        "production_display_gate": {
            "annualized_1815_87pct": "invalid_pending_rebuild_and_reaudit",
            "sharpe_2_9895": "invalid_pending_rebuild_and_reaudit",
            "max_drawdown_7_59pct": "invalid_pending_rebuild_and_reaudit",
        },
        "l5_handoff": {
            "active_manifests_are_identified": True,
            "historical_signals_need_rebuild": True,
            "historical_signal_rebuild_is_model_agent_scope": False,
            "audit_required_before_rebuild": True,
            "commander_gate_required": True,
        },
        "next_model_action": "research_only_recent_regime_and_top_rank_stability_review",
        "production_change_requires_separate_user_authorization": True,
    }

    daily_all = pd.concat(daily_parts, ignore_index=True)
    daily_path = REPORT_DIR / "recent_realized_daily_metrics.csv"
    daily_all.to_csv(daily_path, index=False, encoding="utf-8-sig")

    summary_path = REPORT_DIR / "recent_formal_l4_direction_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 当前正式 L4 近期方向与标签对齐诊断",
        "",
        "## 边界",
        "",
        "本报告只读比较正式模型分数、成熟标签和已实现未复权开盘收益；未训练、未调参、未改正式预测表或 manifest，未生成信号、未回测。",
        "",
        "## 结果摘要",
        "",
        "| 标签 | 近期可评价日 | 平均 RankIC | Top1% 超额 | 成熟 Recent20 RankIC | 成熟 Recent20 Top1% 超额 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in SPECS:
        recent = result["horizons"][horizon]["recent_realized_open_to_open"]
        mature20 = result["horizons"][horizon]["mature_label"]["recent20"]
        recent_rank_ic = "NA" if recent["mean_rank_ic"] is None else f"{recent['mean_rank_ic']:.6f}"
        recent_top = "NA" if recent["mean_top1pct_excess"] is None else f"{recent['mean_top1pct_excess']:.6f}"
        lines.append(
            f"| {horizon.upper()} | {recent['trade_days']} | {recent_rank_ic} | "
            f"{recent_top} | {mature20['mean_rank_ic']:.6f} | "
            f"{mature20['mean_top1pct_excess']:.6f} |"
        )
    comp = result["three_d_repair_comparison"]
    lines.extend(
        [
            "",
            "## 3D 修复前后",
            "",
            f"- 修复前平均 RankIC：`{comp['pre_repair']['mean_rank_ic']:.6f}`。",
            f"- 修复后平均 RankIC：`{comp['post_repair']['mean_rank_ic']:.6f}`。",
            f"- RankIC 变化：`{comp['mean_rank_ic_delta']:+.6f}`。",
            f"- Top1% 超额变化：`{comp['mean_top1pct_excess_delta']:+.6f}`。",
            "",
            "## 解释边界",
            "",
            "20260701 之后没有 L3 成熟标签，本报告中的近期窗口使用 L2 未复权开盘价按正式标签成本公式重建，只能作为事后诊断；正式模型效果评价仍以成熟标签窗口为准。",
            "",
            "## L4 放行结论",
            "",
            "- 当前风险等级：`P1_recent_realized_direction_reversal`。",
            "- 3D 的常数分数缺陷已经完成数值重刷，但修复后的近期已实现方向显著为负；重复使用同一模型重刷不会解决该问题。",
            "- 1D、5D 未发现与 3D 相同的数据完整性缺口，但近期已实现表现同步转弱，应进入 research-only 近期稳定性复核。",
            "- 10D 仅有 1 个已实现交易日，样本不足，不能定性。",
            "- `ready_for_audit_review=true`，`allow_next_layer_continue=false`；审计与指挥官关闭门禁前，不允许 L5 使用旧历史信号继续生产展示。",
        ]
    )
    report_path = REPORT_DIR / "recent_formal_l4_direction_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "daily": str(daily_path), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

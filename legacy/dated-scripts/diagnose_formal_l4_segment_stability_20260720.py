from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA = ROOT / "quant" / "data_file"
MANIFEST_DIR = MAIN / "config" / "prediction_manifests"
REPORT_DIR = DATA / "reports" / "model_agent_formal_l4_segment_stability_20260720"
LABEL_DB = DATA / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"

SPECS = {
    "1d": {
        "manifest": "executable_1d_open_return_l4_formal_20260619.json",
        "sell_offset": 2,
        "label": "executable_1d_open_return",
    },
    "3d": {
        "manifest": "executable_3d_open_return_l4_formal_20260617.json",
        "sell_offset": 4,
        "label": "executable_3d_open_return",
    },
    "5d": {
        "manifest": "executable_5d_open_return_l4_formal_20260620.json",
        "sell_offset": 6,
        "label": "executable_5d_open_return",
    },
    "10d": {
        "manifest": "executable_10d_open_return_l4_formal_20260617.json",
        "sell_offset": 12,
        "label": "executable_10d_open_return",
    },
}

PERIODS = {
    "2022H2": ("20220606", "20221231"),
    "2023": ("20230101", "20231231"),
    "2024": ("20240101", "20241231"),
    "2025": ("20250101", "20251231"),
    "2026YTD": ("20260101", "99991231"),
}


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def resolve_manifest(manifest_name: str) -> tuple[Path, str, dict]:
    path = MANIFEST_DIR / manifest_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    db_path = (path.parent / payload["db_path"]).resolve()
    return db_path, payload["table"], payload


def daily_metrics(db_path: Path, table: str, label: str) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.execute(f"ATTACH '{sql_path(LABEL_DB)}' AS label_db (READ_ONLY)")
    con.execute(f"ATTACH '{sql_path(db_path)}' AS score_db (READ_ONLY)")
    query = f"""
        WITH realized AS (
            SELECT
                s.trade_date,
                s.stock_code,
                CAST(s.pred_prob AS DOUBLE) AS pred_prob,
                CAST(l.{label} AS DOUBLE) AS net_return,
                (CAST(l.{label} AS DOUBLE) + 1.0) * (1.0013 / 0.9982) - 1.0 AS gross_return
            FROM score_db.{table} s
            JOIN label_db.{LABEL_TABLE} l
              ON l.trade_date = s.trade_date AND l.stock_code = s.stock_code
            WHERE s.trade_date >= '20220606'
              AND s.stock_code NOT LIKE '%.BJ'
              AND s.pred_prob IS NOT NULL
              AND l.{label} IS NOT NULL
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY trade_date ORDER BY pred_prob DESC, stock_code
                ) AS score_rank,
                RANK() OVER (
                    PARTITION BY trade_date ORDER BY pred_prob
                ) AS pred_rank,
                RANK() OVER (
                    PARTITION BY trade_date ORDER BY net_return
                ) AS return_rank
            FROM realized
        )
        SELECT
            trade_date,
            COUNT(*) AS rows,
            COUNT(DISTINCT pred_prob) AS distinct_pred_prob,
            STDDEV_POP(pred_prob) AS std_pred_prob,
            CORR(pred_rank, return_rank) AS rank_ic,
            AVG(gross_return) AS market_gross,
            AVG(net_return) AS market_net,
            AVG(gross_return) FILTER (WHERE score_rank <= 1) AS top1_gross,
            AVG(net_return) FILTER (WHERE score_rank <= 1) AS top1_net,
            AVG(gross_return) FILTER (WHERE score_rank <= 3) AS top3_gross,
            AVG(net_return) FILTER (WHERE score_rank <= 3) AS top3_net
        FROM ranked
        GROUP BY trade_date
        ORDER BY trade_date
    """
    result = con.execute(query).fetch_df()
    con.close()
    result["trade_date"] = result["trade_date"].astype(str)
    for top in (1, 3):
        result[f"top{top}_excess"] = result[f"top{top}_gross"] - result["market_gross"]
    return result


def summarize(part: pd.DataFrame) -> dict:
    if part.empty:
        return {"trade_days": 0}
    return {
        "trade_days": int(len(part)),
        "min_trade_date": str(part["trade_date"].min()),
        "max_trade_date": str(part["trade_date"].max()),
        "rank_ic_mean": float(part["rank_ic"].mean()),
        "rank_ic_positive_ratio": float((part["rank_ic"] > 0).mean()),
        "top1_gross_mean": float(part["top1_gross"].mean()),
        "top1_net_mean": float(part["top1_net"].mean()),
        "top1_excess_mean": float(part["top1_excess"].mean()),
        "top1_excess_positive_ratio": float((part["top1_excess"] > 0).mean()),
        "top3_gross_mean": float(part["top3_gross"].mean()),
        "top3_net_mean": float(part["top3_net"].mean()),
        "top3_excess_mean": float(part["top3_excess"].mean()),
        "top3_excess_positive_ratio": float((part["top3_excess"] > 0).mean()),
        "min_distinct_pred_prob": int(part["distinct_pred_prob"].min()),
        "min_std_pred_prob": float(part["std_pred_prob"].min()),
    }


def period_summaries(daily: pd.DataFrame) -> dict:
    result = {}
    for name, (start, end) in PERIODS.items():
        result[name] = summarize(
            daily[(daily["trade_date"] >= start) & (daily["trade_date"] <= end)]
        )
    result["recent60"] = summarize(daily.tail(60))
    return result


def classify_stability(periods: dict) -> dict:
    year_2023 = periods["2023"]
    recent60 = periods["recent60"]
    return {
        "year_2023_rank_ic_negative": year_2023.get("rank_ic_mean", 0.0) < 0,
        "year_2023_top1_excess_negative": year_2023.get("top1_excess_mean", 0.0) < 0,
        "year_2023_top3_excess_negative": year_2023.get("top3_excess_mean", 0.0) < 0,
        "recent60_rank_ic_negative": recent60.get("rank_ic_mean", 0.0) < 0,
        "recent60_top1_excess_negative": recent60.get("top1_excess_mean", 0.0) < 0,
        "recent60_top3_excess_negative": recent60.get("top3_excess_mean", 0.0) < 0,
    }


def write_markdown(payload: dict) -> None:
    lines = [
        "# 当前 formal L4 分段稳定性只读诊断",
        "",
        "## 口径",
        "",
        "- 预测：当前 active formal L4 1D / 3D / 5D / 10D。",
            "- 收益：使用 active L3 中经审计的 executable open-return 标签；该标签由 L2 未复权开盘按各期限生成。",
            "- 净收益：直接使用模型标签；毛收益按同一成本公式逆推；Top1 / Top3 超额使用毛收益减全市场毛收益。",
        "- 本报告只做模型评价，不生成信号、不跑策略回测、不修改正式资产。",
        "",
        "## 分段结果",
        "",
        "| 标签 | 区间 | 交易日 | RankIC | Top1毛收益 | Top1超额 | Top3毛收益 | Top3超额 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon, data in payload["horizons"].items():
        for period in ("2022H2", "2023", "2024", "2025", "2026YTD", "recent60"):
            row = data["periods"][period]
            lines.append(
                f"| {horizon} | {period} | {row.get('trade_days', 0)} | "
                f"{row.get('rank_ic_mean', float('nan')):.6f} | "
                f"{row.get('top1_gross_mean', float('nan')):.4%} | "
                f"{row.get('top1_excess_mean', float('nan')):.4%} | "
                f"{row.get('top3_gross_mean', float('nan')):.4%} | "
                f"{row.get('top3_excess_mean', float('nan')):.4%} |"
            )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- `ready_for_audit_review=true`。",
            "- `allow_next_layer_continue=false`，等待审计与指挥官收口。",
        ]
    )
    (REPORT_DIR / "formal_l4_segment_stability_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "actor": "model-agent",
        "scope": "read_only_active_formal_l4_segment_stability",
        "evaluation_semantics": {
            "market_price": "active L3 executable label derived from L2 raw open, no adjustment",
            "buy": "signal date + 1 trading day open",
            "sell_offsets": {h: spec["sell_offset"] for h, spec in SPECS.items()},
            "net_formula": "sell_open*(1-0.0003-0.0005-0.001)/(buy_open*(1+0.0003+0.001))-1",
            "top_definition": "daily highest pred_prob Top1 / Top3",
        },
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
        "review_gate": {
            "ready_for_audit_review": True,
            "allow_next_layer_continue": False,
        },
    }
    daily_parts = []
    flat_rows = []
    for horizon, spec in SPECS.items():
        db_path, table, manifest = resolve_manifest(spec["manifest"])
        daily = daily_metrics(db_path, table, spec["label"])
        daily.insert(0, "horizon", horizon)
        daily_parts.append(daily)
        periods = period_summaries(daily)
        payload["horizons"][horizon] = {
            "manifest": str(MANIFEST_DIR / spec["manifest"]),
            "manifest_status": manifest.get("approval_status"),
            "formal_asset": f"{db_path}::{table}",
            "periods": periods,
            "stability_flags": classify_stability(periods),
        }
        for period, summary in periods.items():
            flat_rows.append({"horizon": horizon, "period": period, **summary})

    (REPORT_DIR / "formal_l4_segment_stability_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(flat_rows).to_csv(
        REPORT_DIR / "formal_l4_segment_stability_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(daily_parts, ignore_index=True).to_csv(
        REPORT_DIR / "formal_l4_daily_open_return_metrics.csv", index=False, encoding="utf-8-sig"
    )
    write_markdown(payload)
    print(json.dumps({"report_dir": str(REPORT_DIR), "horizons": list(SPECS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

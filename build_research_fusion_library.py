from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


ROOT = Path(r"D:\work\quant\quant_mcp")
DATA_DIR = ROOT / "quant" / "data_file"
MANIFEST_DIR = ROOT / "quant" / "main" / "config" / "prediction_manifests"
DEFAULT_MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
DEFAULT_MARKET_DB = DATA_DIR / "STOCK_DAILY_DATA.db"
DEFAULT_OUTPUT_DIR = (
    DATA_DIR / "reports" / "strategy_agent_model_application_20260619" / "published_asset_fusions_0618_refresh"
)

DEFAULT_TABLE_3D = "stock_predict_data_model_agent_toprank_industryfix_20260618_executable_3d_open_return_v1"
DEFAULT_TABLE_5D = "stock_predict_data_model_agent_toprank_industryfix_20260618_executable_5d_open_return_v1"
DEFAULT_TABLE_10D = "stock_predict_data_model_agent_toprank_industryfix_20260618_executable_10d_open_return_v1"

PREDICTION_LABEL_TO_KEY = {
    "executable_3d_open_return": "3d",
    "executable_5d_open_return": "5d",
    "executable_10d_open_return": "10d",
}
ALLOWED_APPROVALS = {"approved_for_l5", "approved_for_l4_only"}

COMBO_FORMULAS = {
    "combo_rank_min_consensus": "MIN(rank_3d, rank_5d, rank_10d)",
    "combo_rank_max_any": "MAX(rank_3d, rank_5d, rank_10d)",
    "combo_rank_eq_3d5d10d": "(rank_3d + rank_5d + rank_10d) / 3.0",
    "combo_rank_10d60_5d30_3d10": "(rank_10d * 0.60) + (rank_5d * 0.30) + (rank_3d * 0.10)",
    "combo_rank_10d70_5d20_3d10": "(rank_10d * 0.70) + (rank_5d * 0.20) + (rank_3d * 0.10)",
    "combo_rank_10d80_5d20": "(rank_10d * 0.80) + (rank_5d * 0.20)",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a research fusion score library from 3d/5d/10d prediction tables.")
    parser.add_argument("--model-db", default=None)
    parser.add_argument("--market-db", default=str(DEFAULT_MARKET_DB))
    parser.add_argument("--table-3d", default=None)
    parser.add_argument("--table-5d", default=None)
    parser.add_argument("--table-10d", default=None)
    parser.add_argument("--prediction-manifest-dir", default=str(MANIFEST_DIR))
    parser.add_argument("--min-trade-date", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-db-name", default="fusion_combos.db")
    return parser.parse_args(argv)


def _manifest_sort_key(manifest: dict) -> tuple[str, str, str]:
    return (
        str(manifest.get("max_trade_date") or ""),
        str(manifest.get("generated_at") or ""),
        str(manifest.get("table") or ""),
    )


def resolve_prediction_inputs(
    manifest_dir: Path = MANIFEST_DIR,
    min_trade_date: str | None = None,
) -> dict[str, object]:
    best_by_label: dict[str, tuple[tuple[str, str, str], dict[str, object]]] = {}
    for manifest_path in sorted(Path(manifest_dir).glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        label = str(manifest.get("label") or "")
        approval_status = str(manifest.get("approval_status") or "")
        table = str(manifest.get("table") or "")
        raw_db_path = manifest.get("db_path")
        if label not in PREDICTION_LABEL_TO_KEY:
            continue
        if approval_status not in ALLOWED_APPROVALS:
            continue
        if str(manifest.get("source_type") or "sqlite_table") != "sqlite_table":
            continue
        if not raw_db_path or not table:
            continue
        resolved = {
            "label": label,
            "db_path": (manifest_path.parent / str(raw_db_path)).resolve(),
            "table": table,
            "approval_status": approval_status,
            "max_trade_date": str(manifest.get("max_trade_date") or ""),
            "manifest_path": manifest_path.resolve(),
        }
        sort_key = _manifest_sort_key(manifest)
        current = best_by_label.get(label)
        if current is None or sort_key > current[0]:
            best_by_label[label] = (sort_key, resolved)

    missing = [label for label in PREDICTION_LABEL_TO_KEY if label not in best_by_label]
    if missing:
        raise RuntimeError(f"missing prediction manifest for {', '.join(sorted(missing))}")

    rows = {label: best_by_label[label][1] for label in PREDICTION_LABEL_TO_KEY}
    if min_trade_date:
        stale = [row for row in rows.values() if str(row.get("max_trade_date") or "") < str(min_trade_date)]
        if stale:
            stale_desc = ", ".join(
                f"{row['label']}={row.get('max_trade_date') or 'missing'}" for row in stale
            )
            raise RuntimeError(f"stale prediction manifest for min_trade_date={min_trade_date}: {stale_desc}")

    db_paths = {Path(str(row["db_path"])) for row in rows.values()}
    if len(db_paths) != 1:
        raise RuntimeError(f"prediction manifests resolve to multiple db paths: {sorted(str(path) for path in db_paths)}")
    only_db = next(iter(db_paths))
    return {
        "model_db": only_db,
        "table_3d": str(rows["executable_3d_open_return"]["table"]),
        "table_5d": str(rows["executable_5d_open_return"]["table"]),
        "table_10d": str(rows["executable_10d_open_return"]["table"]),
        "manifest_3d": Path(rows["executable_3d_open_return"]["manifest_path"]),
        "manifest_5d": Path(rows["executable_5d_open_return"]["manifest_path"]),
        "manifest_10d": Path(rows["executable_10d_open_return"]["manifest_path"]),
        "approval_status_3d": str(rows["executable_3d_open_return"]["approval_status"]),
        "approval_status_5d": str(rows["executable_5d_open_return"]["approval_status"]),
        "approval_status_10d": str(rows["executable_10d_open_return"]["approval_status"]),
        "max_trade_date_3d": str(rows["executable_3d_open_return"]["max_trade_date"]),
        "max_trade_date_5d": str(rows["executable_5d_open_return"]["max_trade_date"]),
        "max_trade_date_10d": str(rows["executable_10d_open_return"]["max_trade_date"]),
    }


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _rank_pct_sql(column: str) -> str:
    return f"""(
        CAST(RANK() OVER (PARTITION BY trade_date ORDER BY {column} ASC) AS REAL)
        + (CAST(COUNT(*) OVER (PARTITION BY trade_date, {column}) AS REAL) - 1.0) / 2.0
    ) / CAST(COUNT(*) OVER (PARTITION BY trade_date) AS REAL)"""


def _table_stats(conn: sqlite3.Connection, table: str) -> dict[str, object]:
    row = conn.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT trade_date), COUNT(DISTINCT stock_code), MIN(trade_date), MAX(trade_date) FROM {_quote_ident(table)}"
    ).fetchone()
    return {
        "table": table,
        "row_count": int(row[0] or 0),
        "trade_days": int(row[1] or 0),
        "stock_count": int(row[2] or 0),
        "min_trade_date": row[3],
        "max_trade_date": row[4],
    }


def build_fusion_library(args) -> dict[str, object]:
    resolved_inputs = None
    if not all([args.table_3d, args.table_5d, args.table_10d]):
        resolved_inputs = resolve_prediction_inputs(
            manifest_dir=Path(args.prediction_manifest_dir),
            min_trade_date=(None if args.min_trade_date in (None, "") else str(args.min_trade_date)),
        )

    model_db = Path(
        args.model_db
        or (resolved_inputs["model_db"] if resolved_inputs is not None else DEFAULT_MODEL_DB)
    )
    table_3d_name = args.table_3d or str(resolved_inputs["table_3d"])
    table_5d_name = args.table_5d or str(resolved_inputs["table_5d"])
    table_10d_name = args.table_10d or str(resolved_inputs["table_10d"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_db = output_dir / args.output_db_name
    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(output_db)
    try:
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(model_db))} AS model")
        conn.execute(f"ATTACH DATABASE {_quote_literal(str(Path(args.market_db)))} AS market")

        table_3d = _quote_ident(table_3d_name)
        table_5d = _quote_ident(table_5d_name)
        table_10d = _quote_ident(table_10d_name)

        conn.executescript(
            f"""
            DROP TABLE IF EXISTS fusion_rank_base;

            CREATE TABLE fusion_rank_base AS
            WITH joined AS (
                SELECT
                    d10.trade_date AS trade_date,
                    d10.stock_code AS stock_code,
                    d10.pred_prob AS pred_10d,
                    d10."10d_yield_rate" AS "10d_yield_rate",
                    d10.industry AS industry,
                    d10.atr_qfq AS atr_qfq,
                    d10.close_rate AS close_rate,
                    d10.turnover_rate AS turnover_rate,
                    d10.turnover_rate_f AS turnover_rate_f,
                    d10.circ_mv AS circ_mv,
                    d10.total_mv AS total_mv,
                    d10.volume_ratio AS volume_ratio,
                    d10.executable_10d_open_return AS executable_10d_open_return,
                    m.name AS name,
                    d10.close AS close,
                    d10.amount AS amount,
                    d10.pre_close AS pre_close,
                    m.open AS open,
                    m.limit_times AS limit_times,
                    m.st_type AS st_type,
                    d3.pred_prob AS pred_3d,
                    d5.pred_prob AS pred_5d
                FROM model.{table_10d} d10
                INNER JOIN model.{table_3d} d3
                    ON d10.trade_date = d3.trade_date
                   AND d10.stock_code = d3.stock_code
                INNER JOIN model.{table_5d} d5
                    ON d10.trade_date = d5.trade_date
                   AND d10.stock_code = d5.stock_code
                LEFT JOIN market.STOCK_DAILY_DATA m
                    ON d10.trade_date = m.trade_date
                   AND d10.stock_code = m.stock_code
                WHERE d10.pred_prob IS NOT NULL
                  AND d3.pred_prob IS NOT NULL
                  AND d5.pred_prob IS NOT NULL
            ),
            ranked AS (
                SELECT
                    joined.*,
                    {_rank_pct_sql("pred_3d")} AS rank_3d,
                    {_rank_pct_sql("pred_5d")} AS rank_5d,
                    {_rank_pct_sql("pred_10d")} AS rank_10d
                FROM joined
            )
            SELECT * FROM ranked;

            CREATE INDEX idx_fusion_rank_base_trade_stock ON fusion_rank_base(trade_date, stock_code);
            CREATE INDEX idx_fusion_rank_base_trade_pred10d ON fusion_rank_base(trade_date, pred_10d);
            """
        )

        for table_name, formula in COMBO_FORMULAS.items():
            conn.executescript(
                f"""
                DROP TABLE IF EXISTS {_quote_ident(table_name)};
                CREATE TABLE {_quote_ident(table_name)} AS
                SELECT
                    trade_date,
                    stock_code,
                    {formula} AS pred_prob,
                    "10d_yield_rate",
                    industry,
                    atr_qfq,
                    close_rate,
                    turnover_rate,
                    turnover_rate_f,
                    circ_mv,
                    total_mv,
                    volume_ratio,
                    executable_10d_open_return,
                    name,
                    close,
                    amount,
                    pre_close,
                    open,
                    limit_times,
                    st_type,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    rank_3d,
                    rank_5d,
                    rank_10d
                FROM fusion_rank_base;
                CREATE INDEX idx_{table_name}_trade_stock ON {_quote_ident(table_name)}(trade_date, stock_code);
                CREATE INDEX idx_{table_name}_trade_pred ON {_quote_ident(table_name)}(trade_date, pred_prob DESC);
                """
            )

        manifest_rows = [
            {
                "combo_name": table_name,
                "formula": formula,
                **_table_stats(conn, table_name),
            }
            for table_name, formula in COMBO_FORMULAS.items()
        ]
        base_stats = _table_stats(conn, "fusion_rank_base")
        conn.execute("DROP TABLE IF EXISTS fusion_combo_manifest")
        conn.execute(
            """
            CREATE TABLE fusion_combo_manifest (
                combo_name TEXT,
                formula TEXT,
                table_name TEXT,
                row_count INTEGER,
                trade_days INTEGER,
                stock_count INTEGER,
                min_trade_date TEXT,
                max_trade_date TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO fusion_combo_manifest (
                combo_name, formula, table_name, row_count, trade_days, stock_count, min_trade_date, max_trade_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["combo_name"],
                    row["formula"],
                    row["table"],
                    row["row_count"],
                    row["trade_days"],
                    row["stock_count"],
                    row["min_trade_date"],
                    row["max_trade_date"],
                )
                for row in manifest_rows
            ],
        )
        conn.commit()

        manifest = {
            "generated_from": {
                "model_db": str(model_db),
                "market_db": str(Path(args.market_db)),
                "table_3d": table_3d_name,
                "table_5d": table_5d_name,
                "table_10d": table_10d_name,
            },
            "output_db": str(output_db),
            "fusion_rank_base": base_stats,
            "combos": manifest_rows,
        }
        if resolved_inputs is not None:
            manifest["generated_from"].update(
                {
                    "manifest_3d": str(resolved_inputs["manifest_3d"]),
                    "manifest_5d": str(resolved_inputs["manifest_5d"]),
                    "manifest_10d": str(resolved_inputs["manifest_10d"]),
                    "approval_status_3d": resolved_inputs["approval_status_3d"],
                    "approval_status_5d": resolved_inputs["approval_status_5d"],
                    "approval_status_10d": resolved_inputs["approval_status_10d"],
                    "max_trade_date_3d": resolved_inputs["max_trade_date_3d"],
                    "max_trade_date_5d": resolved_inputs["max_trade_date_5d"],
                    "max_trade_date_10d": resolved_inputs["max_trade_date_10d"],
                    "source_mode": "prediction_manifests",
                }
            )
        (output_dir / "fusion_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest
    finally:
        conn.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = build_fusion_library(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
L2_DB = (
    ROOT
    / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"
)
BASE_ACTIONS = (
    ROOT
    / "quant/main/strategy_library/production/"
    "prod_v260_10d_regime_warmup_all4key_v20260724/"
    "signals/full_history_actions_current.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def board_for(stock_code: str) -> str:
    code = stock_code.split(".", 1)[0]
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("300", "301")):
        return "chinext"
    return "main"


def is_valid_stock_code(stock_code: str) -> bool:
    code, _, exchange = stock_code.partition(".")
    if len(code) != 6 or not code.isdigit():
        return False
    if exchange == "SH":
        return code.startswith(("600", "601", "603", "605", "688", "689"))
    if exchange == "SZ":
        return code.startswith(
            ("000", "001", "002", "003", "300", "301")
        )
    return False


def load_board_momentum(lookback: int) -> dict[tuple[str, str], float]:
    with duckdb.connect(str(L2_DB), read_only=True) as con:
        frame = con.execute(
            """
            SELECT
                CAST(trade_date AS VARCHAR) AS trade_date,
                CASE
                    WHEN stock_code LIKE '688%' OR stock_code LIKE '689%'
                        THEN 'star'
                    WHEN stock_code LIKE '300%' OR stock_code LIKE '301%'
                        THEN 'chinext'
                    ELSE 'main'
                END AS board,
                AVG(close_qfq / NULLIF(pre_close_qfq, 0) - 1) AS daily_return
            FROM STOCK_DAILY_DATA
            WHERE close_qfq IS NOT NULL
              AND pre_close_qfq IS NOT NULL
              AND pre_close_qfq > 0
              AND stock_code NOT LIKE '%.BJ'
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchdf()
    pivot = frame.pivot(
        index="trade_date", columns="board", values="daily_return"
    ).sort_index()
    pivot["growth"] = pivot[["star", "chinext"]].mean(axis=1)
    cumulative = (1.0 + pivot).rolling(lookback, min_periods=lookback).apply(
        np.prod, raw=True
    ) - 1.0
    return {
        (str(date), str(board)): float(value)
        for date, row in cumulative.iterrows()
        for board, value in row.items()
        if np.isfinite(value)
    }


def regime(
    value: float | None, weak_below: float, strong_at_or_above: float
) -> str:
    if value is None or not np.isfinite(value):
        return "middle"
    if value < weak_below:
        return "weak"
    if value >= strong_at_or_above:
        return "strong"
    return "middle"


def generate_case(
    base: pd.DataFrame,
    momentum: dict[tuple[str, str], float],
    case: dict,
) -> tuple[pd.DataFrame, dict]:
    positions: dict[str, tuple[str, float]] = {}
    output_rows: list[dict] = []
    scaled = 0
    skipped = 0
    orphan_sells_dropped = 0
    invalid_code_buys_dropped = 0
    below_round_lot_buys_dropped = 0
    regime_counts = {"weak": 0, "middle": 0, "strong": 0}

    for raw in base.to_dict(orient="records"):
        action = str(raw["action"]).upper()
        stock = str(raw["stock_code"])
        if action == "SELL":
            if stock in positions:
                positions.pop(stock)
                output_rows.append(raw)
            else:
                orphan_sells_dropped += 1
            continue

        board = board_for(stock)
        if not is_valid_stock_code(stock):
            invalid_code_buys_dropped += 1
            skipped += 1
            continue
        signal_date = str(raw["signal_date"])
        board_value = momentum.get((signal_date, board))
        board_regime = regime(
            board_value,
            float(case["weak_below"]),
            float(case["strong_at_or_above"]),
        )
        regime_counts[board_regime] += 1
        growth_value = momentum.get((signal_date, "growth"))
        growth_regime = regime(
            growth_value,
            float(case["weak_below"]),
            float(case["strong_at_or_above"]),
        )

        single_cap = float(case["single_caps"][board_regime])
        board_budget = sum(
            target
            for held_board, target in positions.values()
            if held_board == board
        )
        growth_budget = sum(
            target
            for held_board, target in positions.values()
            if held_board in {"star", "chinext"}
        )
        limits = [float(raw["target_pct"]), single_cap]
        if board == "star":
            limits.append(
                max(
                    float(case["star_caps"][board_regime]) - board_budget,
                    0.0,
                )
            )
        if board in {"star", "chinext"}:
            limits.append(
                max(
                    float(case["growth_caps"][growth_regime])
                    - growth_budget,
                    0.0,
                )
            )
        target = min(limits)
        open_price = float(raw["execution_open_raw"])
        minimum_lot = 200 if board == "star" else 100
        minimum_executable_target = (
            minimum_lot
            * open_price
            * (1.0 + float(case["buy_cost_buffer"]))
            / float(case["initial_cash"])
        )
        if target < minimum_executable_target:
            below_round_lot_buys_dropped += 1
            skipped += 1
            continue
        if target < float(case["min_target_pct"]):
            skipped += 1
            continue
        if target < float(raw["target_pct"]) - 1e-12:
            scaled += 1
        row = dict(raw)
        row["target_pct"] = target
        output_rows.append(row)
        positions[stock] = (board, target)

    output = pd.DataFrame(output_rows, columns=list(base.columns))
    return output, {
        "output_rows": int(len(output)),
        "buy_rows": int((output["action"] == "BUY").sum()),
        "sell_rows": int((output["action"] == "SELL").sum()),
        "scaled_buy_rows": int(scaled),
        "skipped_buy_rows": int(skipped),
        "orphan_sell_rows_dropped": int(orphan_sells_dropped),
        "invalid_code_buys_dropped": int(invalid_code_buys_dropped),
        "below_round_lot_buys_dropped": int(
            below_round_lot_buys_dropped
        ),
        "ending_open_positions": int(len(positions)),
        "regime_counts": regime_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hash = str(protocol["base_actions_sha256"]).upper()
    if sha256(BASE_ACTIONS) != expected_hash:
        raise RuntimeError("冻结的生产动作哈希发生漂移")

    base = pd.read_csv(
        BASE_ACTIONS,
        dtype={"signal_date": str, "buy_date": str, "stock_code": str},
    )
    output_dir = protocol_path.parent
    actions_dir = output_dir / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    momentum_cache: dict[int, dict[tuple[str, str], float]] = {}

    for case in protocol["cases"]:
        lookback = int(case["lookback"])
        if lookback not in momentum_cache:
            momentum_cache[lookback] = load_board_momentum(lookback)
        output, audit = generate_case(base, momentum_cache[lookback], case)
        path = actions_dir / f"{case['case_id']}.csv"
        output.to_csv(path, index=False, encoding="utf-8-sig")
        summaries.append(
            {
                "case_id": case["case_id"],
                "definition": case,
                "action_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "action_sha256": sha256(path),
                **audit,
            }
        )

    manifest = {
        "status": "research_only_dynamic_style_risk_actions_generated",
        "protocol_sha256": sha256(protocol_path),
        "base_actions_path": str(BASE_ACTIONS.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "base_actions_sha256": expected_hash,
        "uses_realized_return_for_selection": False,
        "uses_future_data_in_daily_decision": False,
        "momentum_contract": (
            "仅使用信号日及以前的L2前复权close_qfq/pre_close_qfq，"
            "计算板块等权滚动复合收益"
        ),
        "production_change_allowed": False,
        "cases": summaries,
    }
    manifest_path = output_dir / "action_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "case_count": len(summaries),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from model_experiment_grid import write_rows
from selection_module import SelectionConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune backfilled parquet prediction folds with official Juejin backtests."
    )
    parser.add_argument("--data-dir", default=r"D:\work\quant\quant_mcp\quant\data_file")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--strategy-dir", default=r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="20240604")
    parser.add_argument("--end", default="20260615")
    parser.add_argument("--backtest-start", default="2024-06-05 09:00:00")
    parser.add_argument("--backtest-end", default="2026-06-24 15:30:00")
    parser.add_argument("--initial-cash", type=float, default=600000)
    parser.add_argument("--slippage-ratio", type=float, default=0.0015)
    parser.add_argument("--include-bj", action="store_true", help="Include Beijing exchange stocks. Default excludes BJ.")
    parser.add_argument("--top-k-list", default="1,2,3")
    parser.add_argument("--holding-days-list", default="3,5")
    parser.add_argument("--max-positions-list", default="1,2,3")
    parser.add_argument("--min-amount-list", default="none,500000,800000,1200000")
    parser.add_argument("--min-turnover-list", default="none,1.5,2.0")
    parser.add_argument("--max-total-mv-list", default="none,1000000,2000000")
    parser.add_argument("--min-pred-list", default="none")
    parser.add_argument("--min-pred-quantile-list", default="none")
    parser.add_argument("--limit", type=int, help="Run only the first N grid rows for smoke tests.")
    return parser.parse_args(argv)


def _parse_float_list(text: str) -> list[float | None]:
    values: list[float | None] = []
    for part in str(text).split(","):
        part = part.strip().lower()
        if part in {"", "none", "null"}:
            values.append(None)
        else:
            values.append(float(part))
    return values


def _parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _safe(value: Any) -> str:
    return str(value).replace(".", "p").replace("None", "none").replace(" ", "")


def _extract_indicator(text: str):
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(text.splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            try:
                return eval(payload, {"__builtins__": {}}, {"datetime": dt})
            except Exception:
                return None
    return None


def _load_predictions(prediction_dir: Path, start: str, end: str, include_bj: bool = False) -> pd.DataFrame:
    files = sorted(prediction_dir.glob("fold*.parquet"))
    if not files:
        raise SystemExit(f"No fold parquet files under {prediction_dir}")
    frames = [pd.read_parquet(path) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame = frame[(frame["trade_date"] >= str(start)) & (frame["trade_date"] <= str(end))].copy()
    required = {"trade_date", "stock_code", "pred_prob"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Prediction data missing columns: {missing}")
    frame = frame.dropna(subset=["trade_date", "stock_code", "pred_prob"])
    if not include_bj:
        frame = frame[~frame["stock_code"].astype(str).str.endswith(".BJ")].copy()
    frame = frame.sort_values(["trade_date", "pred_prob"], ascending=[True, False])
    before = len(frame)
    frame = frame.drop_duplicates(["trade_date", "stock_code"], keep="first")
    if len(frame) != before:
        print(f"WARN duplicate prediction keys removed: {before - len(frame)}", flush=True)
    return frame


def _top_rows(frame: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    top = frame.groupby("trade_date", group_keys=False).head(int(top_n))
    return top.to_dict("records")


def _signal_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def _run_juejin(
    strategy_dir: Path,
    signal_file: Path,
    log_file: Path,
    max_positions: int,
    holding_days: int,
    args,
) -> tuple[int, dict | None]:
    env = os.environ.copy()
    env.update(
        {
            "GM_SIGNAL_FILE": str(signal_file),
            "GM_MAX_POSITIONS": str(max_positions),
            "GM_HOLDING_DAYS": str(holding_days),
            "GM_OPEN_DAILY_SCORE_EXIT": "0",
            "GM_SYNC_POSITIONS": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_VERBOSE_TRADES": "0",
            "GM_SELL_SCHEDULE_TIME": "09:30:00",
            "GM_BUY_SCHEDULE_TIME": "09:31:00",
            "GM_BACKTEST_START": args.backtest_start,
            "GM_BACKTEST_END": args.backtest_end,
            "GM_BACKTEST_ADJUST": "qfq",
            "GM_BACKTEST_INITIAL_CASH": str(args.initial_cash),
            "GM_BACKTEST_SLIPPAGE_RATIO": str(args.slippage_ratio),
        }
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(strategy_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    return proc.returncode, _extract_indicator(text)


def _grid(args) -> list[dict[str, Any]]:
    rows = []
    for top_k in _parse_int_list(args.top_k_list):
        for holding_days in _parse_int_list(args.holding_days_list):
            for max_positions in _parse_int_list(args.max_positions_list):
                if max_positions < top_k:
                    continue
                for min_amount in _parse_float_list(args.min_amount_list):
                    for min_turnover_rate in _parse_float_list(args.min_turnover_list):
                        for max_total_mv in _parse_float_list(args.max_total_mv_list):
                            for min_pred_prob in _parse_float_list(args.min_pred_list):
                                for min_pred_quantile in _parse_float_list(args.min_pred_quantile_list):
                                    rows.append(
                                        {
                                            "top_k": top_k,
                                            "holding_days": holding_days,
                                            "max_positions": max_positions,
                                            "min_amount": min_amount,
                                            "min_turnover_rate": min_turnover_rate,
                                            "max_total_mv": max_total_mv,
                                            "min_pred_prob": min_pred_prob,
                                            "min_pred_quantile": min_pred_quantile,
                                        }
                                    )
    if args.limit:
        return rows[: int(args.limit)]
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    db_path = data_dir / "odb.db"
    prediction_dir = Path(args.prediction_dir)
    strategy_dir = Path(args.strategy_dir)
    output_dir = Path(args.output_dir)
    signal_dir = output_dir / "signals"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_predictions(prediction_dir, args.start, args.end, include_bj=args.include_bj)
    quality = {
        "rows": int(len(frame)),
        "dates": int(frame["trade_date"].nunique()),
        "stocks": int(frame["stock_code"].nunique()),
        "min_trade_date": str(frame["trade_date"].min()),
        "max_trade_date": str(frame["trade_date"].max()),
        "null_pred_prob": int(frame["pred_prob"].isna().sum()),
    }
    (output_dir / "prediction_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(quality, ensure_ascii=False), flush=True)

    market_rows = load_market_rows_by_trade_date(db_path, args.start, args.end)
    rows_cache = _top_rows(frame, top_n=300)
    results: list[dict[str, Any]] = []
    all_rows_path = output_dir / "juejin_grid_all.csv"
    sorted_path = output_dir / "juejin_grid_sorted.csv"
    dd45_path = output_dir / "juejin_grid_dd45_sorted.csv"

    for idx, params in enumerate(_grid(args), start=1):
        suffix = (
            f"top{params['top_k']}_h{params['holding_days']}_maxpos{params['max_positions']}"
            f"_amt{_safe(params['min_amount'])}_turn{_safe(params['min_turnover_rate'])}_mv{_safe(params['max_total_mv'])}"
            f"_pred{_safe(params['min_pred_prob'])}_q{_safe(params['min_pred_quantile'])}"
        )
        signal_file = signal_dir / f"{suffix}.csv"
        log_file = log_dir / f"{suffix}.log"
        print(f"RUN {idx}: {suffix}", flush=True)
        row = {**params, "signal_file": str(signal_file), "log_file": str(log_file)}
        try:
            signals = build_gm_signal_rows(
                rows_cache,
                SelectionConfig(
                    top_k=int(params["top_k"]),
                    min_pred_prob=params["min_pred_prob"],
                    min_pred_quantile=params["min_pred_quantile"],
                    max_atr_ratio=None,
                    min_amount=params["min_amount"],
                    min_turnover_rate=params["min_turnover_rate"],
                    max_total_mv=params["max_total_mv"],
                    max_per_industry=999,
                    exclude_st=True,
                    exclude_current_limit=True,
                ),
                market_rows_by_trade_date=market_rows,
                holding_days=int(params["holding_days"]),
                max_positions=int(params["max_positions"]),
                target_total_pct=0.98,
            )
            write_gm_signals_csv(signals, signal_file)
            returncode, indicator = _run_juejin(
                strategy_dir,
                signal_file,
                log_file,
                int(params["max_positions"]),
                int(params["holding_days"]),
                args,
            )
            row["signal_count"] = _signal_count(signal_file)
            row["returncode"] = returncode
            if indicator:
                row.update(
                    {
                        "pnl_ratio": indicator.get("pnl_ratio"),
                        "pnl_ratio_annual": indicator.get("pnl_ratio_annual"),
                        "sharp_ratio": indicator.get("sharp_ratio"),
                        "max_drawdown": indicator.get("max_drawdown"),
                        "open_count": indicator.get("open_count"),
                        "close_count": indicator.get("close_count"),
                        "win_ratio": indicator.get("win_ratio"),
                        "calmar_ratio": indicator.get("calmar_ratio"),
                    }
                )
            else:
                row["error"] = "missing_indicator"
        except Exception as exc:
            row["returncode"] = -1
            row["error"] = repr(exc)
        results.append(row)
        write_rows(results, all_rows_path)
        result_frame = pd.DataFrame(results)
        if "pnl_ratio_annual" in result_frame:
            result_frame.sort_values("pnl_ratio_annual", ascending=False).to_csv(
                sorted_path, index=False, encoding="utf-8-sig"
            )
            constrained = result_frame[pd.to_numeric(result_frame["max_drawdown"], errors="coerce") <= 0.45]
            constrained.sort_values("pnl_ratio_annual", ascending=False).to_csv(
                dd45_path, index=False, encoding="utf-8-sig"
            )

    result_frame = pd.DataFrame(results)
    if "pnl_ratio_annual" in result_frame:
        print(result_frame.sort_values("pnl_ratio_annual", ascending=False).head(20).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

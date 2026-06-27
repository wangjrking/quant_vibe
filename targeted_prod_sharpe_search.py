from __future__ import annotations

import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

from backtest_module import read_prediction_rows
from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from model_experiment_grid import write_rows
from selection_module import SelectionConfig


DATA_DIR = Path(r"D:\work\quant\quant_mcp\quant\data_file")
REPORT_DIR = DATA_DIR / "reports" / "prod_liq_prime_sharpe_20260617"
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
RUN_JUEJIN = Path(__file__).resolve().parent / "run_juejin_signal_backtest.py"
REPRODUCTION_FILE = Path(__file__).resolve().parent / "strategy_library" / "production" / "prod_liq_prime_one_v20260612" / "reproduction_v1_0.json"
TARGET_ANNUAL = 3.0
TARGET_SHARPE = 2.0
PREDICTION_START = "20240605"
PREDICTION_END = "20260616"
BACKTEST_START = "2024-06-05 09:00:00"
BACKTEST_END = "2026-07-10 15:30:00"
BACKTEST_ADJUST = "none"
BACKTEST_INITIAL_CASH = 600000
BACKTEST_SLIPPAGE_RATIO = 0.0015

PROD_BASELINE = {
    "strategy_id": "prod_liq_prime_one_v20260612",
    "prediction_db": DATA_DIR / "odb.db",
    "prediction_table": "stock_predict_data_executable_5d_open_return_prod_aligned_202206_202606",
    "market_db": DATA_DIR / "STOCK_DAILY_DATA.db",
    "top_k": 1,
    "max_positions": 1,
    "holding_days": 5,
    "score_exit_entry_ratio": 0.95,
    "min_holding_days_before_score_exit": 2,
    "min_amount": 800000.0,
    "min_turnover_rate": 2.0,
    "max_total_mv": 1000000.0,
    "min_pred_quantile": 0.95,
    "max_atr_ratio": None,
    "stop_loss_pct": 0.08,
    "score_continue_entry_ratio": 1.0,
}


def build_param_grid() -> list[dict]:
    rows = []
    for holding_days in [4, 5, 6]:
        for score_exit_entry_ratio in [0.88, 0.90, 0.92, 0.93, 0.95, 0.97, 1.0]:
            for min_holding_days_before_score_exit in [1, 2, 3]:
                rows.append(
                    {
                        "stage": "exit",
                        "top_k": PROD_BASELINE["top_k"],
                        "max_positions": PROD_BASELINE["max_positions"],
                        "holding_days": holding_days,
                        "score_exit_entry_ratio": score_exit_entry_ratio,
                        "min_holding_days_before_score_exit": min_holding_days_before_score_exit,
                        "min_amount": PROD_BASELINE["min_amount"],
                        "min_turnover_rate": PROD_BASELINE["min_turnover_rate"],
                        "max_total_mv": PROD_BASELINE["max_total_mv"],
                        "min_pred_quantile": PROD_BASELINE["min_pred_quantile"],
                        "max_atr_ratio": PROD_BASELINE["max_atr_ratio"],
                        "stop_loss_pct": PROD_BASELINE["stop_loss_pct"],
                        "score_continue_entry_ratio": PROD_BASELINE["score_continue_entry_ratio"],
                    }
                )
    return rows


def pick_qualified_candidates(rows: list[dict]) -> list[dict]:
    qualified = [
        row
        for row in rows
        if float(row.get("pnl_ratio_annual") or 0.0) >= TARGET_ANNUAL
        and float(row.get("sharp_ratio") or 0.0) >= TARGET_SHARPE
    ]
    return sorted(
        qualified,
        key=lambda row: (
            -float(row.get("sharp_ratio") or 0.0),
            -float(row.get("pnl_ratio_annual") or 0.0),
        ),
    )


def expand_followup_grid(base_rows: list[dict], stage: str) -> list[dict]:
    rows = []
    for base in base_rows:
        if stage == "liquidity":
            for min_amount in [800000.0, 1000000.0, 1200000.0]:
                for min_turnover_rate in [2.0, 3.0, 4.0]:
                    for max_total_mv in [800000.0, 1000000.0, 1200000.0]:
                        rows.append(
                            {
                                **base,
                                "stage": "liquidity",
                                "min_amount": min_amount,
                                "min_turnover_rate": min_turnover_rate,
                                "max_total_mv": max_total_mv,
                            }
                        )
        elif stage == "quantile":
            for min_pred_quantile in [0.95, 0.96, 0.97, 0.98]:
                rows.append(
                    {
                        **base,
                        "stage": "quantile",
                        "min_pred_quantile": min_pred_quantile,
                    }
                )
        elif stage == "stability":
            for max_atr_ratio in [None, 0.10, 0.08, 0.06]:
                for stop_loss_pct in [0.08, 0.06, 0.05]:
                    for score_continue_entry_ratio in [0.95, 1.0, 1.05]:
                        rows.append(
                            {
                                **base,
                                "stage": "stability",
                                "max_atr_ratio": max_atr_ratio,
                                "stop_loss_pct": stop_loss_pct,
                                "score_continue_entry_ratio": score_continue_entry_ratio,
                            }
                        )
        else:
            raise ValueError(f"unsupported stage: {stage}")
    return rows


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("None", "none").replace(" ", "")


def resolve_juejin_python(reproduction_file: Path = REPRODUCTION_FILE) -> Path:
    if reproduction_file.exists():
        payload = json.loads(reproduction_file.read_text(encoding="utf-8"))
        python_path = ((payload.get("workspace") or {}).get("python") or "").strip()
        if python_path:
            return Path(python_path)
    return Path(sys.executable)


def normalize_indicator(indicator):
    if isinstance(indicator, dict):
        return indicator
    if indicator in (None, ""):
        return None
    text = str(indicator)
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    try:
        return eval(text, {"__builtins__": {}}, {"datetime": __import__("datetime")})
    except Exception:
        return None


def _result_sort_key(row: dict) -> tuple[float, float]:
    return (
        float(row.get("sharp_ratio") or 0.0),
        float(row.get("pnl_ratio_annual") or 0.0),
    )


def select_seed_rows(results: list[dict], limit: int = 3) -> list[dict]:
    qualified = pick_qualified_candidates(results)
    if qualified:
        source = qualified
    else:
        source = sorted(results, key=_result_sort_key, reverse=True)
    picked = []
    seen = set()
    for row in source:
        key = (
            row.get("holding_days"),
            row.get("score_exit_entry_ratio"),
            row.get("min_holding_days_before_score_exit"),
            row.get("min_amount"),
            row.get("max_total_mv"),
            row.get("min_pred_quantile"),
            row.get("max_atr_ratio"),
            row.get("stop_loss_pct"),
            row.get("score_continue_entry_ratio"),
        )
        if key in seen:
            continue
        seen.add(key)
        picked.append(
            {
                "top_k": 1,
                "max_positions": 1,
                "holding_days": row.get("holding_days"),
                "score_exit_entry_ratio": row.get("score_exit_entry_ratio"),
                "min_holding_days_before_score_exit": row.get("min_holding_days_before_score_exit"),
                "min_amount": row.get("min_amount"),
                "min_turnover_rate": row.get("min_turnover_rate"),
                "max_total_mv": row.get("max_total_mv"),
                "min_pred_quantile": row.get("min_pred_quantile"),
                "max_atr_ratio": row.get("max_atr_ratio"),
                "stop_loss_pct": row.get("stop_loss_pct", PROD_BASELINE["stop_loss_pct"]),
                "score_continue_entry_ratio": row.get(
                    "score_continue_entry_ratio",
                    PROD_BASELINE["score_continue_entry_ratio"],
                ),
            }
        )
        if len(picked) >= limit:
            break
    return picked


def _extract_indicator(log_text: str):
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_text.splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            return None
    return None


def _signal_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def _param_slug(params: dict) -> str:
    slug = (
        f"h{params['holding_days']}"
        f"_exit{_safe(params['score_exit_entry_ratio'])}"
        f"_mh{params['min_holding_days_before_score_exit']}"
        f"_amt{_safe(params['min_amount'])}"
        f"_turn{_safe(params['min_turnover_rate'])}"
        f"_mv{_safe(params['max_total_mv'])}"
        f"_q{_safe(params['min_pred_quantile'])}"
    )
    if "max_atr_ratio" in params:
        slug += f"_atr{_safe(params.get('max_atr_ratio'))}"
    if "stop_loss_pct" in params:
        slug += f"_sl{_safe(params.get('stop_loss_pct'))}"
    if "score_continue_entry_ratio" in params:
        slug += f"_cont{_safe(params.get('score_continue_entry_ratio'))}"
    return slug


def _build_result_row(params: dict, signal_file: Path, log_file: Path, indicator: dict | None, returncode: int) -> dict:
    indicator = normalize_indicator(indicator)
    row = {
        **params,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "signal_count": _signal_count(signal_file) if signal_file.exists() else 0,
        "returncode": returncode,
    }
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
    return row


def _existing_result(params: dict, signal_file: Path, log_file: Path) -> dict | None:
    if not log_file.exists() or not signal_file.exists():
        return None
    indicator = _extract_indicator(log_file.read_text(encoding="utf-8", errors="ignore"))
    indicator = normalize_indicator(indicator)
    if not indicator:
        return None
    return _build_result_row(params, signal_file, log_file, indicator, 0)


def _build_signals(params: dict, prediction_rows: list[dict], market_rows_by_trade_date: dict[str, dict[str, dict]]) -> list[dict]:
    return build_gm_signal_rows(
        prediction_rows,
        SelectionConfig(
            top_k=int(params["top_k"]),
            min_pred_prob=None,
            min_pred_quantile=float(params["min_pred_quantile"]),
            max_atr_ratio=params.get("max_atr_ratio"),
            min_amount=float(params["min_amount"]),
            min_turnover_rate=float(params["min_turnover_rate"]),
            max_total_mv=float(params["max_total_mv"]),
        ),
        market_rows_by_trade_date=market_rows_by_trade_date,
        holding_days=int(params["holding_days"]),
        max_positions=int(params["max_positions"]),
        weight_mode="equal",
        target_total_pct=0.98,
    )


def build_juejin_command(signal_file: Path, log_file: Path, params: dict) -> list[str]:
    command = [
        str(resolve_juejin_python()),
        str(RUN_JUEJIN),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(params["max_positions"]),
        "--holding-days",
        str(params["holding_days"]),
        "--score-db",
        str(PROD_BASELINE["prediction_db"]),
        "--score-table",
        str(PROD_BASELINE["prediction_table"]),
        "--score-exit-entry-ratio",
        str(params["score_exit_entry_ratio"]),
        "--min-holding-days-before-score-exit",
        str(params["min_holding_days_before_score_exit"]),
        "--score-continue-entry-ratio",
        str(params.get("score_continue_entry_ratio", PROD_BASELINE["score_continue_entry_ratio"])),
        "--max-holding-days",
        str(params["holding_days"]),
        "--backtest-start",
        BACKTEST_START,
        "--backtest-end",
        BACKTEST_END,
        "--backtest-adjust",
        BACKTEST_ADJUST,
        "--backtest-initial-cash",
        str(BACKTEST_INITIAL_CASH),
        "--backtest-slippage-ratio",
        str(BACKTEST_SLIPPAGE_RATIO),
    ]
    stop_loss_pct = params.get("stop_loss_pct")
    if stop_loss_pct is not None:
        command.extend(["--stop-loss-pct", str(stop_loss_pct)])
    return command


def _run_juejin(signal_file: Path, log_file: Path, params: dict) -> tuple[int, dict | None]:
    command = build_juejin_command(signal_file, log_file, params)
    proc = subprocess.run(command, cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "run_juejin_signal_backtest failed")
    payload = json.loads(proc.stdout)
    return proc.returncode, payload.get("indicator")


def _run_stage(
    stage_name: str,
    params_rows: list[dict],
    prediction_rows: list[dict],
    market_rows_by_trade_date: dict[str, dict[str, dict]],
    all_results: list[dict],
) -> list[dict]:
    stage_dir = REPORT_DIR / stage_name
    signal_dir = stage_dir / "signals"
    log_dir = stage_dir / "logs"
    signal_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_results = []
    for index, params in enumerate(params_rows, start=1):
        params = dict(params)
        params["stage"] = stage_name
        slug = _param_slug(params)
        signal_file = signal_dir / f"{slug}.csv"
        log_file = log_dir / f"{slug}.log"
        existing = _existing_result(params, signal_file, log_file)
        if existing:
            result = existing
        else:
            signals = _build_signals(params, prediction_rows, market_rows_by_trade_date)
            write_gm_signals_csv(signals, signal_file)
            returncode, indicator = _run_juejin(signal_file, log_file, params)
            result = _build_result_row(params, signal_file, log_file, indicator, returncode)
        stage_results.append(result)
        all_results.append(result)
        write_rows(stage_results, stage_dir / "results.csv")
        write_rows(sorted(stage_results, key=_result_sort_key, reverse=True), stage_dir / "results_sorted.csv")
        write_rows(all_results, REPORT_DIR / "all_results.csv")
        write_rows(sorted(all_results, key=_result_sort_key, reverse=True), REPORT_DIR / "all_results_sorted.csv")
        print(f"[{stage_name}] {index}/{len(params_rows)} {slug} sharpe={result.get('sharp_ratio')} annual={result.get('pnl_ratio_annual')}")
    return stage_results


def _write_summary(all_results: list[dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    qualified = pick_qualified_candidates(all_results)
    write_rows(qualified, REPORT_DIR / "qualified_results.csv")
    best_overall = max(all_results, key=_result_sort_key) if all_results else None
    lines = [
        "# Prod Liq Prime Sharpe 20260617",
        "",
        "## Baseline",
        "",
        f"- strategy_id: `{PROD_BASELINE['strategy_id']}`",
        f"- baseline_annual: `23.872377681987334`",
        f"- baseline_sharpe: `1.5466297275045484`",
        "",
        "## Goal",
        "",
        f"- annual >= `{TARGET_ANNUAL}` (300%)",
        f"- sharpe >= `{TARGET_SHARPE}`",
        "",
        "## Best Overall",
        "",
    ]
    if best_overall:
        lines.extend(
            [
                f"- stage: `{best_overall.get('stage')}`",
                f"- annual: `{best_overall.get('pnl_ratio_annual')}`",
                f"- sharpe: `{best_overall.get('sharp_ratio')}`",
                f"- drawdown: `{best_overall.get('max_drawdown')}`",
                f"- params: `h={best_overall.get('holding_days')} exit={best_overall.get('score_exit_entry_ratio')} mh={best_overall.get('min_holding_days_before_score_exit')} amt={best_overall.get('min_amount')} turn={best_overall.get('min_turnover_rate')} mv={best_overall.get('max_total_mv')} q={best_overall.get('min_pred_quantile')}`",
                "",
            ]
        )
    lines.extend(["## Qualified Candidates", ""])
    if qualified:
        for row in qualified[:10]:
            lines.append(
                f"- `{row.get('stage')}` annual=`{row.get('pnl_ratio_annual')}` sharpe=`{row.get('sharp_ratio')}` "
                f"params=`h={row.get('holding_days')} exit={row.get('score_exit_entry_ratio')} mh={row.get('min_holding_days_before_score_exit')} "
                f"amt={row.get('min_amount')} turn={row.get('min_turnover_rate')} mv={row.get('max_total_mv')} q={row.get('min_pred_quantile')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Conclusion", ""])
    if qualified:
        lines.append("- 本轮已找到满足年化>=300%、夏普>=2 的掘金研究候选。")
    else:
        lines.append("- 本轮尚未找到同时满足年化>=300%、夏普>=2 的掘金研究候选。")
        if best_overall:
            lines.append("- 下一轮应继续围绕当前最佳阶段参数，进一步收缩流动性门槛或退出节奏。")
    summary_path = REPORT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def run_search() -> list[dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_rows = read_prediction_rows(
        PROD_BASELINE["prediction_db"],
        PROD_BASELINE["prediction_table"],
        PREDICTION_START,
        PREDICTION_END,
    )
    market_rows_by_trade_date = load_market_rows_by_trade_date(
        PROD_BASELINE["market_db"],
        PREDICTION_START,
        PREDICTION_END,
    )
    all_results: list[dict] = []

    exit_results = _run_stage("exit", build_param_grid(), prediction_rows, market_rows_by_trade_date, all_results)
    liquidity_seed = select_seed_rows(exit_results, limit=3)
    liquidity_results = _run_stage(
        "liquidity",
        expand_followup_grid(liquidity_seed, stage="liquidity"),
        prediction_rows,
        market_rows_by_trade_date,
        all_results,
    )
    stability_seed = select_seed_rows(liquidity_results, limit=3)
    stability_results = _run_stage(
        "stability",
        expand_followup_grid(stability_seed, stage="stability"),
        prediction_rows,
        market_rows_by_trade_date,
        all_results,
    )
    quantile_seed = select_seed_rows(stability_results, limit=3)
    _run_stage(
        "quantile",
        expand_followup_grid(quantile_seed, stage="quantile"),
        prediction_rows,
        market_rows_by_trade_date,
        all_results,
    )
    _write_summary(all_results)
    return all_results


def main() -> int:
    run_search()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

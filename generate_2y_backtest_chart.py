from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

from backtest_module import BacktestConfig, read_prediction_rows, run_backtest


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_file"
REPORT_DIR = DATA / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DB = DATA / "odb.db"
TABLE = "stock_predict_data_10d_yield_rate_oos_2y_ic160"
START = "20240604"
END = "20260604"


def configure_fonts():
    for family in ["Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei"]:
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
            font = FontProperties(fname=path)
            plt.rcParams["font.family"] = font.get_name()
            plt.rcParams["font.sans-serif"] = [font.get_name(), "Microsoft YaHei", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
            return font
        except ValueError:
            continue
    return None


def daily_frame(result):
    frame = pd.DataFrame(result.daily_returns)
    if frame.empty:
        return pd.DataFrame(columns=["date", "return", "equity", "drawdown"])
    frame["date"] = pd.to_datetime(frame["trade_date"])
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1
    return frame


def monthly_returns(daily):
    if daily.empty:
        return pd.DataFrame(columns=["month", "return"])
    frame = daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    return frame.groupby("month")["return"].apply(lambda s: (1 + s).prod() - 1).reset_index()


def pct(value):
    return f"{value * 100:.2f}%"


def run_strategy(top_k, min_pred, max_atr):
    rows = read_prediction_rows(DB, TABLE, START, END)
    config = BacktestConfig(
        top_k=top_k,
        min_pred_prob=min_pred,
        max_atr_ratio=max_atr,
        return_col="10d_yield_rate",
        holding_period_days=10,
        start_date=START,
        end_date=END,
    )
    return run_backtest(rows, config)


def make_chart():
    font = configure_fonts()
    font_kwargs = {"fontproperties": font} if font else {}

    strategies = {
        "两年覆盖型 k=3,p=0.01,ATR=0.10": run_strategy(3, 0.01, 0.10),
        "低波动覆盖型 k=3,p=0.01,ATR=0.08": run_strategy(3, 0.01, 0.08),
        "高阈值少交易 k=5,p=0.02,ATR=0.10": run_strategy(5, 0.02, 0.10),
        "单票激进 k=1,p=0.02,ATR=0.10": run_strategy(1, 0.02, 0.10),
    }
    daily = {name: daily_frame(result) for name, result in strategies.items()}
    primary = "两年覆盖型 k=3,p=0.01,ATR=0.10"
    primary_daily = daily[primary]
    primary_metrics = strategies[primary].metrics
    monthly = monthly_returns(primary_daily)
    actual_start = primary_daily["trade_date"].min()
    actual_end = primary_daily["trade_date"].max()

    fig = plt.figure(figsize=(18, 14), dpi=180, facecolor="#f6f8fb")
    gs = fig.add_gridspec(4, 4, hspace=0.48, wspace=0.32)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.text(0.01, 0.78, "两年 OOS 回测图：IC160 因子策略", fontsize=25, weight="bold", **font_kwargs)
    ax.text(
        0.01,
        0.47,
        f"实际预测区间：{actual_start} 至 {actual_end}    样本：485 个交易日 / 238,915 行预测    标签：10日收益",
        fontsize=12,
        color="#3d4a5c",
        **font_kwargs,
    )
    ax.text(
        0.01,
        0.20,
        "可信度口径：因子筛选只使用 2024-05-24 前训练期；下方默认采用覆盖型参数，避免只交易少数日期的高年化幻觉。",
        fontsize=13,
        color="#a33b20",
        **font_kwargs,
    )

    cards = [
        ("年化收益", pct(primary_metrics["annualized_return"])),
        ("累计收益", pct(primary_metrics["cumulative_return"])),
        ("最大回撤", pct(primary_metrics["max_drawdown"])),
        ("夏普", f"{primary_metrics['sharpe']:.2f}"),
        ("Calmar", f"{primary_metrics['calmar']:.2f}"),
        ("胜率", pct(primary_metrics["win_rate"])),
        ("交易日/笔数", f"{primary_metrics['trade_day_count']:.0f}/{primary_metrics['trade_count']:.0f}"),
        ("平均单笔", pct(primary_metrics["avg_trade_return"])),
    ]
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    colors = ["#0b6e69", "#1f7a8c", "#9b2226", "#335c67", "#6a4c93", "#2d6a4f", "#5f6c7b", "#1b998b"]
    for i, (title, value) in enumerate(cards):
        x = 0.01 + i * 0.123
        ax.add_patch(plt.Rectangle((x, 0.12), 0.112, 0.72, color=colors[i], transform=ax.transAxes))
        ax.text(x + 0.012, 0.62, title, color="white", fontsize=10, **font_kwargs)
        ax.text(x + 0.012, 0.32, value, color="white", fontsize=15, weight="bold", **font_kwargs)

    ax = fig.add_subplot(gs[2, :2])
    for name, frame in daily.items():
        if not frame.empty:
            ax.plot(frame["date"], frame["equity"], label=name, linewidth=2)
    ax.axhline(1, color="#555", linewidth=0.8)
    ax.set_title("资金曲线对比", fontsize=14, weight="bold", **font_kwargs)
    ax.grid(alpha=0.25)
    ax.legend(prop=font, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)

    ax = fig.add_subplot(gs[2, 2:])
    ax.plot(primary_daily["date"], primary_daily["drawdown"] * 100, color="#9b2226", linewidth=2)
    ax.fill_between(primary_daily["date"], primary_daily["drawdown"] * 100, 0, color="#9b2226", alpha=0.15)
    ax.set_title("默认覆盖型参数回撤", fontsize=14, weight="bold", **font_kwargs)
    ax.set_ylabel("回撤 %", **font_kwargs)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)

    ax = fig.add_subplot(gs[3, :2])
    ax.bar(monthly["month"], monthly["return"] * 100, color=["#2a9d8f" if v >= 0 else "#d62828" for v in monthly["return"]])
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title("默认覆盖型参数月度收益", fontsize=14, weight="bold", **font_kwargs)
    ax.set_ylabel("%", **font_kwargs)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(gs[3, 2:])
    ax.axis("off")
    rows = []
    for name, result in strategies.items():
        m = result.metrics
        rows.append([name, pct(m["annualized_return"]), pct(m["cumulative_return"]), pct(m["max_drawdown"]), f"{m['trade_day_count']:.0f}", f"{m['trade_count']:.0f}"])
    table = ax.table(cellText=rows, colLabels=["策略", "年化", "累计", "回撤", "交易日", "笔数"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.75)
    for (row, _), cell in table.get_celld().items():
        if font:
            cell.get_text().set_fontproperties(font)
        if row == 0:
            cell.set_facecolor("#102033")
            cell.get_text().set_color("white")

    png = REPORT_DIR / f"backtest_2y_oos_ic160_{actual_start}_{actual_end}.png"
    pdf = REPORT_DIR / f"backtest_2y_oos_ic160_{actual_start}_{actual_end}.pdf"
    trades = pd.DataFrame(strategies[primary].trades)
    if not trades.empty:
        trades.to_csv(REPORT_DIR / f"backtest_2y_oos_ic160_trades_{actual_start}_{actual_end}.csv", index=False, encoding="utf-8-sig")
    fig.savefig(png, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    make_chart()

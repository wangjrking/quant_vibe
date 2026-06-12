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
START = "20250604"
END = "20260604"


def configure_fonts():
    for family in ["Microsoft YaHei", "SimHei", "SimSun", "PingFang SC", "WenQuanYi Micro Hei"]:
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
            font = FontProperties(fname=path)
            plt.rcParams["font.family"] = font.get_name()
            plt.rcParams["font.sans-serif"] = [font.get_name(), "Microsoft YaHei", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
            return font, FontProperties(fname=path)
        except ValueError:
            continue
    return None, None


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
    return frame.groupby("month")["return"].apply(lambda values: (1 + values).prod() - 1).reset_index()


def fmt_pct(value):
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def make_chart():
    font, bold = configure_fonts()
    font_kwargs = {"fontproperties": font} if font else {}
    bold_kwargs = {"fontproperties": bold} if bold else {}

    strategies = {
        "IC160高年化\nk=1,p=0.01,ATR=0.08": (
            "stock_predict_data_10d_yield_rate",
            BacktestConfig(
                top_k=1,
                min_pred_prob=0.01,
                max_atr_ratio=0.08,
                return_col="10d_yield_rate",
                holding_period_days=10,
                start_date=START,
                end_date=END,
            ),
        ),
        "全因子备份\nk=1,p=0.03,ATR=0.05": (
            "stock_predict_data_10d_yield_rate_fullfactor_backup",
            BacktestConfig(
                top_k=1,
                min_pred_prob=0.03,
                max_atr_ratio=0.05,
                return_col="10d_yield_rate",
                holding_period_days=10,
                start_date=START,
                end_date=END,
            ),
        ),
        "风险调整标签\nk=2,p=0.60,ATR=0.06": (
            "stock_predict_data_risk_adjusted_10d_yield_rate",
            BacktestConfig(
                top_k=2,
                min_pred_prob=0.6,
                max_atr_ratio=0.06,
                return_col="10d_yield_rate",
                holding_period_days=10,
                start_date=START,
                end_date=END,
            ),
        ),
    }

    results = {}
    daily = {}
    for name, (table, config) in strategies.items():
        rows = read_prediction_rows(DB, table, START, END)
        result = run_backtest(rows, config)
        results[name] = result
        daily[name] = daily_frame(result)

    best_name = "IC160高年化\nk=1,p=0.01,ATR=0.08"
    best_daily = daily[best_name]
    best_monthly = monthly_returns(best_daily)
    best_metrics = results[best_name].metrics
    actual_start = best_daily["trade_date"].min() if not best_daily.empty else START
    actual_end = best_daily["trade_date"].max() if not best_daily.empty else END

    fig = plt.figure(figsize=(18, 13), dpi=180, facecolor="#f6f8fb")
    gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.32)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.01, 0.78, "高年化选股策略回测图", fontsize=26, weight="bold", **bold_kwargs)
    ax_title.text(
        0.01,
        0.47,
        f"实际回测区间：{actual_start} 至 {actual_end}    真实收益口径：10日收益    交易成本：佣金0.03% + 印花税0.05% + 滑点0.10%",
        fontsize=12,
        color="#3d4a5c",
        **font_kwargs,
    )
    ax_title.text(
        0.01,
        0.2,
        "当前采用：10日收益标签 + 近一年IC筛选160因子 + 年化收益/回撤调参。注意：top_k=1 收益目标更高，但个股集中风险更高。",
        fontsize=13,
        color="#a33b20",
        **bold_kwargs,
    )

    cards = [
        ("年化收益", fmt_pct(best_metrics["annualized_return"])),
        ("累计收益", fmt_pct(best_metrics["cumulative_return"])),
        ("最大回撤", fmt_pct(best_metrics["max_drawdown"])),
        ("夏普", f"{best_metrics['sharpe']:.2f}"),
        ("Calmar", f"{best_metrics['calmar']:.2f}"),
        ("胜率", fmt_pct(best_metrics["win_rate"])),
        ("交易日/笔数", f"{best_metrics['trade_day_count']:.0f}/{best_metrics['trade_count']:.0f}"),
        ("平均单笔", fmt_pct(best_metrics["avg_trade_return"])),
    ]
    ax_cards = fig.add_subplot(gs[1, :])
    ax_cards.axis("off")
    colors = ["#0b6e69", "#1f7a8c", "#9b2226", "#335c67", "#6a4c93", "#2d6a4f", "#5f6c7b", "#1b998b"]
    for index, (title, value) in enumerate(cards):
        x = 0.01 + index * 0.123
        ax_cards.add_patch(plt.Rectangle((x, 0.12), 0.112, 0.72, color=colors[index], transform=ax_cards.transAxes))
        ax_cards.text(x + 0.012, 0.62, title, color="white", fontsize=10, **bold_kwargs)
        ax_cards.text(x + 0.012, 0.32, value, color="white", fontsize=15, weight="bold", **bold_kwargs)

    ax_equity = fig.add_subplot(gs[2, :2])
    line_colors = ["#0077b6", "#f77f00", "#6a4c93"]
    for color, (name, frame) in zip(line_colors, daily.items()):
        if not frame.empty:
            ax_equity.plot(frame["date"], frame["equity"], label=name, linewidth=2.4, color=color)
    ax_equity.axhline(1, color="#555", linewidth=0.8)
    ax_equity.set_title("资金曲线对比", fontsize=14, weight="bold", **bold_kwargs)
    ax_equity.grid(alpha=0.25)
    ax_equity.legend(prop=font, fontsize=8)
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_equity.tick_params(axis="x", rotation=35)

    ax_dd = fig.add_subplot(gs[2, 2:])
    if not best_daily.empty:
        ax_dd.plot(best_daily["date"], best_daily["drawdown"] * 100, color="#9b2226", linewidth=2)
        ax_dd.fill_between(best_daily["date"], best_daily["drawdown"] * 100, 0, color="#9b2226", alpha=0.15)
    ax_dd.set_title("当前采用策略回撤曲线", fontsize=14, weight="bold", **bold_kwargs)
    ax_dd.set_ylabel("回撤 %", **font_kwargs)
    ax_dd.grid(alpha=0.25)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_dd.tick_params(axis="x", rotation=35)

    ax_month = fig.add_subplot(gs[3, :2])
    bar_colors = ["#2a9d8f" if value >= 0 else "#d62828" for value in best_monthly["return"]]
    ax_month.bar(best_monthly["month"], best_monthly["return"] * 100, color=bar_colors)
    ax_month.axhline(0, color="#333", linewidth=0.8)
    ax_month.set_title("当前采用策略月度收益", fontsize=14, weight="bold", **bold_kwargs)
    ax_month.set_ylabel("%", **font_kwargs)
    ax_month.tick_params(axis="x", rotation=45)
    ax_month.grid(axis="y", alpha=0.25)

    ax_table = fig.add_subplot(gs[3, 2:])
    ax_table.axis("off")
    table_rows = []
    for name, result in results.items():
        metrics = result.metrics
        table_rows.append(
            [
                name.replace("\n", " "),
                fmt_pct(metrics["annualized_return"]),
                fmt_pct(metrics["cumulative_return"]),
                fmt_pct(metrics["max_drawdown"]),
                f"{metrics['sharpe']:.2f}",
                f"{metrics['trade_day_count']:.0f}",
            ]
        )
    table = ax_table.table(
        cellText=table_rows,
        colLabels=["策略", "年化", "累计", "回撤", "夏普", "交易日"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.7)
    for (row, _), cell in table.get_celld().items():
        if font:
            cell.get_text().set_fontproperties(font)
        if row == 0:
            cell.set_facecolor("#102033")
            cell.get_text().set_color("white")
            if bold:
                cell.get_text().set_fontproperties(bold)

    png = REPORT_DIR / f"backtest_high_annual_ic160_{actual_start}_{actual_end}.png"
    pdf = REPORT_DIR / f"backtest_high_annual_ic160_{actual_start}_{actual_end}.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    trades = pd.DataFrame(results[best_name].trades)
    if not trades.empty:
        trades.to_csv(REPORT_DIR / f"backtest_high_annual_ic160_trades_{actual_start}_{actual_end}.csv", index=False, encoding="utf-8-sig")

    print(png)
    print(pdf)


if __name__ == "__main__":
    make_chart()

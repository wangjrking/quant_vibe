from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

import alpha191 as official_alpha191


GTJA_OFFICIAL_RAW_COLUMNS = [
    "trade_date",
    "stock_code",
    "industry",
    "act_ent_type",
    "open",
    "close",
    "high",
    "low",
    "pre_close",
    "vol",
    "amount",
    "total_mv",
    "pb",
    "index_2000_open",
    "index_2000_close",
]


def _rolling_last_rank(window_values: np.ndarray) -> float:
    series = pd.Series(window_values)
    if series.empty:
        return np.nan
    return float(series.rank(pct=True).iloc[-1])


def _rolling_weighted_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    weights = np.arange(1, window + 1, dtype=float)
    weight_sum = weights.sum()

    def apply_fn(values: np.ndarray) -> float:
        current_weights = weights[-len(values) :]
        return float(np.dot(values, current_weights) / current_weights.sum())

    return frame.rolling(window, min_periods=1).apply(apply_fn, raw=True)


class OfficialGtjaContext:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        cross_sectional_rank_mode: str = "rank",
        benchmark_prefix: str = "index_2000",
    ) -> None:
        if cross_sectional_rank_mode not in {"rank", "identity"}:
            raise ValueError("cross_sectional_rank_mode must be 'rank' or 'identity'")
        self.rank_mode = cross_sectional_rank_mode
        self.benchmark_prefix = benchmark_prefix

        data = frame.copy()
        data["trade_date"] = data["trade_date"].astype(str)
        data["stock_code"] = data["stock_code"].astype(str)
        data.sort_values(["trade_date", "stock_code"], inplace=True)

        self.trade_dates = sorted(data["trade_date"].unique().tolist())
        self.stock_codes = sorted(data["stock_code"].unique().tolist())
        self.template = pd.DataFrame(index=self.trade_dates, columns=self.stock_codes, dtype=float)
        self.long_keys = data[["trade_date", "stock_code"]].drop_duplicates().reset_index(drop=True)
        self._fields: dict[str, pd.DataFrame] = {}
        self._source = data

        self._fields["OPEN"] = self._pivot_value("open")
        self._fields["CLOSE"] = self._pivot_value("close")
        self._fields["HIGH"] = self._pivot_value("high")
        self._fields["LOW"] = self._pivot_value("low")
        self._fields["VOLUME"] = self._pivot_value("vol")
        self._fields["AMOUNT"] = self._pivot_value("amount")
        self._fields["VWAP"] = self._derive_vwap()
        self._fields["RET"] = self._derive_returns()
        self._fields["SEQUENCE"] = self._broadcast_series(
            pd.Series(np.arange(1, len(self.trade_dates) + 1, dtype=float), index=self.trade_dates)
        )
        self._fields["BANCHMARKINDEXOPEN"] = self._broadcast_group_value(f"{benchmark_prefix}_open")
        self._fields["BANCHMARKINDEXCLOSE"] = self._broadcast_group_value(f"{benchmark_prefix}_close")
        self._fields["DTM"] = self._derive_dtm()
        self._fields["DBM"] = self._derive_dbm()
        self._fields["HD"] = self._derive_hd()
        self._fields["LD"] = self._derive_ld()
        self._fields["TR"] = self._derive_tr()
        self._fields["MKT"], self._fields["SMB"], self._fields["HML"] = self._derive_ff3_factors()

    def _blank(self) -> pd.DataFrame:
        return self.template.copy()

    def _pivot_value(self, column: str) -> pd.DataFrame:
        if column not in self._source.columns:
            return self._blank()
        pivot = (
            self._source[["trade_date", "stock_code", column]]
            .pivot(index="trade_date", columns="stock_code", values=column)
            .reindex(index=self.trade_dates, columns=self.stock_codes)
        )
        return pivot.apply(pd.to_numeric, errors="coerce")

    def _broadcast_series(self, series: pd.Series) -> pd.DataFrame:
        values = np.repeat(series.reindex(self.trade_dates).to_numpy()[:, None], len(self.stock_codes), axis=1)
        return pd.DataFrame(values, index=self.trade_dates, columns=self.stock_codes)

    def _broadcast_group_value(self, column: str) -> pd.DataFrame:
        if column not in self._source.columns:
            return self._blank()
        series = (
            self._source[["trade_date", column]]
            .drop_duplicates(subset=["trade_date"], keep="last")
            .set_index("trade_date")[column]
        )
        series = pd.to_numeric(series.reindex(self.trade_dates), errors="coerce")
        return self._broadcast_series(series)

    def _derive_vwap(self) -> pd.DataFrame:
        amount = self._fields["AMOUNT"]
        volume = self._fields["VOLUME"]
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = (amount * 10.0) / volume.replace(0, np.nan)
        return vwap.replace([np.inf, -np.inf], np.nan)

    def _derive_returns(self) -> pd.DataFrame:
        close = self._fields["CLOSE"]
        pre_close = self._pivot_value("pre_close")
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = close / pre_close.replace(0, np.nan) - 1.0
        return returns.replace([np.inf, -np.inf], np.nan)

    def _derive_dtm(self) -> pd.DataFrame:
        open_price = self._fields["OPEN"]
        high = self._fields["HIGH"]
        delay_open = open_price.shift(1)
        return pd.DataFrame(
            np.where(open_price <= delay_open, 0.0, np.maximum(high - open_price, open_price - delay_open)),
            index=self.trade_dates,
            columns=self.stock_codes,
        )

    def _derive_dbm(self) -> pd.DataFrame:
        open_price = self._fields["OPEN"]
        low = self._fields["LOW"]
        delay_open = open_price.shift(1)
        return pd.DataFrame(
            np.where(open_price >= delay_open, 0.0, np.maximum(open_price - low, open_price - delay_open)),
            index=self.trade_dates,
            columns=self.stock_codes,
        )

    def _derive_hd(self) -> pd.DataFrame:
        high = self._fields["HIGH"]
        return high - high.shift(1)

    def _derive_ld(self) -> pd.DataFrame:
        low = self._fields["LOW"]
        return low.shift(1) - low

    def _derive_tr(self) -> pd.DataFrame:
        high = self._fields["HIGH"]
        low = self._fields["LOW"]
        pre_close = self._pivot_value("pre_close")
        values = np.maximum.reduce(
            [
                (high - low).to_numpy(dtype=float),
                (high - pre_close).abs().to_numpy(dtype=float),
                (low - pre_close).abs().to_numpy(dtype=float),
            ]
        )
        return pd.DataFrame(values, index=self.trade_dates, columns=self.stock_codes)

    def _derive_ff3_factors(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        required = {"trade_date", "stock_code", "close", "pre_close", "total_mv", "pb"}
        if not required.issubset(self._source.columns):
            blank = self._blank()
            return blank, blank.copy(), blank.copy()

        raw = self._source[list(required)].copy()
        raw["trade_date"] = raw["trade_date"].astype(str)
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        raw["pre_close"] = pd.to_numeric(raw["pre_close"], errors="coerce")
        raw["total_mv"] = pd.to_numeric(raw["total_mv"], errors="coerce")
        raw["pb"] = pd.to_numeric(raw["pb"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            raw["ret"] = raw["close"] / raw["pre_close"].replace(0, np.nan) - 1.0
            raw["bm"] = 1.0 / raw["pb"].replace(0, np.nan)
        raw.replace([np.inf, -np.inf], np.nan, inplace=True)

        factor_rows: list[dict[str, float | str]] = []
        for trade_date, day in raw.groupby("trade_date", sort=True):
            valid = day.dropna(subset=["ret", "total_mv", "bm"]).copy()
            if valid.empty:
                factor_rows.append({"trade_date": trade_date, "MKT": np.nan, "SMB": np.nan, "HML": np.nan})
                continue

            market_weights = valid["total_mv"].clip(lower=0)
            market_weight_sum = market_weights.sum()
            market_ret = float(np.average(valid["ret"], weights=market_weights)) if market_weight_sum > 0 else np.nan

            size_median = valid["total_mv"].median()
            bm_low = valid["bm"].quantile(1 / 3)
            bm_high = valid["bm"].quantile(2 / 3)
            sized = valid.assign(
                size_bucket=np.where(valid["total_mv"] <= size_median, "S", "B"),
                bm_bucket=np.select(
                    [valid["bm"] <= bm_low, valid["bm"] >= bm_high],
                    ["L", "H"],
                    default="M",
                ),
            )

            portfolio_returns: dict[str, float] = {}
            for size_bucket in ("S", "B"):
                for bm_bucket in ("L", "M", "H"):
                    bucket = sized[(sized["size_bucket"] == size_bucket) & (sized["bm_bucket"] == bm_bucket)]
                    key = f"{size_bucket}{bm_bucket}"
                    if bucket.empty:
                        portfolio_returns[key] = np.nan
                        continue
                    weights = bucket["total_mv"].clip(lower=0)
                    weight_sum = weights.sum()
                    portfolio_returns[key] = float(np.average(bucket["ret"], weights=weights)) if weight_sum > 0 else np.nan

            small_mean = np.nanmean([portfolio_returns["SL"], portfolio_returns["SM"], portfolio_returns["SH"]])
            big_mean = np.nanmean([portfolio_returns["BL"], portfolio_returns["BM"], portfolio_returns["BH"]])
            high_mean = np.nanmean([portfolio_returns["SH"], portfolio_returns["BH"]])
            low_mean = np.nanmean([portfolio_returns["SL"], portfolio_returns["BL"]])
            smb = float(small_mean - big_mean) if not (np.isnan(small_mean) or np.isnan(big_mean)) else np.nan
            hml = float(high_mean - low_mean) if not (np.isnan(high_mean) or np.isnan(low_mean)) else np.nan
            factor_rows.append({"trade_date": trade_date, "MKT": market_ret, "SMB": smb, "HML": hml})

        factor_frame = pd.DataFrame(factor_rows).set_index("trade_date").reindex(self.trade_dates)
        return (
            self._broadcast_series(factor_frame["MKT"]),
            self._broadcast_series(factor_frame["SMB"]),
            self._broadcast_series(factor_frame["HML"]),
        )

    def _to_frame(self, value) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.reindex(index=self.trade_dates, columns=self.stock_codes)
        if isinstance(value, pd.Series):
            if list(value.index.astype(str)) == self.trade_dates:
                return self._broadcast_series(pd.to_numeric(value, errors="coerce"))
            if list(value.index.astype(str)) == self.stock_codes:
                values = np.repeat(pd.to_numeric(value, errors="coerce").to_numpy()[None, :], len(self.trade_dates), axis=0)
                return pd.DataFrame(values, index=self.trade_dates, columns=self.stock_codes)
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = np.full(self.template.shape, arr.item())
        elif arr.shape == (len(self.trade_dates),):
            arr = np.repeat(arr[:, None], len(self.stock_codes), axis=1)
        elif arr.shape == (len(self.stock_codes),):
            arr = np.repeat(arr[None, :], len(self.trade_dates), axis=0)
        if arr.shape != self.template.shape:
            raise ValueError(f"cannot coerce value with shape {arr.shape} into GTJA frame")
        return pd.DataFrame(arr, index=self.trade_dates, columns=self.stock_codes)

    def to_long(self, value, column_name: str, output_dates: Iterable[str] | None = None) -> pd.DataFrame:
        frame = self._to_frame(value)
        if output_dates is not None:
            keep_dates = {str(date) for date in output_dates}
            frame = frame.loc[frame.index.isin(keep_dates)]
        try:
            series = frame.stack(future_stack=True).rename(column_name)
        except TypeError:
            # pandas 1.x does not support future_stack; keep NaN rows so target-date
            # alignment matches the newer implementation closely enough for merges.
            series = frame.stack(dropna=False).rename(column_name)
        return series.reset_index().rename(columns={"level_0": "trade_date", "level_1": "stock_code"})

    def __call__(self, field: str) -> pd.DataFrame:
        return self._fields.get(field, self._blank())

    def ABS(self, value) -> pd.DataFrame:
        return self._to_frame(value).abs()

    def LOG(self, value) -> pd.DataFrame:
        frame = self._to_frame(value).astype(float)
        return pd.DataFrame(np.log(frame.where(frame > 0)), index=self.trade_dates, columns=self.stock_codes)

    def SIGN(self, value) -> pd.DataFrame:
        return pd.DataFrame(np.sign(self._to_frame(value)), index=self.trade_dates, columns=self.stock_codes)

    def DELAY(self, value, periods: int = 1) -> pd.DataFrame:
        return self._to_frame(value).shift(int(periods))

    def DELTA(self, value, periods: int = 1) -> pd.DataFrame:
        frame = self._to_frame(value)
        return frame - frame.shift(int(periods))

    def RANK(self, value) -> pd.DataFrame:
        frame = self._to_frame(value)
        if self.rank_mode == "identity":
            return frame
        return frame.rank(axis=1, pct=True)

    def SUM(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).sum()

    def SUMIF(self, value, window: int, condition) -> pd.DataFrame:
        frame = self._to_frame(value).where(self._to_frame(condition).astype(bool))
        return frame.rolling(int(window), min_periods=1).sum()

    def PROD(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).apply(np.prod, raw=True)

    def MEAN(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).mean()

    def MA(self, value, window: int) -> pd.DataFrame:
        return self.MEAN(value, window)

    def STD(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).std()

    def CORR(self, left, right, window: int) -> pd.DataFrame:
        lhs = self._to_frame(left).astype(float)
        rhs = self._to_frame(right).astype(float)
        return lhs.rolling(int(window), min_periods=1).corr(rhs)

    def COV(self, left, right, window: int) -> pd.DataFrame:
        lhs = self._to_frame(left).astype(float)
        rhs = self._to_frame(right).astype(float)
        return lhs.rolling(int(window), min_periods=1).cov(rhs)

    def TSMAX(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).max()

    def TSMIN(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).min()

    def MAX(self, left, right):
        if np.isscalar(right) and int(right) == right and int(right) > 1:
            return self.TSMAX(left, int(right))
        lhs = self._to_frame(left)
        rhs = self._to_frame(right)
        return pd.DataFrame(np.maximum(lhs, rhs), index=self.trade_dates, columns=self.stock_codes)

    def MIN(self, left, right):
        if np.isscalar(right) and int(right) == right and int(right) > 1:
            return self.TSMIN(left, int(right))
        lhs = self._to_frame(left)
        rhs = self._to_frame(right)
        return pd.DataFrame(np.minimum(lhs, rhs), index=self.trade_dates, columns=self.stock_codes)

    def TSRANK(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).apply(_rolling_last_rank, raw=True)

    def SMA(self, value, window: int, weight: int = 1) -> pd.DataFrame:
        alpha = float(weight) / float(window)
        return self._to_frame(value).ewm(alpha=alpha, adjust=False).mean()

    def WMA(self, value, window: int) -> pd.DataFrame:
        return _rolling_weighted_mean(self._to_frame(value), int(window))

    def DECAYLINEAR(self, value, window: int) -> pd.DataFrame:
        return _rolling_weighted_mean(self._to_frame(value), int(window))

    def COUNT(self, condition, window: int) -> pd.DataFrame:
        frame = self._to_frame(condition).astype(float)
        return frame.rolling(int(window), min_periods=1).sum()

    def FILTER(self, value, condition) -> pd.DataFrame:
        return self._to_frame(value).where(self._to_frame(condition).astype(bool))

    def HIGHDAY(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).apply(
            lambda values: float(len(values) - np.argmax(values)),
            raw=True,
        )

    def LOWDAY(self, value, window: int) -> pd.DataFrame:
        return self._to_frame(value).rolling(int(window), min_periods=1).apply(
            lambda values: float(len(values) - np.argmin(values)),
            raw=True,
        )

    def SCAN_MUL(self, value, condition) -> pd.DataFrame:
        values = self._to_frame(value).astype(float)
        cond = self._to_frame(condition).astype(bool)
        out = pd.DataFrame(1.0, index=self.trade_dates, columns=self.stock_codes)
        for idx in range(len(self.trade_dates)):
            if idx == 0:
                out.iloc[idx] = np.where(cond.iloc[idx], values.iloc[idx], 1.0)
            else:
                out.iloc[idx] = np.where(cond.iloc[idx], values.iloc[idx] * out.iloc[idx - 1], out.iloc[idx - 1])
        return out

    def REGBETA(self, y, x, window: int) -> pd.DataFrame:
        lhs = self._to_frame(y).astype(float)
        rhs = self._to_frame(x).astype(float)
        cov = lhs.rolling(int(window), min_periods=1).cov(rhs)
        var = rhs.rolling(int(window), min_periods=1).var()
        return cov / var.replace(0, np.nan)

    def REGRESI(self, y, *args):
        if len(args) < 2:
            raise ValueError("REGRESI requires predictors plus a trailing window")
        *predictors, window = args
        target = self._to_frame(y).astype(float)
        xs = [self._to_frame(p).astype(float) for p in predictors]
        window = int(window)
        if any(frame.notna().sum().sum() == 0 for frame in xs):
            return self._blank()
        out = self._blank()
        for col in self.stock_codes:
            y_col = target[col]
            x_cols = [frame[col] for frame in xs]
            for row_idx in range(len(self.trade_dates)):
                start = max(0, row_idx - window + 1)
                y_win = y_col.iloc[start : row_idx + 1]
                x_win = pd.concat([series.iloc[start : row_idx + 1] for series in x_cols], axis=1)
                valid = y_win.notna() & x_win.notna().all(axis=1)
                if valid.sum() <= len(xs):
                    continue
                y_values = y_win[valid].to_numpy(dtype=float)
                x_values = x_win[valid].to_numpy(dtype=float)
                design = np.column_stack([np.ones(len(x_values)), x_values])
                beta, *_ = np.linalg.lstsq(design, y_values, rcond=None)
                latest_x = np.concatenate([[1.0], x_cols[0].iloc[[row_idx]].to_numpy(dtype=float)])
                if len(xs) > 1:
                    latest_x = np.concatenate([[1.0], np.array([series.iloc[row_idx] for series in x_cols], dtype=float)])
                if np.isnan(latest_x).any():
                    continue
                out.at[self.trade_dates[row_idx], col] = y_col.iloc[row_idx] - float(np.dot(latest_x, beta))
        return out


def required_official_gtja_raw_columns() -> list[str]:
    return GTJA_OFFICIAL_RAW_COLUMNS.copy()


def append_official_gtja_alpha(
    frame: pd.DataFrame,
    *,
    cross_sectional_rank_mode: str = "rank",
    benchmark_prefix: str = "index_2000",
    output_dates: Iterable[str] | None = None,
    alpha_batch_size: int = 16,
    progress_callback=None,
) -> pd.DataFrame:
    ctx = OfficialGtjaContext(
        frame,
        cross_sectional_rank_mode=cross_sectional_rank_mode,
        benchmark_prefix=benchmark_prefix,
    )
    batch_size = max(1, int(alpha_batch_size))
    alpha_numbers = list(range(1, 192))
    total_batches = (len(alpha_numbers) + batch_size - 1) // batch_size
    batch_frames: list[pd.DataFrame] = []

    for batch_index, start in enumerate(range(0, len(alpha_numbers), batch_size), start=1):
        batch_numbers = alpha_numbers[start : start + batch_size]
        indexed_parts = []
        for no in batch_numbers:
            fn = getattr(official_alpha191, f"alpha_{no:03d}")
            part = ctx.to_long(fn(ctx), f"gtja_alpha{no:03d}", output_dates=output_dates)
            value_columns = [column for column in part.columns if column not in {"trade_date", "stock_code"}]
            indexed = part.set_index(["trade_date", "stock_code"])
            if len(value_columns) != 1:
                raise ValueError(f"unexpected GTJA projection columns: {value_columns}")
            indexed_parts.append(indexed[value_columns])
        batch_frame = pd.concat(indexed_parts, axis=1, sort=False)
        batch_frames.append(batch_frame)
        if progress_callback is not None:
            progress_callback(
                {
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "alpha_start": batch_numbers[0],
                    "alpha_end": batch_numbers[-1],
                    "batch_columns": len(batch_numbers),
                    "rows": int(batch_frame.shape[0]),
                }
            )

    merged = pd.concat(batch_frames, axis=1, sort=False).reset_index()

    result = frame.copy()
    result["trade_date"] = result["trade_date"].astype(str)
    result["stock_code"] = result["stock_code"].astype(str)
    if output_dates is not None:
        keep_dates = {str(date) for date in output_dates}
        result = result[result["trade_date"].isin(keep_dates)].copy()
    return result.merge(merged, on=["trade_date", "stock_code"], how="left", validate="many_to_one")

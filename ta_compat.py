"""
ta_compat.py - pandas_ta 兼容层

由于 ta 库缺少 pandas_ta 的一些函数，此模块提供兼容实现
"""

import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, WMAIndicator
from ta.momentum import KAMAIndicator
from ta.volatility import AverageTrueRange
from ta.volume import ChaikinMoneyFlowIndicator
from ta.trend import CCIIndicator


def dema(series, length=30):
    """双指数移动平均线"""
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return 2 * ema1 - ema2


def kama(series, length=30):
    """自适应移动平均线"""
    return KAMAIndicator(close=series, window=length).kama()


def sma(series, length=30):
    """简单移动平均线"""
    return SMAIndicator(close=series, window=length).sma_indicator()


def midpoint(series, length=14):
    """中点价"""
    return (series.rolling(window=length).max() + series.rolling(window=length).min()) / 2


def midprice(high, low, length=14):
    """中间价"""
    return (high.rolling(window=length).max() + low.rolling(window=length).min()) / 2


def t3(series, length=5):
    """T3移动平均线"""
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3


def tema(series, length=30):
    """三重指数移动平均线"""
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    return 3 * (ema1 - ema2) + ema3


def trima(series, length=30):
    """三角移动平均线"""
    return series.rolling(window=length).mean().rolling(window=length).mean()


def wma(series, length=30):
    """加权移动平均线"""
    weights = np.arange(1, length + 1)
    return series.rolling(window=length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def ema(series, length=30):
    """指数移动平均线"""
    return EMAIndicator(close=series, window=length).ema_indicator()


def rsi(series, length=14):
    """相对强弱指标"""
    from ta.momentum import RSIIndicator
    return RSIIndicator(close=series, window=length).rsi()


def atr(high, low, close, length=14):
    """平均真实波动范围"""
    return AverageTrueRange(high=high, low=low, close=close, window=length).average_true_range()


def cmf(high, low, close, volume, length=14):
    """资金流量指标"""
    return ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume, window=length).chaikin_money_flow()


def cci(high, low, close, length=14):
    """顺势指标"""
    return CCIIndicator(high=high, low=low, close=close, window=length).cci()

"""Reusable data helpers from the research notebook."""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_yfinance_ohlcv_df(symbol, start='2023-01-01', end='2026-08-31', interval='1d'):
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df = df[['open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col].squeeze(), errors='coerce')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def rolling_zscore(series, window=90):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)

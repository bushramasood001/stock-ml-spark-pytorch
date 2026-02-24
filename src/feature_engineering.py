"""
feature_engineering.py

# PURPOSE: Create ML features from stock price data (returns, moving averages, volatility, next-day target).
"""

import pandas as pd
import numpy as np


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    # PURPOSE: Create Daily_Return and Log_Return from Close price.
    """
    df = df.copy()

    # PURPOSE: Previous day's Close (lag)
    df["Prev_Close"] = df["Close"].shift(1)

    # PURPOSE: Simple daily return
    df["Daily_Return"] = (df["Close"] - df["Prev_Close"]) / df["Prev_Close"]

    # PURPOSE: Log return (more stable for compounding)
    df["Log_Return"] = np.log(df["Close"] / df["Prev_Close"])

    return df


def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    # PURPOSE: Add rolling moving average (MA) and rolling volatility (std of returns).
    """
    df = df.copy()

    # PURPOSE: Rolling mean of Close price (moving average)
    df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()

    # PURPOSE: Rolling volatility of Daily_Return (standard deviation)
    df[f"Volatility_{window}"] = df["Daily_Return"].rolling(window=window).std()

    return df


def add_next_return_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    # PURPOSE: Create Next_Return as the prediction target (next day's Daily_Return).
    """
    df = df.copy()

    # PURPOSE: Shift Daily_Return upwards so today's row has tomorrow's return
    df["Next_Return"] = df["Daily_Return"].shift(-1)

    return df


def build_ml_dataset(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    # PURPOSE: Full feature pipeline + remove rows with nulls created by shifting/rolling.
    """
    df = df.copy()

    # PURPOSE: Ensure sorted by Date (important for time series)
    if "Date" in df.columns:
        df = df.sort_values("Date")

    # PURPOSE: Add features
    df = add_returns(df)
    df = add_rolling_features(df, window=window)
    df = add_next_return_target(df)

    # PURPOSE: Drop rows with NaN created due to lag/rolling/shift
    df = df.dropna().reset_index(drop=True)

    return df

"""Time-series utilities: time-based indexing, resampling, decomposition and diagnostics."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller


def daily_spend(df: pd.DataFrame, exclude_outliers: bool = False) -> pd.Series:
    """Daily total debit spend on a gap-free DatetimeIndex."""
    spend = df[df["type"] == "debit"]
    if exclude_outliers and "is_outlier" in spend:
        spend = spend[~spend["is_outlier"]]
    series = spend.groupby("date")["amount"].sum()
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return series.reindex(full, fill_value=0.0).rename("spend").rename_axis("date")


def daily_income(df: pd.DataFrame) -> pd.Series:
    credit = df[df["type"] == "credit"].groupby("date")["amount"].sum()
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return credit.reindex(full, fill_value=0.0).rename("income")


def resample_spend(series: pd.Series, rule: str = "W") -> pd.Series:
    rule = {"M": "ME", "Q": "QE", "Y": "YE"}.get(rule, rule)
    return series.resample(rule).sum()


def rolling_mean(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def monthly_table(df: pd.DataFrame, exclude_outliers: bool = True) -> pd.DataFrame:
    debit = df[df["type"] == "debit"]
    if exclude_outliers and "is_outlier" in debit:
        debit = debit[~debit["is_outlier"]]
    spend = debit.groupby("month")["amount"].sum().rename("spend")
    income = df[df["type"] == "credit"].groupby("month")["amount"].sum().rename("income")
    table = pd.concat([income, spend], axis=1).fillna(0.0)
    table["savings"] = table["income"] - table["spend"]
    table["savings_rate_pct"] = np.where(table["income"] > 0, 100 * table["savings"] / table["income"], np.nan)
    table["spend_mom_pct"] = table["spend"].pct_change() * 100
    return table


def weekday_profile(series: pd.Series) -> pd.DataFrame:
    frame = series.to_frame("spend")
    frame["weekday"] = frame.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out = frame.groupby("weekday")["spend"].agg(["mean", "median", "sum"]).reindex(order)
    return out


def decompose(series: pd.Series, period: int = 7):
    return seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")


def adf_test(series: pd.Series) -> dict:
    stat, p, lags, nobs, crit, _ = adfuller(series.to_numpy(), autolag="AIC", result_object=False)
    return {"statistic": stat, "p_value": p, "lags": lags, "observations": nobs, "critical_values": crit,
            "stationary": bool(p < 0.05)}


def autocorrelation(series: pd.Series, nlags: int = 35) -> np.ndarray:
    return acf(series.to_numpy(), nlags=nlags, fft=True)


def holt_winters(train: pd.Series, horizon: int = 30) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(train.clip(lower=0), trend=None, seasonal="add", seasonal_periods=7,
                                     initialization_method="estimated").fit()
        forecast = model.forecast(horizon)
    return forecast.clip(lower=0)

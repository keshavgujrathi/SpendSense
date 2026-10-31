"""Linear-regression spend model, accuracy evaluation and next-month forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .timeseries import holt_winters

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def scheduled_amounts(recurring: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Known recurring bills expected on each calendar day (rent, subscriptions, utilities)."""
    total = pd.Series(0.0, index=index)
    for _, row in recurring.iterrows():
        day = np.minimum(int(row["usual_day"]), index.days_in_month)
        total += np.where(index.day == day, row["typical_amount"], 0.0)
    return total


def design_matrix(index: pd.DatetimeIndex, origin: pd.Timestamp, scheduled: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=index)
    X["scheduled_bills"] = scheduled.reindex(index).to_numpy()
    for d in range(1, 7):
        X[f"dow_{DAY_NAMES[d]}"] = (index.dayofweek == d).astype(float)
    X["early_month"] = (index.day <= 7).astype(float)
    X["trend_months"] = (index - origin).days / 30.0
    return X


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    wmape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "wMAPE": wmape,
        "accuracy": max(0.0, 1 - wmape),
    }


def fit_spend_model(daily: pd.Series, recurring: pd.DataFrame, test_days: int = 90) -> dict:
    origin = daily.index.min()
    X = design_matrix(daily.index, origin, scheduled_amounts(recurring, daily.index))
    split = len(daily) - test_days
    X_train, X_test, y_train, y_test = X.iloc[:split], X.iloc[split:], daily.iloc[:split], daily.iloc[split:]

    model = LinearRegression().fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=y_test.index).clip(lower=0)
    baseline = pd.Series(y_train.mean(), index=y_test.index)

    weekly_true, weekly_pred = y_test.resample("W").sum(), pred.resample("W").sum()
    ols = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    coefficients = pd.DataFrame({"coefficient": ols.params, "std_error": ols.bse, "p_value": ols.pvalues})

    cv = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(X):
        m = LinearRegression().fit(X.iloc[train_idx], daily.iloc[train_idx])
        cv.append(evaluate(daily.iloc[val_idx], m.predict(X.iloc[val_idx]).clip(min=0)))
    return {
        "model": model, "origin": origin, "features": list(X.columns),
        "daily": evaluate(y_test, pred), "weekly": evaluate(weekly_true, weekly_pred),
        "baseline": evaluate(y_test, baseline), "cv": pd.DataFrame(cv),
        "predictions": pd.DataFrame({"actual": y_test, "predicted": pred}),
        "weekly_predictions": pd.DataFrame({"actual": weekly_true, "predicted": weekly_pred}),
        "coefficients": coefficients, "ols": ols, "residual_std": float(np.std(y_train - model.predict(X_train))),
        "durbin_watson": float(sm.stats.durbin_watson(ols.resid)),
    }


def _lr_forecast(train: pd.Series, horizon_index: pd.DatetimeIndex, recurring: pd.DataFrame) -> pd.Series:
    origin = train.index.min()
    train_X = design_matrix(train.index, origin, scheduled_amounts(recurring, train.index))
    future_X = design_matrix(horizon_index, origin, scheduled_amounts(recurring, horizon_index))
    model = LinearRegression().fit(train_X, train)
    return pd.Series(model.predict(future_X), index=horizon_index).clip(lower=0)


def complete_months(daily: pd.Series) -> list[pd.Period]:
    periods = daily.index.to_period("M").unique()
    last_day = daily.index.max()
    return [p for p in periods if p.start_time >= daily.index.min() and p.end_time.normalize() <= last_day]


def backtest_monthly(daily: pd.Series, recurring: pd.DataFrame, n_months: int = 4) -> pd.DataFrame | None:
    months = complete_months(daily)[-n_months:]
    rows = []
    for period in months:
        train = daily[daily.index < period.start_time]
        if len(train) < 120:
            continue
        horizon = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
        actual = daily.reindex(horizon).sum()
        naive = train.iloc[-30:].mean() * len(horizon)
        rows.append({
            "month": str(period), "actual": actual, "naive": naive,
            "holt_winters": holt_winters(train, len(horizon)).sum(),
            "linear_regression": _lr_forecast(train, horizon, recurring).sum(),
        })
    if not rows:
        return None
    table = pd.DataFrame(rows).set_index("month")
    table["ensemble"] = table[["holt_winters", "linear_regression"]].mean(axis=1)
    return table


def backtest_errors(table: pd.DataFrame) -> pd.DataFrame:
    models = [c for c in table.columns if c != "actual"]
    rows = [{"model": m, "MAPE_%": 100 * (np.abs(table[m] - table["actual"]) / table["actual"]).mean()}
            for m in models]
    return pd.DataFrame(rows).sort_values("MAPE_%").reset_index(drop=True)


def forecast_next(daily: pd.Series, recurring: pd.DataFrame, horizon: int = 30) -> dict:
    future = pd.date_range(daily.index.max() + pd.offsets.Day(1), periods=horizon, freq="D")
    lr = _lr_forecast(daily, future, recurring)
    hw = holt_winters(daily, horizon)
    hw.index = future
    sigma = float(np.std(daily - _lr_forecast(daily, daily.index, recurring)))
    ensemble = (lr + hw) / 2
    return {"dates": future, "linear_regression": lr, "holt_winters": hw, "ensemble": ensemble,
            "total": float(ensemble.sum()), "low": float(ensemble.sum() - 1.28 * sigma * np.sqrt(horizon)),
            "high": float(ensemble.sum() + 1.28 * sigma * np.sqrt(horizon))}

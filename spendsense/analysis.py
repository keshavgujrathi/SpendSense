"""Univariate, bivariate and multivariate analysis plus statistical summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MEASUREMENT_SCALES = pd.DataFrame(
    [
        ("date", "Temporal", "Interval", "Ordered, equal spacing, no true zero"),
        ("description", "Text", "Nominal", "Free-text narration"),
        ("merchant", "Categorical", "Nominal", "Canonical merchant label"),
        ("category", "Categorical", "Nominal", "Spending taxonomy, no order"),
        ("payment_mode", "Categorical", "Nominal", "UPI / Card / NetBanking / Cash"),
        ("type", "Categorical", "Nominal", "Debit or credit"),
        ("amount_band", "Categorical", "Ordinal", "Micro < Small < Medium < Large < Major"),
        ("weekday", "Categorical", "Ordinal", "Cyclic ordered days"),
        ("amount", "Numerical (continuous)", "Ratio", "True zero, ratios meaningful"),
        ("day_of_month", "Numerical (discrete)", "Interval", "Counts of days, no true zero"),
        ("transactions per day", "Numerical (discrete)", "Ratio", "Counts with true zero"),
    ],
    columns=["variable", "data_type", "measurement_scale", "remark"],
)


def summary_stats(series: pd.Series) -> pd.Series:
    x = series.dropna().astype(float)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    return pd.Series({
        "count": x.size, "mean": x.mean(), "median": x.median(), "std": x.std(),
        "variance": x.var(), "min": x.min(), "q1": q1, "q3": q3, "max": x.max(),
        "range": x.max() - x.min(), "iqr": q3 - q1, "skewness": stats.skew(x),
        "kurtosis": stats.kurtosis(x), "cv": x.std() / x.mean() if x.mean() else np.nan,
    })


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    spend = df[df["type"] == "debit"]
    out = spend.groupby("category")["amount"].agg(["count", "sum", "mean", "median", "std", "max"])
    out["share_pct"] = 100 * out["sum"] / out["sum"].sum()
    out["cv"] = out["std"] / out["mean"]
    return out.sort_values("sum", ascending=False)


def weekly_category_matrix(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    spend = df[df["type"] == "debit"]
    matrix = spend.pivot_table(index=pd.Grouper(key="date", freq=freq), columns="category",
                               values="amount", aggfunc="sum", fill_value=0.0)
    return matrix.loc[:, matrix.std() > 0]


def correlation_matrix(df: pd.DataFrame, method: str = "spearman", freq: str = "W") -> pd.DataFrame:
    return weekly_category_matrix(df, freq).corr(method=method)


def top_correlations(corr: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename_axis(["variable_1", "variable_2"]).rename("correlation").reset_index()
    pairs["abs"] = pairs["correlation"].abs()
    return pairs.sort_values("abs", ascending=False).head(n).drop(columns="abs").reset_index(drop=True)


def contingency_table(df: pd.DataFrame, row: str = "category", col: str = "payment_mode") -> pd.DataFrame:
    spend = df[df["type"] == "debit"]
    return pd.crosstab(spend[row], spend[col])


def chi_square_test(table: pd.DataFrame) -> dict:
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return {"chi2": chi2, "p_value": p, "dof": dof, "cramers_v": float(np.sqrt(chi2 / (n * k))) if k else np.nan}


def group_difference(df: pd.DataFrame, group: str) -> dict:
    """Kruskal-Wallis test for differences in log spend across the groups of a categorical column."""
    spend = df[df["type"] == "debit"]
    samples = [np.log1p(g["amount"].to_numpy()) for _, g in spend.groupby(group) if len(g) > 2]
    h, p = stats.kruskal(*samples)
    return {"H": h, "p_value": p, "groups": len(samples)}


def multi_way_table(df: pd.DataFrame, rows: list[str], cols: list[str], values: str = "amount",
                    aggfunc: str = "sum") -> pd.DataFrame:
    spend = df[df["type"] == "debit"]
    return spend.pivot_table(index=rows, columns=cols, values=values, aggfunc=aggfunc, fill_value=0).round(0)


def weekday_order() -> list[str]:
    return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def spend_by_weekday_category(df: pd.DataFrame) -> pd.DataFrame:
    spend = df[df["type"] == "debit"]
    table = spend.pivot_table(index="category", columns="weekday", values="amount", aggfunc="mean", fill_value=0)
    return table.reindex(columns=weekday_order())

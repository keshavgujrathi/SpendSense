"""Decision-support insights: KPIs, recurring payments, budget review."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import WANTS


def debits(df: pd.DataFrame, exclude_outliers: bool = True) -> pd.DataFrame:
    spend = df[df["type"] == "debit"]
    if exclude_outliers and "is_outlier" in spend:
        spend = spend[~spend["is_outlier"]]
    return spend


def latest_complete_month(df: pd.DataFrame) -> str:
    last = df["date"].max()
    month = last.to_period("M")
    if last.normalize() < month.end_time.normalize():
        month = month - 1
    return str(month)


def kpis(df: pd.DataFrame) -> dict:
    month = latest_complete_month(df)
    prev = str(pd.Period(month) - 1)
    spend_all, income_all = debits(df), df[df["type"] == "credit"]
    one_off = df[(df["type"] == "debit") & df.get("is_outlier", False) & (df["month"] == month)]["amount"].sum()
    cur_spend = spend_all.loc[spend_all["month"] == month, "amount"].sum()
    prev_spend = spend_all.loc[spend_all["month"] == prev, "amount"].sum()
    income = income_all.loc[income_all["month"] == month, "amount"].sum()
    by_cat = spend_all[spend_all["month"] == month].groupby("category")["amount"].sum().sort_values(ascending=False)
    return {
        "month": month, "spend": cur_spend, "income": income, "savings": income - cur_spend,
        "savings_rate": (income - cur_spend) / income * 100 if income else np.nan,
        "spend_change_pct": (cur_spend - prev_spend) / prev_spend * 100 if prev_spend else np.nan,
        "top_category": by_cat.index[0] if len(by_cat) else "-",
        "top_category_amount": float(by_cat.iloc[0]) if len(by_cat) else 0.0,
        "transactions": int((spend_all["month"] == month).sum()),
        "avg_daily": cur_spend / pd.Period(month).days_in_month,
        "one_off_spend": float(one_off),
    }


def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    """Merchants charged at a steady monthly cadence with a stable amount."""
    spend = debits(df)
    rows = []
    for merchant, g in spend.groupby("merchant"):
        if len(g) < 3:
            continue
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.dropna()
        amount_cv = g["amount"].std() / g["amount"].mean() if g["amount"].mean() else np.inf
        if 26 <= gaps.median() <= 35 and gaps.std() <= 6 and amount_cv <= 0.4:
            typical = float(g["amount"].median())
            rows.append({
                "merchant": merchant, "category": g["category"].mode().iat[0], "typical_amount": typical,
                "usual_day": int(g["day_of_month"].median()),
                "charges": len(g), "last_charge": g["date"].max(),
                "next_expected": g["date"].max() + pd.offsets.Day(int(round(gaps.median()))),
                "annual_cost": typical * 12,
            })
    if not rows:
        return pd.DataFrame(columns=["merchant", "category", "typical_amount", "usual_day", "charges", "last_charge",
                                     "next_expected", "annual_cost"])
    return pd.DataFrame(rows).sort_values("annual_cost", ascending=False).reset_index(drop=True)


def budget_review(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    spend = debits(df)
    month = latest_complete_month(df)
    table = spend.pivot_table(index="category", columns="month", values="amount", aggfunc="sum", fill_value=0.0)
    if month not in table.columns:
        return pd.DataFrame()
    prior = [c for c in table.columns if c < month][-lookback:]
    base = table[prior].median(axis=1) if prior else table[month]
    out = pd.DataFrame({"latest": table[month], "typical": base})
    factor = np.where(out.index.isin(WANTS), 0.95, 1.0)
    out["suggested_cap"] = out["typical"] * factor
    out["difference"] = out["latest"] - out["suggested_cap"]
    out["status"] = np.where(out["difference"] > 0.1 * out["suggested_cap"].clip(lower=1), "Over", "Within")
    out["type"] = np.where(out.index.isin(WANTS), "Want", "Need")
    return out.sort_values("latest", ascending=False).round(0)


def top_merchants(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    spend = debits(df)
    out = spend.groupby("merchant").agg(total=("amount", "sum"), transactions=("amount", "size"),
                                         category=("category", lambda s: s.mode().iat[0]))
    return out.sort_values("total", ascending=False).head(n)


def plain_language_insights(df: pd.DataFrame) -> list[str]:
    k = kpis(df)
    notes = []
    if not np.isnan(k["savings_rate"]):
        verdict = "healthy" if k["savings_rate"] >= 20 else "below the 20% guideline"
        notes.append(f"You saved {k['savings_rate']:.0f}% of income in {k['month']}, which is {verdict}.")
    if not np.isnan(k["spend_change_pct"]):
        direction = "up" if k["spend_change_pct"] > 0 else "down"
        notes.append(f"Spending is {direction} {abs(k['spend_change_pct']):.0f}% compared with the previous month.")
    notes.append(f"{k['top_category']} was the largest category at {k['top_category_amount']:,.0f}.")
    recurring = detect_recurring(df)
    subs = recurring[recurring["category"] == "Subscriptions"]
    if len(subs):
        notes.append(f"{len(subs)} subscriptions cost {subs['annual_cost'].sum():,.0f} per year in total.")
    return notes

"""Synthetic bank-statement generator with realistic structure and deliberate data defects."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORY_NOTES, MERCHANTS

SPEND_PROFILE = {
    "Groceries": dict(median=480, sigma=0.55, rate=0.33, weekend=1.35),
    "Food Delivery": dict(median=320, sigma=0.45, rate=0.45, weekend=1.5),
    "Dining": dict(median=650, sigma=0.50, rate=0.18, weekend=1.9),
    "Transport": dict(median=170, sigma=0.50, rate=0.55, weekend=0.7),
    "Fuel": dict(median=1400, sigma=0.30, rate=0.07, weekend=1.2),
    "Shopping": dict(median=1500, sigma=0.80, rate=0.12, weekend=1.5),
    "Entertainment": dict(median=600, sigma=0.50, rate=0.07, weekend=2.6),
    "Health": dict(median=550, sigma=0.70, rate=0.05, weekend=0.9),
    "Travel": dict(median=3200, sigma=0.70, rate=0.02, weekend=1.2),
    "Education": dict(median=900, sigma=0.60, rate=0.02, weekend=1.0),
    "Miscellaneous": dict(median=400, sigma=0.80, rate=0.10, weekend=1.1),
}

MODE_WEIGHTS = {
    "Groceries": {"UPI": 0.65, "Card": 0.2, "Cash": 0.15},
    "Food Delivery": {"UPI": 0.6, "Card": 0.4},
    "Dining": {"UPI": 0.4, "Card": 0.6},
    "Transport": {"UPI": 0.8, "Card": 0.2},
    "Fuel": {"Card": 0.65, "UPI": 0.35},
    "Shopping": {"Card": 0.6, "UPI": 0.35, "NetBanking": 0.05},
    "Entertainment": {"UPI": 0.5, "Card": 0.5},
    "Health": {"UPI": 0.6, "Card": 0.3, "Cash": 0.1},
    "Travel": {"Card": 0.7, "NetBanking": 0.2, "UPI": 0.1},
    "Education": {"Card": 0.5, "UPI": 0.5},
    "Miscellaneous": {"UPI": 0.4, "Cash": 0.5, "Card": 0.1},
}

MODE_VARIANTS = {
    "UPI": ["UPI", "upi", "Upi"],
    "Card": ["Card", "CARD", "Debit Card", "Credit Card"],
    "NetBanking": ["NetBanking", "Net Banking", "NEFT"],
    "Cash": ["Cash", "CASH"],
}
TYPE_VARIANTS = {"debit": ["debit", "Debit", "DEBIT", "DR"], "credit": ["credit", "Credit", "CREDIT", "CR"]}

FACTOR_LOADINGS = {
    "Dining": {"social": 0.8},
    "Entertainment": {"social": 0.9},
    "Miscellaneous": {"social": 0.4},
    "Transport": {"social": 0.4, "home": -0.6, "trip": 0.5},
    "Food Delivery": {"home": 0.9, "social": -0.3},
    "Groceries": {"home": 0.5},
    "Shopping": {"home": 0.6},
    "Fuel": {"home": -0.7, "trip": 0.8},
    "Travel": {"trip": 0.8},
}
RATE_SCALE = 0.5
LOADING_SCALE = 1.5

ANOMALIES = [
    ("Croma Electronics", 58999.0, "Shopping", "Card"),
    ("MakeMyTrip", 47850.0, "Travel", "Card"),
    ("Apollo Hospitals", 32400.0, "Health", "UPI"),
    ("Unknown Merchant INTL", 24999.0, "Miscellaneous", "Card"),
    ("Swiggy", 9800.0, "Food Delivery", "UPI"),
    ("Uber", 6400.0, "Transport", "UPI"),
    ("Amazon", 41200.0, "Shopping", "Card"),
    ("Zomato", 7350.0, "Food Delivery", "UPI"),
]


def _description(merchant: str, mode: str, category: str, rng: np.random.Generator) -> str:
    ref = int(rng.integers(10**8, 10**9))
    base = merchant.upper()
    note = rng.choice(CATEGORY_NOTES.get(category, [""])) if rng.random() < 0.6 else ""
    if merchant == "ATM Cash Withdrawal":
        text = f"ATM WDL {int(rng.integers(1000, 9999))} {note}".strip()
    elif mode == "UPI":
        text = f"UPI/{base}/{ref}/{note}"
    elif mode == "Card":
        text = f"POS {int(rng.integers(1000, 9999))}XX {base} BANGALORE {note}"
    elif mode == "NetBanking":
        text = f"NEFT-{ref}-{base}-{note}"
    else:
        text = f"CASH PAID {base} {note}"
    style = rng.random()
    if style < 0.08:
        text = text.lower()
    elif style < 0.13:
        text = text + "  "
    elif style < 0.18:
        text = text.replace("/", "*", 1)
    return text


def _pick_mode(category: str, rng: np.random.Generator) -> str:
    weights = MODE_WEIGHTS[category]
    return str(rng.choice(list(weights), p=np.array(list(weights.values())) / sum(weights.values())))


def _latent_factors(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    def ar1(phi: float, sd: float) -> np.ndarray:
        noise, x = rng.normal(0, sd, n), np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]
        return x

    trip = np.zeros(n)
    t = 0
    while t < n:
        if rng.random() < 0.02:
            trip[t:t + int(rng.integers(3, 7))] = 1.0
            t += 6
        t += 1
    return {"social": ar1(0.4, 0.8), "home": ar1(0.6, 0.7), "trip": trip}


def generate_transactions(months: int = 18, end: str = "2026-09-30", seed: int = 42):
    """Return (raw_statement, truth) where truth lists the injected anomalies."""
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end)
    start_ts = end_ts + pd.DateOffset(days=1) - pd.DateOffset(months=months)
    days = pd.date_range(start_ts, end_ts, freq="D")
    factors = _latent_factors(len(days), rng)
    rows: list[dict] = []

    def add(date, merchant, amount, category, mode, kind="debit"):
        rows.append(dict(date=date, description=_description(merchant, mode, category, rng),
                         amount=round(float(amount), 2), type=kind, category=category,
                         payment_mode=mode, merchant_true=merchant))

    for day_idx, d in enumerate(days):
        months_in = (d.year - start_ts.year) * 12 + d.month - start_ts.month
        growth = 1.006 ** months_in
        weekend = d.dayofweek >= 5
        post_salary = d.day <= 7
        for category, p in SPEND_PROFILE.items():
            lam = RATE_SCALE * p["rate"] * (p["weekend"] if weekend else 1.0)
            for name, loading in FACTOR_LOADINGS.get(category, {}).items():
                lam *= np.exp(LOADING_SCALE * loading * factors[name][day_idx])
            if post_salary and category in {"Food Delivery", "Shopping", "Dining"}:
                lam *= 1.25
            if category == "Shopping" and d.month in (10, 11):
                lam *= 1.8
            if category == "Travel" and d.month in (5, 12):
                lam *= 2.5
            for _ in range(rng.poisson(lam)):
                merchant = str(rng.choice(MERCHANTS[category]))
                mode = "Cash" if merchant == "ATM Cash Withdrawal" else _pick_mode(category, rng)
                amount = p["median"] * np.exp(p["sigma"] * rng.standard_normal()) * growth
                if category == "Transport" and merchant == "Namma Metro":
                    amount = float(rng.choice([20, 30, 40, 50, 60]))
                add(d, merchant, max(amount, 10), category, mode)

        rent = 18000 if months_in < 12 else 19000
        fixed = {
            1: [("House Rent", rent, "Rent", "NetBanking")],
            3: [("Cult Fit Gym", 1500, "Subscriptions", "UPI")],
            8: [("ACT Fibernet", 1049, "Utilities", "UPI")],
            12: [("Jio Recharge", 719, "Utilities", "UPI")],
            15: [("Netflix", 649, "Subscriptions", "Card")],
            18: [("Spotify", 119, "Subscriptions", "Card")],
            21: [("Disney Hotstar", 299, "Subscriptions", "Card")],
            22: [("YouTube Premium", 129, "Subscriptions", "Card")],
        }
        for merchant, amount, category, mode in fixed.get(d.day, []):
            add(d, merchant, amount, category, mode)
        if d.day == 7:
            summer = 1.55 if d.month in (4, 5, 6) else 1.0
            add(d, "BESCOM Electricity", rng.uniform(850, 1500) * summer * growth, "Utilities", "UPI")
        if d.day == 28:
            salary = 68000 if months_in < 9 else 74000
            add(d, "Acme Technologies Salary", salary, "Income", "NetBanking", "credit")
        if d.day == 30 or (d.month == 2 and d.day == 28):
            add(d, "Savings Interest", rng.uniform(180, 320), "Income", "NetBanking", "credit")
        if rng.random() < 0.06:
            add(d, "Friend Repayment", rng.uniform(300, 3000), "Income", "UPI", "credit")

    truth_rows = []
    chosen = rng.choice(len(days), size=len(ANOMALIES), replace=False)
    for idx, (merchant, amount, category, mode) in zip(chosen, ANOMALIES):
        add(days[idx], merchant, amount, category, mode)
        truth_rows.append((days[idx], rows[-1]["description"]))

    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    truth = pd.DataFrame(truth_rows, columns=["date", "description"])

    debit_idx = df.index[(df["type"] == "debit")]
    protected = set(df.index[df["description"].isin(truth["description"])])
    candidates = np.array([i for i in debit_idx if i not in protected])

    missing_cat = rng.choice(candidates, size=int(0.07 * len(candidates)), replace=False)
    missing_mode = rng.choice(candidates, size=int(0.04 * len(candidates)), replace=False)
    missing_amt = rng.choice(candidates, size=int(0.025 * len(candidates)), replace=False)
    df.loc[missing_cat, "category"] = np.nan
    df.loc[missing_mode, "payment_mode"] = np.nan
    df.loc[missing_amt, "amount"] = np.nan

    noisy_mode = rng.random(len(df)) < 0.35
    df.loc[noisy_mode & df["payment_mode"].notna(), "payment_mode"] = [
        rng.choice(MODE_VARIANTS[m]) for m in df.loc[noisy_mode & df["payment_mode"].notna(), "payment_mode"]
    ]
    df["type"] = [rng.choice(TYPE_VARIANTS[t]) if rng.random() < 0.4 else t for t in df["type"]]
    lower_cat = (rng.random(len(df)) < 0.05) & df["category"].notna()
    df.loc[lower_cat, "category"] = df.loc[lower_cat, "category"].str.lower()

    dup_flag = rng.random(len(df)) < 0.03
    repeats = np.where(dup_flag, 2, 1)
    df = df.loc[np.repeat(df.index.to_numpy(), repeats)].reset_index(drop=True)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    raw = df.drop(columns=["merchant_true"])
    truth["date"] = truth["date"].dt.strftime("%Y-%m-%d")
    return raw, truth


if __name__ == "__main__":
    data, _ = generate_transactions()
    data.to_csv("data/sample_transactions.csv", index=False)
    print(f"{len(data)} rows written")

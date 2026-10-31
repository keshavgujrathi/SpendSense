"""Loading, cleaning and transforming raw bank statements (deduplication, value replacement,
binning, missing-data handling including maximum-likelihood imputation)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import KEYWORD_RULES, MERCHANT_CATEGORY, CATEGORIES

COLUMN_ALIASES = {
    "date": ["date", "txn date", "transaction date", "value date", "posting date", "tran date"],
    "description": ["description", "narration", "details", "particulars", "transaction details",
                    "remarks", "transaction remarks"],
    "amount": ["amount", "transaction amount", "amt"],
    "debit": ["debit", "withdrawal", "withdrawal amt.", "withdrawals", "debit amount"],
    "credit": ["credit", "deposit", "deposit amt.", "deposits", "credit amount"],
    "type": ["type", "dr/cr", "cr/dr", "txn type", "transaction type"],
    "category": ["category"],
    "payment_mode": ["payment_mode", "payment mode", "mode", "channel"],
}

MODE_ALIASES = {
    "upi": "UPI", "card": "Card", "debit card": "Card", "credit card": "Card", "pos": "Card",
    "netbanking": "NetBanking", "net banking": "NetBanking", "neft": "NetBanking",
    "imps": "NetBanking", "rtgs": "NetBanking", "cash": "Cash", "atm": "Cash",
}
TYPE_ALIASES = {
    "debit": "debit", "dr": "debit", "d": "debit", "withdrawal": "debit",
    "credit": "credit", "cr": "credit", "c": "credit", "deposit": "credit",
}
AMOUNT_BAND_EDGES = [0, 100, 500, 2000, 10000, np.inf]
AMOUNT_BAND_LABELS = ["Micro", "Small", "Medium", "Large", "Major"]
IMPUTATION_METHODS = ["drop", "mean", "median", "category_mean", "category_median", "mle_category", "mle"]


@dataclass
class CleaningReport:
    rows_raw: int = 0
    invalid_rows_dropped: int = 0
    duplicates_removed: int = 0
    labels_replaced: int = 0
    merchants_resolved: int = 0
    categories_inferred: int = 0
    categories_by_text: int = 0
    modes_inferred: int = 0
    amounts_imputed: int = 0
    rows_final: int = 0
    imputation_method: str = "mle"
    missing_before: dict = field(default_factory=dict)
    missing_after: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def load_statement(source) -> tuple[pd.DataFrame, list[str]]:
    """Read a CSV path/buffer or DataFrame and map it onto the canonical statement schema."""
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    notes: list[str] = []
    lookup = {}
    for canonical, names in COLUMN_ALIASES.items():
        for name in names:
            lookup[name] = canonical
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in lookup and lookup[key] not in renamed.values():
            renamed[col] = lookup[key]
    df = df.rename(columns=renamed)

    def to_number(series: pd.Series) -> pd.Series:
        text = series.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
        return pd.to_numeric(text, errors="coerce")

    if "amount" not in df.columns and ("debit" in df.columns or "credit" in df.columns):
        debit = to_number(df["debit"]) if "debit" in df.columns else pd.Series(np.nan, index=df.index)
        credit = to_number(df["credit"]) if "credit" in df.columns else pd.Series(np.nan, index=df.index)
        df["amount"] = debit.fillna(credit)
        df["type"] = np.where(debit.notna() & (debit > 0), "debit", "credit")
        notes.append("Separate debit/credit columns were merged into amount and type.")
    elif "amount" in df.columns:
        df["amount"] = to_number(df["amount"])

    missing = [c for c in ("date", "description", "amount") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    if "type" not in df.columns:
        if (df["amount"] < 0).any():
            df["type"] = np.where(df["amount"] < 0, "debit", "credit")
            df["amount"] = df["amount"].abs()
            notes.append("Transaction type was inferred from the sign of the amount.")
        else:
            df["type"] = "debit"
            notes.append("No type column found; all rows were treated as debits.")

    for col in ("category", "payment_mode"):
        if col not in df.columns:
            df[col] = np.nan
    return df[["date", "description", "amount", "type", "category", "payment_mode"]].copy(), notes


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    deduped = df.drop_duplicates(keep="first")
    return deduped.reset_index(drop=True), len(df) - len(deduped)


def _alias_series(series: pd.Series, aliases: dict, title_fallback: bool = False) -> tuple[pd.Series, int]:
    text = series.astype("string").str.strip()
    key = text.str.lower()
    mapped = key.map(aliases)
    if title_fallback:
        canonical = {c.lower(): c for c in CATEGORIES + ["Income", "Uncategorised"]}
        mapped = key.map(canonical)
        mapped = mapped.fillna(text.str.title())
    else:
        mapped = mapped.fillna(text)
    mapped = mapped.where(text.notna())
    changed = int((mapped.fillna("") != series.astype("string").fillna("")).sum())
    return mapped.astype(object).where(mapped.notna(), np.nan), changed


def standardise_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    total = 0
    out["type"], n = _alias_series(out["type"], TYPE_ALIASES)
    total += n
    out["payment_mode"], n = _alias_series(out["payment_mode"], MODE_ALIASES)
    total += n
    out["category"], n = _alias_series(out["category"], {}, title_fallback=True)
    total += n
    return out, total


def _build_merchant_patterns() -> list[tuple[re.Pattern, str]]:
    patterns = []
    for name in MERCHANT_CATEGORY:
        words = re.sub(r"[^A-Za-z0-9 ]", " ", name.replace("'", "")).upper().split()
        patterns.append((len(words), re.compile(r"\b" + r"\s+".join(words) + r"\b"), name))
    patterns.sort(key=lambda t: -t[0])
    return [(p, n) for _, p, n in patterns]


_MERCHANT_PATTERNS = _build_merchant_patterns()
_NOISE_TOKENS = {"UPI", "POS", "NEFT", "IMPS", "RTGS", "ATM", "WDL", "PAYMENT", "PAID", "BANGALORE",
                 "BENGALURU", "INDIA", "PVT", "LTD", "TO", "FROM", "BY", "CASH"}


def normalise_merchant(description: str) -> str:
    """Collapse a noisy bank narration into a canonical merchant name."""
    text = str(description).upper().replace("'", "")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    if text.startswith("ATM WDL"):
        return "ATM Cash Withdrawal"
    for pattern, name in _MERCHANT_PATTERNS:
        if pattern.search(text):
            return name
    tokens = [t for t in text.split() if t not in _NOISE_TOKENS and not re.search(r"\d", t)]
    return " ".join(tokens[:3]).title() if tokens else "Unknown"


def infer_category(merchant: str, description: str) -> str:
    if merchant in MERCHANT_CATEGORY:
        return MERCHANT_CATEGORY[merchant]
    text = str(description).upper()
    for keyword, category in KEYWORD_RULES:
        if keyword in text:
            return category
    return "Uncategorised"


def infer_payment_mode(description: str) -> str | None:
    text = str(description).upper()
    if text.startswith("UPI") or "UPI/" in text:
        return "UPI"
    if text.startswith("POS") or " POS " in text:
        return "Card"
    if text.startswith(("NEFT", "IMPS", "RTGS")):
        return "NetBanking"
    if text.startswith(("ATM", "CASH")):
        return "Cash"
    return None


def _lognormal_mean(obs: pd.DataFrame, col: str, min_n: int = 4) -> pd.Series:
    """Maximum-likelihood lognormal fit per group; returns the fitted median exp(mu)."""
    positive = obs[obs["amount"] > 0].copy()
    positive["log_amount"] = np.log(positive["amount"])
    grouped = positive.groupby(col)["log_amount"]
    mu = grouped.mean()
    sigma2 = grouped.var(ddof=0)
    count = grouped.size()
    fitted = np.exp(mu)
    return fitted[count >= min_n]


def _estimates(obs: pd.DataFrame, target: pd.DataFrame, method: str) -> pd.Series:
    global_mean, global_median = obs["amount"].mean(), obs["amount"].median()
    if method == "mean":
        return pd.Series(global_mean, index=target.index)
    if method == "median":
        return pd.Series(global_median, index=target.index)
    if method in ("category_mean", "category_median"):
        stat = method.split("_")[1]
        table = obs.groupby("category")["amount"].agg(stat)
        return target["category"].map(table).fillna(obs["amount"].agg(stat))
    if method in ("mle", "mle_category"):
        positive = obs[obs["amount"] > 0]
        logs = np.log(positive["amount"])
        fallback = float(np.exp(logs.mean()))
        by_category = _lognormal_mean(obs, "category")
        estimate = target["category"].map(by_category)
        if method == "mle":
            estimate = target["merchant"].map(_lognormal_mean(obs, "merchant")).fillna(estimate)
        return estimate.fillna(fallback)
    raise ValueError(f"Unknown imputation method: {method}")


def impute_amounts(df: pd.DataFrame, method: str = "mle") -> pd.DataFrame:
    out = df.copy()
    missing = out["amount"].isna()
    out["amount_imputed"] = False
    if not missing.any():
        return out
    if method == "drop":
        return out[~missing].reset_index(drop=True)
    estimates = _estimates(out[~missing], out[missing], method)
    out.loc[missing, "amount"] = estimates.round(2)
    out.loc[missing, "amount_imputed"] = True
    return out


def evaluate_imputation(df: pd.DataFrame, fraction: float = 0.1, repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Hide known amounts, re-estimate them with each method and report the error."""
    known = df[(df["type"] == "debit") & df["amount"].notna()]
    if "amount_imputed" in known:
        known = known[~known["amount_imputed"]]
    methods = [m for m in IMPUTATION_METHODS if m != "drop"]
    scores = {m: [] for m in methods}
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        hidden = rng.choice(known.index, size=int(len(known) * fraction), replace=False)
        train = known.drop(index=hidden)
        test = known.loc[hidden]
        for method in methods:
            error = test["amount"] - _estimates(train, test, method)
            scores[method].append((error.abs().mean(), np.sqrt((error ** 2).mean()),
                                   (error.abs() / test["amount"]).median()))
    rows = []
    for method, values in scores.items():
        mae, rmse, mdape = np.mean(values, axis=0)
        rows.append(dict(method=method, MAE=mae, RMSE=rmse, median_abs_pct_error=mdape))
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def bin_amounts(amount: pd.Series, strategy: str = "fixed", bins: int = 5) -> pd.Series:
    if strategy == "fixed":
        return pd.cut(amount, AMOUNT_BAND_EDGES, labels=AMOUNT_BAND_LABELS, right=False)
    if strategy == "quantile":
        return pd.qcut(amount, bins, duplicates="drop")
    if strategy == "equal_width":
        return pd.cut(amount, bins)
    raise ValueError(f"Unknown binning strategy: {strategy}")


def missing_summary(df: pd.DataFrame) -> dict:
    return {c: int(df[c].isna().sum()) for c in ("date", "description", "amount", "type", "category", "payment_mode")}


def clean_transactions(raw: pd.DataFrame, impute_method: str = "mle") -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_raw=len(raw), imputation_method=impute_method)
    df = raw.copy()
    report.missing_before = missing_summary(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed", dayfirst=True)
    before = len(df)
    df = df[df["date"].notna() & df["description"].notna()]
    report.invalid_rows_dropped = before - len(df)
    df["date"] = df["date"].dt.normalize()

    df, report.duplicates_removed = remove_duplicates(df)
    df, report.labels_replaced = standardise_labels(df)

    df["merchant"] = df["description"].map(normalise_merchant)
    report.merchants_resolved = int(df["merchant"].isin(MERCHANT_CATEGORY).sum())

    need_category = df["category"].isna()
    inferred = [infer_category(m, d) for m, d in zip(df["merchant"], df["description"])]
    credit = df["type"].eq("credit")
    inferred = pd.Series(np.where(credit, "Income", inferred), index=df.index)
    df.loc[need_category, "category"] = inferred[need_category]
    report.categories_inferred = int(need_category.sum())

    need_mode = df["payment_mode"].isna()
    guessed = df["description"].map(infer_payment_mode)
    category_mode = df.groupby("category")["payment_mode"].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    fallback = df["category"].map(category_mode).fillna("UPI")
    df.loc[need_mode, "payment_mode"] = guessed[need_mode].fillna(fallback[need_mode])
    report.modes_inferred = int(need_mode.sum())

    report.amounts_imputed = int(df["amount"].isna().sum())
    df = impute_amounts(df, impute_method)
    df["type"] = df["type"].fillna("debit")

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["day_of_month"] = df["date"].dt.day
    df["amount_band"] = bin_amounts(df["amount"], "fixed")

    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    report.rows_final = len(df)
    report.missing_after = missing_summary(df)
    return df, report

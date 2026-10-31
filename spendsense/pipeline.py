"""End-to-end preparation shared by the Streamlit app, the tests and the report builder."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .clustering import categorise_by_clusters, detect_outliers
from .generator import generate_transactions
from .preprocessing import CleaningReport, clean_transactions, load_statement


@dataclass
class Prepared:
    raw: pd.DataFrame
    clean: pd.DataFrame
    report: CleaningReport
    notes: list
    truth: pd.DataFrame | None = None


def prepare(source, impute_method: str = "mle", outlier_method: str = "gmm",
            contamination: float = 0.01, truth: pd.DataFrame | None = None) -> Prepared:
    raw, notes = load_statement(source)
    clean, report = clean_transactions(raw, impute_method)
    unresolved = int((clean["category"] == "Uncategorised").sum())
    if unresolved:
        clean = categorise_by_clusters(clean)
        report.categories_by_text = unresolved - int((clean["category"] == "Uncategorised").sum())
    clean = detect_outliers(clean, outlier_method, contamination)
    return Prepared(raw=raw, clean=clean, report=report, notes=notes, truth=truth)


def prepare_sample(**kwargs) -> Prepared:
    raw, truth = generate_transactions()
    return prepare(raw, truth=truth, **kwargs)

import numpy as np

from spendsense.analysis import chi_square_test, contingency_table, correlation_matrix
from spendsense.bandit import nudge_table, run_experiment
from spendsense.clustering import (categorise_by_clusters, compare_methods, evaluate_outliers, merchant_features,
                                   select_k)
from spendsense.insights import budget_review, detect_recurring, kpis
from spendsense.modeling import backtest_errors, backtest_monthly, evaluate, fit_spend_model, forecast_next
from spendsense.reduction import (category_matrix, intrinsic_dimensionality, run_pca, run_svd, standardise,
                                  train_som)
from spendsense.timeseries import adf_test, daily_spend, resample_spend


def test_cleaning_removes_all_gaps(prepared):
    clean = prepared.clean
    assert clean[["amount", "category", "payment_mode"]].isna().sum().sum() == 0
    assert prepared.report.duplicates_removed > 0
    assert set(clean["type"]) == {"debit", "credit"}


def test_outlier_detection_finds_injected_anomalies(prepared):
    result = evaluate_outliers(prepared.clean, prepared.truth)
    assert result["recall"] >= 0.85 and result["precision"] >= 0.7


def test_contingency_and_correlation(prepared):
    table = contingency_table(prepared.clean)
    assert chi_square_test(table)["p_value"] < 0.05
    corr = correlation_matrix(prepared.clean)
    assert np.allclose(np.diag(corr), 1.0)


def test_time_series_helpers(prepared):
    daily = daily_spend(prepared.clean, exclude_outliers=True)
    assert daily.index.is_monotonic_increasing and (daily >= 0).all()
    assert resample_spend(daily, "M").shape[0] >= 17
    assert "p_value" in adf_test(daily)


def test_clustering_methods(prepared):
    feats, X = merchant_features(prepared.clean)
    table, labels = compare_methods(X, 4)
    assert len(table) == 5 and all(len(set(v)) == 4 for v in labels.values())
    assert len(select_k(X)) > 3


def test_text_clustering_fills_unknown_categories(prepared):
    frame = prepared.clean.copy()
    mask = frame.sample(40, random_state=1).index
    frame.loc[mask, "category"] = np.nan
    filled = categorise_by_clusters(frame)
    assert filled.loc[mask, "category"].notna().mean() > 0.5


def test_dimensionality_reduction(prepared):
    matrix = category_matrix(prepared.clean)
    X = standardise(matrix)
    pca = run_pca(X, matrix.columns)
    assert np.isclose(pca["cumulative"][-1], 1.0)
    svd = run_svd(X)
    assert svd["reconstruction_error"][-1] < 1e-8
    assert intrinsic_dimensionality(X)["variance_90"] <= X.shape[1]
    assert train_som(X, (4, 4), 500).shape == (4, 4, X.shape[1])


def test_regression_and_forecast(prepared):
    daily = daily_spend(prepared.clean, exclude_outliers=True)
    recurring = detect_recurring(prepared.clean)
    model = fit_spend_model(daily, recurring)
    assert model["daily"]["R2"] > 0.5 > model["baseline"]["R2"]
    assert evaluate([1, 2, 3], [1, 2, 3])["accuracy"] == 1.0
    backtest = backtest_monthly(daily, recurring)
    assert backtest_errors(backtest)["MAPE_%"].min() < 25
    assert forecast_next(daily, recurring)["total"] > 0


def test_insights(prepared):
    info = kpis(prepared.clean)
    assert info["income"] > 0 and info["spend"] > 0
    assert {"House Rent", "Netflix"} <= set(detect_recurring(prepared.clean)["merchant"])
    assert not budget_review(prepared.clean).empty


def test_bandit_learns_better_than_random(prepared):
    spend = prepared.clean[prepared.clean["type"] == "debit"].groupby("category")["amount"].sum() / 18
    table = nudge_table(spend)
    result = run_experiment(table["success_prob"].to_numpy(), rounds=300, runs=15)
    final = result["regret"].iloc[-1]
    assert final["Thompson sampling"] < final["Random"] and final["UCB1"] < final["Random"]

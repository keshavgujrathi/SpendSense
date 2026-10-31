"""Runs the full SpendSense analysis on the sample statement and saves the figures used in the report."""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from spendsense import analysis, bandit, clustering, insights, modeling, preprocessing, reduction, timeseries  # noqa: E402
from spendsense.config import CATEGORY_COLORS  # noqa: E402
from spendsense.pipeline import prepare_sample  # noqa: E402

FIG_DIR = ROOT / "report" / "figures"
BLUE, ORANGE, GREEN, RED, GREY = "#2F6FDE", "#E8743B", "#19A979", "#D1495B", "#8A97A8"
sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                     "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 11})
FIGS: dict[str, Path] = {}


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    FIGS[name] = path


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, color, size=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=color,
                                    ec="#1B2430", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color="#1B2430")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.3, color="#1B2430"))

    box(2, 31, 17, 10, "Bank statement\nCSV upload or\nsynthetic generator", "#e8f0fd")
    box(25, 31, 17, 10, "Loader\nschema mapping,\ndate and amount parsing", "#e8f0fd")
    box(48, 31, 22, 10, "Data transformation\ndeduplicate, replace values,\nbin, impute (MLE)", "#fdeee6")
    box(76, 31, 21, 10, "Clustering-based\noutlier detection\nand text categorisation", "#fdeee6")
    arrow(19.5, 36, 24.5, 36)
    arrow(42.5, 36, 47.5, 36)
    arrow(70.5, 36, 75.5, 36)
    arrow(86, 30.5, 86, 24)
    ax.text(50, 26.2, "Analysis engine (spendsense package)", ha="center", fontsize=9.5, weight="bold")
    items = [("Explore\nuni, bi, multi-variate\ncontingency", 3), ("Time series\nresample, decompose\nADF, ACF", 22),
             ("Clustering\nK-Means, spectral,\nEM, MST, ward", 41), ("Dimensionality\nPCA, SVD, FA,\nMDS, SOM", 60),
             ("Model and RL\nregression, forecast,\nbandit nudges", 79)]
    for text, x in items:
        box(x, 10, 18, 11, text, "#e6f6f0", 8.5)
    box(22, 0.8, 56, 5.5, "Streamlit dashboard: nine sections, interactive Plotly charts", "#f1e9fb")
    for _, x in items:
        arrow(x + 9, 9.6, 50, 6.8)
    save(fig, "architecture")


def build_all() -> dict:
    P = prepare_sample()
    clean, report, truth = P.clean, P.report, P.truth
    spend = insights.debits(clean)
    daily = timeseries.daily_spend(clean, exclude_outliers=True)
    recurring = insights.detect_recurring(clean)
    cat_summary = analysis.category_summary(clean)
    facts: dict = {"clean": clean, "report": report, "cat_summary": cat_summary, "recurring": recurring}
    facts["date_min"], facts["date_max"] = clean["date"].min(), clean["date"].max()
    facts["n_final"] = len(clean)
    facts["n_debit"] = int((clean["type"] == "debit").sum())
    facts["n_merchants"] = clean["merchant"].nunique()

    architecture()

    # Module 2: data preparation
    miss = pd.DataFrame({"before": report.missing_before, "after": report.missing_after})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    miss.plot.bar(ax=axes[0], color=[ORANGE, GREEN], width=0.75)
    axes[0].set_title("Missing values per column")
    axes[0].set_ylabel("rows")
    axes[0].tick_params(axis="x", rotation=30)
    stages = ["Raw rows", "After removing\nduplicates", "After value\nstandardisation", "Final cleaned\nrows"]
    values = [report.rows_raw, report.rows_raw - report.invalid_rows_dropped - report.duplicates_removed,
              report.rows_raw - report.invalid_rows_dropped - report.duplicates_removed, report.rows_final]
    axes[1].bar(stages, values, color=[GREY, BLUE, BLUE, GREEN])
    for i, v in enumerate(values):
        axes[1].text(i, v + 10, f"{v:,}", ha="center", fontsize=9)
    axes[1].set_title("Row count through the cleaning stages")
    axes[1].set_ylim(0, max(values) * 1.12)
    fig.tight_layout()
    save(fig, "cleaning")
    facts["missing_before"], facts["missing_after"] = report.missing_before, report.missing_after

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, (strategy, title) in zip(axes, [("equal_width", "Equal width (5 bins)"), ("quantile", "Equal frequency (5 bins)"),
                                            ("fixed", "Domain bands")]):
        counts = preprocessing.bin_amounts(spend["amount"], strategy).value_counts().sort_index()
        labels = [str(i) if strategy == "fixed" else f"B{n + 1}" for n, i in enumerate(counts.index)]
        ax.bar(labels, counts.values, color=BLUE)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("transactions")
    fig.tight_layout()
    save(fig, "binning")
    facts["band_counts"] = preprocessing.bin_amounts(spend["amount"], "fixed").value_counts().sort_index()
    facts["equal_width_top_share"] = float(
        preprocessing.bin_amounts(spend["amount"], "equal_width").value_counts(normalize=True).max())

    imputation = preprocessing.evaluate_imputation(clean)
    facts["imputation"] = imputation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    order = imputation.sort_values("MAE")
    axes[0].barh(order["method"], order["MAE"], color=[GREEN if m == "mle" else GREY for m in order["method"]])
    axes[0].invert_yaxis()
    axes[0].set_title("Imputation error on hidden amounts (MAE, rupees)")
    category = "Shopping"
    values = spend[(spend["category"] == category) & ~spend["amount_imputed"]]["amount"]
    mu, sigma = np.log(values).mean(), np.log(values).std(ddof=0)
    grid = np.linspace(values.min(), values.quantile(0.97), 300)
    axes[1].hist(values[values <= values.quantile(0.97)], bins=30, density=True, color="#b9c9e8", label="observed")
    axes[1].plot(grid, stats.lognorm.pdf(grid, s=sigma, scale=np.exp(mu)), color=RED, lw=2.5, label="MLE lognormal")
    axes[1].set_title(f"Maximum-likelihood lognormal fit: {category}")
    axes[1].set_xlabel("amount (rupees)")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "imputation")
    facts["mle_mu"], facts["mle_sigma"] = float(mu), float(sigma)

    text_eval = clustering.evaluate_document_clustering(clean)
    facts["text_eval"] = text_eval

    # Module 3 and 4: exploratory analysis
    amounts = spend["amount"]
    facts["spend_stats"] = analysis.summary_stats(amounts)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].hist(amounts, bins=60, color=BLUE)
    axes[0].set_title("Amount distribution")
    axes[0].set_xlabel("rupees")
    axes[1].hist(np.log10(amounts), bins=40, color=GREEN)
    axes[1].set_title("Log10 amount distribution")
    axes[1].set_xlabel("log10(rupees)")
    top = cat_summary.index[:8]
    sns.boxplot(data=spend[spend["category"].isin(top)], y="category", x="amount", order=top, ax=axes[2],
                color="#b9c9e8", fliersize=2)
    axes[2].set_xscale("log")
    axes[2].set_title("Spread by category (log scale)")
    axes[2].set_ylabel("")
    fig.tight_layout()
    save(fig, "univariate")

    corr = analysis.correlation_matrix(clean, "spearman")
    facts["top_corr"] = analysis.top_correlations(corr, 6)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".1f", annot_kws={"size": 7}, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Spearman correlation of weekly category spend")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "correlation")

    weekday = analysis.spend_by_weekday_category(clean)
    weekday = weekday.loc[[c for c in cat_summary.index if c in weekday.index]]
    profile = timeseries.weekday_profile(daily)
    facts["weekday_profile"] = profile
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].bar([d[:3] for d in profile.index], profile["mean"], color=[BLUE] * 5 + [ORANGE] * 2)
    axes[0].set_title("Average daily spend by weekday")
    axes[0].set_ylabel("rupees")
    sns.heatmap(weekday, cmap="Blues", ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Average transaction by category and weekday")
    axes[1].set_xticklabels([d[:3] for d in weekday.columns], rotation=0)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.tight_layout()
    save(fig, "weekday")
    facts["group_test"] = analysis.group_difference(clean, "payment_mode")

    table = analysis.contingency_table(clean)
    facts["chi"] = analysis.chi_square_test(table)
    facts["contingency"] = table
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    sns.heatmap(table.loc[[c for c in cat_summary.index if c in table.index]], annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar=False)
    ax.set_title("Contingency table: category by payment mode (debits)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "contingency")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    sample = spend.sample(min(len(spend), 1500), random_state=1)
    for cat, g in sample.groupby("category"):
        axes[0].scatter(g["day_of_month"], g["amount"], s=9, alpha=0.6, color=CATEGORY_COLORS[cat])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("day of month")
    axes[0].set_ylabel("amount (log)")
    axes[0].set_title("Scatter plot")
    ordered = cat_summary.sort_values("sum")
    axes[1].hlines(ordered.index, 0, ordered["sum"], color="#cfd8e6")
    axes[1].plot(ordered["sum"], ordered.index, "o", color=BLUE)
    axes[1].set_title("Dot chart: total spend")
    axes[1].set_xlabel("rupees")
    merchants = insights.top_merchants(clean, 10).sort_values("total")
    axes[2].barh(merchants.index, merchants["total"], color=[CATEGORY_COLORS[c] for c in merchants["category"]])
    axes[2].set_title("Bar plot: top merchants")
    axes[2].set_xlabel("rupees")
    fig.tight_layout()
    save(fig, "visual_toolkit")

    # Module 3: time series
    monthly = timeseries.monthly_table(clean)
    monthly = monthly[monthly["income"] > 0]
    facts["monthly"] = monthly
    weekly = timeseries.resample_spend(daily, "W")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2))
    axes[0].plot(daily.index, daily.values, color="#b9c9e8", lw=0.8, label="daily")
    axes[0].plot(daily.index, timeseries.rolling_mean(daily, 28), color=RED, lw=2, label="28-day mean")
    axes[0].set_title("Daily typical spend with rolling mean")
    axes[0].set_ylabel("rupees")
    axes[0].legend(loc="upper left")
    x = np.arange(len(monthly))
    axes[1].bar(x - 0.2, monthly["income"], 0.4, color="#9db8ee", label="income")
    axes[1].bar(x + 0.2, monthly["spend"], 0.4, color=BLUE, label="spend")
    ax2 = axes[1].twinx()
    ax2.plot(x, monthly["savings_rate_pct"], color=ORANGE, marker="o", lw=2, label="savings rate")
    ax2.set_ylabel("savings rate (%)")
    ax2.grid(False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m[2:] for m in monthly.index], rotation=45, fontsize=8)
    axes[1].set_title("Monthly resample: income, spend and savings rate")
    axes[1].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "timeseries")
    facts["weekly_mean"], facts["weekly_std"] = float(weekly.mean()), float(weekly.std())

    parts = timeseries.decompose(daily)
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    for ax, (name, values) in zip(axes, [("Observed", parts.observed), ("Trend", parts.trend),
                                         ("Weekly seasonality", parts.seasonal), ("Residual", parts.resid)]):
        ax.plot(values.index, values.values, color=BLUE, lw=1)
        ax.set_ylabel(name, fontsize=9)
    fig.suptitle("Additive decomposition of daily spend (period = 7 days)", fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "decomposition")
    facts["seasonal_range"] = float(parts.seasonal.max() - parts.seasonal.min())
    facts["seasonal_peak_day"] = parts.seasonal.iloc[:7].idxmax().day_name()
    facts["adf"] = timeseries.adf_test(daily)
    ac = timeseries.autocorrelation(daily, 35)
    facts["acf7"], facts["acf1"] = float(ac[7]), float(ac[1])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(ac)), ac, color=BLUE)
    ax.axhline(1.96 / np.sqrt(len(daily)), color=RED, ls="--", lw=1)
    ax.axhline(-1.96 / np.sqrt(len(daily)), color=RED, ls="--", lw=1)
    ax.set_xlabel("lag (days)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Autocorrelation of daily spend")
    save(fig, "acf")

    # Module 5: clustering
    feats, X = clustering.merchant_features(clean)
    facts["n_feat_merchants"] = len(feats)
    ks = clustering.select_k(X)
    facts["select_k"] = ks
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].plot(ks["k"], ks["silhouette"], "o-", color=BLUE)
    axes[0].set_title("Silhouette score")
    axes[1].plot(ks["k"], ks["inertia"], "o-", color=GREEN)
    axes[1].set_title("K-Means inertia (elbow)")
    axes[2].plot(ks["k"], ks["bic"], "o-", color=ORANGE)
    axes[2].set_title("GMM BIC")
    for ax in axes:
        ax.set_xlabel("clusters (k)")
    fig.tight_layout()
    save(fig, "cluster_selection")

    k = 4
    comparison, label_sets = clustering.compare_methods(X, k)
    facts["cluster_comparison"] = comparison
    labels = label_sets["K-Means"]
    names = clustering.name_clusters(feats, labels)
    facts["cluster_profile"] = clustering.cluster_profile(feats, labels, names)
    coords = PCA(n_components=2).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    palette = [BLUE, ORANGE, GREEN, "#945ECF"]
    for i, cluster in enumerate(sorted(set(labels))):
        mask = labels == cluster
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=48, color=palette[i], label=names[cluster], alpha=0.85)
    for i, merchant in enumerate(feats.index):
        if feats.loc[merchant, "txn_count"] > 45 or feats.loc[merchant, "avg_amount"] > 5000:
            axes[0].annotate(merchant, coords[i], fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_title("K-Means merchant personas (PCA projection)")
    axes[0].legend(fontsize=7, loc="best")
    dendrogram(clustering.linkage_matrix(X), labels=list(feats.index), orientation="right", ax=axes[1],
               leaf_font_size=6.5, color_threshold=0.7 * clustering.linkage_matrix(X)[-1, 2])
    axes[1].set_title("Ward hierarchical dendrogram")
    fig.tight_layout()
    save(fig, "personas")

    outlier_eval = {}
    for method in ("gmm", "kmeans", "dbscan"):
        flagged = clustering.detect_outliers(clean.drop(columns=["is_outlier", "outlier_score"]), method, 0.01)
        outlier_eval[method] = clustering.evaluate_outliers(flagged, truth)
    facts["outlier_eval"] = outlier_eval
    debit = clean[clean["type"] == "debit"]
    fig, ax = plt.subplots(figsize=(10, 4))
    normal = debit[~debit["is_outlier"]]
    odd = debit[debit["is_outlier"]]
    ax.scatter(normal["date"], normal["amount"], s=7, color="#b9c9e8", label="typical")
    ax.scatter(odd["date"], odd["amount"], s=55, color=RED, label="flagged outlier", edgecolor="white", zorder=3)
    for _, row in odd.iterrows():
        ax.annotate(row["merchant"], (row["date"], row["amount"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_title("Clustering-based outlier detection (Gaussian mixture, EM)")
    ax.set_ylabel("amount (rupees, log scale)")
    ax.legend()
    save(fig, "outliers")
    facts["n_outliers"] = int(clean["is_outlier"].sum())
    facts["outlier_total"] = float(odd["amount"].sum())

    # Module 6: dimensionality reduction
    matrix = reduction.category_matrix(clean, "W")
    Xs = reduction.standardise(matrix)
    facts["matrix_shape"] = Xs.shape
    pca = reduction.run_pca(Xs, matrix.columns)
    facts["pca"] = {"ratio": pca["ratio"], "n_90": pca["n_90"], "cum": pca["cumulative"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    comps = np.arange(1, len(pca["ratio"]) + 1)
    axes[0].bar(comps, pca["ratio"], color="#b9c9e8")
    axes[0].plot(comps, pca["cumulative"], "o-", color=RED)
    axes[0].axhline(0.9, color=GREY, ls="--", lw=1)
    axes[0].set_title("Scree plot and cumulative variance")
    axes[0].set_xlabel("component")
    sns.heatmap(pca["loadings"].iloc[:, :4], cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7},
                ax=axes[1], cbar=False)
    axes[1].set_title("Loadings of the first four components")
    fig.tight_layout()
    save(fig, "pca")

    svd = reduction.run_svd(Xs)
    facts["svd_rank90"] = int(np.searchsorted(np.cumsum(svd["energy"]), 0.90) + 1)
    facts["svd_err"] = svd["reconstruction_error"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].bar(np.arange(1, len(svd["s"]) + 1), svd["s"], color=BLUE)
    axes[0].set_title("Singular values")
    axes[0].set_xlabel("index")
    axes[1].plot(np.arange(1, len(svd["s"]) + 1), svd["reconstruction_error"], "o-", color=ORANGE)
    axes[1].set_title("Relative reconstruction error by rank")
    axes[1].set_xlabel("rank")
    fig.tight_layout()
    save(fig, "svd")

    fa = reduction.run_factor_analysis(Xs, matrix.columns, 3)
    facts["fa_loadings"] = fa["loadings"]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    sns.heatmap(fa["loadings"], cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax, cbar=False)
    ax.set_title("Varimax-rotated factor loadings")
    save(fig, "factor")
    facts["intrinsic"] = reduction.intrinsic_dimensionality(Xs)

    comparison_embed, embeddings = reduction.embedding_comparison(Xs)
    facts["embedding"] = comparison_embed
    quarters = matrix.index.to_period("Q").astype(str)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    qcolors = dict(zip(sorted(set(quarters)), sns.color_palette("viridis", len(set(quarters)))))
    for ax, (name, emb) in zip(axes, embeddings.items()):
        ax.scatter(emb[:, 0], emb[:, 1], s=16, c=[qcolors[q] for q in quarters])
        score = comparison_embed.set_index("method").loc[name, "trustworthiness"]
        ax.set_title(f"{name} (T={score:.2f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "embeddings")

    weights = reduction.train_som(Xs)
    u = reduction.som_u_matrix(weights)
    hits = np.zeros(weights.shape[:2])
    for i, j in reduction.som_bmus(Xs, weights):
        hits[i, j] += 1
    facts["som_qe"] = reduction.quantisation_error(Xs, weights)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    sns.heatmap(u, cmap="viridis", ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("SOM U-matrix")
    sns.heatmap(hits, cmap="Blues", annot=True, fmt=".0f", ax=axes[1], cbar=False)
    axes[1].set_title("SOM hit map")
    fig.tight_layout()
    save(fig, "som")

    # Module 7: model, forecast and reinforcement learning
    model = modeling.fit_spend_model(daily, recurring)
    facts["model"] = model
    pred = model["weekly_predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].plot(pred.index, pred["actual"], "o-", color=BLUE, label="actual")
    axes[0].plot(pred.index, pred["predicted"], "s--", color=ORANGE, label="predicted")
    axes[0].set_title("Weekly spend on the held-out last 90 days")
    axes[0].set_ylabel("rupees")
    axes[0].legend()
    residual = model["predictions"]["actual"] - model["predictions"]["predicted"]
    axes[1].hist(residual, bins=25, color=GREEN)
    axes[1].set_title("Daily residuals")
    axes[1].set_xlabel("rupees")
    fig.tight_layout()
    save(fig, "regression")

    backtest = modeling.backtest_monthly(daily, recurring)
    facts["backtest"], facts["backtest_errors"] = backtest, modeling.backtest_errors(backtest)
    forecast = modeling.forecast_next(daily, recurring)
    facts["forecast"] = forecast
    history = daily.iloc[-120:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    axes[0].plot(history.index, timeseries.rolling_mean(history, 7), color=BLUE, label="history (7-day mean)")
    axes[0].plot(forecast["dates"], forecast["ensemble"].rolling(7, min_periods=1).mean(), color=ORANGE, ls="--",
                 label="forecast (7-day mean)")
    axes[0].set_title("Daily spend and 30-day forecast")
    axes[0].legend()
    idx = np.arange(len(backtest))
    width = 0.2
    for i, col in enumerate(["actual", "naive", "holt_winters", "linear_regression"]):
        axes[1].bar(idx + (i - 1.5) * width, backtest[col], width, label=col.replace("_", " "))
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(backtest.index, fontsize=8)
    axes[1].set_title("Rolling backtest of monthly totals")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "forecast")

    monthly_spend = spend.groupby("category")["amount"].sum() / spend["month"].nunique()
    nudges = bandit.nudge_table(monthly_spend)
    facts["nudges"] = nudges
    experiment = bandit.run_experiment(nudges["success_prob"].to_numpy(), 400, 60)
    facts["bandit"] = experiment
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = {"Random": GREY, "Epsilon-greedy": BLUE, "UCB1": GREEN, "Thompson sampling": ORANGE}
    for policy in bandit.POLICIES:
        axes[0].plot(experiment["regret"][policy], label=policy, color=colors[policy], lw=2)
    axes[0].set_title("Cumulative regret (60 runs)")
    axes[0].set_xlabel("round")
    axes[0].legend(fontsize=8)
    pulls = experiment["pulls"].copy()
    pulls.index = [f"Tip {i + 1}" for i in range(len(pulls))]
    pulls[["Random", "Thompson sampling"]].plot.bar(ax=axes[1], color=[GREY, ORANGE], width=0.75)
    axes[1].set_title("Share of suggestions per tip")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save(fig, "bandit")

    facts["kpis"] = insights.kpis(clean)
    facts["budget"] = insights.budget_review(clean)
    facts["figs"] = FIGS
    return facts


if __name__ == "__main__":
    out = build_all()
    print({k: str(v)[:60] for k, v in out["figs"].items()})

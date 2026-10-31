"""SpendSense: personal expense intelligence built on exploratory data analysis."""
import io
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA

from spendsense import analysis, bandit, clustering, insights, modeling, preprocessing, reduction, timeseries
from spendsense.config import CATEGORY_COLORS, CURRENCY
from spendsense.pipeline import prepare
from spendsense.generator import generate_transactions

warnings.filterwarnings("ignore")
px.defaults.template = "plotly_white"

st.set_page_config(page_title="SpendSense", page_icon="💸", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {background: #f6f8fb; border: 1px solid #e3e8ef; padding: 12px 16px; border-radius: 10px;}
    div[data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #55637a; white-space: normal;}
    div[data-testid="stMetricValue"] {font-size: 1.55rem;}
    div[data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
    div[data-testid="stMetricDelta"] {white-space: normal;}
    </style>
    """,
    unsafe_allow_html=True,
)


def money(x: float) -> str:
    return f"{CURRENCY}{x:,.0f}"


def show(fig, height: int | None = None):
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner="Preparing data...")
def load_prepared(kind: str, payload: bytes | None, impute: str, outlier: str, contamination: float):
    if kind == "sample":
        raw, truth = generate_transactions()
        return prepare(raw, impute, outlier, contamination, truth=truth)
    return prepare(io.BytesIO(payload), impute, outlier, contamination)


@st.cache_data(show_spinner="Computing embeddings...")
def cached_embeddings(X: np.ndarray):
    return reduction.embedding_comparison(X)


@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(daily: pd.Series, recurring: pd.DataFrame):
    return modeling.backtest_monthly(daily, recurring)


@st.cache_data
def cached_imputation_eval(clean: pd.DataFrame):
    return preprocessing.evaluate_imputation(clean)


@st.cache_data(show_spinner="Clustering descriptions...")
def cached_text_eval(clean: pd.DataFrame):
    return clustering.evaluate_document_clustering(clean)


def template_csv() -> bytes:
    raw, _ = generate_transactions(months=3)
    return raw[["date", "description", "amount", "type"]].head(25).to_csv(index=False).encode()


st.sidebar.title("💸 SpendSense")
st.sidebar.caption("Personal expense intelligence")
source = st.sidebar.radio("Data source", ["Sample statement", "Upload my CSV"])
upload = None
if source == "Upload my CSV":
    upload = st.sidebar.file_uploader("Bank statement (CSV)", type="csv")
    st.sidebar.download_button("Download CSV template", template_csv(), "statement_template.csv", "text/csv")
    st.sidebar.caption("Needs date, description and amount columns. Type, category and payment mode are optional. "
                       "Data stays in this session and is never stored.")
impute = st.sidebar.selectbox("Missing-amount strategy", preprocessing.IMPUTATION_METHODS, index=6,
                              help="Applied to transactions whose amount is blank.")
outlier_method = st.sidebar.selectbox("Outlier detector", ["gmm", "kmeans", "dbscan"],
                                      format_func={"gmm": "Gaussian mixture (EM)", "kmeans": "K-Means distance",
                                                   "dbscan": "DBSCAN noise"}.get)
contamination = st.sidebar.slider("Outlier sensitivity", 0.005, 0.05, 0.01, 0.005, format="%.3f")

if source == "Upload my CSV" and upload is None:
    st.title("Upload a bank statement")
    st.info("Choose a CSV in the sidebar, or switch to the sample statement to explore the app.")
    st.stop()

try:
    P = load_prepared("sample" if source == "Sample statement" else "upload",
                      None if upload is None else upload.getvalue(), impute, outlier_method, contamination)
except Exception as exc:
    st.error(f"The statement could not be processed: {exc}")
    st.stop()

clean, report = P.clean, P.report
if clean.empty or clean["date"].nunique() < 60:
    st.warning("At least two months of transactions are needed for the full analysis.")
    st.stop()

PAGES = ["Overview", "Data preparation", "Explore", "Time series", "Patterns and anomalies",
         "Dimensionality", "Forecast and model", "Smart nudges", "About"]
page = st.sidebar.radio("Section", PAGES)
st.sidebar.divider()
st.sidebar.caption("BCSE331L · Exploratory Data Analysis\n\nKeshav Gujrathi · 23BDS0225\n\nGuide: Dr. Vijayanand C.")

spend_df = insights.debits(clean)
daily = timeseries.daily_spend(clean, exclude_outliers=True)
recurring = insights.detect_recurring(clean)
cat_order = analysis.category_summary(clean).index.tolist()


def page_overview():
    st.title("Where is my money going?")
    k = insights.kpis(clean)
    st.caption(f"Latest complete month: {k['month']} · {len(clean):,} transactions analysed "
               f"from {clean['date'].min():%d %b %Y} to {clean['date'].max():%d %b %Y}")
    c = st.columns(5)
    c[0].metric("Spend", money(k["spend"]), None if np.isnan(k["spend_change_pct"]) else f"{k['spend_change_pct']:+.0f}% vs prior",
                delta_color="inverse")
    c[1].metric("Income", money(k["income"]))
    c[2].metric("Savings rate", "n/a" if np.isnan(k["savings_rate"]) else f"{k['savings_rate']:.0f}%")
    c[3].metric("Top category", k["top_category"], money(k["top_category_amount"]), delta_color="off")
    c[4].metric("One-off big spends", money(k["one_off_spend"]), help="Flagged as outliers and excluded from typical spend")

    for note in insights.plain_language_insights(clean):
        st.markdown(f"- {note}")

    monthly = timeseries.monthly_table(clean)
    monthly = monthly[monthly["income"] > 0]
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_bar(x=monthly.index, y=monthly["income"], name="Income", marker_color="#9db8ee")
        fig.add_bar(x=monthly.index, y=monthly["spend"], name="Spend", marker_color="#2F6FDE")
        fig.add_scatter(x=monthly.index, y=monthly["savings_rate_pct"], name="Savings rate %", yaxis="y2",
                        mode="lines+markers", line=dict(color="#E8743B", width=2))
        fig.update_layout(title="Income, spend and savings rate", barmode="group",
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, ticksuffix="%"),
                          xaxis=dict(type="category"), legend=dict(orientation="h", y=-0.2))
        show(fig, 380)
    with right:
        latest = spend_df[spend_df["month"] == k["month"]].groupby("category")["amount"].sum().reset_index()
        fig = px.pie(latest, names="category", values="amount", hole=0.55, color="category",
                     color_discrete_map=CATEGORY_COLORS, title=f"Category mix · {k['month']}")
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
        show(fig, 380)

    by_month = spend_df.groupby(["month", "category"])["amount"].sum().reset_index()
    fig = px.bar(by_month, x="month", y="amount", color="category", color_discrete_map=CATEGORY_COLORS,
                 category_orders={"category": cat_order}, title="Monthly spend by category")
    fig.update_xaxes(type="category")
    show(fig, 380)

    a, b = st.columns(2)
    with a:
        st.subheader("Recurring payments")
        if recurring.empty:
            st.write("No steady monthly payments detected.")
        else:
            view = recurring[["merchant", "category", "typical_amount", "next_expected", "annual_cost"]].copy()
            view["next_expected"] = view["next_expected"].dt.strftime("%d %b %Y")
            st.dataframe(view.rename(columns={"typical_amount": "amount", "annual_cost": "per year"}).round(0),
                         hide_index=True, use_container_width=True)
    with b:
        st.subheader("Unusual transactions")
        flagged = clean[clean["is_outlier"]][["date", "merchant", "category", "amount"]].sort_values("amount", ascending=False)
        if flagged.empty:
            st.write("Nothing unusual flagged.")
        else:
            flagged = flagged.assign(date=flagged["date"].dt.strftime("%d %b %Y"))
            st.dataframe(flagged.round(0), hide_index=True, use_container_width=True)

    st.subheader("Budget check for the latest month")
    review = insights.budget_review(clean)
    if not review.empty:
        view = review[["type", "latest", "typical", "suggested_cap", "status"]]
        st.dataframe(view.style.apply(lambda r: ["background-color: #fde8e8" if r["status"] == "Over" else "" for _ in r], axis=1),
                     use_container_width=True)
        st.caption("Suggested cap is the median of the previous three months, trimmed by 5% for discretionary categories.")


def page_preparation():
    st.title("Data preparation")
    st.write("Raw statements are messy. Every step below is applied automatically to the data you loaded.")
    c = st.columns(6)
    c[0].metric("Raw rows", f"{report.rows_raw:,}")
    c[1].metric("Duplicates removed", report.duplicates_removed)
    c[2].metric("Labels standardised", report.labels_replaced)
    c[3].metric("Categories inferred", report.categories_inferred)
    c[4].metric("Modes inferred", report.modes_inferred)
    c[5].metric("Amounts imputed", report.amounts_imputed)
    for note in P.notes:
        st.info(note)

    tabs = st.tabs(["Missing data", "Replacing values", "Binning", "Imputation and MLE", "Text clustering"])
    with tabs[0]:
        miss = pd.DataFrame({"before": report.missing_before, "after": report.missing_after}).reset_index()
        miss = miss.melt(id_vars="index", var_name="stage", value_name="missing").rename(columns={"index": "column"})
        fig = px.bar(miss, x="column", y="missing", color="stage", barmode="group", title="Missing values per column",
                     color_discrete_map={"before": "#E8743B", "after": "#19A979"})
        show(fig, 340)
        st.write("Categories are recovered from the merchant dictionary and keyword rules, payment modes from the "
                 "narration pattern (UPI, POS, NEFT, ATM), and missing amounts with the selected strategy.")
    with tabs[1]:
        sample = P.raw.assign(merchant=P.raw["description"].map(preprocessing.normalise_merchant)).sample(
            min(12, len(P.raw)), random_state=3)[["description", "merchant"]]
        st.markdown("**Noisy narration to canonical merchant**")
        st.dataframe(sample, hide_index=True, use_container_width=True)
        a, b = st.columns(2)
        before = P.raw["payment_mode"].fillna("(missing)").value_counts().rename("rows").reset_index()
        after = clean["payment_mode"].value_counts().rename("rows").reset_index()
        a.plotly_chart(px.bar(before, x="payment_mode", y="rows", title="Payment mode labels, raw"), use_container_width=True)
        b.plotly_chart(px.bar(after, x="payment_mode", y="rows", title="Payment mode labels, cleaned"), use_container_width=True)
    with tabs[2]:
        strategy = st.radio("Binning strategy", ["fixed", "quantile", "equal_width"], horizontal=True,
                            format_func={"fixed": "Domain bands", "quantile": "Equal frequency",
                                         "equal_width": "Equal width"}.get)
        amounts = spend_df["amount"]
        bands = preprocessing.bin_amounts(amounts, strategy)
        counts = bands.value_counts().sort_index()
        counts.index = counts.index.astype(str)
        fig = px.bar(x=counts.index, y=counts.values, labels={"x": "band", "y": "transactions"},
                     title="Transactions per amount band")
        show(fig, 340)
        st.caption("Equal-width bins collapse almost every purchase into the first bin because spending is heavily "
                   "right-skewed. Domain bands (Micro to Major) are far more informative.")
    with tabs[3]:
        evaluation = cached_imputation_eval(clean)
        st.markdown("**Hide 10% of known amounts, re-estimate them, compare the error**")
        a, b = st.columns([2, 3])
        a.dataframe(evaluation.round(2), hide_index=True, use_container_width=True)
        fig = px.bar(evaluation, x="method", y="MAE", title="Mean absolute error by strategy", color="method")
        fig.update_layout(showlegend=False)
        with b:
            show(fig, 320)
        category = st.selectbox("Inspect the lognormal fit for", cat_order)
        values = spend_df[(spend_df["category"] == category) & ~spend_df["amount_imputed"]]["amount"]
        values = values[values > 0]
        if len(values) >= 5:
            mu, sigma = np.log(values).mean(), np.log(values).std(ddof=0)
            grid = np.linspace(values.min(), values.quantile(0.98), 200)
            fig = px.histogram(values[values <= values.quantile(0.98)], nbins=30, histnorm="probability density",
                               title=f"{category}: maximum-likelihood lognormal fit (mu={mu:.2f}, sigma={sigma:.2f})")
            fig.add_scatter(x=grid, y=stats.lognorm.pdf(grid, s=sigma, scale=np.exp(mu)), mode="lines",
                            name="MLE fit", line=dict(color="#D1495B", width=3))
            show(fig, 340)
    with tabs[4]:
        result = cached_text_eval(clean)
        st.write("Descriptions are vectorised with TF-IDF and grouped with K-Means. Each cluster adopts the majority "
                 "category of its labelled members, which lets the app categorise merchants it has never seen.")
        if report.categories_by_text:
            st.info(f"{report.categories_by_text} transactions with unknown merchants were categorised this way.")
        c = st.columns(3)
        c[0].metric("Hidden labels", result["hidden"])
        c[1].metric("Coverage", f"{result['coverage']:.0%}")
        c[2].metric("Accuracy where assigned", f"{result['accuracy']:.0%}")


def page_explore():
    st.title("Explore")
    tabs = st.tabs(["Univariate", "Bivariate", "Contingency and n-D", "Visual toolkit", "Summary and scales"])
    with tabs[0]:
        choice = st.selectbox("Amounts for", ["All spending"] + cat_order)
        data = spend_df if choice == "All spending" else spend_df[spend_df["category"] == choice]
        log_scale = st.checkbox("Logarithmic x-axis", value=True)
        a, b = st.columns(2)
        fig = px.histogram(data, x="amount", nbins=50, log_x=log_scale, title="Distribution")
        a.plotly_chart(fig, use_container_width=True)
        fig = px.box(data, x="amount", log_x=log_scale, points="outliers", title="Box plot")
        b.plotly_chart(fig, use_container_width=True)
        summary = analysis.summary_stats(data["amount"]).round(2).rename("value").to_frame()
        st.dataframe(summary.T, use_container_width=True)
        st.caption("Skewness well above zero and a mean far above the median confirm right-skewed spending.")
    with tabs[1]:
        a, b = st.columns(2)
        profile = timeseries.weekday_profile(daily).reset_index()
        a.plotly_chart(px.bar(profile, x="weekday", y="mean", title="Average daily spend by weekday"),
                       use_container_width=True)
        grid = analysis.spend_by_weekday_category(clean)
        grid = grid.loc[[c for c in cat_order if c in grid.index]]
        b.plotly_chart(px.imshow(grid, color_continuous_scale="Blues", aspect="auto",
                                 title="Average transaction by category and weekday"), use_container_width=True)
        method = st.radio("Correlation method", ["spearman", "pearson"], horizontal=True)
        corr = analysis.correlation_matrix(clean, method)
        fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                        title=f"Weekly category spend, {method} correlation")
        show(fig, 560)
        st.dataframe(analysis.top_correlations(corr).round(2), hide_index=True, use_container_width=True)
        test = analysis.group_difference(clean, "payment_mode")
        st.caption(f"Kruskal-Wallis across payment modes: H = {test['H']:.1f}, p = {test['p_value']:.3g}")
    with tabs[2]:
        a, b = st.columns(2)
        row = a.selectbox("Rows", ["category", "payment_mode", "weekday", "amount_band"], index=0)
        col = b.selectbox("Columns", ["payment_mode", "weekday", "amount_band", "category"], index=0)
        if row == col:
            st.warning("Choose two different variables.")
        else:
            table = analysis.contingency_table(clean, row, col)
            st.dataframe(table, use_container_width=True)
            test = analysis.chi_square_test(table)
            st.write(f"Chi-square = {test['chi2']:.1f}, dof = {test['dof']}, p = {test['p_value']:.3g}, "
                     f"Cramér's V = {test['cramers_v']:.2f}")
            show(px.imshow(table, color_continuous_scale="Blues", aspect="auto", text_auto=True,
                           title="Contingency heatmap"), 480)
        st.markdown("**Three-way table: total spend by category, payment mode and weekend flag**")
        st.dataframe(analysis.multi_way_table(clean, ["category"], ["payment_mode", "is_weekend"]),
                     use_container_width=True)
    with tabs[3]:
        a, b = st.columns(2)
        sample = spend_df.sample(min(len(spend_df), 1500), random_state=1)
        fig = px.scatter(sample, x="day_of_month", y="amount", color="category", log_y=True,
                         color_discrete_map=CATEGORY_COLORS, opacity=0.7, title="Scatter: amount against day of month")
        a.plotly_chart(fig, use_container_width=True)
        totals = analysis.category_summary(clean).reset_index().sort_values("sum")
        fig = px.scatter(totals, x="sum", y="category", title="Dot chart: total spend per category",
                         color_discrete_sequence=["#2F6FDE"])
        fig.update_traces(marker=dict(size=12))
        b.plotly_chart(fig, use_container_width=True)
        merchants = insights.top_merchants(clean, 12).reset_index().sort_values("total")
        fig = px.bar(merchants, x="total", y="merchant", orientation="h", color="category",
                     color_discrete_map=CATEGORY_COLORS, title="Bar plot: top merchants")
        show(fig, 420)
    with tabs[4]:
        st.markdown("**Statistical summary by category**")
        st.dataframe(analysis.category_summary(clean).round(1), use_container_width=True)
        st.markdown("**Data types and measurement scales**")
        st.dataframe(analysis.MEASUREMENT_SCALES, hide_index=True, use_container_width=True)


def page_timeseries():
    st.title("Time series")
    a, b, c = st.columns(3)
    rule = a.selectbox("Aggregate by", ["D", "W", "M"], index=1, format_func={"D": "Day", "W": "Week", "M": "Month"}.get)
    window = b.slider("Rolling window (periods)", 2, 30, 4 if rule == "W" else 7)
    span = c.date_input("Date range", (daily.index.min().date(), daily.index.max().date()))
    series = daily
    if len(span) == 2:
        series = daily.loc[pd.Timestamp(span[0]):pd.Timestamp(span[1])]
    agg = timeseries.resample_spend(series, rule)
    if rule == "M" and len(agg) > 0:
        agg = agg.iloc[:-1] if series.index.max().day < series.index.max().days_in_month else agg
    fig = go.Figure()
    fig.add_bar(x=agg.index, y=agg.values, name="Spend", marker_color="#9db8ee")
    fig.add_scatter(x=agg.index, y=timeseries.rolling_mean(agg, window), name=f"{window}-period mean",
                    line=dict(color="#D1495B", width=3))
    fig.update_layout(title="Spend over time (typical spend, outliers excluded)", legend=dict(orientation="h", y=-0.2))
    show(fig, 380)

    left, right = st.columns(2)
    with left:
        profile = timeseries.weekday_profile(series).reset_index()
        show(px.bar(profile, x="weekday", y="mean", title="Weekday pattern"), 320)
    with right:
        cats = spend_df.set_index("date").groupby([pd.Grouper(freq="ME"), "category"])["amount"].sum().reset_index()
        show(px.area(cats, x="date", y="amount", color="category", color_discrete_map=CATEGORY_COLORS,
                     title="Grouped by month and category"), 320)

    st.subheader("Decomposition and diagnostics")
    parts = timeseries.decompose(daily)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["Observed", "Trend", "Weekly seasonality", "Residual"])
    for i, (name, values) in enumerate([("obs", parts.observed), ("trend", parts.trend), ("seasonal", parts.seasonal),
                                         ("resid", parts.resid)], start=1):
        fig.add_scatter(x=values.index, y=values, mode="lines", row=i, col=1, showlegend=False,
                        line=dict(color="#2F6FDE", width=1.5))
    show(fig, 640)
    a, b = st.columns(2)
    adf = timeseries.adf_test(daily)
    a.metric("Augmented Dickey-Fuller p-value", f"{adf['p_value']:.4f}",
             "stationary" if adf["stationary"] else "non-stationary", delta_color="off")
    ac = timeseries.autocorrelation(daily, 35)
    b.plotly_chart(px.bar(x=np.arange(len(ac)), y=ac, labels={"x": "lag (days)", "y": "autocorrelation"},
                          title="Autocorrelation function"), use_container_width=True)
    monthly = timeseries.monthly_table(clean)
    st.markdown("**Month-by-month summary**")
    st.dataframe(monthly.round(1), use_container_width=True)


def page_patterns():
    st.title("Patterns and anomalies")
    feats, X = clustering.merchant_features(clean)
    if len(feats) < 8:
        st.warning("Not enough distinct merchants for clustering.")
        return
    st.subheader("Merchant spending personas")
    ks = clustering.select_k(X)
    a, b = st.columns(2)
    a.plotly_chart(px.line(ks, x="k", y="silhouette", markers=True, title="Silhouette by number of clusters"),
                   use_container_width=True)
    b.plotly_chart(px.line(ks, x="k", y="bic", markers=True, title="GMM BIC by number of clusters"),
                   use_container_width=True)
    c1, c2 = st.columns(2)
    k = c1.slider("Number of clusters", 2, min(8, len(feats) - 1), 4)
    method = c2.selectbox("Method", ["K-Means", "Spectral", "GMM (EM)", "Hierarchical (Ward)", "MST"])
    table, label_sets = clustering.compare_methods(X, k)
    st.dataframe(table.round(3), hide_index=True, use_container_width=True)
    labels = label_sets[method]
    names = clustering.name_clusters(feats, labels)
    coords = PCA(n_components=2).fit_transform(X)
    plot = pd.DataFrame(coords, columns=["x", "y"], index=feats.index)
    plot["persona"] = [names[c] for c in labels]
    plot["merchant"] = feats.index
    plot["avg_amount"] = feats["avg_amount"].round(0)
    fig = px.scatter(plot, x="x", y="y", color="persona", hover_name="merchant", hover_data=["avg_amount"],
                     title=f"{method} personas (PCA projection of merchant behaviour)")
    fig.update_traces(marker=dict(size=11))
    show(fig, 440)
    st.dataframe(clustering.cluster_profile(feats, labels, names), use_container_width=True)
    with st.expander("Hierarchical dendrogram"):
        dendro = ff.create_dendrogram(X, labels=list(feats.index), linkagefun=lambda m: linkage(m, "ward"))
        show(dendro, 520)

    st.subheader("Outlier detection")
    flagged = clean[clean["is_outlier"]]
    st.write(f"{len(flagged)} transactions flagged by **{outlier_method}**. Features are the log amount and a "
             "category-relative robust z-score; only unusually large debits are flagged.")
    debit = clean[clean["type"] == "debit"]
    fig = px.scatter(debit, x="date", y="amount", log_y=True, color="is_outlier",
                     color_discrete_map={False: "#b9c4d6", True: "#D1495B"}, hover_data=["merchant", "category"],
                     title="Debits over time, outliers highlighted")
    show(fig, 420)
    if P.truth is not None:
        score = clustering.evaluate_outliers(clean, P.truth)
        d = st.columns(4)
        d[0].metric("Injected anomalies", score["true_anomalies"])
        d[1].metric("Caught", score["caught"])
        d[2].metric("Precision", f"{score['precision']:.0%}")
        d[3].metric("Recall", f"{score['recall']:.0%}")
    st.dataframe(flagged[["date", "merchant", "category", "amount", "outlier_score"]].round(2)
                 .sort_values("amount", ascending=False), hide_index=True, use_container_width=True)


def page_dimensionality():
    st.title("Dimensionality reduction")
    freq = st.radio("Observation window", ["W", "2W"], horizontal=True, format_func={"W": "Weekly", "2W": "Fortnightly"}.get)
    matrix = reduction.category_matrix(clean, freq)
    if len(matrix) < 20 or matrix.shape[1] < 4:
        st.warning("Not enough periods or categories for this section.")
        return
    X = reduction.standardise(matrix)
    st.caption(f"{X.shape[0]} periods described by {X.shape[1]} spending categories (log-scaled and standardised).")
    tabs = st.tabs(["PCA", "SVD", "Factor analysis", "Intrinsic dimension", "Non-linear methods"])
    pca = reduction.run_pca(X, matrix.columns)
    with tabs[0]:
        a, b = st.columns(2)
        comps = np.arange(1, len(pca["ratio"]) + 1)
        fig = go.Figure()
        fig.add_bar(x=comps, y=pca["ratio"], name="Variance")
        fig.add_scatter(x=comps, y=pca["cumulative"], name="Cumulative", mode="lines+markers")
        fig.update_layout(title="Explained variance", xaxis_title="component", legend=dict(orientation="h", y=-0.25))
        a.plotly_chart(fig, use_container_width=True)
        quarter = matrix.index.to_period("Q").astype(str)
        scatter = pd.DataFrame(pca["scores"][:, :2], columns=["PC1", "PC2"]).assign(quarter=quarter)
        b.plotly_chart(px.scatter(scatter, x="PC1", y="PC2", color="quarter", title="Periods in PC space"),
                       use_container_width=True)
        st.write(f"{pca['n_90']} of {X.shape[1]} components are needed to retain 90% of the variance.")
        show(px.imshow(pca["loadings"].iloc[:, :4], color_continuous_scale="RdBu_r", zmin=-0.7, zmax=0.7,
                       aspect="auto", title="Loadings of the first four components"), 460)
    with tabs[1]:
        svd = reduction.run_svd(X)
        a, b = st.columns(2)
        a.plotly_chart(px.bar(x=np.arange(1, len(svd["s"]) + 1), y=svd["s"],
                              labels={"x": "index", "y": "singular value"}, title="Singular values"),
                       use_container_width=True)
        b.plotly_chart(px.line(x=np.arange(1, len(svd["s"]) + 1), y=svd["reconstruction_error"], markers=True,
                               labels={"x": "rank", "y": "relative error"}, title="Reconstruction error by rank"),
                       use_container_width=True)
        rank = st.slider("Rank", 1, len(svd["s"]), min(5, len(svd["s"])))
        st.write(f"A rank-{rank} approximation keeps {svd['energy'][:rank].sum():.0%} of the energy with a relative "
                 f"error of {svd['reconstruction_error'][rank - 1]:.2f}.")
    with tabs[2]:
        n = st.slider("Number of factors", 2, 5, 3)
        fa = reduction.run_factor_analysis(X, matrix.columns, n)
        show(px.imshow(fa["loadings"], color_continuous_scale="RdBu_r", zmin=-0.8, zmax=0.8, aspect="auto",
                       title="Varimax-rotated factor loadings"), 460)
        st.dataframe(fa["communality"].round(2).to_frame().T, use_container_width=True)
        st.caption("Factors group categories that rise and fall together, such as staying in (delivery, "
                   "groceries) versus going out (dining, entertainment, transport).")
    with tabs[3]:
        dims = reduction.intrinsic_dimensionality(X)
        table = pd.DataFrame({"estimator": ["Components for 90% variance", "Components for 95% variance",
                                            "Kaiser criterion (eigenvalue > 1)", "Levina-Bickel local MLE"],
                              "dimension": [dims["variance_90"], dims["variance_95"], dims["kaiser"],
                                            round(dims["levina_bickel"], 1)]})
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.write(f"The data has {dims['ambient']} observed variables, but the estimators suggest a much smaller "
                 "number of independent behaviours.")
    with tabs[4]:
        comparison, embeddings = cached_embeddings(X)
        a, b = st.columns([2, 3])
        a.dataframe(comparison.round(3), hide_index=True, use_container_width=True)
        a.caption("Trustworthiness measures how well local neighbourhoods are preserved in 2-D.")
        method = b.selectbox("Embedding", list(embeddings))
        emb = pd.DataFrame(embeddings[method], columns=["x", "y"]).assign(
            quarter=matrix.index.to_period("Q").astype(str))
        show(px.scatter(emb, x="x", y="y", color="quarter", title=f"{method} embedding"), 420)
        weights = reduction.train_som(X)
        u = reduction.som_u_matrix(weights)
        a, b = st.columns(2)
        a.plotly_chart(px.imshow(u, color_continuous_scale="Viridis", title="SOM U-matrix (cluster boundaries)"),
                       use_container_width=True)
        bmu = reduction.som_bmus(X, weights)
        hits = np.zeros(weights.shape[:2])
        for i, j in bmu:
            hits[i, j] += 1
        b.plotly_chart(px.imshow(hits, color_continuous_scale="Blues", text_auto=True, title="SOM hit map"),
                       use_container_width=True)
        st.caption(f"Quantisation error: {reduction.quantisation_error(X, weights):.2f}")


def page_forecast():
    st.title("Forecast and model")
    model = modeling.fit_spend_model(daily, recurring)
    tabs = st.tabs(["Next 30 days", "Linear regression", "Understanding accuracy"])
    with tabs[0]:
        fc = modeling.forecast_next(daily, recurring)
        c = st.columns(3)
        c[0].metric("Expected spend, next 30 days", money(fc["total"]))
        c[1].metric("Likely range (80%)", f"{money(max(fc['low'], 0))} to {money(fc['high'])}")
        recent = daily.iloc[-30:].sum()
        c[2].metric("Last 30 days", money(recent), f"{(fc['total'] / recent - 1) * 100:+.0f}% expected", delta_color="off")
        history = daily.iloc[-120:]
        fig = go.Figure()
        fig.add_scatter(x=history.index, y=timeseries.rolling_mean(history, 7), name="History (7-day mean)",
                        line=dict(color="#2F6FDE"))
        fig.add_scatter(x=fc["dates"], y=fc["ensemble"].rolling(7, min_periods=1).mean(), name="Forecast (7-day mean)",
                        line=dict(color="#E8743B", dash="dash"))
        fig.update_layout(title="Daily spend and forecast", legend=dict(orientation="h", y=-0.2))
        show(fig, 360)
        backtest = cached_backtest(daily, recurring)
        if backtest is not None:
            st.markdown("**Rolling backtest on the last complete months**")
            errors = modeling.backtest_errors(backtest)
            a, b = st.columns(2)
            a.dataframe(backtest.round(0), use_container_width=True)
            b.dataframe(errors.round(1), hide_index=True, use_container_width=True)
    with tabs[1]:
        d, w, base = model["daily"], model["weekly"], model["baseline"]
        c = st.columns(4)
        c[0].metric("Daily R²", f"{d['R2']:.2f}", f"baseline {base['R2']:.2f}", delta_color="off")
        c[1].metric("Daily MAE", money(d["MAE"]))
        c[2].metric("Weekly R²", f"{w['R2']:.2f}")
        c[3].metric("Weekly accuracy", f"{w['accuracy']:.0%}")
        pred = model["weekly_predictions"]
        fig = go.Figure()
        fig.add_scatter(x=pred.index, y=pred["actual"], name="Actual", mode="lines+markers")
        fig.add_scatter(x=pred.index, y=pred["predicted"], name="Predicted", mode="lines+markers")
        fig.update_layout(title="Weekly spend on the held-out last 90 days", legend=dict(orientation="h", y=-0.2))
        show(fig, 340)
        coefs = model["coefficients"].round(3)
        coefs["significant"] = np.where(coefs["p_value"] < 0.05, "yes", "")
        st.dataframe(coefs, use_container_width=True)
        residual = model["predictions"]["actual"] - model["predictions"]["predicted"]
        show(px.histogram(residual, nbins=30, title="Residuals on held-out days"), 300)
    with tabs[2]:
        st.markdown(
            """
            - **R²** is the share of variance the model explains. Zero matches simply predicting the average.
            - **MAE** is the typical miss in rupees; **RMSE** punishes large misses more.
            - **Accuracy** here is 1 minus weighted MAPE, so 70% means total absolute error is 30% of total spend.
            - Daily spend is naturally noisy, so accuracy improves sharply when predictions are summed to weeks or months.
            """
        )
        cv = model["cv"].mean().round(3).rename("time-series cross-validation (mean of 5 folds)").to_frame()
        st.dataframe(cv.T, use_container_width=True)


def page_nudges():
    st.title("Smart nudges")
    st.write("Which saving tip should the app suggest first? Each tip is an arm of a multi-armed bandit. "
             "The app tries a tip, observes whether it worked, and gradually favours the best one.")
    monthly_spend = spend_df.groupby("category")["amount"].sum() / max(spend_df["month"].nunique(), 1)
    table = bandit.nudge_table(monthly_spend)
    if table.empty:
        st.warning("No eligible spending categories found.")
        return
    view = table[["text", "category", "monthly_spend", "accept", "potential_saving", "expected_saving"]].copy()
    view.columns = ["tip", "category", "avg monthly spend", "acceptance rate", "potential saving", "expected saving"]
    st.dataframe(view.round(2), hide_index=True, use_container_width=True)
    st.caption("Acceptance rates are modelled assumptions used to simulate user feedback. In a live product they would "
               "be learned from real responses.")
    a, b, c = st.columns(3)
    rounds = a.slider("Rounds", 100, 1000, 400, 50)
    runs = b.slider("Simulation runs", 10, 100, 40, 10)
    epsilon = c.slider("Epsilon (exploration)", 0.01, 0.3, 0.1, 0.01)
    result = bandit.run_experiment(table["success_prob"].to_numpy(), rounds, runs, epsilon)
    regret = result["regret"].reset_index().melt(id_vars="index", var_name="policy", value_name="regret")
    show(px.line(regret, x="index", y="regret", color="policy", labels={"index": "round", "regret": "cumulative regret"},
                 title="Cumulative regret (lower is better)"), 380)
    pulls = result["pulls"]
    pulls.index = [t[:38] + ("..." if len(t) > 38 else "") for t in table["text"]]
    pulls = pulls.reset_index().melt(id_vars="index", var_name="policy", value_name="share")
    show(px.bar(pulls, x="share", y="index", color="policy", barmode="group", orientation="h",
                labels={"index": "", "share": "share of suggestions"}, title="How often each tip is suggested"), 460)
    best = table.iloc[0]
    st.success(f"Start with: **{best['text']}**. Expected saving about {money(best['expected_saving'])} per month.")
    with st.expander("Supervised learning versus reinforcement learning"):
        st.markdown(
            """
            | | Supervised learning | Reinforcement learning |
            |---|---|---|
            | Feedback | Correct label for every example | Reward only for the action taken |
            | Data | Fixed, historical | Generated by the agent's own choices |
            | Challenge | Generalise to unseen data | Balance exploration and exploitation |
            | In this app | Regression on daily spend | Choosing which nudge to show |
            """
        )


def page_about():
    st.title("About SpendSense")
    st.write("SpendSense turns a raw bank statement into decisions: where money goes, what repeats every month, "
             "what looks unusual, and what next month is likely to cost. It is the course project for "
             "BCSE331L Exploratory Data Analysis by Keshav Gujrathi (23BDS0225) under Dr. Vijayanand C.")
    coverage = pd.DataFrame([
        ("1", "Data types, scales, EDA workflow, tools", "Explore > Summary and scales", "analysis.py"),
        ("2", "Deduplication, value replacement, binning", "Data preparation", "preprocessing.py"),
        ("2", "Missing data and maximum likelihood estimation", "Data preparation > Imputation and MLE", "preprocessing.py"),
        ("3", "Univariate, bivariate, multivariate analysis", "Explore", "analysis.py"),
        ("3", "Time-based indexing, resampling, decomposition", "Time series", "timeseries.py"),
        ("4", "Summaries, contingency tables, scatter, dot, bar plots", "Explore", "analysis.py"),
        ("5", "Spectral, EM/GMM, hierarchical, MST, document clustering", "Patterns and anomalies", "clustering.py"),
        ("5", "Outlier detection using clustering", "Patterns and anomalies", "clustering.py"),
        ("6", "PCA, SVD, factor analysis, intrinsic dimension", "Dimensionality", "reduction.py"),
        ("6", "MDS, Isomap, t-SNE, self-organising map", "Dimensionality > Non-linear methods", "reduction.py"),
        ("7", "Linear regression, evaluation, accuracy", "Forecast and model", "modeling.py"),
        ("7", "Reinforcement learning (bandits)", "Smart nudges", "bandit.py"),
    ], columns=["module", "topic", "where in the app", "source"])
    st.dataframe(coverage, hide_index=True, use_container_width=True)
    st.subheader("Using your own statement")
    st.markdown(
        "Upload a CSV with a date, a description and an amount. Separate debit and credit columns are merged "
        "automatically, and signed amounts are supported. Processing happens in the current session only."
    )


{"Overview": page_overview, "Data preparation": page_preparation, "Explore": page_explore,
 "Time series": page_timeseries, "Patterns and anomalies": page_patterns, "Dimensionality": page_dimensionality,
 "Forecast and model": page_forecast, "Smart nudges": page_nudges, "About": page_about}[page]()

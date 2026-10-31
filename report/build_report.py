"""Builds the project report (Word) from the live analysis results.

Usage: python report/build_report.py --repo URL --video URL --app URL
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import ROOT, build_all  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="https://github.com/<username>/SpendSense")
parser.add_argument("--video", default="https://drive.google.com/<demo-video-link>")
parser.add_argument("--app", default="https://<app-name>.streamlit.app")
parser.add_argument("--out", default=str(ROOT / "report" / "SpendSense_Project_Report.docx"))
args = parser.parse_args()

F = build_all()
R = F["report"]
doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.5)
    s.top_margin = s.bottom_margin = Cm(2.3)

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.25
for name, size in (("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 12)):
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0x1B, 0x2A, 0x49)
    st.paragraph_format.space_before = Pt(14 if name != "Heading 1" else 18)
    st.paragraph_format.space_after = Pt(6)

counter = {"fig": 0, "tab": 0}


def para(text, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, after=None):
    p = doc.add_paragraph()
    p.alignment = align
    run = p.add_run(text)
    run.bold, run.italic = bold, italic
    if size:
        run.font.size = Pt(size)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if isinstance(item, tuple):
            p.add_run(item[0]).bold = True
            p.add_run(item[1])
        else:
            p.add_run(item)


def heading(text, level=1):
    return doc.add_heading(text, level)


def shade(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(10)


def figure(name, text, width=16):
    counter["fig"] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(F["figs"][name]), width=Cm(width))
    caption(f"Figure {counter['fig']}: {text}")


def table(df, text, widths=None, size=10):
    counter["tab"] += 1
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = True
    r = cap.add_run(f"Table {counter['tab']}: {text}")
    r.italic = True
    r.font.size = Pt(10.5)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, col in enumerate(df.columns):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(str(col))
        run.bold = True
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(255, 255, 255)
        shade(c, "1B2A49")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            para_ = cells[i].paragraphs[0]
            para_.paragraph_format.space_after = Pt(0)
            para_.paragraph_format.line_spacing = 1.0
            run = para_.add_run(str(v))
            run.font.size = Pt(size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def money(x):
    return f"Rs. {x:,.0f}"


def pb():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- title page
para("VELLORE INSTITUTE OF TECHNOLOGY", bold=True, size=15, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
para("School of Computer Science and Engineering", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
para("BCSE331L - Exploratory Data Analysis", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
para("Course Project Report", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=40)
para("SpendSense", bold=True, size=30, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
para("Personal Expense Intelligence through Exploratory Data Analysis", italic=True, size=14,
     align=WD_ALIGN_PARAGRAPH.CENTER, after=50)
for label, value in (("Student", "Keshav Gujrathi"), ("Registration number", "23BDS0225"),
                     ("Course", "BCSE331L - Exploratory Data Analysis"), ("Faculty guide", "Dr. Vijayanand C.")):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"{label}: ").bold = True
    p.add_run(value)
    p.paragraph_format.space_after = Pt(3)
pb()

heading("Project Links", 1)
para("The repository, the demonstration video and the hosted application are accessible to anyone with the link.")
table(pd.DataFrame([("GitHub repository", args.repo), ("Demonstration video", args.video),
                    ("Hosted application", args.app)], columns=["Resource", "Link"]), "Project resources",
      widths=[4.5, 11.5], size=11)

heading("Declaration", 1)
para("I hereby declare that this report titled \"SpendSense: Personal Expense Intelligence through Exploratory Data "
     "Analysis\" is a record of the work carried out by me for the course BCSE331L - Exploratory Data Analysis under "
     "the guidance of Dr. Vijayanand C. All analyses reported here were produced by the accompanying source code, "
     "and every figure and number in this document can be regenerated by running it.")
para("Keshav Gujrathi (23BDS0225)", bold=True, align=WD_ALIGN_PARAGRAPH.RIGHT)

# ---------------------------------------------------------------- abstract
heading("Abstract", 1)
para(f"Most people can tell roughly what they earn but not where it goes. Bank statements hold the answer, yet they "
     f"are long, noisy and full of cryptic narrations, duplicate entries and blanks. SpendSense is a web application "
     f"that applies the complete exploratory data analysis workflow to a personal bank statement and converts it into "
     f"decisions: where the money goes, which payments repeat, which transactions look unusual, what the next month "
     f"will probably cost and which saving tip is worth trying first.")
para(f"The system was developed and evaluated on an 18-month synthetic statement of {R.rows_raw:,} raw rows that "
     f"deliberately contains the defects found in real exports. The pipeline removed {R.duplicates_removed} duplicates, "
     f"standardised {R.labels_replaced} inconsistent labels, resolved {R.merchants_resolved:,} narrations to canonical "
     f"merchants and filled every missing category, payment mode and amount. Maximum-likelihood imputation gave the "
     f"lowest error among six strategies. A Gaussian-mixture detector recovered "
     f"{F['outlier_eval']['gmm']['caught']} of {F['outlier_eval']['gmm']['true_anomalies']} injected anomalies with "
     f"precision {F['outlier_eval']['gmm']['precision']:.0%}. A linear regression anchored on known recurring bills "
     f"explained {F['model']['daily']['R2']:.0%} of the variance in held-out daily spend and predicted weekly spend "
     f"with {F['model']['weekly']['accuracy']:.0%} accuracy. A multi-armed bandit identified the most valuable "
     f"budgeting nudge with far lower regret than random suggestion.")
para("Keywords: exploratory data analysis, data transformation, maximum likelihood estimation, time series, clustering, "
     "dimensionality reduction, linear regression, reinforcement learning, personal finance.", italic=True)
pb()

heading("Contents", 1)
toc = ["1. Introduction", "2. Foundations of Exploratory Data Analysis", "3. System Design", "4. Dataset",
       "5. Data Transformation", "6. Correlation Analysis and Time Series", "7. Data Summarisation and Visualisation",
       "8. Clustering and Outlier Detection", "9. Dimensionality Reduction", "10. Model Development and Evaluation",
       "11. Application and Deployment", "12. Contemporary Issues", "13. Results and Discussion",
       "14. Conclusion and Future Work", "References", "Appendix A: Syllabus Coverage",
       "Appendix B: Repository Structure"]
for item in toc:
    para(item, align=WD_ALIGN_PARAGRAPH.LEFT, after=2)
pb()

# ---------------------------------------------------------------- 1 introduction
heading("1. Introduction", 1)
heading("1.1 Problem statement", 2)
para("Personal money management is a daily problem. Salaried people and students alike discover at month end that "
     "savings fell short, without knowing which habits caused it. Banking apps list transactions but offer little "
     "analysis, and spreadsheet tracking demands manual categorisation that few sustain. Three questions come up "
     "repeatedly: where did the money go, what is going to be charged again, and is anything wrong?")
heading("1.2 Objectives", 2)
bullets(["Clean a raw, noisy statement automatically and report exactly what was repaired.",
         "Describe spending through univariate, bivariate and multivariate analysis with proper statistical tests.",
         "Expose time structure: trend, weekly rhythm, month-end effects and recurring payments.",
         "Group merchants into behavioural personas and flag suspicious transactions using clustering.",
         "Reduce the many spending categories to a few underlying behaviours.",
         "Forecast the next 30 days and quantify accuracy honestly.",
         "Recommend which saving tip to try first using reinforcement learning."])
heading("1.3 Scope and contributions", 2)
para("The project covers Modules 1 to 7 of the syllabus end to end in a single working application. Its main "
     "contributions are a robust statement loader that accepts common Indian bank layouts, a transformation pipeline "
     "that treats missing data with maximum-likelihood estimation, a clustering-based approach to both categorisation "
     "and anomaly detection, a regression forecaster that exploits recurring bills, and a bandit module that "
     "personalises advice. Everything runs in the browser session; no statement is stored.")

# ---------------------------------------------------------------- 2 foundations
heading("2. Foundations of Exploratory Data Analysis", 1)
heading("2.1 Steps in EDA and how SpendSense follows them", 2)
table(pd.DataFrame([
    ("1. Define the question", "Where does money go, what repeats, what is unusual, what comes next"),
    ("2. Load and inspect", "Loader maps varied column names to date, description, amount, type"),
    ("3. Clean and transform", "Deduplicate, replace values, bin, impute (Section 5)"),
    ("4. Summarise and visualise", "Statistics, contingency tables, plots (Sections 6 and 7)"),
    ("5. Discover structure", "Clustering, outliers, dimensionality reduction (Sections 8 and 9)"),
    ("6. Model and communicate", "Regression, forecast, bandit nudges, dashboard (Sections 10 and 11)"),
], columns=["EDA step", "Realisation in SpendSense"]), "EDA workflow mapped to the project", widths=[5, 11])
heading("2.2 Data types and measurement scales", 2)
para("Numerical data may be discrete (a count such as transactions per day) or continuous (an amount in rupees). "
     "Categorical data is nominal (merchant, category, payment mode) or ordinal (amount band, weekday). The four "
     "measurement scales apply to the dataset as follows.")
from spendsense.analysis import MEASUREMENT_SCALES  # noqa: E402
table(MEASUREMENT_SCALES.rename(columns={"data_type": "type", "measurement_scale": "scale"}),
      "Variables, data types and measurement scales", widths=[3.2, 3.6, 2.2, 7], size=9)
para("The scale matters for the technique: ratio data such as amounts supports means, ratios and logarithms, whereas "
     "nominal data allows only counts and chi-square tests, and ordinal data permits rank-based tests.")
heading("2.3 EDA compared with classical and Bayesian analysis", 2)
table(pd.DataFrame([
    ("Starting point", "Data first, hypotheses emerge", "Model and hypothesis first", "Prior belief first"),
    ("Method", "Plots, summaries, robust statistics", "Fit assumed model, test hypotheses", "Update prior with likelihood"),
    ("Output", "Insight and questions", "Parameter estimates, p-values", "Posterior distributions"),
    ("Role here", "Whole application", "Chi-square, Kruskal-Wallis, OLS, ADF", "Thompson sampling in nudges"),
], columns=["Aspect", "EDA", "Classical", "Bayesian"]), "Comparison of analytical approaches", widths=[2.8, 4.4, 4.4, 4.4], size=9.5)
heading("2.4 Software tools", 2)
para("Python was used throughout: pandas and NumPy for data handling, SciPy and statsmodels for statistics, "
     "scikit-learn for clustering, decomposition and regression, Plotly and Matplotlib for visualisation, and "
     "Streamlit for the interface. Pytest verifies the pipeline with automated tests.")

# ---------------------------------------------------------------- 3 design
heading("3. System Design", 1)
para("SpendSense is organised as a Python package whose modules correspond to the syllabus modules, with a thin "
     "Streamlit layer on top. Data flows from the loader through transformation and outlier detection into the "
     "analysis modules, and the interface calls them on demand with caching.")
figure("architecture", "SpendSense architecture")
table(pd.DataFrame([
    ("generator.py", "Synthetic statement with latent behaviours and injected defects"),
    ("preprocessing.py", "Loader, deduplication, value replacement, binning, imputation, MLE"),
    ("analysis.py", "Summary statistics, correlation, contingency tables, tests"),
    ("timeseries.py", "Indexing, resampling, decomposition, ADF, ACF"),
    ("clustering.py", "Five clustering methods, text clustering, outlier detection"),
    ("reduction.py", "PCA, SVD, factor analysis, MDS, Isomap, t-SNE, SOM"),
    ("modeling.py", "Regression, accuracy metrics, backtest, forecast"),
    ("bandit.py", "Multi-armed bandit nudges"),
    ("insights.py", "KPIs, recurring payments, budget review"),
], columns=["Module", "Responsibility"]), "Package modules", widths=[4, 12])

# ---------------------------------------------------------------- 4 dataset
heading("4. Dataset", 1)
para(f"Real statements contain private information, so the demonstration uses a synthetic statement from "
     f"{F['date_min']:%d %B %Y} to {F['date_max']:%d %B %Y}. It models a salaried professional with fixed monthly "
     f"obligations (rent, utilities, subscriptions), variable spending in eleven categories, a monthly salary and "
     f"occasional repayments from friends. Variable spending follows lognormal amounts, weekend effects, a festive "
     f"shopping season in October and November, holiday travel peaks and three latent behavioural factors (going out, "
     f"staying home, travelling) that make categories move together, as in real life.")
para("Realistic defects were injected so that each data-preparation technique has something to repair: exact duplicate "
     "rows, blank categories, payment modes and amounts, inconsistent labels (\"DR\", \"Debit\", \"upi\", \"Net Banking\"), "
     "noisy narrations such as \"POS 4274XX ZOMATO BANGALORE biryani\", and eight one-off large purchases whose "
     "locations are kept as ground truth for evaluating anomaly detection. The application also accepts any user CSV "
     "with date, description and amount columns.")
cs = F["cat_summary"].copy()
cs_view = pd.DataFrame({"category": cs.index, "transactions": cs["count"].astype(int),
                        "total (Rs.)": cs["sum"].map("{:,.0f}".format), "median (Rs.)": cs["median"].map("{:,.0f}".format),
                        "share %": cs["share_pct"].map("{:.1f}".format)})
table(cs_view, "Debit transactions by category after cleaning (including flagged one-offs)", widths=[4, 2.8, 3.2, 3, 2.5], size=9.5)

# ---------------------------------------------------------------- 5 transformation
heading("5. Data Transformation", 1)
para("This chapter covers Module 2: deduplication, replacing values, discretisation, and the treatment of missing data "
     "including maximum-likelihood estimation.")
heading("5.1 Deduplication and replacing values", 2)
para(f"Of {R.rows_raw:,} raw rows, {R.duplicates_removed} were exact duplicates, typical of overlapping statement exports, "
     f"and were removed, leaving {R.rows_final:,}. Replacement operated at three levels. Labels for type and payment mode "
     f"were mapped to canonical values ({R.labels_replaced} cells changed). Each narration was reduced to a canonical "
     f"merchant using word-boundary patterns that tolerate prefixes, reference numbers, case and punctuation, resolving "
     f"{R.merchants_resolved / R.rows_final:.0%} of rows to a known merchant. Finally, {R.modes_inferred} missing payment "
     f"modes were inferred from narration prefixes (UPI, POS, NEFT, ATM) and {R.categories_inferred} missing categories "
     f"from the merchant dictionary and keyword rules.")
figure("cleaning", "Missing values before and after cleaning, and row counts per stage", 16)
heading("5.2 Discretisation and binning", 2)
para(f"Amounts are strongly right-skewed. Equal-width binning places {F['equal_width_top_share']:.0%} of all debits in "
     f"a single bin, which conveys almost nothing. Equal-frequency bins balance counts but have meaningless edges. The "
     f"application therefore uses domain bands - Micro (under Rs. 100), Small, Medium, Large (Rs. 2,000 to 10,000) and "
     f"Major - which are both balanced enough to compare and meaningful to a user. The same idea bins the day of month "
     f"into early, middle and late periods for the regression model.")
figure("binning", "Three binning strategies on transaction amounts")
heading("5.3 Missing data: traditional methods", 2)
para(f"{R.amounts_imputed} amounts were blank. Listwise deletion would discard those transactions and understate spend, "
     f"and is available as an option. Mean imputation inserts the global average, which is dominated by large purchases "
     f"and is wrong for almost every individual transaction. Median and category-wise mean or median are progressively "
     f"better because they condition on context. All are implemented and compared.")
heading("5.4 Maximum likelihood estimation", 2)
para(f"Spending amounts within a merchant or category are well described by a lognormal distribution: the logarithm of "
     f"the amount is normal. For observations x1..xn, the log-likelihood is maximised in closed form by "
     f"mu = mean(ln x) and sigma squared = mean((ln x - mu)^2). The fitted median exp(mu) is then the imputed value. "
     f"SpendSense fits this model per merchant when at least four observations exist, falls back to the category, and "
     f"finally to the global fit. For the Shopping category the fit gives mu = {F['mle_mu']:.2f} and sigma = "
     f"{F['mle_sigma']:.2f}, matching the observed histogram closely (see the figure below). Fixed bills such as rent have sigma near "
     f"zero, so their blanks are recovered exactly.")
imp = F["imputation"].copy()
imp["MAE"] = imp["MAE"].map("{:,.0f}".format)
imp["RMSE"] = imp["RMSE"].map("{:,.0f}".format)
imp["median_abs_pct_error"] = (imp["median_abs_pct_error"] * 100).map("{:.0f}%".format)
table(imp.rename(columns={"median_abs_pct_error": "median % error"}),
      "Imputation accuracy when 10% of known amounts are hidden (mean of 5 repeats)", widths=[4, 3, 3, 4])
figure("imputation", "Imputation error by method and the maximum-likelihood lognormal fit")
best = F["imputation"].iloc[0]
para(f"Hiding known amounts and re-estimating them shows that '{best['method']}' has the lowest mean absolute error "
     f"(Rs. {best['MAE']:,.0f}), roughly {F['imputation'].set_index('method').loc['mean', 'MAE'] / best['MAE']:.1f} times "
     f"better than mean imputation. Conditioning on the merchant is what makes the difference, and maximum likelihood "
     f"supplies a principled, robust estimate even from a handful of observations.")
heading("5.5 Categorising unknown merchants with text clustering", 2)
para(f"For merchants outside the dictionary, descriptions are vectorised with TF-IDF and grouped by K-Means. Each cluster "
     f"adopts the majority category of its labelled members when purity is at least 60%. Hiding 20% of labels, the method "
     f"assigned {F['text_eval']['coverage']:.0%} of them with {F['text_eval']['accuracy']:.0%} accuracy, so unfamiliar "
     f"narrations are categorised without any hand-written rule.")

# ---------------------------------------------------------------- 6 correlation, time series
heading("6. Correlation Analysis and Time Series", 1)
heading("6.1 Univariate analysis", 2)
ss = F["spend_stats"]
para(f"The {F['n_debit']:,} debit amounts have mean Rs. {ss['mean']:,.0f}, median Rs. {ss['median']:,.0f}, standard "
     f"deviation Rs. {ss['std']:,.0f}, skewness {ss['skewness']:.1f} and kurtosis {ss['kurtosis']:.1f}. The mean far above "
     f"the median and the large skewness show a long right tail produced by rent, travel and shopping. On a log scale "
     f"the distribution becomes close to symmetric, which justifies lognormal modelling and log-scaled features.")
figure("univariate", "Distribution of amounts and spread by category")
heading("6.2 Bivariate analysis", 2)
tc = F["top_corr"].copy()
tc["correlation"] = tc["correlation"].map("{:+.2f}".format)
para("Weekly spend per category was correlated using Spearman's coefficient, which is robust to the heavy tails. The "
     "strongest relationships reveal behaviour rather than arithmetic.")
table(tc, "Strongest rank correlations between weekly category spend", widths=[4.5, 4.5, 3])
figure("correlation", "Spearman correlation matrix of weekly category spend", 12)
gt = F["group_test"]
para(f"The sign pattern separates staying in from going out: food delivery moves against transport "
     f"({F['top_corr'].iloc[0]['correlation']:+.2f}) and against dining, so weeks heavy on deliveries are light on outings. "
     f"Rent and subscriptions correlate positively only because both are fixed bills falling in the first weeks of "
     f"each month, a reminder that correlation here partly reflects the calendar. Spend also differs significantly "
     f"across payment modes (Kruskal-Wallis H = {gt['H']:.1f}, p = {gt['p_value']:.2g}).")
heading("6.3 Multivariate analysis", 2)
wp = F["weekday_profile"]
para(f"Three-way tables relate category, payment mode and weekend flag. Across weekdays, average daily spend is highest "
     f"on Sunday ({money(wp.loc['Sunday', 'mean'])}) and Saturday ({money(wp.loc['Saturday', 'mean'])}) and lowest on "
     f"Friday ({money(wp.loc['Friday', 'mean'])}). The heat map on the right shows the typical transaction size for each "
     f"category and weekday.")
figure("weekday", "Average daily spend by weekday and average transaction by category and weekday")
heading("6.4 Time series fundamentals and time-based indexing", 2)
para("A time series is a sequence of observations indexed by time. Indexing the transactions with a DatetimeIndex "
     "enables label-based slicing, such as selecting one quarter in a single expression, frequency conversion, and "
     "calendar-aware grouping. The daily series was made gap-free by reindexing to every calendar day and filling days "
     "without spending with zero. Flagged one-off purchases are excluded so that the series describes typical behaviour.")
heading("6.5 Visualising, grouping and resampling", 2)
m = F["monthly"]
para(f"Resampling daily spend to months shows a typical monthly outflow of {money(m['spend'].median())} against income of "
     f"about {money(m['income'].median())}, a median savings rate of {m['savings_rate_pct'].median():.0f}%. The two weakest "
     f"months were {m['savings_rate_pct'].idxmin()} ({m['savings_rate_pct'].min():.0f}%) and "
     f"{m['savings_rate_pct'].sort_values().index[1]} ({m['savings_rate_pct'].sort_values().iloc[1]:.0f}%), consistent "
     f"with the festive shopping and holiday travel seasons that were built into the data and recovered by the analysis.")
figure("timeseries", "Daily spend with rolling mean, and monthly income, spend and savings rate")
heading("6.6 Characteristics of the series", 2)
adf = F["adf"]
para(f"Decomposition with a seven-day period separates trend, seasonality and residual. The weekly component ranges over "
     f"{money(F['seasonal_range'])} and peaks on {F['seasonal_peak_day']}. The Augmented Dickey-Fuller test gives "
     f"p = {adf['p_value']:.4f}, so the series is {'stationary' if adf['stationary'] else 'non-stationary'}; its mean "
     f"does not drift, which keeps the regression formulation simple. The autocorrelation function shows a "
     f"lag-7 value of {F['acf7']:.2f} and {F['acf1']:.2f} at lag 1: there is essentially no serial dependence. Daily "
     f"spend is therefore driven by the calendar (bill dates and weekends) rather than by momentum, which is why a "
     f"calendar-feature regression is appropriate here instead of an autoregressive model.")
figure("decomposition", "Additive decomposition of daily spend")
figure("acf", "Autocorrelation function with 95% significance bounds", 11)

# ---------------------------------------------------------------- 7 summarisation
heading("7. Data Summarisation and Visualisation", 1)
heading("7.1 Statistical summary measures", 2)
cs2 = F["cat_summary"].head(8)
view = pd.DataFrame({"category": cs2.index, "mean": cs2["mean"].map("{:,.0f}".format), "median": cs2["median"].map("{:,.0f}".format),
                     "std": cs2["std"].map("{:,.0f}".format), "CV": cs2["cv"].map("{:.2f}".format)})
para("Beyond the mean, the coefficient of variation (standard deviation over mean) summarises how predictable a "
     "category is. Rent has a coefficient of variation near zero, while shopping and travel are volatile.")
table(view, "Summary statistics for the eight largest categories (rupees, including flagged one-offs)", widths=[4.2, 3, 3, 3, 2.5])
heading("7.2 One-, two- and n-dimensional analysis", 2)
para("One-dimensional analysis studies a single variable through histograms, boxplots and summary measures (see the univariate figure in Section 6.1). "
     "Two-dimensional analysis relates pairs, as with the correlation matrix and the weekday by category matrix. "
     "n-dimensional analysis uses pivot tables with several keys, and in Sections 8 and 9 whole feature vectors.")
heading("7.3 Contingency tables", 2)
ch = F["chi"]
para(f"A contingency table counts transactions for each category and payment mode. The chi-square test of independence "
     f"gives chi2 = {ch['chi2']:.0f} on {ch['dof']} degrees of freedom (p = {ch['p_value']:.2g}) with Cramer's V = "
     f"{ch['cramers_v']:.2f}. Payment habit is therefore clearly related to category: fuel and dining lean towards "
     f"cards, small daily purchases towards UPI, and rent goes through net banking.")
figure("contingency", "Contingency table of category by payment mode", 11)
heading("7.4 Scatter plots, dot charts and bar plots", 2)
para("The scatter plot shows amount against day of month on a log axis, where the rent and bill cluster at the start of "
     "the month and the spread of discretionary purchases are visible. The dot chart ranks categories by total spend, "
     "which is easier to compare than a pie chart. The bar plot lists the merchants that take the most money.")
figure("visual_toolkit", "Scatter plot, dot chart and bar plot")

# ---------------------------------------------------------------- 8 clustering
heading("8. Clustering and Outlier Detection", 1)
heading("8.1 Merchant personas", 2)
para(f"Each of {F['n_feat_merchants']} merchants with at least three transactions was described by six standardised "
     f"features: log transaction count, log average amount, coefficient of variation, weekend share, early-month share and "
     f"share of months active. Five algorithms were compared: K-Means, spectral clustering on a nearest-neighbour graph, a "
     f"Gaussian mixture fitted by the expectation-maximisation (EM) algorithm, Ward's hierarchical agglomerative "
     f"clustering, and minimum-spanning-tree clustering, which removes the longest edges of the tree.")
figure("cluster_selection", "Choosing the number of clusters")
cc = F["cluster_comparison"].copy()
for col in ("silhouette", "davies_bouldin", "calinski_harabasz"):
    cc[col] = cc[col].map("{:.2f}".format)
table(cc, "Cluster quality at k = 4 (higher silhouette and Calinski-Harabasz, lower Davies-Bouldin, is better)", widths=[4, 2, 3, 3.5, 3.5], size=9.5)
para("K-Means gave the best balance of cohesion and separation. MST clustering has a deceptively low Davies-Bouldin "
     "value because it isolates a few distant merchants and leaves the remainder merged, which its low silhouette exposes.")
prof = F["cluster_profile"].reset_index(drop=True)
prof["avg_amount"] = prof["avg_amount"].map("{:,.0f}".format)
table(prof[["persona", "merchants", "avg_txn_count", "avg_amount", "amount_cv", "weekend_share"]].rename(
    columns={"avg_txn_count": "avg txns", "avg_amount": "avg amount", "amount_cv": "CV", "weekend_share": "weekend share"}),
    "Personas found by K-Means", widths=[4.5, 2.3, 2.3, 2.6, 1.8, 2.6], size=9.5)
figure("personas", "K-Means personas projected with PCA, and the Ward dendrogram", 16.5)
para("The personas are interpretable. Rent, the gym and electricity form a fixed-bill group with near-constant amounts. "
     "Food delivery, cabs, the metro, quick-commerce and streaming or telecom bills form an everyday-habits group with "
     "frequent, similar-sized charges. Dining, shopping and entertainment form a weekend leisure group with the highest "
     "weekend share and variable amounts. Fuel, pharmacies, cash withdrawals and local stores are occasional essentials.")
heading("8.2 Document clustering", 2)
para("Section 5.5 applies the same idea to text: TF-IDF vectors of the narrations are clustered and labelled by majority "
     "vote, allowing the application to categorise merchants it has never seen.")
heading("8.3 Outlier detection using clustering", 2)
para("Transactions are described by their log amount and a robust z-score relative to their own category (median and "
     "median absolute deviation, so that outliers do not distort the baseline). A Gaussian mixture models normal "
     "behaviour by EM, and transactions with the lowest likelihood are candidates; only unusually large debits are "
     "flagged. The K-Means variant uses the distance to the nearest centroid, and DBSCAN labels low-density points as "
     "noise. The eight injected anomalies provide ground truth.")
oe = pd.DataFrame([(k, v["flagged"], v["caught"], f"{v['precision']:.0%}", f"{v['recall']:.0%}")
                   for k, v in F["outlier_eval"].items()], columns=["method", "flagged", "caught of 8", "precision", "recall"])
table(oe, "Outlier detection against the injected anomalies", widths=[4, 3, 3, 3, 3])
figure("outliers", "Debits over time with detected outliers highlighted")
para(f"The mixture model flagged {F['n_outliers']} transactions worth {money(F['outlier_total'])} in total, which are "
     f"reported as one-off large spends and excluded from typical spending, forecasts and budgets.")

# ---------------------------------------------------------------- 9 dimensionality
heading("9. Dimensionality Reduction", 1)
r, c = F["matrix_shape"]
para(f"For this module each week is an observation and each of the {c} spending categories a variable ({r} weeks), log "
     f"scaled and standardised so that large categories do not dominate.")
heading("9.1 Principal component analysis", 2)
pc = F["pca"]
para(f"The first three components explain {pc['cum'][2]:.0%} of the variance and {pc['n_90']} are needed for 90%. "
     f"Because many spending decisions are independent, the variance is spread across many components rather than "
     f"concentrated, so the data is only moderately compressible. The loadings show that the first component contrasts "
     f"going-out categories (transport, dining, entertainment) with staying-in ones (food delivery, groceries, "
     f"shopping).")
figure("pca", "Scree plot and component loadings")
heading("9.2 Singular value decomposition", 2)
para(f"SVD factorises the standardised matrix as U times S times V transpose; the squared singular values equal the "
     f"energy captured by each rank. A rank-{F['svd_rank90']} approximation retains 90% of the energy. Because PCA of a "
     f"centred matrix is its SVD, the two views agree on the explained variance.")
figure("svd", "Singular values and reconstruction error")
heading("9.3 Factor analysis and intrinsic dimensionality", 2)
ind = F["intrinsic"]
para(f"A three-factor model with varimax rotation yields interpretable latent behaviours. Factor 1 isolates shopping, "
     f"which has its own seasonal driver. Factor 2 contrasts going out (dining, entertainment, transport, subscriptions) "
     f"with staying in (food delivery, groceries). Factor 3 separates fixed bills (utilities against rent). "
     f"Intrinsic-dimension estimates differ by method, as expected: {ind['kaiser']} factors by the Kaiser criterion "
     f"(eigenvalue above one), {ind['levina_bickel']:.1f} by the Levina-Bickel local maximum-likelihood estimator and "
     f"{ind['variance_90']} components for 90% of the variance, against {ind['ambient']} observed variables. Spending "
     f"is therefore driven by a handful of strong behaviours plus considerable independent variation.")
figure("factor", "Factor loadings", 8.5)
heading("9.4 Non-linear methods", 2)
emb = F["embedding"].copy()
emb["trustworthiness"] = emb["trustworthiness"].map("{:.3f}".format)
para("Multidimensional scaling preserves pairwise distances, Isomap preserves geodesic distances along a "
     "neighbourhood graph, t-SNE preserves local neighbourhoods probabilistically, and a self-organising map (SOM) "
     "learns a grid of prototype vectors. Trustworthiness, the fraction of 2-D neighbours that are also true neighbours "
     "in the original space, ranks them.")
table(emb, "Trustworthiness of 2-D embeddings (10 neighbours)", widths=[4, 4])
figure("embeddings", "Two-dimensional embeddings coloured by quarter", 16.5)
figure("som", f"Self-organising map U-matrix and hit map (quantisation error {F['som_qe']:.2f})", 12)
para(f"All non-linear methods preserve neighbourhoods better than the linear PCA projection, with "
     f"{F['embedding'].iloc[0]['method']} highest, although the margins are modest, which indicates only mild curvature "
     f"in the data.")

# ---------------------------------------------------------------- 10 model
heading("10. Model Development and Evaluation", 1)
heading("10.1 Constructing the linear regression model", 2)
mdl = F["model"]
para("The target is daily typical spend. The design matrix contains the amount of recurring bills expected on that date "
     "(from the recurring-payment detector), weekday indicators, an early-month flag and a linear trend. Using known "
     "bills as a feature lets the model concentrate on the variable part of spending. The last 90 days were held out in "
     "chronological order, and time-series cross-validation with five expanding folds checked stability.")
coef = mdl["coefficients"].copy()
coef = coef.reset_index().rename(columns={"index": "term"})
coef["coefficient"] = coef["coefficient"].map("{:,.1f}".format)
coef["std_error"] = coef["std_error"].map("{:,.1f}".format)
coef["p_value"] = coef["p_value"].map("{:.3f}".format)
table(coef, "Ordinary least squares coefficients (training period)", widths=[4.5, 3.5, 3.5, 3], size=9.5)
sig = mdl["coefficients"]
para(f"The coefficient on scheduled bills is {sig.loc['scheduled_bills', 'coefficient']:.2f}, essentially one, as it "
     f"should be. Weekend days add roughly Rs. {sig.loc['dow_Sun', 'coefficient']:,.0f} on Sundays over Mondays "
     f"(p = {sig.loc['dow_Sun', 'p_value']:.3f}), which is the quantified weekend effect. The Durbin-Watson statistic "
     f"is {mdl['durbin_watson']:.2f}.")
heading("10.2 Evaluation and understanding accuracy", 2)
rows = []
for label, key in (("Daily, linear regression", "daily"), ("Weekly, linear regression", "weekly"), ("Daily, mean baseline", "baseline")):
    v = mdl[key]
    rows.append((label, f"{v['MAE']:,.0f}", f"{v['RMSE']:,.0f}", f"{v['R2']:.2f}", f"{v['wMAPE']:.0%}", f"{v['accuracy']:.0%}"))
table(pd.DataFrame(rows, columns=["model", "MAE", "RMSE", "R2", "wMAPE", "accuracy"]),
      "Held-out performance (rupees)", widths=[5, 2.2, 2.2, 2, 2.2, 2.4])
para(f"R2 is the share of variance explained; zero means no better than predicting the training mean. The baseline "
     f"scores {mdl['baseline']['R2']:.2f}, while the model reaches {mdl['daily']['R2']:.2f} daily and "
     f"{mdl['weekly']['R2']:.2f} weekly. Accuracy here is one minus weighted MAPE: daily spend is inherently noisy "
     f"({mdl['daily']['accuracy']:.0%}), but errors partly cancel when summed, giving {mdl['weekly']['accuracy']:.0%} for weeks. "
     f"Reporting both avoids the misleading impression that a single number gives.")
figure("regression", "Weekly predictions on held-out data and residual distribution")
heading("10.3 Forecasting next month", 2)
fc, bt = F["forecast"], F["backtest"]
be = F["backtest_errors"].copy()
be["MAPE_%"] = be["MAPE_%"].map("{:.1f}".format)
para(f"A rolling-origin backtest forecasts each of the last complete months using only earlier data and compares the "
     f"monthly total with the truth. Holt-Winters exponential smoothing with weekly seasonality, the regression model, "
     f"their average and a naive last-30-days baseline were compared; Holt-Winters had the lowest error and every model "
     f"beat the naive baseline. The application reports the average of regression and Holt-Winters so that the "
     f"regression's interpretable drivers inform the figure. For the next 30 days the application reports "
     f"{money(fc['total'])}, with an 80% range of {money(max(fc['low'], 0))} to {money(fc['high'])}.")
table(be, "Backtest error on monthly totals (mean absolute percentage error)", widths=[5, 4])
figure("forecast", "Thirty-day forecast and monthly backtest")
heading("10.4 Reinforcement learning: nudges as a bandit", 2)
para("Supervised learning learns from labelled examples; the correct answer is known for every training row. "
     "Reinforcement learning (RL) instead learns from the consequences of its own actions: an agent chooses, receives a "
     "reward, and must balance exploring unknown options with exploiting the best known one. Choosing which saving tip "
     "to show a user is exactly this problem, a multi-armed bandit, because feedback arrives only for the tip shown.")
nd = F["nudges"][["text", "category", "monthly_spend", "accept", "expected_saving"]].copy()
nd["monthly_spend"] = nd["monthly_spend"].map("{:,.0f}".format)
nd["expected_saving"] = nd["expected_saving"].map("{:,.0f}".format)
nd["accept"] = nd["accept"].map("{:.0%}".format)
table(nd.rename(columns={"text": "tip", "monthly_spend": "avg monthly spend", "accept": "acceptance", "expected_saving": "expected saving"}),
      "Candidate tips, ranked by expected monthly saving (rupees)", widths=[6.2, 2.6, 2.6, 2.2, 2.4], size=9)
para("The arm's success probability is its acceptance rate times its potential saving, normalised. Four policies were "
     "simulated for 400 rounds averaged over 60 runs: random choice, epsilon-greedy, UCB1 (optimism in the face of "
     "uncertainty) and Thompson sampling (sampling from Beta posteriors, a Bayesian approach).")
reg = F["bandit"]["regret"].iloc[-1]
figure("bandit", "Cumulative regret and how often each tip is suggested")
para(f"After 400 rounds the cumulative regret was {reg['Random']:.0f} for random choice, {reg['UCB1']:.0f} for UCB1, "
     f"{reg['Thompson sampling']:.0f} for Thompson sampling and {reg['Epsilon-greedy']:.0f} for epsilon-greedy. The learning "
     f"policies concentrate their suggestions on the tip with the highest expected saving, "
     f"'{F['nudges'].iloc[0]['text']}', whereas random choice spreads suggestions evenly. The acceptance rates are "
     f"modelled assumptions standing in for user feedback, which a deployed system would collect.")
heading("10.5 Applications of reinforcement learning", 2)
para("The same pattern appears widely: recommender systems and ad selection (bandits), dynamic pricing, portfolio and "
     "inventory management, energy-grid control, robotics and game playing (deep RL), and personalised health "
     "interventions. In personal finance it can adapt reminders, savings-goal schedules and alert thresholds to the "
     "individual rather than applying one rule to everyone.")

# ---------------------------------------------------------------- 11 application
heading("11. Application and Deployment", 1)
para("The dashboard has nine sections: Overview (key indicators, income against spend, category mix, recurring payments, "
     "unusual transactions and a budget check), Data preparation, Explore, Time series, Patterns and anomalies, "
     "Dimensionality, Forecast and model, Smart nudges and About. The sidebar chooses the data source and three "
     "settings: the missing-amount strategy, the outlier detector and its sensitivity.")
k = F["kpis"]
para(f"For the latest complete month ({k['month']}) the overview reports spend of {money(k['spend'])}, income of "
     f"{money(k['income'])} and a savings rate of {k['savings_rate']:.0f}%. The recurring-payment detector found "
     f"{len(F['recurring'])} steady monthly charges costing {money(F['recurring']['annual_cost'].sum())} a year, "
     f"including every subscription.")
rec = F["recurring"][["merchant", "category", "typical_amount", "annual_cost"]].copy()
rec["typical_amount"] = rec["typical_amount"].map("{:,.0f}".format)
rec["annual_cost"] = rec["annual_cost"].map("{:,.0f}".format)
table(rec, "Recurring payments detected", widths=[4.5, 3.5, 3.5, 3.5], size=9.5)
heading("11.1 Quality assurance", 2)
para("Eighteen automated tests check merchant normalisation, payment-mode inference, deduplication, flexible statement "
     "loading, binning, MLE imputation, outlier recall, statistical tests, clustering, text categorisation, "
     "dimensionality reduction, regression accuracy and the bandit's learning behaviour. Every page of the application was "
     "also exercised headlessly with Streamlit's testing framework.")
heading("11.2 Deployment", 2)
para("The repository is deployable on Streamlit Community Cloud directly from GitHub by selecting app.py as the "
     "entry point; the pinned dependencies are listed in requirements.txt. Analysis runs inside the user's session and "
     "uploaded statements are not written to disk.")

# ---------------------------------------------------------------- 12 issues
heading("12. Contemporary Issues", 1)
bullets([("Privacy. ", "Bank statements are among the most sensitive personal data. SpendSense processes uploads in memory "
          "only and the sample statement is synthetic. A production system would add encryption and on-device processing."),
         ("Data quality and bias. ", "Models trained on one user's habits may not generalise, and merchant dictionaries "
          "reflect the markets they were built for. Transparency about repairs, as in the cleaning report, lets users judge."),
         ("Explainability. ", "Linear coefficients, named personas and plain-language insights were chosen over opaque "
          "models so that advice can be justified."),
         ("Responsible advice. ", "Nudges are suggestions, not financial advice; the bandit's rewards are simulated here "
          "and real deployment requires consent and care that tips do not cause harm."),
         ("Synthetic data. ", "Synthetic data protects privacy and gives ground truth, but results should be validated "
          "on real statements before drawing behavioural conclusions.")])

# ---------------------------------------------------------------- 13 results
heading("13. Results and Discussion", 1)
summary = pd.DataFrame([
    ("Data cleaning", f"{R.duplicates_removed} duplicates removed; all {sum(R.missing_before.values())} blanks filled"),
    ("Imputation", f"MLE MAE Rs. {best['MAE']:,.0f} vs mean Rs. {F['imputation'].set_index('method').loc['mean', 'MAE']:,.0f}"),
    ("Time series", f"weekly peak on {F['seasonal_peak_day']}; ADF p = {adf['p_value']:.3f}"),
    ("Clustering", f"4 personas, best silhouette {F['cluster_comparison']['silhouette'].max():.2f}"),
    ("Outliers", f"{F['outlier_eval']['gmm']['caught']} of {F['outlier_eval']['gmm']['true_anomalies']} anomalies caught, precision {F['outlier_eval']['gmm']['precision']:.0%}"),
    ("Dimensionality", f"{pc['n_90']} components for 90% variance; best embedding {F['embedding'].iloc[0]['method']}"),
    ("Regression", f"daily R2 {mdl['daily']['R2']:.2f}; weekly accuracy {mdl['weekly']['accuracy']:.0%}"),
    ("Bandit", f"regret {reg['Thompson sampling']:.0f} (Thompson) vs {reg['Random']:.0f} (random)"),
], columns=["Component", "Key result"])
table(summary, "Summary of results", widths=[4, 12])
para("Three lessons stand out. First, preparation determines everything downstream: resolving merchants and repairing "
     "amounts made the later analyses meaningful. Second, honest evaluation matters; daily spend is noisy and the "
     "report shows both daily and weekly accuracy rather than selecting the flattering one. Third, simple, "
     "interpretable methods applied with domain knowledge, such as including known bills in the regression, beat more "
     "elaborate approaches. Limitations are that the evaluation data is synthetic, the statement covers a single "
     "person, and the bandit's feedback is simulated.")

# ---------------------------------------------------------------- 14 conclusion
heading("14. Conclusion and Future Work", 1)
para("SpendSense demonstrates that the exploratory data analysis toolbox, from deduplication to reinforcement learning, "
     "can be assembled into a practical tool that answers an everyday question. It cleans messy statements, explains "
     "behaviour, spots anomalies, forecasts and advises, all in an interactive application.")
bullets(["Validate on anonymised real statements and add bank-specific parsers.",
         "Learn merchant categories continuously from user corrections.",
         "Collect genuine accept or ignore feedback to drive the bandit, and add contextual features.",
         "Support budgets and goals with alerts, and multi-account views."])

heading("References", 1)
for ref in ["Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.",
            "Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical Learning. Springer.",
            "McKinney, W. (2022). Python for Data Analysis, 3rd ed. O'Reilly.",
            "Levina, E., Bickel, P. (2004). Maximum likelihood estimation of intrinsic dimension. NIPS.",
            "Kohonen, T. (2001). Self-Organizing Maps. Springer.",
            "Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed. MIT Press.",
            "Hyndman, R. J., Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd ed. OTexts.",
            "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. JMLR 12.",
            "Seabold, S., Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. SciPy."]:
    para(ref, align=WD_ALIGN_PARAGRAPH.LEFT, after=3)

heading("Appendix A: Syllabus Coverage", 1)
table(pd.DataFrame([
    ("1", "EDA steps, data types, scales, tools", "Sections 2, 7.1; Explore tab"),
    ("2", "Deduplication, replacing values, binning", "Sections 5.1, 5.2"),
    ("2", "Missing data and MLE", "Sections 5.3, 5.4"),
    ("3", "Uni-, bi-, multivariate analysis", "Section 6.1 to 6.3"),
    ("3", "Time series, indexing, resampling", "Sections 6.4 to 6.6"),
    ("4", "Summary measures, contingency tables, plots", "Section 7"),
    ("5", "Spectral, EM, hierarchical, MST, document clustering", "Sections 8.1, 8.2"),
    ("5", "Outlier detection using clustering", "Section 8.3"),
    ("6", "PCA, SVD, factor analysis, intrinsic dimension", "Sections 9.1 to 9.3"),
    ("6", "MDS, manifold learning, SOM", "Section 9.4"),
    ("7", "Linear regression, evaluation, accuracy", "Sections 10.1 to 10.3"),
    ("7", "Reinforcement learning and applications", "Sections 10.4, 10.5"),
    ("8", "Contemporary issues", "Section 12"),
], columns=["Module", "Topic", "Where covered"]), "Mapping from syllabus to report", widths=[2, 8, 6])

heading("Appendix B: Repository Structure", 1)
for line in ["SpendSense/", "  app.py                      Streamlit application", "  spendsense/                 analysis package",
             "  tests/                      automated tests", "  data/sample_transactions.csv  sample statement",
             "  report/                     report builder and figures", "  requirements.txt            dependencies",
             "  .streamlit/config.toml      theme and server settings"]:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(line)
    r.font.name = "Consolas"
    r.font.size = Pt(10)

# page numbers
for section in doc.sections:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.text = text
        run._r.append(el)

doc.save(args.out)
print("saved", args.out, counter)

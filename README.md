# SpendSense

Personal expense intelligence built on exploratory data analysis.
Course project for **BCSE331L - Exploratory Data Analysis**.

Student: Keshav Gujrathi (23BDS0225) · Guide: Dr. Vijayanand C.

SpendSense takes a bank statement and answers three everyday questions: where does my money go, what will be charged
again, and is anything unusual? It cleans the statement, analyses it, flags anomalies, forecasts the next 30 days and
recommends which saving tip to try first.

## Features

| Section | What it does | Syllabus module |
|---|---|---|
| Overview | KPIs, income vs spend, category mix, recurring payments, unusual transactions, budget check | 1 |
| Data preparation | Deduplication, value replacement, binning, missing-data strategies, maximum-likelihood imputation, text clustering | 2 |
| Explore | Univariate, bivariate and n-D analysis, correlation, contingency tables, scatter, dot and bar plots | 3, 4 |
| Time series | Resampling, rolling means, decomposition, ADF test, autocorrelation | 3 |
| Patterns and anomalies | K-Means, spectral, EM/GMM, hierarchical and MST clustering; outlier detection | 5 |
| Dimensionality | PCA, SVD, factor analysis, intrinsic dimension, MDS, Isomap, t-SNE, SOM | 6 |
| Forecast and model | Linear regression, accuracy metrics, backtested 30-day forecast | 7 |
| Smart nudges | Multi-armed bandit (epsilon-greedy, UCB1, Thompson sampling) | 7 |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens with a generated sample statement. To analyse your own, choose **Upload my CSV** in the sidebar. The file
needs a date, a description and an amount column; separate debit/credit columns and signed amounts are handled.
Uploaded data is processed in memory and never stored.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Rebuild the project report

```bash
python report/build_report.py --repo <github-url> --video <video-url> --app <app-url>
```

This reruns the whole analysis, regenerates every figure and writes `report/SpendSense_Project_Report.docx`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. On share.streamlit.io choose **New app**, select the repository and set the main file to `app.py`.
3. Deploy. Dependencies are read from `requirements.txt`.

## Structure

```
app.py                  Streamlit application
spendsense/             analysis package (one module per topic)
tests/                  automated tests
data/                   sample statement
report/                 report builder and figures
```

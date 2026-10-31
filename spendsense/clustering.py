"""Clustering of merchants and descriptions, and clustering-based outlier detection."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["log_txn_count", "log_avg_amount", "amount_cv", "weekend_share",
                   "month_start_share", "active_month_share"]


def merchant_features(df: pd.DataFrame, min_txn: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    spend = df[(df["type"] == "debit") & ~df.get("is_outlier", pd.Series(False, index=df.index))]
    n_months = max(spend["month"].nunique(), 1)
    grouped = spend.groupby("merchant")
    feats = pd.DataFrame({
        "txn_count": grouped.size(),
        "avg_amount": grouped["amount"].mean(),
        "amount_cv": grouped["amount"].std().fillna(0) / grouped["amount"].mean(),
        "weekend_share": grouped["is_weekend"].mean(),
        "month_start_share": grouped["day_of_month"].apply(lambda s: (s <= 7).mean()),
        "active_month_share": grouped["month"].nunique() / n_months,
    })
    feats = feats[feats["txn_count"] >= min_txn].copy()
    feats["category"] = grouped["category"].agg(lambda s: s.mode().iat[0]).reindex(feats.index)
    feats["log_txn_count"] = np.log1p(feats["txn_count"])
    feats["log_avg_amount"] = np.log(feats["avg_amount"])
    X = StandardScaler().fit_transform(feats[FEATURE_COLUMNS])
    return feats, X


def kmeans_labels(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(X)


def spectral_labels(X: np.ndarray, k: int) -> np.ndarray:
    neighbours = max(2, min(8, len(X) - 1))
    return SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=neighbours,
                              random_state=0, assign_labels="kmeans").fit_predict(X)


def gmm_labels(X: np.ndarray, k: int):
    model = GaussianMixture(n_components=k, covariance_type="diag", n_init=5, random_state=0).fit(X)
    return model.predict(X), model


def hierarchical_labels(X: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def mst_labels(X: np.ndarray, k: int) -> np.ndarray:
    dist = squareform(pdist(X)) + 1e-9
    np.fill_diagonal(dist, 0)
    mst = minimum_spanning_tree(dist).toarray()
    rows, cols = np.nonzero(mst)
    order = np.argsort(mst[rows, cols])[::-1]
    for idx in order[: k - 1]:
        mst[rows[idx], cols[idx]] = 0
    return connected_components(mst, directed=False)[1]


def linkage_matrix(X: np.ndarray) -> np.ndarray:
    return linkage(X, method="ward")


def compare_methods(X: np.ndarray, k: int) -> tuple[pd.DataFrame, dict]:
    label_sets = {
        "K-Means": kmeans_labels(X, k),
        "Spectral": spectral_labels(X, k),
        "GMM (EM)": gmm_labels(X, k)[0],
        "Hierarchical (Ward)": hierarchical_labels(X, k),
        "MST": mst_labels(X, k),
    }
    rows = []
    for name, labels in label_sets.items():
        valid = 1 < len(set(labels)) < len(X)
        rows.append({
            "method": name, "clusters": len(set(labels)),
            "silhouette": silhouette_score(X, labels) if valid else np.nan,
            "davies_bouldin": davies_bouldin_score(X, labels) if valid else np.nan,
            "calinski_harabasz": calinski_harabasz_score(X, labels) if valid else np.nan,
        })
    return pd.DataFrame(rows), label_sets


def select_k(X: np.ndarray, k_range=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        if k >= len(X):
            break
        km = kmeans_labels(X, k)
        gm = GaussianMixture(n_components=k, covariance_type="diag", n_init=3, random_state=0).fit(X)
        rows.append({"k": k, "silhouette": silhouette_score(X, km), "inertia": KMeans(
            n_clusters=k, n_init=10, random_state=0).fit(X).inertia_, "bic": gm.bic(X)})
    return pd.DataFrame(rows)


def name_clusters(feats: pd.DataFrame, labels: np.ndarray) -> dict:
    """Assign interpretable persona names to clusters from their average behaviour."""
    profile = feats.assign(cluster=labels).groupby("cluster")[
        ["txn_count", "avg_amount", "amount_cv", "month_start_share", "weekend_share"]].mean()
    names: dict = {}
    remaining = list(profile.index)

    def take(column: str, ascending: bool, label: str):
        if remaining:
            ordered = profile.loc[remaining, column].sort_values(ascending=ascending)
            pick = ordered.index[0]
            names[pick] = label
            remaining.remove(pick)

    take("amount_cv", True, "Fixed recurring bills")
    take("txn_count", False, "Everyday habits")
    take("weekend_share", False, "Weekend leisure and shopping")
    for c in remaining:
        names[c] = "Occasional essentials" if len(remaining) == 1 else f"Occasional essentials {c}"
    return names


def cluster_profile(feats: pd.DataFrame, labels: np.ndarray, names: dict) -> pd.DataFrame:
    frame = feats.assign(cluster=labels)
    out = frame.groupby("cluster").agg(
        merchants=("txn_count", "size"), avg_txn_count=("txn_count", "mean"),
        avg_amount=("avg_amount", "mean"), amount_cv=("amount_cv", "mean"),
        weekend_share=("weekend_share", "mean"))
    out.insert(0, "persona", [names[c] for c in out.index])
    return out.round(2)


def _outlier_features(spend: pd.DataFrame) -> np.ndarray:
    log_amount = np.log(spend["amount"].clip(lower=1))
    grouped = log_amount.groupby(spend["category"])
    median = grouped.transform("median")
    mad = grouped.transform(lambda s: (s - s.median()).abs().median()).clip(lower=0.15)
    robust_z = (log_amount - median) / (1.4826 * mad)
    features = np.column_stack([log_amount, robust_z])
    return StandardScaler().fit_transform(features), robust_z.to_numpy()


def detect_outliers(df: pd.DataFrame, method: str = "gmm", contamination: float = 0.01) -> pd.DataFrame:
    """Flag unusually large debits using cluster density (GMM/EM), centroid distance or DBSCAN noise."""
    out = df.copy()
    out["outlier_score"] = 0.0
    out["is_outlier"] = False
    spend = out[out["type"] == "debit"]
    X, robust_z = _outlier_features(spend)

    if method == "gmm":
        model = GaussianMixture(n_components=3, covariance_type="full", n_init=3, random_state=0).fit(X)
        score = -model.score_samples(X)
        flagged = score >= np.quantile(score, 1 - contamination)
    elif method == "kmeans":
        model = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
        score = model.transform(X).min(axis=1)
        flagged = score >= np.quantile(score, 1 - contamination)
    elif method == "dbscan":
        model = DBSCAN(eps=0.55, min_samples=10).fit(X)
        score = (model.labels_ == -1).astype(float)
        flagged = model.labels_ == -1
    else:
        raise ValueError(f"Unknown outlier method: {method}")

    flagged = flagged & (robust_z > 2.5)
    out.loc[spend.index, "outlier_score"] = score
    out.loc[spend.index, "is_outlier"] = flagged
    return out


def evaluate_outliers(df: pd.DataFrame, truth: pd.DataFrame) -> dict:
    truth = truth.assign(date=pd.to_datetime(truth["date"]))
    key = df.assign(_key=list(zip(df["date"], df["description"])))
    truth_keys = set(zip(truth["date"], truth["description"]))
    actual = key["_key"].isin(truth_keys)
    predicted = key["is_outlier"]
    tp = int((actual & predicted).sum())
    return {"flagged": int(predicted.sum()), "true_anomalies": int(actual.sum()), "caught": tp,
            "precision": tp / max(int(predicted.sum()), 1), "recall": tp / max(int(actual.sum()), 1)}


_NOISE = {"UPI", "POS", "XX", "NEFT", "BANGALORE", "ATM", "WDL", "PAID", "CASH"}


def _clean_text(description: str) -> str:
    tokens = re.findall(r"[A-Za-z]+", str(description).upper().replace("'", ""))
    return " ".join(t for t in tokens if t not in _NOISE and len(t) > 1).lower()


def document_clusters(descriptions: pd.Series, k: int):
    texts = descriptions.map(_clean_text)
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix)
    return labels


def categorise_by_clusters(df: pd.DataFrame, k: int | None = None, min_purity: float = 0.6) -> pd.DataFrame:
    """Assign categories to unlabelled rows using the majority label of their text cluster."""
    out = df.copy()
    known = out["category"].notna() & (out["category"] != "Uncategorised") & (out["type"] == "debit")
    k = k or int(min(40, max(8, 2 * out.loc[known, "category"].nunique())))
    labels = document_clusters(out["description"], k)
    out["text_cluster"] = labels
    votes = out[known].groupby("text_cluster")["category"].agg(
        lambda s: (s.mode().iat[0], (s == s.mode().iat[0]).mean()))
    unknown = (out["category"].isna() | (out["category"] == "Uncategorised")) & (out["type"] == "debit")
    for idx in out.index[unknown]:
        cluster = out.at[idx, "text_cluster"]
        if cluster in votes.index and votes[cluster][1] >= min_purity:
            out.at[idx, "category"] = votes[cluster][0]
    return out


def evaluate_document_clustering(df: pd.DataFrame, hide: float = 0.2, seed: int = 0) -> dict:
    spend = df[(df["type"] == "debit") & df["category"].notna()].copy()
    truth = spend["category"].copy()
    rng = np.random.default_rng(seed)
    hidden = rng.choice(spend.index, size=int(len(spend) * hide), replace=False)
    spend.loc[hidden, "category"] = np.nan
    result = categorise_by_clusters(spend)
    predicted = result.loc[hidden, "category"]
    assigned = predicted.notna()
    accuracy = float((predicted[assigned] == truth.loc[hidden][assigned]).mean()) if assigned.any() else 0.0
    return {"hidden": len(hidden), "assigned": int(assigned.sum()), "accuracy": accuracy,
            "coverage": float(assigned.mean()),
            "ari": float(adjusted_rand_score(truth, result["text_cluster"]))}

"""Dimensionality reduction: PCA, SVD, factor analysis, intrinsic dimensionality, MDS,
manifold learning and a compact self-organising map."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import MDS, TSNE, Isomap, trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def category_matrix(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Periods x categories matrix of log-scaled spend (outliers excluded)."""
    spend = df[df["type"] == "debit"]
    if "is_outlier" in spend:
        spend = spend[~spend["is_outlier"]]
    matrix = spend.pivot_table(index=pd.Grouper(key="date", freq=freq), columns="category", values="amount",
                               aggfunc="sum", fill_value=0.0)
    matrix = matrix.iloc[1:-1] if len(matrix) > 12 else matrix
    matrix = matrix.loc[:, matrix.std() > 0]
    return np.log1p(matrix)


def standardise(matrix: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(matrix)


def run_pca(X: np.ndarray, columns) -> dict:
    pca = PCA().fit(X)
    ratio = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=columns,
                            columns=[f"PC{i + 1}" for i in range(len(ratio))])
    return {"model": pca, "scores": pca.transform(X), "ratio": ratio, "cumulative": np.cumsum(ratio),
            "loadings": loadings, "n_90": int(np.searchsorted(np.cumsum(ratio), 0.90) + 1)}


def run_svd(X: np.ndarray) -> dict:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    energy = s ** 2 / np.sum(s ** 2)
    errors = []
    for rank in range(1, len(s) + 1):
        approx = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        errors.append(np.linalg.norm(X - approx) / np.linalg.norm(X))
    return {"U": U, "s": s, "Vt": Vt, "energy": energy, "reconstruction_error": np.array(errors)}


def run_factor_analysis(X: np.ndarray, columns, n_factors: int = 3) -> dict:
    fa = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=0).fit(X)
    loadings = pd.DataFrame(fa.components_.T, index=columns, columns=[f"Factor {i + 1}" for i in range(n_factors)])
    communality = (loadings ** 2).sum(axis=1).rename("communality")
    return {"model": fa, "loadings": loadings, "communality": communality, "scores": fa.transform(X),
            "log_likelihood": float(fa.score(X))}


def levina_bickel(X: np.ndarray, k: int = 10, seed: int = 0) -> float:
    """Maximum-likelihood estimate of local intrinsic dimension (Levina & Bickel, 2004)."""
    rng = np.random.default_rng(seed)
    jittered = X + rng.normal(scale=1e-6, size=X.shape)
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(jittered).kneighbors(jittered)
    distances = distances[:, 1:]
    logs = np.log(distances[:, -1:] / distances[:, :-1])
    local = (k - 1) / logs.sum(axis=1)
    return float(np.mean(local))


def intrinsic_dimensionality(X: np.ndarray) -> dict:
    pca = PCA().fit(X)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return {
        "variance_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "variance_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "kaiser": int((pca.explained_variance_ > 1).sum()),
        "levina_bickel": levina_bickel(X),
        "ambient": X.shape[1],
    }


def run_mds(X: np.ndarray) -> np.ndarray:
    return MDS(n_components=2, n_init=1, max_iter=200, random_state=0, normalized_stress="auto").fit_transform(X)


def run_isomap(X: np.ndarray, neighbours: int = 12) -> np.ndarray:
    return Isomap(n_neighbors=neighbours, n_components=2).fit_transform(X)


def run_tsne(X: np.ndarray, perplexity: float = 15) -> np.ndarray:
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(X)


def train_som(X: np.ndarray, grid: tuple[int, int] = (6, 6), iterations: int = 6000, lr0: float = 0.5,
              seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gx, gy = grid
    weights = X[rng.integers(0, len(X), size=gx * gy)].reshape(gx, gy, -1).copy()
    coords = np.stack(np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij"), axis=-1)
    sigma0 = max(grid) / 2
    for t in range(iterations):
        frac = t / iterations
        lr = lr0 * np.exp(-3 * frac)
        sigma = max(sigma0 * np.exp(-3 * frac), 0.5)
        x = X[rng.integers(0, len(X))]
        bmu = np.unravel_index(np.argmin(((weights - x) ** 2).sum(axis=2)), (gx, gy))
        dist2 = ((coords - np.array(bmu)) ** 2).sum(axis=2)
        influence = np.exp(-dist2 / (2 * sigma ** 2))[..., None]
        weights += lr * influence * (x - weights)
    return weights


def som_bmus(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    flat = weights.reshape(-1, weights.shape[-1])
    nearest = np.argmin(((X[:, None, :] - flat[None]) ** 2).sum(axis=2), axis=1)
    return np.column_stack(np.unravel_index(nearest, weights.shape[:2]))


def som_u_matrix(weights: np.ndarray) -> np.ndarray:
    gx, gy, _ = weights.shape
    u = np.zeros((gx, gy))
    for i in range(gx):
        for j in range(gy):
            dists = [np.linalg.norm(weights[i, j] - weights[i + di, j + dj])
                     for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= i + di < gx and 0 <= j + dj < gy]
            u[i, j] = np.mean(dists)
    return u


def quantisation_error(X: np.ndarray, weights: np.ndarray) -> float:
    flat = weights.reshape(-1, weights.shape[-1])
    return float(np.mean(np.min(np.linalg.norm(X[:, None, :] - flat[None], axis=2), axis=1)))


def embedding_comparison(X: np.ndarray, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Embed X in 2-D with several methods and rank them by trustworthiness."""
    rng = np.random.default_rng(seed)
    embeddings = {
        "PCA": PCA(n_components=2).fit_transform(X),
        "MDS": run_mds(X),
        "Isomap": run_isomap(X),
        "t-SNE": run_tsne(X),
    }
    weights = train_som(X)
    bmu = som_bmus(X, weights).astype(float) + rng.normal(scale=0.05, size=(len(X), 2))
    embeddings["SOM"] = bmu
    rows = [{"method": name, "trustworthiness": trustworthiness(X, emb, n_neighbors=10)}
            for name, emb in embeddings.items()]
    return pd.DataFrame(rows).sort_values("trustworthiness", ascending=False).reset_index(drop=True), embeddings

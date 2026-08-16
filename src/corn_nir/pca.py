"""Ajustement ACP partage par les graphiques de scores/loadings du notebook d'EDA."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def compute_pca(X, n_components: int | None = None, variance_threshold: float = 0.95) -> dict:
    """Ajuste une ACP sur `X`, avec `n_components` fixe explicitement ou choisi
    comme le plus petit nombre atteignant `variance_threshold` de variance
    expliquee cumulee.

    Retourne les scores, les loadings, la moyenne et *toutes* les valeurs
    propres, pour que l'ACP puisse etre ajustee une seule fois et partagee
    entre les graphiques de scores et de loadings.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    max_components = min(n_samples - 1, n_features)

    full_pca = PCA(n_components=max_components, random_state=0).fit(X)

    if n_components is None:
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_components = max(1, min(n_components, max_components - 1))

    return {
        "n_components": n_components,
        "explained_variance_ratio": full_pca.explained_variance_ratio_,
        "eigenvalues": full_pca.explained_variance_,
        "scores": full_pca.transform(X)[:, :n_components],
        "components": full_pca.components_[:n_components, :],
        "mean": full_pca.mean_,
    }

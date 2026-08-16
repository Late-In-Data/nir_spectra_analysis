import numpy as np

from corn_nir.pca import compute_pca


def _make_synthetic_data(rng, n_samples=60, n_features=30):
    # Un signal de rang faible (3 directions latentes) plus un bruit faible,
    # pour que quelques CP captent presque toute la variance, a l'image de
    # la structure des spectres NIR.
    latent = rng.normal(size=(n_samples, 3))
    loadings = rng.normal(size=(3, n_features))
    X = latent @ loadings + rng.normal(scale=0.05, size=(n_samples, n_features))
    return X


def test_compute_pca_shapes():
    rng = np.random.default_rng(0)
    X = _make_synthetic_data(rng)
    pca_result = compute_pca(X, n_components=3)
    assert pca_result["n_components"] == 3
    assert pca_result["scores"].shape == (60, 3)
    assert pca_result["components"].shape == (3, 30)
    assert pca_result["mean"].shape == (30,)
    assert pca_result["eigenvalues"].shape[0] >= 3


def test_variance_threshold_selection():
    rng = np.random.default_rng(4)
    X = _make_synthetic_data(rng)
    pca_result = compute_pca(X, n_components=None, variance_threshold=0.95)
    cumulative = np.cumsum(pca_result["explained_variance_ratio"])
    assert cumulative[pca_result["n_components"] - 1] >= 0.95
    if pca_result["n_components"] > 1:
        assert cumulative[pca_result["n_components"] - 2] < 0.95

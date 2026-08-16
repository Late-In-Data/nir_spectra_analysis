import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from corn_nir.variable_selection import (
    compute_vip,
    literature_band_overlap,
    select_wavelengths_by_vip,
)


@pytest.fixture
def synthetic_data():
    """X has one strongly informative column and many pure-noise columns."""
    rng = np.random.default_rng(0)
    n, p = 100, 30
    informative_col = 5
    X = rng.normal(size=(n, p))
    y = 3.0 * X[:, informative_col] + 0.05 * rng.normal(size=n)
    return X, y, informative_col


def test_vip_shape_and_nonnegative(synthetic_data):
    X, y, _ = synthetic_data
    pls = PLSRegression(n_components=3, scale=False).fit(X, y)
    vip = compute_vip(pls, X)
    assert vip.shape == (X.shape[1],)
    assert np.all(vip >= 0)


def test_vip_ranks_informative_feature_highest(synthetic_data):
    X, y, informative_col = synthetic_data
    pls = PLSRegression(n_components=3, scale=False).fit(X, y)
    vip = compute_vip(pls, X)
    assert np.argmax(vip) == informative_col
    assert vip[informative_col] > 1.0


def test_vip_feature_count_mismatch_raises(synthetic_data):
    X, y, _ = synthetic_data
    pls = PLSRegression(n_components=3, scale=False).fit(X, y)
    with pytest.raises(ValueError):
        compute_vip(pls, X[:, :-1])


def test_select_wavelengths_by_vip(synthetic_data):
    X, y, informative_col = synthetic_data
    pls = PLSRegression(n_components=3, scale=False).fit(X, y)
    vip = compute_vip(pls, X)
    wavelength_nm = np.arange(X.shape[1])
    mask, selected = select_wavelengths_by_vip(vip, wavelength_nm, threshold=1.0)
    assert mask.dtype == bool
    assert informative_col in selected


def test_literature_band_overlap():
    selected = np.array([1900, 1950, 2000, 2400])
    result = literature_band_overlap(selected, (1908, 2108))
    assert result["n_selected"] == 4
    assert result["n_inside_band"] == 2
    assert result["fraction_inside_band"] == pytest.approx(0.5)


def test_literature_band_overlap_empty_selection():
    result = literature_band_overlap(np.array([]), (1908, 2108))
    assert result["n_selected"] == 0
    assert np.isnan(result["fraction_inside_band"])

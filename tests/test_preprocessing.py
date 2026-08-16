import numpy as np
import pytest

from corn_nir.preprocessing import (
    MSC,
    SNV,
    CropWavelengths,
    Detrend,
    MeanCenter,
    SavitzkyGolay,
    build_preprocessing,
)


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return rng.normal(loc=5.0, scale=2.0, size=(10, 50))


def test_mean_center_learns_train_mean_only(X):
    X_train, X_test = X[:6], X[6:]
    mc = MeanCenter().fit(X_train)
    out = mc.transform(X_test)
    np.testing.assert_allclose(out, X_test - X_train.mean(axis=0))


def test_snv_row_stats(X):
    out = SNV().fit_transform(X)
    means = out.mean(axis=1)
    stds = out.std(axis=1, ddof=1)
    np.testing.assert_allclose(means, np.zeros(X.shape[0]), atol=1e-10)
    np.testing.assert_allclose(stds, np.ones(X.shape[0]), atol=1e-10)


def test_msc_reference_is_fixed_from_train(X):
    X_train, X_test = X[:6], X[6:]
    msc = MSC().fit(X_train)
    reference_before = msc.reference_.copy()
    msc.transform(X_test)
    np.testing.assert_allclose(msc.reference_, reference_before)


def test_savitzky_golay_output_shape(X):
    sg = SavitzkyGolay(window_length=9, polyorder=2, deriv=1)
    out = sg.fit_transform(X)
    assert out.shape == X.shape


def test_crop_wavelengths_keeps_only_columns_above_threshold():
    wavelength_nm = np.linspace(1100, 2500, 50)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 50))
    out = CropWavelengths(wavelength_nm=wavelength_nm, crop_min_nm=1450).fit_transform(X)
    expected_n_cols = int((wavelength_nm >= 1450).sum())
    assert out.shape == (10, expected_n_cols)
    np.testing.assert_allclose(out, X[:, wavelength_nm >= 1450])


def test_detrend_removes_linear_trend():
    wavelength_nm = np.linspace(1100, 2500, 50)
    rng = np.random.default_rng(0)
    slopes = rng.normal(size=(10, 1))
    intercepts = rng.normal(size=(10, 1))
    X = slopes * wavelength_nm + intercepts  # pure linear baselines, no signal
    out = Detrend(wavelength_nm=wavelength_nm, degree=1).fit_transform(X)
    np.testing.assert_allclose(out, np.zeros_like(X), atol=1e-8)


def test_detrend_matches_shape_of_cropped_input():
    wavelength_nm = np.linspace(1100, 2500, 50)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 50))
    cropped_wl = wavelength_nm[wavelength_nm >= 1450]
    X_cropped = X[:, wavelength_nm >= 1450]
    out = Detrend(wavelength_nm=cropped_wl, degree=1).fit_transform(X_cropped)
    assert out.shape == X_cropped.shape
    assert np.isfinite(out).all()


def test_build_preprocessing_raw_is_none():
    assert build_preprocessing("raw") is None


def test_build_preprocessing_unknown_raises():
    with pytest.raises(ValueError):
        build_preprocessing("not_a_real_preprocessing")


@pytest.mark.parametrize(
    "name", ["mean_center", "snv", "msc", "sg_smooth", "sg_deriv1", "sg_deriv2"]
)
def test_build_preprocessing_runs_on_real_shape(name, X):
    transformer = build_preprocessing(name)
    out = transformer.fit_transform(X)
    assert out.shape == X.shape
    assert np.isfinite(out).all()

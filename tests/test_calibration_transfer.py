import numpy as np
import pytest

from corn_nir.calibration_transfer import (
    DirectStandardization,
    PiecewiseDirectStandardization,
    SlopeBiasCorrection,
)


@pytest.fixture
def paired_spectra():
    """Slave spectra are the master spectra (smooth, bounded, spectrum-like
    curves) plus a smooth wavelength-dependent multiplicative/additive
    offset — a case DS/PDS should be able to largely correct."""
    rng = np.random.default_rng(0)
    n, p = 20, 50
    wavelength = np.linspace(0, 1, p)
    bumps = np.stack([
        np.exp(-((wavelength - center) ** 2) / (2 * 0.05 ** 2))
        for center in (0.2, 0.5, 0.8)
    ])
    amplitudes = rng.uniform(0.5, 1.5, size=(n, 3))
    X_master = amplitudes @ bumps + 0.01 * rng.normal(size=(n, p))
    offset = 0.3 + 0.2 * np.sin(wavelength * 6)
    X_slave = X_master * 0.9 + offset
    return X_slave, X_master


def test_direct_standardization_reduces_discrepancy(paired_spectra):
    X_slave, X_master = paired_spectra
    ds = DirectStandardization(n_components=5).fit(X_slave, X_master)
    X_corrected = ds.transform(X_slave)

    error_before = np.mean(np.abs(X_slave - X_master))
    error_after = np.mean(np.abs(X_corrected - X_master))
    assert error_after < error_before


def test_direct_standardization_output_shape(paired_spectra):
    X_slave, X_master = paired_spectra
    ds = DirectStandardization(n_components=5).fit(X_slave, X_master)
    X_corrected = ds.transform(X_slave[:5])
    assert X_corrected.shape == (5, X_master.shape[1])


def test_direct_standardization_handles_few_transfer_samples(paired_spectra):
    X_slave, X_master = paired_spectra
    # Only 3 transfer samples (n_components must be clamped internally).
    ds = DirectStandardization(n_components=10).fit(X_slave[:3], X_master[:3])
    X_corrected = ds.transform(X_slave)
    assert X_corrected.shape == X_master.shape
    assert np.isfinite(X_corrected).all()


def test_piecewise_ds_reduces_discrepancy(paired_spectra):
    X_slave, X_master = paired_spectra
    pds = PiecewiseDirectStandardization(window=3, n_components=2).fit(X_slave, X_master)
    X_corrected = pds.transform(X_slave)

    error_before = np.mean(np.abs(X_slave - X_master))
    error_after = np.mean(np.abs(X_corrected - X_master))
    assert error_after < error_before


def test_piecewise_ds_output_shape(paired_spectra):
    X_slave, X_master = paired_spectra
    pds = PiecewiseDirectStandardization(window=3, n_components=2).fit(X_slave, X_master)
    X_corrected = pds.transform(X_slave[:5])
    assert X_corrected.shape == (5, X_master.shape[1])


@pytest.fixture
def paired_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(5, 15, size=20)
    y_pred_slave = 0.7 * y_true + 2.0 + 0.05 * rng.normal(size=20)  # linear offset/scale
    return y_pred_slave, y_true


def test_slope_bias_correction_reduces_error(paired_predictions):
    y_pred_slave, y_true = paired_predictions
    sbc = SlopeBiasCorrection().fit(y_pred_slave, y_true)
    y_corrected = sbc.transform(y_pred_slave)

    error_before = np.mean(np.abs(y_pred_slave - y_true))
    error_after = np.mean(np.abs(y_corrected - y_true))
    assert error_after < error_before


def test_slope_bias_correction_recovers_known_linear_relationship():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_slave = 2.0 * y_true + 1.0  # exact linear relationship, no noise
    sbc = SlopeBiasCorrection().fit(y_pred_slave, y_true)
    np.testing.assert_allclose(sbc.slope_, 0.5, atol=1e-8)
    np.testing.assert_allclose(sbc.intercept_, -0.5, atol=1e-8)
    np.testing.assert_allclose(sbc.transform(y_pred_slave), y_true, atol=1e-8)

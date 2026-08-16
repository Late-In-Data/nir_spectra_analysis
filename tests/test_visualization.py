from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from corn_nir.data import load_corn_mat
from corn_nir.visualization import (
    compute_instrument_differences,
    plot_difference_curves,
    plot_nbs_standards,
    plot_parity,
    plot_pca_by_instrument,
    plot_pls_scores_by_target,
    compute_spectra_vs_mean_diagnostic,
    plot_spectra_colored_by_all_targets,
    plot_spectra_colored_by_target,
    plot_pca_loadings,
    plot_pca_scores_by_target,
    plot_spectra_overlay,
    plot_spectra_vs_mean_spectrum,
    plot_target_correlation_matrix,
    plot_target_distributions,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "corn.mat"


@pytest.fixture(scope="module")
def dataset():
    return load_corn_mat(DATA_PATH)


def test_spectra_overlay_runs(dataset):
    fig = plot_spectra_overlay(dataset.wavelength_nm, dataset.spectra)
    assert fig is not None


def test_target_distributions_runs(dataset):
    fig = plot_target_distributions(dataset.targets)
    assert fig is not None


def test_nbs_standards_runs(dataset):
    fig = plot_nbs_standards(dataset.wavelength_nm, dataset.nbs)
    assert fig is not None


def test_pca_by_instrument_runs(dataset):
    fig, pca = plot_pca_by_instrument(dataset.spectra)
    assert pca.explained_variance_ratio_.shape == (2,)


def test_compute_instrument_differences(dataset):
    diffs, stats = compute_instrument_differences(dataset.spectra, reference="m5")
    assert set(diffs.keys()) == {"mp5", "mp6"}
    for name in ("mp5", "mp6"):
        assert diffs[name].shape == dataset.spectra["m5"].shape
        assert stats[name]["mean_abs_diff"] >= 0
        assert stats[name]["max_abs_diff"] >= stats[name]["mean_abs_diff"]


def test_difference_curves_runs(dataset):
    fig, stats = plot_difference_curves(dataset.wavelength_nm, dataset.spectra)
    assert set(stats.keys()) == {"mp5", "mp6"}


def test_target_correlation_matrix_runs(dataset):
    fig, corr = plot_target_correlation_matrix(dataset.targets)
    assert corr.shape == (4, 4)
    for col in dataset.target_names:
        assert corr.loc[col, col] == pytest.approx(1.0)


def test_pls_scores_by_target_runs(dataset):
    fig, pls = plot_pls_scores_by_target(dataset.spectra["m5"], dataset.targets, n_components=2)
    assert pls.x_scores_.shape == (80, 2)


def test_plot_parity_runs(dataset):
    y = dataset.targets["Moisture"].values
    fig = plot_parity(y, y + 0.1, title="test parity")
    assert fig is not None


def test_spectra_vs_mean_spectrum_runs(dataset):
    fig = plot_spectra_vs_mean_spectrum(dataset.spectra["m5"], title="test diagnostic")
    assert fig is not None


def test_plot_calibration_vs_loocv_curve_runs():
    import numpy as np

    from corn_nir.visualization import plot_calibration_vs_loocv_curve

    rmse_cal = np.array([0.5, 0.4, 0.3, 0.25])
    rmse_loocv = np.array([0.6, 0.5, 0.45, 0.44])
    fig = plot_calibration_vs_loocv_curve(rmse_cal, rmse_loocv, title="test curve")
    assert fig is not None


def test_plot_pca_scores_by_target_runs(dataset):
    from corn_nir.pca import compute_pca

    pca_result = compute_pca(dataset.spectra["m5"], n_components=2)
    fig = plot_pca_scores_by_target(
        pca_result["scores"], dataset.targets["Moisture"].values,
        pca_result["explained_variance_ratio"], target_name="Moisture",
    )
    assert fig is not None


def test_plot_pca_loadings_runs(dataset):
    from corn_nir.pca import compute_pca

    pca_result = compute_pca(dataset.spectra["m5"], n_components=3)
    fig = plot_pca_loadings(dataset.wavelength_nm, pca_result["components"])
    assert fig is not None


def test_spectra_vs_mean_spectrum_subsamples_large_input():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 700))
    fig = plot_spectra_vs_mean_spectrum(X, max_points=500)
    assert fig is not None


def _spread_low_vs_high(x_vals, y_vals):
    import numpy as np

    x_vals, y_vals = np.asarray(x_vals), np.asarray(y_vals)
    residual = y_vals - x_vals
    low = x_vals < np.median(x_vals)
    return residual[low].std(), residual[~low].std()


def test_diagnostic_additive_effect_gives_constant_spread():
    # Additive effect: a per-sample offset added to every wavelength -> the
    # scatter's vertical spread around y=x (computed by the diagnostic) stays
    # roughly constant regardless of the mean-spectrum value ("millefeuille").
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.normal(loc=1.0, scale=0.05, size=(80, 50)) + np.linspace(0, 2, 50)
    additive = base + rng.normal(scale=0.3, size=(80, 1))

    x_vals, y_vals = compute_spectra_vs_mean_diagnostic(additive, max_points=None)
    spread_low, spread_high = _spread_low_vs_high(x_vals, y_vals)
    assert spread_high == pytest.approx(spread_low, rel=0.5)


def test_diagnostic_multiplicative_effect_gives_widening_spread():
    # Multiplicative effect: a per-sample scale factor -> the diagnostic's spread
    # grows with the mean-spectrum value ("cone").
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.normal(loc=1.0, scale=0.05, size=(80, 50)) + np.linspace(0, 2, 50)
    multiplicative = base * (1 + rng.normal(scale=0.3, size=(80, 1)))

    x_vals, y_vals = compute_spectra_vs_mean_diagnostic(multiplicative, max_points=None)
    spread_low, spread_high = _spread_low_vs_high(x_vals, y_vals)
    assert spread_high > spread_low * 1.5


def test_spectra_colored_by_target_runs(dataset):
    fig = plot_spectra_colored_by_target(
        dataset.wavelength_nm, dataset.spectra["m5"], dataset.targets["Moisture"].values,
        target_name="Moisture",
    )
    assert fig is not None
    assert len(fig.axes) == 2  # spectra axis + colorbar axis


def test_spectra_colored_by_all_targets_runs(dataset):
    fig = plot_spectra_colored_by_all_targets(dataset.wavelength_nm, dataset.spectra["m5"], dataset.targets)
    assert fig is not None
    assert len(fig.axes) == 2 * dataset.targets.shape[1]  # one spectra + one colorbar axis per target


def test_diagnostic_subsamples_large_input():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 700))
    x_vals, y_vals = compute_spectra_vs_mean_diagnostic(X, max_points=500)
    assert len(x_vals) == len(y_vals) == 500

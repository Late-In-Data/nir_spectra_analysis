from pathlib import Path

import numpy as np
import pytest

from corn_nir.data import (
    N_SAMPLES,
    N_WAVELENGTHS,
    load_corn_mat,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "corn.mat"


@pytest.fixture(scope="module")
def dataset():
    return load_corn_mat(DATA_PATH)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_corn_mat("does/not/exist.mat")


def test_wavelength_axis(dataset):
    assert len(dataset.wavelength_nm) == N_WAVELENGTHS
    assert dataset.wavelength_nm[0] == 1100
    assert dataset.wavelength_nm[-1] == 2498
    np.testing.assert_array_equal(
        dataset.wavelength_nm, np.arange(1100, 2500, 2, dtype=float)
    )


def test_spectra_shapes(dataset):
    assert set(dataset.spectra.keys()) == {"m5", "mp5", "mp6"}
    for instrument, matrix in dataset.spectra.items():
        assert matrix.shape == (N_SAMPLES, N_WAVELENGTHS), instrument
        assert not np.isnan(matrix).any(), instrument


def test_targets(dataset):
    assert dataset.targets.shape == (N_SAMPLES, 4)
    assert dataset.target_names == ["Moisture", "Oil", "Protein", "Starch"]
    assert list(dataset.targets.columns) == dataset.target_names
    assert not dataset.targets.isna().any().any()


def test_nbs_shapes(dataset):
    assert set(dataset.nbs.keys()) == {"m5", "mp5", "mp6"}
    for instrument, matrix in dataset.nbs.items():
        assert matrix.shape[1] == N_WAVELENGTHS, instrument
        assert matrix.shape[0] > 0, instrument
        assert not np.isnan(matrix).any(), instrument

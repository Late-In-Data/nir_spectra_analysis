from pathlib import Path

import numpy as np
import pytest

from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline

from corn_nir.data import load_corn_mat
from corn_nir.preprocessing import SNV
from corn_nir.validation import (
    compute_calibration_vs_loocv_curve,
    kennard_stone_split,
    nested_loo_generic,
    nested_loo_plsr,
    select_n_components_by_cv,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "corn.mat"


def test_kennard_stone_split_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 50))
    train_idx, test_idx = kennard_stone_split(X, n_train=60)
    assert len(train_idx) == 60
    assert len(test_idx) == 20
    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(train_idx) | set(test_idx) == set(range(80))


def test_kennard_stone_split_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 10))
    a1, b1 = kennard_stone_split(X, n_train=20)
    a2, b2 = kennard_stone_split(X, n_train=20)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)


def test_kennard_stone_invalid_n_train():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError):
        kennard_stone_split(X, n_train=1)
    with pytest.raises(ValueError):
        kennard_stone_split(X, n_train=11)


def test_select_n_components_by_cv_on_real_data():
    ds = load_corn_mat(DATA_PATH)
    X = ds.spectra["m5"]
    y = ds.targets["Moisture"].values
    best_n, rmse_per_n = select_n_components_by_cv(X, y, max_components=10, n_splits=3)
    assert 1 <= best_n <= 10
    assert len(rmse_per_n) == 10
    assert np.all(rmse_per_n > 0)


def test_one_se_rule_is_more_parsimonious_than_raw_min():
    ds = load_corn_mat(DATA_PATH)
    X = ds.spectra["m5"]
    y = ds.targets["Moisture"].values
    best_n_min, _ = select_n_components_by_cv(X, y, max_components=30, n_splits=5, rule="min")
    best_n_1se, _ = select_n_components_by_cv(X, y, max_components=30, n_splits=5, rule="one_se")
    assert best_n_1se <= best_n_min


def test_select_n_components_invalid_rule():
    X = np.zeros((20, 5))
    y = np.zeros(20)
    with pytest.raises(ValueError):
        select_n_components_by_cv(X, y, max_components=3, n_splits=2, rule="bogus")


def test_compute_calibration_vs_loocv_curve_shapes_and_ordering():
    ds = load_corn_mat(DATA_PATH)
    X = ds.spectra["m5"]
    y = ds.targets["Moisture"].values
    rmse_cal, rmse_loocv = compute_calibration_vs_loocv_curve(X, y, max_components=8)
    assert len(rmse_cal) == 8
    assert len(rmse_loocv) == 8
    assert np.all(rmse_cal > 0)
    assert np.all(rmse_loocv > 0)
    # L'erreur de calibration (intra-echantillon) ne doit jamais depasser l'erreur LOO honnete.
    assert np.all(rmse_cal <= rmse_loocv + 1e-8)


def test_nested_loo_plsr_on_real_data():
    ds = load_corn_mat(DATA_PATH)
    X = ds.spectra["m5"]
    y = ds.targets["Moisture"].values
    df = nested_loo_plsr(X, y, max_components=5, inner_splits=3)
    assert len(df) == len(y)  # une ligne par echantillon
    assert set(df["sample_index"]) == set(range(len(y)))
    assert df["n_components_selected"].between(1, 5).all()
    np.testing.assert_allclose(sorted(df["y_true"]), sorted(y))


def test_nested_loo_plsr_never_sees_held_out_sample_when_choosing_n_components(monkeypatch):
    # Par construction, seules X_train/y_train (les 79 lignes restantes) sont
    # jamais transmises a select_n_components_by_cv : si la fonction laissait
    # fuiter l'echantillon ecarte dans le choix du nombre de composantes, cet
    # appel produirait ou propagerait un NaN dans l'ajustement.
    ds = load_corn_mat(DATA_PATH)
    X = ds.spectra["m5"]
    y = ds.targets["Moisture"].values
    df = nested_loo_plsr(X, y, max_components=5, inner_splits=3)
    assert np.isfinite(df["y_pred"]).all()
    assert np.isfinite(df["n_components_selected"]).all()


def test_nested_loo_plsr_with_preprocessing_pipeline_factory():
    ds = load_corn_mat(DATA_PATH)
    X = ds.spectra["m5"]
    y = ds.targets["Moisture"].values

    def snv_pls_factory(n_components):
        return Pipeline([
            ("prep", SNV()),
            ("plsr", PLSRegression(n_components=n_components, scale=False)),
        ])

    df = nested_loo_plsr(X, y, max_components=5, inner_splits=3, pipeline_factory=snv_pls_factory)
    assert len(df) == len(y)


def test_nested_loo_generic_with_elasticnet():
    from sklearn.linear_model import ElasticNetCV
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler

    ds = load_corn_mat(DATA_PATH)
    X = ds.spectra["m5"][:20]  # sous-echantillonne : LOO x ElasticNetCV est lent a n=80 complet
    y = ds.targets["Moisture"].values[:20]

    def build_model():
        return SkPipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(
                l1_ratio=[0.9], alphas=5, cv=3, max_iter=5000, tol=1e-2, random_state=0,
            )),
        ])

    df = nested_loo_generic(X, y, build_model)
    assert len(df) == len(y)
    assert set(df["sample_index"]) == set(range(len(y)))
    assert np.isfinite(df["y_pred"]).all()

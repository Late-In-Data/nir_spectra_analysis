import numpy as np
import pytest

from corn_nir.models import (
    build_gradient_boosting,
    build_plsr,
    build_random_forest,
    build_ridge,
    build_svr_rbf,
    fit_plsr_with_cv_components,
)


@pytest.fixture
def Xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 20))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=40)
    return X, y


def test_build_plsr_fits_and_predicts(Xy):
    X, y = Xy
    model = build_plsr(n_components=3)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]


def test_build_ridge_fits_and_predicts(Xy):
    X, y = Xy
    model = build_ridge()
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]


def test_build_svr_rbf_fits_and_predicts(Xy):
    X, y = Xy
    model = build_svr_rbf(cv=3)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]


def test_build_random_forest_fits_and_predicts(Xy):
    X, y = Xy
    model = build_random_forest(cv=3, n_iter=3, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]


def test_build_gradient_boosting_fits_and_predicts(Xy):
    X, y = Xy
    model = build_gradient_boosting(cv=3, n_iter=3, random_state=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]


def test_fit_plsr_with_cv_components(Xy):
    X, y = Xy
    model, best_n = fit_plsr_with_cv_components(X, y, max_components=10, inner_splits=3)
    assert 1 <= best_n <= 10
    pred = model.predict(X)
    assert pred.shape[0] == X.shape[0]

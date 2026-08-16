import numpy as np
import pytest

from corn_nir.evaluation import regression_metrics


def test_regression_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])

    m = regression_metrics(y_true, y_pred)

    assert m["RMSE"] == pytest.approx(0.5, rel=1e-6)
    assert m["MAE"] == pytest.approx(0.25, rel=1e-6)
    assert m["R2"] == pytest.approx(0.8, rel=1e-6)
    assert m["RPD"] == pytest.approx(1.290994 / 0.5, rel=1e-4)


def test_regression_metrics_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y_true, y_true)
    assert m["RMSE"] == pytest.approx(0.0)
    assert m["MAE"] == pytest.approx(0.0)
    assert m["R2"] == pytest.approx(1.0)
    assert m["RPD"] == float("inf")

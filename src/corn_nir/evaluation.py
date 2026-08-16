"""Metriques de regression pour le benchmark Corn NIR."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Calcule RMSE, MAE, R2 et RPD pour un ensemble de predictions.

    Le RPD (Residual Predictive Deviation) est defini ici comme
    ``std(y_true, ddof=1) / RMSE`` : le rapport entre l'ecart-type de la
    population de reference et l'erreur de prediction. C'est une convention
    de chimiometrie, pas une metrique scikit-learn, d'ou son calcul explicite
    ici plutot qu'un import direct.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    std_ref = float(np.std(y_true, ddof=1))
    rpd = std_ref / rmse if rmse > 0 else float("inf")

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "RPD": rpd}

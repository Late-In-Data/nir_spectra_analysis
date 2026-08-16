"""Constructeurs de modeles pour le benchmark Corn NIR."""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .validation import select_n_components_by_cv


def build_plsr(n_components: int, scale: bool = False) -> PLSRegression:
    """Construit un modele PLSR. `scale=False` conserve les unites
    d'absorbance brutes (la PLSR centre quand meme X et y en interne) ; la
    mise a l'echelle par longueur d'onde est un choix de pretraitement gere
    explicitement dans la comparaison de pretraitements, pas integre par
    defaut dans le modele de base."""
    return PLSRegression(n_components=n_components, scale=scale)


def fit_plsr_with_cv_components(X, y, max_components: int = 10, inner_splits: int = 5,
                                 random_state: int = 0):
    """Choisit le nombre de composantes PLS par CV (regle du 1-ecart-type,
    voir `validation.select_n_components_by_cv`) puis ajuste le modele final
    sur l'ensemble de (X, y). Retourne (model, best_n_components).

    C'est le schema "ajuster un modele PLSR final/deployable" reutilise a
    plusieurs endroits du projet (modeles d'interpretabilite en Phase E,
    baselines inter-instruments en Phases G/H, demo Streamlit) : factorise
    ici une seule fois pour que chaque appelant partage exactement la meme
    regle de selection de composantes et la meme convention `scale=False`.
    """
    best_n, _ = select_n_components_by_cv(
        X, y, max_components=max_components, n_splits=inner_splits, random_state=random_state,
    )
    model = build_plsr(best_n)
    model.fit(X, y)
    return model, best_n


def build_ridge() -> Pipeline:
    """Regression Ridge avec sa propre recherche d'alpha par validation
    croisee generalisee (forme fermee efficace, sans reajustement explicite)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 4, 15))),
    ])


def build_svr_rbf(cv: int = 3) -> GridSearchCV:
    """SVR a noyau RBF ; C/gamma/epsilon regles par une recherche par
    grille interne (grille volontairement modeste pour que la CV imbriquee
    reste praticable)."""
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    param_grid = {
        "svr__C": [1, 10, 100],
        "svr__gamma": ["scale", 0.01, 0.001],
        "svr__epsilon": [0.01, 0.1],
    }
    return GridSearchCV(pipe, param_grid, cv=cv, scoring="neg_root_mean_squared_error")


def build_random_forest(cv: int = 3, n_iter: int = 8, random_state: int = 0) -> RandomizedSearchCV:
    """Foret aleatoire avec une petite recherche aleatoire d'hyperparametres."""
    rf = RandomForestRegressor(random_state=random_state)
    param_dist = {
        "n_estimators": [100, 300],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", 0.3, 1.0],
    }
    return RandomizedSearchCV(
        rf, param_dist, n_iter=n_iter, cv=cv, random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )


def build_gradient_boosting(cv: int = 3, n_iter: int = 8, random_state: int = 0) -> RandomizedSearchCV:
    """Gradient Boosting (implementation native de scikit-learn, sans
    dependance supplementaire) avec une petite recherche aleatoire
    d'hyperparametres."""
    gbm = GradientBoostingRegressor(random_state=random_state)
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    return RandomizedSearchCV(
        gbm, param_dist, n_iter=n_iter, cv=cv, random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )

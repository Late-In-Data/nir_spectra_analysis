"""Outils de validation sans fuite pour le benchmark Corn NIR.

Toute fonction qui regle un hyperparametre (ex. le nombre de composantes PLS)
ne doit jamais voir que le pli d'entrainement avec lequel elle est appelee :
c'est a l'appelant de ne jamais lui transmettre de donnees de test mises de cote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    cross_val_predict,
    cross_val_score,
)

from .evaluation import regression_metrics


def kennard_stone_split(X, n_train: int):
    """Echantillonnage sequentiel de Kennard-Stone (Kennard & Stone, 1969).

    Selectionne deterministement `n_train` echantillons qui maximisent la
    couverture spectrale, donnant un split calibration/prediction
    reproductible, comparable aux splits utilises dans la litterature sur ce
    jeu de donnees (ex. 60/20). Retourne (train_idx, test_idx), le premier
    dans l'ordre de selection et le second par ordre croissant.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if not (2 <= n_train <= n):
        raise ValueError(f"n_train doit etre dans [2, {n}], recu {n_train}")

    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [int(i), int(j)]
    remaining = set(range(n)) - {i, j}
    min_dist_to_selected = np.minimum(dist[i], dist[j])

    while len(selected) < n_train:
        remaining_list = np.fromiter(remaining, dtype=int)
        next_idx = int(remaining_list[np.argmax(min_dist_to_selected[remaining_list])])
        selected.append(next_idx)
        remaining.discard(next_idx)
        min_dist_to_selected = np.minimum(min_dist_to_selected, dist[next_idx])

    train_idx = np.array(selected)
    test_idx = np.array(sorted(remaining))
    return train_idx, test_idx


def _default_pipeline_factory(n_components: int):
    return PLSRegression(n_components=n_components, scale=False)


def select_n_components_by_cv(
    X, y, max_components: int = 10, n_splits: int = 5, random_state: int = 0,
    rule: str = "one_se", pipeline_factory=None,
):
    """Choisit un nombre de composantes PLS a partir du RMSE de CV sur (X, y).
    A appeler uniquement sur un pli d'entrainement. Retourne
    (best_n_components, rmse_per_n).

    Avec 700 longueurs d'onde fortement colineaires et n<=80 echantillons, la
    courbe RMSE-de-CV en fonction du nombre de composantes ne plafonne jamais
    sur ce jeu de donnees (verifie jusqu'a 30 composantes) : elle continue de
    s'ameliorer regulierement, et la regle du 1-ecart-type seule ne peut pas
    detecter un coude qui n'existe pas ; elle empeche seulement le
    depassement *a l'interieur* du plafond `max_components` donne, tout en
    suivant ce plafond de facon quasi lineaire (ex. mode 9/12/14/17
    composantes selectionnees pour des plafonds de 10/12/15/20 sur Moisture).
    Le defaut de 10 ne vient donc pas d'un point ou la courbe repart vers le
    haut (elle ne le fait pas, dans la plage testee), mais du rapport
    echantillons/variables (regle empirique de CheMOOCs Grain 11 d'au moins
    10 echantillons par dimension du modele, plus difficile a satisfaire ici
    avec ~60-64 echantillons d'entrainement par pli) et de l'alignement avec
    le plafond de composantes utilise dans la litterature sur ce meme jeu de
    donnees (Cataltas & Tutuncu, 2023, PeerJ Computer Science, PLSR plafonne
    a 10 variables latentes). `rule="one_se"` (par defaut) applique la regle
    standard du 1-ecart-type (Hastie, Tibshirani & Friedman, *The Elements of
    Statistical Learning*, section 7.10) a l'interieur de ce plafond, en
    choisissant le plus petit nombre de composantes dont le RMSE moyen de CV
    est a moins d'un ecart-type du minimum atteint. `rule="min"` revient au
    minimiseur brut (toujours plafonne a `max_components`).

    `pipeline_factory(n_components) -> estimateur` construit le modele a
    evaluer pour un nombre de composantes donne ; par defaut, une simple
    `PLSRegression`. Passer une factory qui enveloppe une etape de
    pretraitement (ex. `Pipeline([("prep", SNV()), ("plsr", PLSRegression(...))])`)
    pour comparer des variantes de pretraitement avec exactement le meme
    moteur de CV : l'etape de pretraitement est reajustee sur chaque pli
    d'entrainement par `cross_val_score`, donc elle ne voit jamais les
    donnees du pli de validation.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    max_components = max(1, min(max_components, X.shape[0] - 1, X.shape[1]))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipeline_factory = pipeline_factory or _default_pipeline_factory

    rmse_per_fold = np.empty((max_components, n_splits))
    for n in range(1, max_components + 1):
        model = pipeline_factory(n)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        rmse_per_fold[n - 1] = -scores

    rmse_per_n = rmse_per_fold.mean(axis=1)

    if rule == "min":
        best_n = int(np.argmin(rmse_per_n)) + 1
    elif rule == "one_se":
        sem_per_n = rmse_per_fold.std(axis=1, ddof=1) / np.sqrt(n_splits)
        min_idx = int(np.argmin(rmse_per_n))
        threshold = rmse_per_n[min_idx] + sem_per_n[min_idx]
        best_n = int(np.argmax(rmse_per_n <= threshold)) + 1
    else:
        raise ValueError(f"Regle '{rule}' inconnue, attendu 'one_se' ou 'min'")

    return best_n, rmse_per_n


def compute_calibration_vs_loocv_curve(X, y, max_components: int = 15, pipeline_factory=None):
    """RMSE de calibration (intra-echantillon) vs RMSE de CV leave-one-out,
    pour chaque nombre de composantes de 1 a `max_components`. C'est la
    courbe classique calibration-vs-validation utilisee en chimiometrie pour
    choisir un nombre de composantes PLS a l'oeil (ex. CheMOOCs Grain 11,
    fig. 5), avec le leave-one-out comme schema de validation plutot que le
    k-fold : avec seulement `len(y)` echantillons, la LOO utilise chaque
    echantillon comme validation exactement une fois tout en entrainant sur
    n-1 echantillons a chaque fois, ce qui est le choix standard quand
    l'effectif est trop faible pour se permettre de perdre des donnees
    d'entrainement en le decoupant davantage.

    Retourne (rmse_calibration, rmse_loocv), deux tableaux de longueur
    `max_components`. Le RMSE de calibration est ajuste et evalue sur
    l'ensemble complet (X, y) sans aucune donnee mise de cote : ce n'est pas
    une estimation de generalisation, seulement l'autre moitie de la
    comparaison (un petit ecart avec la courbe LOO suggere que l'ajustement
    n'est pas mauvais ; un ecart large et croissant avec le nombre de
    composantes est la signature classique du surajustement).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    max_components = max(1, min(max_components, X.shape[0] - 1, X.shape[1]))
    pipeline_factory = pipeline_factory or _default_pipeline_factory
    loo = LeaveOneOut()

    rmse_calibration = np.empty(max_components)
    rmse_loocv = np.empty(max_components)
    for n in range(1, max_components + 1):
        model = pipeline_factory(n)
        model.fit(X, y)
        y_pred_cal = model.predict(X)
        rmse_calibration[n - 1] = np.sqrt(np.mean((y - y_pred_cal.ravel()) ** 2))

        y_pred_loo = cross_val_predict(pipeline_factory(n), X, y, cv=loo)
        rmse_loocv[n - 1] = np.sqrt(np.mean((y - y_pred_loo.ravel()) ** 2))

    return rmse_calibration, rmse_loocv


def nested_loo_plsr(
    X, y, max_components: int = 10, inner_splits: int = 5, random_state: int = 0,
    rule: str = "one_se", pipeline_factory=None,
) -> pd.DataFrame:
    """Validation croisee imbriquee leave-one-out pour une baseline PLSR
    (precedee optionnellement d'une etape de pretraitement, voir
    `pipeline_factory` dans `select_n_components_by_cv`).

    La boucle externe est un leave-one-out (LOO) : adapte a cette taille
    d'echantillon (n=80) puisque chaque echantillon sert de validation
    exactement une fois tout en entrainant sur les 79 autres a chaque fois,
    sans mettre de cote un split separe. Pour chacune des 80 iterations
    externes, une CV interne (`select_n_components_by_cv`, meme `rule`)
    choisit le nombre de composantes PLS en utilisant seulement les 79
    echantillons d'entrainement : l'echantillon laisse de cote n'influence
    jamais le `n_components` choisi, seulement le residu unique qu'il
    produit une fois predit.

    Retourne un DataFrame avec une ligne par echantillon laisse de cote
    (`sample_index`, `y_true`, `y_pred`, `n_components_selected`) ; le
    RMSE/MAE/R2/RPD n'ont pas de sens sur une seule ligne (n=1) : les agreger
    avec `corn_nir.evaluation.regression_metrics(df["y_true"], df["y_pred"])`
    pour une estimation globale honnete, et resumer `n_components_selected`
    separement (ex. son mode).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    pipeline_factory = pipeline_factory or _default_pipeline_factory
    loo = LeaveOneOut()

    rows = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_n, _ = select_n_components_by_cv(
            X_train, y_train, max_components=max_components, n_splits=inner_splits,
            random_state=random_state, rule=rule, pipeline_factory=pipeline_factory,
        )
        model = pipeline_factory(best_n)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({
            "sample_index": int(test_idx[0]),
            "y_true": float(y_test[0]),
            "y_pred": float(np.ravel(y_pred)[0]),
            "n_components_selected": best_n,
        })

    return pd.DataFrame(rows)


def nested_loo_generic(X, y, build_model) -> pd.DataFrame:
    """CV leave-one-out pour un estimateur avec sa propre recherche
    d'hyperparametres integree (ex. `ElasticNetCV`, `GridSearchCV`, un
    `Pipeline` enveloppant l'un des deux).

    `build_model()` doit retourner un estimateur neuf et non ajuste a chaque
    appel. Seul le decoupage LOO externe est fait ici : la CV interne propre
    de l'estimateur (pour alpha, l1_ratio, etc.) est ajustee exclusivement
    sur les 79 echantillons d'entrainement de chaque iteration, puisque
    `build_model()` n'est jamais `.fit()` que sur cet ensemble
    d'entrainement.

    Retourne un DataFrame avec une ligne par echantillon laisse de cote
    (`sample_index`, `y_true`, `y_pred`) ; les agreger avec
    `corn_nir.evaluation.regression_metrics` pour une estimation globale
    honnete ; le RMSE/MAE/R2/RPD n'ont pas de sens sur une seule ligne.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    loo = LeaveOneOut()

    rows = []
    for train_idx, test_idx in loo.split(X):
        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        rows.append({
            "sample_index": int(test_idx[0]),
            "y_true": float(y[test_idx][0]),
            "y_pred": float(np.ravel(y_pred)[0]),
        })

    return pd.DataFrame(rows)

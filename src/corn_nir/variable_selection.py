"""Outils d'importance des longueurs d'onde / variables pour le benchmark Corn NIR."""

from __future__ import annotations

import numpy as np


def compute_vip(pls, X) -> np.ndarray:
    """Scores VIP (Variable Importance in Projection) pour un modele
    `sklearn.cross_decomposition.PLSRegression` ajuste (Wold, Sjostrom &
    Eriksson, 2001 ; Chong & Jun, 2005) :

        VIP_j = sqrt( p * somme_k( w_jk^2 * SSY_k ) / somme_k(SSY_k) )

    ou `p` est le nombre de predicteurs, `w_jk` est le poids PLS (norme
    unitaire) de la variable j sur la composante k, et
    `SSY_k = q_k^2 * (t_k^T t_k)` est la part de variance de Y expliquee par
    la composante k (`t_k` le vecteur de scores, `q_k` le loading en y). Une
    variable avec VIP > 1 contribue plus que la variable moyenne au pouvoir
    explicatif du modele, le seuil de selection conventionnel (heuristique,
    pas statistiquement exact).

    `X` sert uniquement a valider le nombre de variables contre `pls` ; le
    calcul du VIP lui-meme n'a besoin que des attributs PLS deja ajustes du
    modele.
    """
    X = np.asarray(X, dtype=float)
    t = pls.x_scores_          # (n_echantillons, n_composantes)
    w = pls.x_weights_         # (n_variables, n_composantes)
    q = pls.y_loadings_        # (n_cibles, n_composantes)

    p, h = w.shape
    if X.shape[1] != p:
        raise ValueError(f"X a {X.shape[1]} variables, le modele a ete ajuste sur {p}")

    # SSY_k = q_k^2 * ||t_k||^2, en utilisant la premiere (seule, pour une cible unique) reponse
    ssy = (np.sum(t ** 2, axis=0)) * (q[0] ** 2)  # forme (h,)
    total_ssy = np.sum(ssy)

    weight_sq = (w / np.linalg.norm(w, axis=0, keepdims=True)) ** 2  # (p, h)
    vip = np.sqrt(p * (weight_sq @ ssy) / total_ssy)
    return vip


def select_wavelengths_by_vip(vip: np.ndarray, wavelength_nm, threshold: float = 1.0):
    """Retourne (mask, longueurs_d_onde_selectionnees) pour les longueurs
    d'onde dont le VIP depasse `threshold` (l'heuristique conventionnelle
    VIP > 1)."""
    vip = np.asarray(vip, dtype=float)
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    mask = vip > threshold
    return mask, wavelength_nm[mask]


def literature_band_overlap(selected_wavelengths, band_range: tuple[float, float]) -> dict:
    """Comparaison descriptive (non statistique) entre
    `selected_wavelengths` et une bande informative rapportee dans la
    litterature `(low_nm, high_nm)` : combien / quelle fraction de nos
    longueurs d'onde selectionnees tombent a l'interieur."""
    selected = np.asarray(selected_wavelengths, dtype=float)
    low, high = band_range
    inside = (selected >= low) & (selected <= high)
    return {
        "n_selected": int(len(selected)),
        "n_inside_band": int(inside.sum()),
        "fraction_inside_band": float(inside.mean()) if len(selected) else float("nan"),
    }

"""Methodes de transfert de calibration (Direct Standardization, Piecewise
Direct Standardization) pour projeter les spectres d'un instrument esclave
dans l'espace spectral de l'instrument maitre, a partir d'une poignee
d'echantillons de transfert apparies.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression


class DirectStandardization:
    """Direct Standardization (dans l'esprit de Wang & Kowalski, 1991) :
    apprend une projection esclave -> maitre a partir d'echantillons de
    transfert apparies.

    Une DS litterale resout une matrice de transformation complete
    (n_longueurs_d_onde x n_longueurs_d_onde), ce qui est totalement
    sous-determine avec seulement une poignee d'echantillons de transfert
    (700 longueurs d'onde contre au plus quelques dizaines d'echantillons).
    Ici, la projection est a la place une regression PLS2 des spectres
    maitres sur les spectres esclaves : la reduction de dimension integree
    de la PLS rend le probleme bien pose pour de petits ensembles de
    transfert, tout en apprenant une correction multivariee sur l'ensemble
    du spectre (pas independante longueur d'onde par longueur d'onde).
    """

    def __init__(self, n_components: int = 5):
        self.n_components = n_components

    def fit(self, X_slave_transfer, X_master_transfer):
        X_slave_transfer = np.asarray(X_slave_transfer, dtype=float)
        X_master_transfer = np.asarray(X_master_transfer, dtype=float)
        n_components = max(1, min(self.n_components, X_slave_transfer.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=n_components, scale=False)
        self.pls_.fit(X_slave_transfer, X_master_transfer)
        return self

    def transform(self, X_slave):
        return self.pls_.predict(np.asarray(X_slave, dtype=float))


class SlopeBiasCorrection:
    """Correction pente-biais (Slope and Bias Correction), la methode de
    transfert de calibration la plus simple, generalement la premiere
    essayee avant DS/PDS dans la litterature.

    Contrairement a DS/PDS, elle ne touche pas du tout au spectre : elle
    corrige les *predictions* du modele. A partir d'une poignee
    d'echantillons de transfert, elle prend les predictions que les spectres
    esclaves non corriges produisent deja (via le modele entraine sur m5) et
    ajuste une simple relation lineaire contre les vraies valeurs de
    reference : `y_true = pente * y_pred + ordonnee`. Cette seule droite est
    ensuite appliquee pour corriger toute nouvelle prediction.

    Seulement 2 parametres, contre une transformation spectrale complete
    pour DS/PDS : necessite beaucoup moins d'echantillons de transfert pour
    etre ajustee de facon fiable, mais ne corrige qu'un decalage/une echelle
    lineaire globale sur la sortie. Elle ne peut pas corriger une distorsion
    spectrale dependante de la longueur d'onde comme le font DS/PDS.
    """

    def fit(self, y_pred_transfer, y_true_transfer):
        y_pred_transfer = np.asarray(y_pred_transfer, dtype=float).ravel()
        y_true_transfer = np.asarray(y_true_transfer, dtype=float).ravel()
        self.slope_, self.intercept_ = np.polyfit(y_pred_transfer, y_true_transfer, deg=1)
        return self

    def transform(self, y_pred):
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        return self.slope_ * y_pred + self.intercept_


class PiecewiseDirectStandardization:
    """Piecewise Direct Standardization (Wang, Veltkamp & Kowalski, 1991) :
    pour chaque longueur d'onde maitre, une regression *locale* sur une
    fenetre de longueurs d'onde esclaves voisines est ajustee a partir des
    echantillons de transfert apparies, plutot qu'une seule projection
    globale sur tout le spectre. Capture une derive instrumentale dependante
    de la longueur d'onde (ex. deplacement de position de bande) qu'une
    transformation globale unique ne peut pas capturer.

    Chaque fenetre locale est ajustee par une PLS a faible `n_components`
    (pas par MCO) : la fenetre peut etre plus large que le nombre
    d'echantillons de transfert, donc une regression par fenetre non
    regularisee serait singuliere.
    """

    def __init__(self, window: int = 5, n_components: int = 2):
        self.window = window
        self.n_components = n_components

    def fit(self, X_slave_transfer, X_master_transfer):
        X_slave_transfer = np.asarray(X_slave_transfer, dtype=float)
        X_master_transfer = np.asarray(X_master_transfer, dtype=float)
        n_samples, n_wavelengths = X_slave_transfer.shape

        self._windows = []
        for j in range(n_wavelengths):
            lo = max(0, j - self.window)
            hi = min(n_wavelengths, j + self.window + 1)
            n_components = max(1, min(self.n_components, n_samples - 1, hi - lo))
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_slave_transfer[:, lo:hi], X_master_transfer[:, j])
            self._windows.append((lo, hi, pls))
        return self

    def transform(self, X_slave):
        X_slave = np.asarray(X_slave, dtype=float)
        n_samples = X_slave.shape[0]
        n_wavelengths = len(self._windows)
        X_corrected = np.empty((n_samples, n_wavelengths))
        for j, (lo, hi, pls) in enumerate(self._windows):
            X_corrected[:, j] = pls.predict(X_slave[:, lo:hi]).ravel()
        return X_corrected

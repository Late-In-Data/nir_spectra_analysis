"""Transformateurs de pretraitement spectral sans fuite pour le benchmark
Corn NIR : spectres bruts, centrage, SNV, MSC, lissage et derivees de
Savitzky-Golay. Tous les transformateurs implementent l'API fit/transform de
scikit-learn pour pouvoir s'inserer dans une `Pipeline` et n'etre ajustes que
sur le pli d'entrainement.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class MeanCenter(BaseEstimator, TransformerMixin):
    """Centrage par colonne (par longueur d'onde). La moyenne est apprise
    uniquement sur les donnees d'ajustement, donc sans risque dans une
    Pipeline ajustee pli par pli. A noter que `PLSRegression` centre deja ses
    donnees en interne : cette etape sert surtout de test de coherence,
    elle doit reproduire la baseline "brute" presque a l'identique.
    """

    def fit(self, X, y=None):
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.mean_


class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate : centrage et mise a l'echelle par echantillon
    (par ligne). Sans etat entre echantillons, aucun parametre inter-echantillon
    n'est appris, donc aucun risque de fuite quel que soit le pli sur lequel
    on l'applique.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, ddof=1, keepdims=True)
        return (X - mean) / std


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction contre un spectre de reference. La
    reference (spectre moyen des donnees d'ajustement) est apprise
    uniquement sur le pli d'ajustement puis reutilisee, jamais recalculee,
    lors de la transformation de nouvelles donnees. C'est la facon
    sans-fuite d'appliquer la MSC a l'interieur d'une validation croisee.
    """

    def fit(self, X, y=None):
        self.reference_ = np.asarray(X, dtype=float).mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_corrected = np.empty_like(X)
        for i in range(X.shape[0]):
            a, b = np.polyfit(self.reference_, X[i, :], deg=1)
            X_corrected[i, :] = (X[i, :] - b) / a
        return X_corrected


class CropWavelengths(BaseEstimator, TransformerMixin):
    """Ne garde que les colonnes dont la longueur d'onde est >= `crop_min_nm`.
    Un hyperparametre fixe (la borne de coupure), pas appris a partir des
    donnees, donc sans risque de fuite. Pensee pour se placer avant `Detrend`
    dans une Pipeline quand l'ajustement de la ligne de base ne doit voir
    qu'une sous-plage (ex. isoler la region a dominante additive d'une region
    a dominante multiplicative avant d'ajuster la ligne de base) : la coupure
    doit se faire avant l'ajustement, pas apres, car un ajustement par
    moindres carres est influence par chaque point qu'il voit.
    """

    def __init__(self, wavelength_nm, crop_min_nm: float):
        self.wavelength_nm = wavelength_nm
        self.crop_min_nm = crop_min_nm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mask = np.asarray(self.wavelength_nm, dtype=float) >= self.crop_min_nm
        return X[:, mask]


class Detrend(BaseEstimator, TransformerMixin):
    """Retrait de ligne de base polynomiale par echantillon (Barnes, Dhanoa &
    Lister, 1989) : pour chaque spectre, ajuste un polynome de degre `degree`
    contre la longueur d'onde par moindres carres et le soustrait. `degree=1`
    (par defaut) soustrait la droite de meilleur ajustement, ce qui distingue
    ce transformateur de `MeanCenter` (une constante, pas une droite), des
    derivees SG (une fenetre locale, pas un ajustement global) et de
    SNV/MSC (qui remettent aussi a l'echelle, pas seulement decalent la ligne
    de base). Sans etat entre echantillons comme SNV, aucun parametre
    inter-echantillon n'est appris, donc aucun risque de fuite.

    `wavelength_nm` doit correspondre aux colonnes du `X` transmis : si
    utilise apres avoir coupe une plage de longueurs d'onde, passer l'axe
    deja coupe, car un ajustement par moindres carres depend de chaque point
    qu'il voit : ajuster sur la plage complete puis couper apres n'est pas
    equivalent a couper d'abord et n'ajuster que sur la region conservee.
    """

    def __init__(self, wavelength_nm, degree: int = 1):
        self.wavelength_nm = wavelength_nm
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        wavelength_nm = np.asarray(self.wavelength_nm, dtype=float)
        design = np.vander(wavelength_nm, N=self.degree + 1, increasing=True)
        coeffs, _, _, _ = np.linalg.lstsq(design, X.T, rcond=None)
        baseline = (design @ coeffs).T
        return X - baseline


class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """Filtre de lissage / derivation de Savitzky-Golay. `window_length`,
    `polyorder` et `deriv` sont des hyperparametres fixes, pas appris a
    partir des donnees, donc ce transformateur ne presente pas non plus de
    risque de fuite entre plis.
    """

    def __init__(self, window_length: int = 13, polyorder: int = 2, deriv: int = 0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return savgol_filter(
            X, window_length=self.window_length, polyorder=self.polyorder,
            deriv=self.deriv, axis=1,
        )


# Variantes de pretraitement comparees dans le projet. Les parametres
# Savitzky-Golay sont fixes et documentes ici plutot que laisses implicites :
# window_length=13 (26 nm au pas de 2 nm du jeu de donnees) / polyorder=2
# pour le lissage et la 1ere derivee ; un window_length=17 / polyorder=3 plus
# large pour la 2e derivee, car la derivee seconde d'un polynome de degre 2
# est une constante par fenetre et serait sinon une estimation trop grossiere.
PREPROCESSING_NAMES = (
    "raw",
    "mean_center",
    "snv",
    "msc",
    "sg_smooth",
    "sg_deriv1",
    "sg_deriv2",
)


def build_preprocessing(name: str):
    """Retourne une instance neuve du transformateur pour un nom de
    `PREPROCESSING_NAMES`, ou None pour "raw" (aucune etape de pretraitement
    dans la pipeline)."""
    if name == "raw":
        return None
    if name == "mean_center":
        return MeanCenter()
    if name == "snv":
        return SNV()
    if name == "msc":
        return MSC()
    if name == "sg_smooth":
        return SavitzkyGolay(window_length=13, polyorder=2, deriv=0)
    if name == "sg_deriv1":
        return SavitzkyGolay(window_length=13, polyorder=2, deriv=1)
    if name == "sg_deriv2":
        return SavitzkyGolay(window_length=17, polyorder=3, deriv=2)
    raise ValueError(f"Pretraitement '{name}' inconnu, attendu l'un de {PREPROCESSING_NAMES}")

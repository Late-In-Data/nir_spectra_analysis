"""Chargeur du jeu de donnees Corn NIR (fichier .mat Cargill / Eigenvector Research)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

SPECTRA_KEYS = ("m5spec", "mp5spec", "mp6spec")
NBS_KEYS = ("m5nbs", "mp5nbs", "mp6nbs")
TARGETS_KEY = "propvals"
N_SAMPLES = 80
N_WAVELENGTHS = 700
WAVELENGTH_START_NM = 1100
WAVELENGTH_STOP_NM = 2500
WAVELENGTH_STEP_NM = 2


@dataclass
class CornDataset:
    """Conteneur pour le jeu de donnees benchmark Corn NIR."""

    wavelength_nm: np.ndarray
    spectra: dict[str, np.ndarray] = field(default_factory=dict)
    targets: pd.DataFrame = None
    target_names: list[str] = field(default_factory=list)
    nbs: dict[str, np.ndarray] = field(default_factory=dict)


def _extract_matrix(entry) -> np.ndarray:
    """Retourne une matrice float64 a partir d'un struct Dataset PLS_Toolbox ou d'un tableau brut."""
    if isinstance(entry, mat_struct):
        return np.asarray(entry.data, dtype=float)
    return np.asarray(entry, dtype=float)


def _extract_target_names(entry, n_targets: int) -> list[str]:
    """Lit les noms de colonnes depuis le champ `.label` d'un struct Dataset
    PLS_Toolbox, avec un repli numerique si le fichier ne porte pas de labels
    (ex. une variante en matrice brute)."""
    if isinstance(entry, mat_struct):
        try:
            raw_labels = entry.label[1][0]
            names = [str(name).strip() for name in np.asarray(raw_labels).ravel()]
            if len(names) == n_targets and all(names):
                return names
        except (AttributeError, IndexError, TypeError):
            pass
    return [f"target_{i}" for i in range(n_targets)]


def load_corn_mat(path: str | Path) -> CornDataset:
    """Charge le fichier `.mat` Corn NIR dans un :class:`CornDataset` valide.

    Gere a la fois les structs Dataset PLS_Toolbox d'origine
    (`scipy.io.loadmat` avec ``struct_as_record=False, squeeze_me=True``) et
    une variante en matrice NumPy brute du meme fichier, pour que le
    chargeur continue de fonctionner si le fichier source est ré-exporte.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier .mat Corn introuvable : {path}")

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    wavelength_nm = np.arange(
        WAVELENGTH_START_NM, WAVELENGTH_STOP_NM, WAVELENGTH_STEP_NM, dtype=float
    )
    if len(wavelength_nm) != N_WAVELENGTHS:
        raise AssertionError(
            f"{N_WAVELENGTHS} longueurs d'onde attendues, {len(wavelength_nm)} calculees"
        )

    spectra: dict[str, np.ndarray] = {}
    for key in SPECTRA_KEYS:
        if key not in mat:
            raise KeyError(f"Champ de spectres attendu '{key}' manquant dans {path}")
        instrument = key.replace("spec", "")
        matrix = _extract_matrix(mat[key])
        if matrix.shape != (N_SAMPLES, N_WAVELENGTHS):
            raise ValueError(
                f"'{key}' a la forme {matrix.shape}, ({N_SAMPLES}, {N_WAVELENGTHS}) attendue"
            )
        spectra[instrument] = matrix

    if TARGETS_KEY not in mat:
        raise KeyError(f"Champ de cibles attendu '{TARGETS_KEY}' manquant dans {path}")
    targets_matrix = _extract_matrix(mat[TARGETS_KEY])
    if targets_matrix.shape[0] != N_SAMPLES:
        raise ValueError(
            f"'{TARGETS_KEY}' a {targets_matrix.shape[0]} lignes, {N_SAMPLES} attendues"
        )
    target_names = _extract_target_names(mat[TARGETS_KEY], targets_matrix.shape[1])
    targets = pd.DataFrame(targets_matrix, columns=target_names)

    nbs: dict[str, np.ndarray] = {}
    for key in NBS_KEYS:
        if key not in mat:
            raise KeyError(f"Champ NBS attendu '{key}' manquant dans {path}")
        instrument = key.replace("nbs", "")
        matrix = _extract_matrix(mat[key])
        if matrix.shape[1] != N_WAVELENGTHS:
            raise ValueError(
                f"'{key}' a {matrix.shape[1]} colonnes de longueur d'onde, {N_WAVELENGTHS} attendues"
            )
        nbs[instrument] = matrix

    return CornDataset(
        wavelength_nm=wavelength_nm,
        spectra=spectra,
        targets=targets,
        target_names=target_names,
        nbs=nbs,
    )

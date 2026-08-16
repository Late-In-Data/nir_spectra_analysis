"""Fonctions graphiques reutilisables pour le projet Corn NIR (EDA + rapport)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


def plot_spectra_overlay(wavelength_nm, spectra: dict[str, np.ndarray], ax=None, sharey: bool = True, figsize=[5, 6]):
    """Superpose le spectre de chaque echantillon par panneau, plus sa
    moyenne. Un panneau par entree du dict `spectra` : fonctionne pour des
    instruments, mais aussi pour comparer par exemple des spectres bruts et
    une version pretraitee des memes donnees. `sharey=False` quand les
    panneaux sont a des echelles tres differentes (ex. absorbance brute vs
    une derivee)."""
    if ax is None:
        fig, ax = plt.subplots(1, len(spectra), figsize=(figsize[0] * len(spectra), figsize[1]), sharey=sharey)
        ax = np.atleast_1d(ax)
    else:
        fig = ax[0].figure

    colors = plt.get_cmap("tab10").colors
    for i, (label, X) in enumerate(spectra.items()):
        color = colors[i % len(colors)]
        for row in X:
            ax[i].plot(wavelength_nm, row, color=color, alpha=0.4, lw=1)
        mean = X.mean(axis=0)
        ax[i].plot(wavelength_nm, mean, color="black", lw=1.6, label="mean")
        ax[i].set_title(f"{label} (n={X.shape[0]})", fontsize=15)
        ax[i].set_xlabel("Wavelength (nm)", fontsize=15)
        if i == 0 or not sharey:
            ax[i].set_ylabel("Absorbance", fontsize=15)
        ax[i].legend(fontsize=15)
    fig.tight_layout()
    return fig


def plot_spectra_colored_by_target(wavelength_nm, X: np.ndarray, y, target_name: str = "", ax=None):
    """Superpose tous les spectres colores selon leur valeur de reference
    (viridis, du bas vers le haut) : un graphique classique d'EDA en NIR qui
    teste visuellement si une tendance spectrale (ex. un effet de
    niveau/echelle global) suit la propriete a predire, plutot que d'etre
    purement instrumentale. Complete le diagnostic additif/multiplicatif :
    un degrade de couleur visible et monotone le long des spectres signifie
    que la cible n'est pas seulement encodee dans la forme fine du spectre,
    mais aussi dans une caracteristique plus grossiere, de type niveau."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(y)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    for i in order:
        ax.plot(wavelength_nm, X[i], color=cmap(norm(y[i])), lw=0.8, alpha=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=target_name or "target value")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance")
    ax.set_title(f"Spectra colored by {target_name}" if target_name else "Spectra colored by target value", fontweight='bold')
    if owns_figure:
        fig.tight_layout()
    return fig


def plot_spectra_colored_by_all_targets(wavelength_nm, X: np.ndarray, targets: pd.DataFrame):
    """Version en grille de `plot_spectra_colored_by_target`, un panneau par
    colonne cible, sur 2 lignes."""
    n_cols = targets.shape[1] // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    for ax, col in zip(axes.flat, targets.columns):
        plot_spectra_colored_by_target(wavelength_nm, X, targets[col].values, target_name=col, ax=ax)
    fig.tight_layout()
    return fig


def compute_spectra_vs_mean_diagnostic(X: np.ndarray, max_points: int | None = 20000,
                                        random_state: int = 0):
    """Preparation des donnees pour le diagnostic additif-vs-multiplicatif
    (CheMOOCs Grain 10, §2) : pour chaque paire (echantillon, longueur
    d'onde), la valeur du spectre moyen a cette longueur d'onde (`x_vals`)
    associee a la valeur propre de l'echantillon a ce point (`y_vals`).
    Trace, un nuage en "millefeuille" (dispersion verticale a peu pres
    constante autour de y=x, quelle que soit l'intensite) indique un effet
    additif ; un nuage en "cone" (dispersion qui s'elargit en s'eloignant de
    l'origine) indique un effet multiplicatif. `max_points` sous-echantillonne
    les paires (echantillon x longueur d'onde) pour la taille/vitesse du
    graphique ; None les garde toutes.
    """
    X = np.asarray(X, dtype=float)
    mean_spectrum = X.mean(axis=0)
    x_vals = np.tile(mean_spectrum, X.shape[0])
    y_vals = X.ravel()

    if max_points is not None and len(x_vals) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(x_vals), size=max_points, replace=False)
        x_vals, y_vals = x_vals[idx], y_vals[idx]
    return x_vals, y_vals


def plot_spectra_vs_mean_spectrum(X: np.ndarray, ax=None, title: str | None = None,
                                   max_points: int = 20000, random_state: int = 0):
    """Trace le diagnostic additif-vs-multiplicatif, voir
    `compute_spectra_vs_mean_diagnostic` pour la lecture "millefeuille vs cone"."""
    x_vals, y_vals = compute_spectra_vs_mean_diagnostic(X, max_points=max_points, random_state=random_state)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    ax.scatter(x_vals, y_vals, s=3, alpha=0.4, color="steelblue", edgecolors="none")
    lo, hi = min(x_vals.min(), y_vals.min()), max(x_vals.max(), y_vals.max())
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--", label="y = x")
    ax.set_xlabel("Mean spectrum value")
    ax.set_ylabel("Individual spectrum value")
    ax.set_title(title or "Spectra vs. mean spectrum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pca_scores_by_target(scores: np.ndarray,  y, explained_var : np.ndarray, target_name: str = "", ax=None):
    """Graphique de scores ACP (PC1 vs PC2) colores par une valeur cible continue."""
    y = np.asarray(y, dtype=float).ravel()

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    sc = ax.scatter(scores[:, 0], scores[:, 1], c=y, cmap="jet", s=25, edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, ax=ax, label=target_name or "target value")
    ax.set_xlabel(f"PC1 ({explained_var[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained_var[1] * 100:.1f}% var.)")
    ax.set_title(f"Scores ACP colorés par {target_name}" if target_name else "Scores ACP")
    if owns_figure:
        fig.tight_layout()
    return fig


def plot_pca_loadings(wavelength_nm, components: np.ndarray, ax=None):
    """Loadings ACP (une courbe par composante retenue) vs longueur d'onde."""
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    for i, loading in enumerate(components):
        ax.plot(wavelength_nm, loading, label=f"PC{i + 1}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Loading")
    ax.set_title("Loadings ACP")
    ax.legend(fontsize=8)
    if owns_figure:
        fig.tight_layout()
    return fig


def plot_calibration_vs_loocv_curve(rmse_calibration, rmse_loocv, ax=None, title: str | None = None):
    """RMSE de calibration (intra-echantillon) vs RMSE de CV leave-one-out
    en fonction du nombre de composantes : la courbe classique utilisee pour
    choisir un nombre de composantes PLS a l'oeil (voir
    `corn_nir.validation.compute_calibration_vs_loocv_curve`)."""
    n_components = np.arange(1, len(rmse_calibration) + 1)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.plot(n_components, rmse_calibration, marker="o", ms=4, label="Calibration (RMSEC)")
    ax.plot(n_components, rmse_loocv, marker="o", ms=4, label="Validation LOO (RMSECV)")
    ax.set_xlabel("Nombre de composantes PLS")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    if owns_figure:
        fig.tight_layout()
    return fig


def plot_target_distributions(targets: pd.DataFrame):
    """Histogramme de chaque propriete cible."""
    n = targets.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    for ax, col in zip(axes, targets.columns):
        ax.hist(targets[col], bins=15, color="steelblue", edgecolor="white")
        ax.set_xlabel(col, fontsize=15, fontweight='bold')
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_nbs_standards(wavelength_nm, nbs: dict[str, np.ndarray]):
    """Trace les spectres des standards de verre NBS mesures sur chaque instrument."""
    fig, axes = plt.subplots(1, len(nbs), figsize=(5 * len(nbs), 6), sharey=True)
    colors = plt.get_cmap("tab10").colors
    for i, (instrument, X) in enumerate(nbs.items()):
        for j, row in enumerate(X):
            axes[i].plot(wavelength_nm, row, color=colors[j % len(colors)], lw=1.2, label=f"standard {j}")
        axes[i].set_title(f"NBS standards - {instrument} (n={X.shape[0]})", fontsize=13, fontweight='bold')
        axes[i].set_xlabel("Wavelength (nm)", fontsize=15)
        if i == 0:
            axes[i].set_ylabel("Absorbance", fontsize=15)
        axes[i].legend(fontsize=15)
    fig.tight_layout()
    return fig


def compute_instrument_differences(spectra: dict[str, np.ndarray], reference: str = "m5"):
    """Difference spectrale par echantillon de chaque instrument non-reference
    par rapport a l'instrument de reference, plus des statistiques resumant
    le decalage instrumental."""
    diffs = {}
    stats = {}
    ref = spectra[reference]
    for name, X in spectra.items():
        if name == reference:
            continue
        diff = ref - X
        diffs[name] = diff
        stats[name] = {
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "std_diff": float(np.std(diff)),
        }
    return diffs, stats


def plot_difference_curves(wavelength_nm, spectra: dict[str, np.ndarray], reference: str = "m5"):
    """Trace la courbe de difference moyenne +/- 1 ecart-type de chaque
    instrument par rapport a l'instrument de reference (m5 par defaut), pour
    visualiser le decalage instrumental."""
    diffs, stats = compute_instrument_differences(spectra, reference=reference)
    names = list(diffs.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), sharey=True)
    if len(names) == 1:
        axes = [axes]
    colors = plt.get_cmap("tab10").colors
    for i, name in enumerate(names):
        diff = diffs[name]
        mean = diff.mean(axis=0)
        std = diff.std(axis=0, ddof=1)
        axes[i].axhline(0, color="grey", lw=0.8, ls="--")
        axes[i].plot(wavelength_nm, mean, color=colors[i], lw=1.6, label="mean diff.")
        axes[i].fill_between(wavelength_nm, mean - std, mean + std, color=colors[i], alpha=0.2, label="+/- 1 std")
        axes[i].set_title(f"{reference} - {name}", fontsize=15)
        axes[i].set_xlabel("Wavelength (nm)", fontsize=15)
        if i == 0:
            axes[i].set_ylabel("Absorbance difference", fontsize=13)
        axes[i].legend(fontsize=15)
    fig.tight_layout()
    return fig, stats


def plot_target_correlation_matrix(targets: pd.DataFrame):
    """Carte de chaleur de la correlation de Pearson entre les proprietes cibles."""
    corr = targets.corr()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontweight='bold')
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontweight='bold')
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, corr


def plot_pls_scores_by_target(
    X: np.ndarray, targets: pd.DataFrame, n_components: int = 2
):
    """Scores PLS2 exploratoires (PC1 vs PC2) colores par chaque propriete
    cible, pour relier la structure spectrale a la chimie. C'est une
    visualisation d'EDA uniquement : `n_components` est fixe arbitrairement
    et NON regle par validation croisee ici, le modele PLSR de base regle
    par CV est construit separement dans la phase de modelisation."""
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X, targets.values)
    scores = pls.x_scores_

    n_targets = targets.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(4.5 * n_targets, 4))
    for i, col in enumerate(targets.columns):
        sc = axes[i].scatter(scores[:, 0], scores[:, 1], c=targets[col], cmap="viridis", s=28)
        axes[i].set_title(col)
        axes[i].set_xlabel("PLS component 1")
        if i == 0:
            axes[i].set_ylabel("PLS component 2")
        fig.colorbar(sc, ax=axes[i], fraction=0.046, pad=0.04)
    fig.suptitle("Exploratory PLS2 scores colored by target value (m5, not CV-tuned)")
    fig.tight_layout()
    return fig, pls


def plot_pca_by_instrument(spectra: dict[str, np.ndarray], n_components: int = 2):
    """Ajuste une seule ACP sur les spectres regroupes de tous les
    instruments et trace les deux premiers scores colores par instrument,
    pour visualiser le decalage de domaine instrumental."""
    instruments = list(spectra.keys())
    X_pooled = np.vstack([spectra[name] for name in instruments])
    labels = np.concatenate(
        [np.full(spectra[name].shape[0], name) for name in instruments]
    )

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X_pooled)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.get_cmap("tab10").colors
    for i, name in enumerate(instruments):
        mask = labels == name
        ax.scatter(scores[mask, 0], scores[mask, 1], color=colors[i], label=name, alpha=0.75, s=28)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)")
    ax.set_title("PCA of raw spectra, colored by instrument")
    ax.legend()
    fig.tight_layout()
    return fig, pca


def plot_vip(wavelength_nm, vip: np.ndarray, threshold: float = 1.0, title: str = "VIP scores", ax=None):
    """Trace les scores VIP en fonction de la longueur d'onde, avec la ligne
    conventionnelle VIP > threshold."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    above = vip > threshold
    ax.plot(wavelength_nm, vip, color="steelblue", lw=1.2)
    ax.fill_between(wavelength_nm, 0, vip, where=above, color="firebrick", alpha=0.35, label=f"VIP > {threshold}")
    ax.axhline(threshold, color="grey", ls="--", lw=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("VIP score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_elasticnet_selection(wavelength_nm, coefficients: np.ndarray, title: str = "Elastic Net selected wavelengths", ax=None):
    """Trace les coefficients Elastic Net en fonction de la longueur d'onde,
    en mettant en evidence les longueurs d'onde non nulles (selectionnees) :
    le pendant parcimonieux du VIP."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    coefficients = np.asarray(coefficients).ravel()
    selected = coefficients != 0
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.plot(wavelength_nm, coefficients, color="lightgrey", lw=0.8, zorder=1)
    ax.scatter(
        wavelength_nm[selected], coefficients[selected],
        color="seagreen", s=10, zorder=2, label=f"selected (n={selected.sum()})",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Elastic Net coefficient")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_parity(y_true, y_pred, title: str = "Parity plot", ax=None):
    """Nuage predit vs reel avec la droite identite y=x."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", ls="--", lw=1, label="y = x")
    ax.scatter(y_true, y_pred, alpha=0.7, s=28, color="steelblue")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

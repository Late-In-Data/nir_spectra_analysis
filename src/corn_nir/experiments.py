"""Fonctions de calcul de haut niveau pour chaque phase du benchmark Corn NIR.

Chaque fonction ici est pure vis-a-vis du systeme de fichiers : elle retourne
des DataFrames (et, quand c'est utile, les tableaux/modeles bruts derriere)
mais n'ecrit jamais de CSV ni de figure elle-meme. Ce sont les notebooks qui
les appellent qui enregistrent eux-memes leurs figures et tableaux dans
`reports/`, une fois les resultats calcules.

Chaque fonction `run_*` correspond a une phase du projet (voir les notebooks
pour le recit complet). `verbose=True` (par defaut) affiche la progression,
utile en terminal comme en notebook ; cela n'ecrit jamais de fichier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration_transfer import (
    DirectStandardization,
    PiecewiseDirectStandardization,
    SlopeBiasCorrection,
)
from .evaluation import regression_metrics
from .models import (
    build_ridge,
    build_svr_rbf,
)
from .preprocessing import PREPROCESSING_NAMES, build_preprocessing
from .validation import (
    kennard_stone_split,
    nested_loo_generic,
    nested_loo_plsr,
    select_n_components_by_cv,
)
from .variable_selection import compute_vip, literature_band_overlap, select_wavelengths_by_vip

METRIC_NAMES = ["RMSE", "MAE", "R2", "RPD"]
ML_MODEL_NAMES = ["ridge", "svr_rbf", "random_forest", "gradient_boosting"]

# Bandes informatives rapportees dans la litterature pour les constituants du
# mais (Fatemi, Singh & Kamruzzaman, 2022, Food Chemistry 383, 132442 : VIP +
# algorithme genetique). Verifiees a partir du resume de l'article via deux
# sources independantes (Semantic Scholar + portail de recherche Illinois
# Experts) le 2026-07-26.
# Wu et al. (2023, Food Chemistry: X 18, 100666) n'est PAS inclus ici : ses
# bandes de longueurs d'onde precises n'ont pas pu etre verifiees de facon
# independante a partir de sources accessibles, donc aucune comparaison
# chiffree n'est faite avec cette etude (la regle du projet de ne jamais
# fabriquer un resultat s'etend aux citations).
FATEMI_2022_BANDS_NM = {
    "Moisture": (1908, 2108),
    "Oil": (2176, 2304),
    "Protein": (2130, 2190),
    "Starch": (1452, 1770),
}


def _summarize_cv(cv_df: pd.DataFrame, extra: dict) -> dict:
    """Resume un DataFrame de CV imbriquee (une ligne par pli) en moyenne/ecart-type par metrique.

    Resume aussi les colonnes intra-echantillon `{metric}_train` quand elles
    sont presentes (voir `nested_loo_plsr`/`nested_loo_generic`), pour que
    l'appelant puisse comparer directement calibration et validation depuis
    la ligne de resume.
    """
    summary = dict(extra)
    for m in METRIC_NAMES:
        summary[f"{m}_mean"] = cv_df[m].mean()
        summary[f"{m}_std"] = cv_df[m].std()
        train_col = f"{m}_train"
        if train_col in cv_df.columns:
            summary[f"{train_col}_mean"] = cv_df[train_col].mean()
            summary[f"{train_col}_std"] = cv_df[train_col].std()
    return summary


def make_preprocessing_pipeline_factory(preprocessing_name: str):
    """Construit une `pipeline_factory(n_components) -> estimateur` pour une
    variante de pretraitement donnee, pour reutiliser `nested_loo_plsr` sans
    modification."""

    def factory(n_components: int):
        prep = build_preprocessing(preprocessing_name)
        model = PLSRegression(n_components=n_components, scale=False)
        if prep is None:
            return model
        return Pipeline([("prep", prep), ("plsr", model)])

    return factory


def make_ml_pipeline_factory_loo(prep_name: str, model_name: str, inner_splits: int, random_state: int):
    """Factory `build_model() -> estimateur` adaptee au LOO : Random Forest et
    Gradient Boosting utilisent des hyperparametres fixes et raisonnables
    plutot qu'une recherche aleatoire integree. Sous leave-one-out, cette
    recherche tournerait sinon une fois par echantillon laisse de cote
    (~80x), ce qui n'est pas un usage proportionne du temps de calcul pour le
    gain probable a cette taille d'echantillon. Ridge et SVR-RBF ne sont pas
    concernes, tous deux restent assez rapides sous LOO avec leur propre
    recherche interne inchangee."""

    def build_model():
        prep = build_preprocessing(prep_name)
        prep_step = ("prep", prep) if prep is not None else ("passthrough", "passthrough")
        if model_name == "ridge":
            model_step = ("model", build_ridge())
        elif model_name == "svr_rbf":
            model_step = ("model", build_svr_rbf(cv=inner_splits))
        elif model_name == "random_forest":
            model_step = ("model", RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=random_state,
            ))
        elif model_name == "gradient_boosting":
            model_step = ("model", GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state,
            ))
        else:
            raise ValueError(f"Modele '{model_name}' inconnu")
        return Pipeline([prep_step, model_step])

    return build_model


def run_baseline_plsr_loo(ds, preprocessing_name: str = "sg_deriv1", max_components: int = 10,
                           inner_splits: int = 5, random_state: int = 0,
                           verbose: bool = True, pipeline_factory=None):
    """Phase C : baseline PLSR intra-instrument sur m5 pour les 4 cibles,
    par validation croisee imbriquee leave-one-out (`nested_loo_plsr`) : chaque
    echantillon valide exactement une fois, sans reserver de split separe,
    ce qui convient a cette taille d'echantillon (n=80). Voir la docstring de
    `nested_loo_plsr` pour le detail de pourquoi la boucle externe LOO ne
    laisse jamais fuiter l'echantillon laisse de cote dans le choix de
    `n_components`.

    `pipeline_factory`, si fourni, remplace entierement la factory par
    defaut de `preprocessing_name` (ex. pour utiliser un pretraitement hors
    de `PREPROCESSING_NAMES`, comme crop+Detrend) ; `preprocessing_name`
    n'est alors pas utilise pour l'ajustement lui-meme.
    Retourne (loo_all_df, summary_df).
    """
    X = ds.spectra["m5"]
    pipeline_factory = pipeline_factory or make_preprocessing_pipeline_factory(preprocessing_name)
    loo_frames, summary_rows = [], []

    for target in ds.target_names:
        y = ds.targets[target].values
        if verbose:
            print(f"[baseline PLSR, LOO] cible={target} : CV imbriquee leave-one-out "
                  f"({len(y)} iterations, interne={inner_splits})...")

        loo_df = nested_loo_plsr(
            X, y, max_components=max_components, inner_splits=inner_splits,
            random_state=random_state, pipeline_factory=pipeline_factory,
        )
        loo_df.insert(0, "target", target)
        loo_frames.append(loo_df)

        metrics = regression_metrics(loo_df["y_true"], loo_df["y_pred"])
        summary = {
            "target": target, **metrics,
            "n_components_selected_mode": int(loo_df["n_components_selected"].mode().iloc[0]),
            "n_components_selected_min": int(loo_df["n_components_selected"].min()),
            "n_components_selected_max": int(loo_df["n_components_selected"].max()),
        }
        summary_rows.append(summary)

        if verbose:
            print(
                f"    LOO : RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}  RPD={metrics['RPD']:.2f}  "
                f"(composantes : mode={summary['n_components_selected_mode']}, "
                f"plage=[{summary['n_components_selected_min']},{summary['n_components_selected_max']}])"
            )

    loo_all = pd.concat(loo_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return loo_all, summary_df


def run_preprocessing_comparison_loo(ds, max_components: int = 10, inner_splits: int = 5,
                                      random_state: int = 0, verbose: bool = True,
                                      preprocessing_names=PREPROCESSING_NAMES):
    """Phase D (variante leave-one-out) : meme moteur de CV imbriquee LOO que
    `run_baseline_plsr_loo`, en comparant toutes les variantes de
    pretraitement sur m5 plutot qu'une seule baseline.
    Retourne (loo_all_df, summary_df, best_per_target_df)."""
    X = ds.spectra["m5"]
    loo_frames, summary_rows = [], []

    for name in preprocessing_names:
        factory = make_preprocessing_pipeline_factory(name)
        for target in ds.target_names:
            y = ds.targets[target].values
            if verbose:
                print(f"[pretraitement={name}, LOO] cible={target} : CV imbriquee leave-one-out "
                      f"({len(y)} iterations, interne={inner_splits})...")

            loo_df = nested_loo_plsr(
                X, y, max_components=max_components, inner_splits=inner_splits,
                random_state=random_state, pipeline_factory=factory,
            )
            loo_df.insert(0, "preprocessing", name)
            loo_df.insert(0, "target", target)
            loo_frames.append(loo_df)

            metrics = regression_metrics(loo_df["y_true"], loo_df["y_pred"])
            summary_rows.append({
                "target": target, "preprocessing": name, **metrics,
                "n_components_selected_mode": int(loo_df["n_components_selected"].mode().iloc[0]),
            })

            if verbose:
                print(
                    f"    LOO : RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}  RPD={metrics['RPD']:.2f}  "
                    f"(composantes mode={summary_rows[-1]['n_components_selected_mode']})"
                )

    loo_all = pd.concat(loo_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    best = summary_df.loc[summary_df.groupby("target")["RMSE"].idxmin()]
    return loo_all, summary_df, best


def run_variable_selection_loo(ds, best_prep: dict, max_components: int = 10, inner_splits: int = 5,
                                random_state: int = 0, verbose: bool = True,
                                enet_l1_ratio=(0.9,), enet_n_alphas: int = 10,
                                enet_cv: int = 3, enet_max_iter: int = 5000, enet_tol: float = 1e-2):
    """Phase E : scores VIP + coefficients PLS (sur le meilleur pretraitement
    de chaque cible) et une selection parcimonieuse par Elastic Net, comparee
    aux bandes informatives de Fatemi et al. (2022). La performance de
    l'Elastic Net est estimee par leave-one-out (`nested_loo_generic`).

    `best_prep` : {cible: nom_pretraitement}, typiquement le gagnant de la
    Phase D, mais l'appelant peut passer n'importe quel mapping (ex. un
    notebook explorant d'autres alternatives).

    LOO multiplie par ~80x le nombre d'ajustements `ElasticNetCV` par rapport
    a une poignee de plis externes, donc la grille de recherche par defaut
    ici est plus legere (`enet_l1_ratio`, `enet_n_alphas`, `enet_max_iter`,
    `enet_tol`) : passer des valeurs plus riches pour une recherche plus
    poussee (et plus lente).

    Retourne un dict avec les cles "vip_summary" (DataFrame), "enet_cv"
    (DataFrame), "enet_summary" (DataFrame) et "per_target" ({cible: {...}})
    qui contient les tableaux/modeles bruts necessaires pour tracer (vip,
    pls_coef, enet_coefs) ; le tracé lui-meme est laisse a l'appelant (notebook).
    """
    X_m5 = ds.spectra["m5"]
    vip_rows, enet_loo_frames, enet_summary_rows = [], [], []
    per_target = {}

    for target in ds.target_names:
        y = ds.targets[target].values
        prep_name = best_prep[target]
        if verbose:
            print(f"[selection de variables, LOO] cible={target} : pretraitement={prep_name}, "
                  f"ajustement du PLSR final sur l'ensemble du jeu m5...")

        prep = build_preprocessing(prep_name)
        X = prep.fit_transform(X_m5) if prep is not None else X_m5.copy()

        best_n, _ = select_n_components_by_cv(
            X, y, max_components=max_components, n_splits=inner_splits, random_state=random_state,
        )
        pls = PLSRegression(n_components=best_n, scale=False).fit(X, y)

        vip = compute_vip(pls, X)
        mask, selected_wavelengths = select_wavelengths_by_vip(vip, ds.wavelength_nm, threshold=1.0)
        band = FATEMI_2022_BANDS_NM[target]
        overlap = literature_band_overlap(selected_wavelengths, band)

        if verbose:
            print(
                f"    n_components={best_n}, VIP>1 : {mask.sum()}/{len(mask)} longueurs d'onde "
                f"({overlap['n_inside_band']}/{overlap['n_selected']} = "
                f"{overlap['fraction_inside_band']:.0%} tombent dans la bande "
                f"{band[0]}-{band[1]} nm de Fatemi et al. 2022 pour {target})"
            )

        vip_rows.append({
            "target": target, "preprocessing": prep_name, "n_components": best_n,
            "n_wavelengths_vip_above_1": int(mask.sum()),
            "fatemi_2022_band_nm": f"{band[0]}-{band[1]}",
            "n_vip_wavelengths_inside_fatemi_band": overlap["n_inside_band"],
            "fraction_vip_wavelengths_inside_fatemi_band": overlap["fraction_inside_band"],
        })

        # --- Elastic Net : selection parcimonieuse + performance honnete par leave-one-out ---
        if verbose:
            print(f"    Elastic Net : CV leave-one-out ({len(y)} iterations)...")

        def build_enet_model(prep_name=prep_name, random_state=random_state):
            prep_step = ("prep", build_preprocessing(prep_name)) if prep_name != "raw" else ("passthrough", "passthrough")
            return Pipeline([
                prep_step,
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(
                    l1_ratio=list(enet_l1_ratio), alphas=enet_n_alphas, cv=enet_cv,
                    max_iter=enet_max_iter, tol=enet_tol, random_state=random_state,
                )),
            ])

        enet_loo_df = nested_loo_generic(X_m5, y, build_enet_model)
        enet_loo_df.insert(0, "target", target)
        enet_loo_frames.append(enet_loo_df)

        enet_final = build_enet_model().fit(X_m5, y)
        enet_coefs = enet_final.named_steps["enet"].coef_
        n_selected_enet = int(np.sum(enet_coefs != 0))

        enet_metrics = regression_metrics(enet_loo_df["y_true"], enet_loo_df["y_pred"])
        enet_summary = {
            "target": target, "preprocessing": prep_name, **enet_metrics,
            "n_wavelengths_selected": n_selected_enet,
            "alpha_selected": enet_final.named_steps["enet"].alpha_,
            "l1_ratio_selected": enet_final.named_steps["enet"].l1_ratio_,
        }
        enet_summary_rows.append(enet_summary)

        if verbose:
            print(
                f"    Elastic Net : RMSE={enet_metrics['RMSE']:.3f}  R2={enet_metrics['R2']:.3f}  "
                f"({n_selected_enet}/700 longueurs d'onde a coefficient non nul)"
            )

        per_target[target] = {
            "preprocessing": prep_name, "n_components": best_n,
            "vip": vip, "pls_coef": pls.coef_, "enet_coefs": enet_coefs,
        }

    return {
        "vip_summary": pd.DataFrame(vip_rows),
        "enet_cv": pd.concat(enet_loo_frames, ignore_index=True),
        "enet_summary": pd.DataFrame(enet_summary_rows),
        "per_target": per_target,
    }


def run_ml_comparison_loo(ds, best_prep: dict, inner_splits: int = 3, random_state: int = 0,
                           verbose: bool = True, model_names=ML_MODEL_NAMES,
                           plsr_reference: pd.DataFrame | None = None):
    """Phase F : Ridge / SVR-RBF / Random Forest / Gradient Boosting, chacun
    sur le meilleur pretraitement de sa cible, evalues par leave-one-out
    (`nested_loo_generic`). Random Forest et Gradient Boosting utilisent des
    hyperparametres fixes sous LOO (voir la docstring de
    `make_ml_pipeline_factory_loo`) ; Ridge et SVR-RBF gardent leur propre
    recherche interne.

    `plsr_reference`, si fourni, doit avoir des colonnes
    `RMSE`/`MAE`/`R2`/`RPD` par cible (ex. le resume de `run_baseline_plsr_loo`
    ou de `run_preprocessing_comparison_loo`).
    Retourne (loo_all_df, summary_df, best_per_target_df).
    """
    X_m5 = ds.spectra["m5"]
    loo_frames, summary_rows = [], []

    for target in ds.target_names:
        y = ds.targets[target].values
        prep_name = best_prep[target]

        for model_name in model_names:
            if verbose:
                print(f"[comparaison ML, LOO] cible={target} : modele={model_name} "
                      f"(pretraitement={prep_name}), CV leave-one-out ({len(y)} iterations)...")

            build_model = make_ml_pipeline_factory_loo(prep_name, model_name, inner_splits, random_state)
            loo_df = nested_loo_generic(X_m5, y, build_model)
            loo_df.insert(0, "model", model_name)
            loo_df.insert(0, "target", target)
            loo_frames.append(loo_df)

            metrics = regression_metrics(loo_df["y_true"], loo_df["y_pred"])
            summary_rows.append({"target": target, "model": model_name, "preprocessing": prep_name, **metrics})

            if verbose:
                print(f"    RMSE={metrics['RMSE']:.3f}  R2={metrics['R2']:.3f}  RPD={metrics['RPD']:.2f}")

        if plsr_reference is not None:
            plsr_row = plsr_reference[plsr_reference["target"] == target].iloc[0]
            summary_rows.append({
                "target": target, "model": "plsr", "preprocessing": prep_name,
                "RMSE": plsr_row["RMSE"], "MAE": plsr_row["MAE"],
                "R2": plsr_row["R2"], "RPD": plsr_row["RPD"],
            })

    loo_all = pd.concat(loo_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    best = summary_df.loc[summary_df.groupby("target")["RMSE"].idxmin()]
    return loo_all, summary_df, best


def fit_m5_models(ds, best_prep: dict, max_components: int = 10, inner_splits: int = 5,
                   random_state: int = 0):
    """Ajuste un modele PLSR par cible sur l'ensemble *complet* du jeu m5
    (meilleur pretraitement, composantes choisies par CV) : le point de
    depart partage des Phases G et H. Retourne
    {cible: (pretraitement_ajuste, modele_ajuste, best_n)}."""
    models = {}
    for target in ds.target_names:
        y = ds.targets[target].values
        prep_name = best_prep[target]
        prep = build_preprocessing(prep_name)
        X_t = prep.fit_transform(ds.spectra["m5"]) if prep is not None else ds.spectra["m5"].copy()
        best_n, _ = select_n_components_by_cv(
            X_t, y, max_components=max_components, n_splits=inner_splits, random_state=random_state,
        )
        model = PLSRegression(n_components=best_n, scale=False).fit(X_t, y)
        models[target] = (prep, model, best_n)
    return models


def run_cross_instrument_robustness(ds, best_prep: dict, max_components: int = 10,
                                     inner_splits: int = 5, random_state: int = 0,
                                     verbose: bool = True, in_domain_reference: pd.DataFrame | None = None):
    """Phase G : entraine un modele PLSR sur m5 uniquement, puis l'applique
    TEL QUEL, sans aucune correction, directement aux spectres mp5/mp6
    correspondants des memes 80 echantillons physiques, et mesure la chute
    de performance.

    Le transformateur de pretraitement est ajuste une seule fois sur m5 puis
    seulement `.transform()`-e (jamais reajuste) sur mp5/mp6 : le reajuster
    sur l'instrument cible serait deja, en soi, une forme (legere) de
    correction, ce qui irait a l'encontre du but de cette baseline
    sans-correction. La Phase H est l'endroit ou la correction deliberee
    (DS/PDS/SBC) intervient.

    Retourne (results_df, models) ou `models` est la sortie de
    `fit_m5_models` (reutilisable par les appelants qui veulent aussi lancer
    la Phase H sur les memes ajustements).
    """
    models = fit_m5_models(ds, best_prep, max_components=max_components,
                            inner_splits=inner_splits, random_state=random_state)
    rows = []

    for target in ds.target_names:
        y = ds.targets[target].values
        prep, model, best_n = models[target]
        prep_name = best_prep[target]

        in_domain_rmse = in_domain_reference.set_index("target").loc[target, "RMSE_mean"] \
            if in_domain_reference is not None else None
        in_domain_r2 = in_domain_reference.set_index("target").loc[target, "R2_mean"] \
            if in_domain_reference is not None else None

        if verbose:
            msg = f"[robustesse inter-instruments] cible={target} (pretraitement={prep_name}, n_components={best_n})"
            if in_domain_rmse is not None:
                msg += f" : reference intra-domaine RMSE={in_domain_rmse:.3f}, R2={in_domain_r2:.3f}"
            print(msg)

        for other in ("mp5", "mp6"):
            X_other_t = prep.transform(ds.spectra[other]) if prep is not None else ds.spectra[other].copy()
            y_pred = model.predict(X_other_t)
            metrics = regression_metrics(y, y_pred)
            row = {
                "target": target, "test_instrument": other, "preprocessing": prep_name,
                "n_components": best_n,
            }
            if in_domain_rmse is not None:
                row["in_domain_RMSE_mean"] = in_domain_rmse
                row["in_domain_R2_mean"] = in_domain_r2
            row.update(metrics)
            rows.append(row)

            if verbose:
                ratio = f" (x{metrics['RMSE'] / in_domain_rmse:.1f} vs intra-domaine)" if in_domain_rmse else ""
                print(
                    f"    modele entraine sur m5, applique a {other} (sans correction) : RMSE={metrics['RMSE']:.3f}{ratio} "
                    f"R2={metrics['R2']:.3f} RPD={metrics['RPD']:.2f}"
                )

    return pd.DataFrame(rows), models


def run_calibration_transfer(ds, best_prep: dict, max_components: int = 10, inner_splits: int = 5,
                              n_transfer: int = 25, random_state: int = 0, verbose: bool = True,
                              models: dict | None = None):
    """Phase H : Direct Standardization (DS), Piecewise DS (PDS) et la
    correction pente-biais (SBC), apprises sur un sous-ensemble disjoint
    d'echantillons de mais (selectionnes par Kennard-Stone pour la couverture
    spectrale) et evaluees sur les echantillons RESTANTS mis de cote : la
    transformation de transfert ne voit jamais les echantillons sur lesquels
    elle est ensuite jugee.

    La SBC corrige les predictions du modele par un simple ajustement
    lineaire (2 parametres) plutot que les spectres (DS/PDS), voir la
    docstring de `SlopeBiasCorrection`. Elle est ajustee sur les MEMES
    echantillons de transfert que DS/PDS, en utilisant les predictions que
    les spectres slave non corriges produisent deja via le modele entraine
    sur m5, contre les vraies valeurs de reference de ces echantillons.

    Tente aussi, a titre exploratoire, une DS a partir des seuls standards de
    verre NBS : m5nbs a 3 lignes contre 4 pour mp5nbs/mp6nbs (un ecart de
    comptage inexplique dans le fichier source, on ne peut pas confirmer
    qu'il s'agit des 3 memes standards physiques dans le meme ordre), et le
    verre a une matrice optique differente du mais, donc toute correction
    apprise a partir de ces standards est un signal exploratoire plus
    faible, pas une alternative equivalente au transfert par echantillons de
    mais ci-dessus.

    `models`, si fourni (typiquement la sortie de `fit_m5_models` de la
    Phase G), evite de reajuster les modeles m5. Retourne (results_df,
    pivot_df, examples) ou `examples[(target, slave)]` contient
    `(y_eval, y_pred_no_correction, y_pred_ds, y_pred_pds, y_pred_sbc)` pour
    les graphiques de parite.
    """
    if models is None:
        models = fit_m5_models(ds, best_prep, max_components=max_components,
                                inner_splits=inner_splits, random_state=random_state)

    X_m5 = ds.spectra["m5"]

    transfer_splits = {}
    for slave in ("mp5", "mp6"):
        transfer_idx, eval_idx = kennard_stone_split(ds.spectra[slave], n_train=n_transfer)
        transfer_splits[slave] = (transfer_idx, eval_idx)
        if verbose:
            print(f"[transfert de calibration] {slave} : {len(transfer_idx)} echantillons de transfert "
                  f"(selectionnes par Kennard-Stone), {len(eval_idx)} mis de cote pour l'evaluation")

    nbs_available = all(ds.nbs[i].shape[0] >= 2 for i in ("m5", "mp5", "mp6"))
    if nbs_available:
        n_nbs_pairs = min(ds.nbs["m5"].shape[0], ds.nbs["mp5"].shape[0], ds.nbs["mp6"].shape[0])
        if verbose:
            print(
                f"[transfert de calibration] verification bonus NBS : m5 a {ds.nbs['m5'].shape[0]} standards, "
                f"mp5/mp6 en ont {ds.nbs['mp5'].shape[0]}/{ds.nbs['mp6'].shape[0]} ; on utilise seulement les "
                f"{n_nbs_pairs} premiers de chacun via un appariement positionnel NAIF (non verifie) pour un "
                "test exploratoire de DS a partir des standards de verre."
            )

    rows = []
    examples = {}

    for target in ds.target_names:
        y = ds.targets[target].values
        prep, model, best_n = models[target]
        prep_name = best_prep[target]

        for slave in ("mp5", "mp6"):
            transfer_idx, eval_idx = transfer_splits[slave]
            X_slave_raw = ds.spectra[slave]

            def predict_on(X_slave_corrected_raw, prep=prep, model=model):
                X_t = prep.transform(X_slave_corrected_raw) if prep is not None else X_slave_corrected_raw.copy()
                return model.predict(X_t)

            y_true_eval = y[eval_idx]

            y_pred_no_corr = predict_on(X_slave_raw[eval_idx])
            m_no_corr = regression_metrics(y_true_eval, y_pred_no_corr)

            ds_model = DirectStandardization(n_components=min(10, len(transfer_idx) - 1)).fit(
                X_slave_raw[transfer_idx], X_m5[transfer_idx],
            )
            y_pred_ds = predict_on(ds_model.transform(X_slave_raw[eval_idx]))
            m_ds = regression_metrics(y_true_eval, y_pred_ds)

            pds_model = PiecewiseDirectStandardization(window=5, n_components=2).fit(
                X_slave_raw[transfer_idx], X_m5[transfer_idx],
            )
            y_pred_pds = predict_on(pds_model.transform(X_slave_raw[eval_idx]))
            m_pds = regression_metrics(y_true_eval, y_pred_pds)

            y_pred_no_corr_transfer = predict_on(X_slave_raw[transfer_idx])
            sbc_model = SlopeBiasCorrection().fit(y_pred_no_corr_transfer, y[transfer_idx])
            y_pred_sbc = sbc_model.transform(y_pred_no_corr)
            m_sbc = regression_metrics(y_true_eval, y_pred_sbc)

            examples[(target, slave)] = (y_true_eval, y_pred_no_corr, y_pred_ds, y_pred_pds, y_pred_sbc)

            for method, m in (
                ("no_correction", m_no_corr), ("DS", m_ds), ("PDS", m_pds), ("SBC", m_sbc),
            ):
                rows.append({
                    "target": target, "slave_instrument": slave, "method": method,
                    "n_transfer": len(transfer_idx), "n_eval": len(eval_idx), **m,
                })

            if verbose:
                print(
                    f"    {target} / {slave} : sans_correction RMSE={m_no_corr['RMSE']:.3f} R2={m_no_corr['R2']:.3f}  |  "
                    f"DS RMSE={m_ds['RMSE']:.3f} R2={m_ds['R2']:.3f}  |  PDS RMSE={m_pds['RMSE']:.3f} R2={m_pds['R2']:.3f}  |  "
                    f"SBC RMSE={m_sbc['RMSE']:.3f} R2={m_sbc['R2']:.3f}"
                )

            if nbs_available:
                ds_nbs_model = DirectStandardization(n_components=min(2, n_nbs_pairs - 1)).fit(
                    ds.nbs[slave][:n_nbs_pairs], ds.nbs["m5"][:n_nbs_pairs],
                )
                m_ds_nbs = regression_metrics(
                    y_true_eval, predict_on(ds_nbs_model.transform(X_slave_raw[eval_idx]))
                )
                rows.append({
                    "target": target, "slave_instrument": slave, "method": "DS_nbs_standards_exploratory",
                    "n_transfer": n_nbs_pairs, "n_eval": len(eval_idx), **m_ds_nbs,
                })
                if verbose:
                    print(f"      DS depuis le verre NBS (exploratoire, {n_nbs_pairs} paires) : "
                          f"RMSE={m_ds_nbs['RMSE']:.3f} R2={m_ds_nbs['R2']:.3f}")

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["target", "slave_instrument"], columns="method", values="RMSE").reset_index()
    return df, pivot, examples

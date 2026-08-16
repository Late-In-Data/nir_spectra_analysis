"""Demo interactive du projet Corn NIR Calibration Transfer.

Lancement :
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from corn_nir.data import load_corn_mat  # noqa: E402
from corn_nir.evaluation import regression_metrics  # noqa: E402
from corn_nir.models import (  # noqa: E402
    build_gradient_boosting,
    build_random_forest,
    build_ridge,
    build_svr_rbf,
    fit_plsr_with_cv_components,
)
from corn_nir.preprocessing import PREPROCESSING_NAMES, build_preprocessing  # noqa: E402
from corn_nir.validation import kennard_stone_split  # noqa: E402
from corn_nir.variable_selection import compute_vip, select_wavelengths_by_vip  # noqa: E402
from corn_nir.visualization import (  # noqa: E402
    plot_parity,
    plot_spectra_colored_by_target,
    plot_spectra_overlay,
    plot_vip,
)

DATA_PATH = ROOT / "data" / "raw" / "corn.mat"

# "PLSR" est traite a part dans fit_interactive_model (son nombre de
# composantes est choisi par CV, pas fixe), donc il n'a pas de constructeur
# ici, seuls les autres modeles en ont besoin. Le menu liste toujours "PLSR"
# en premier quel que soit l'ordre du dict ci-dessous.
MODEL_BUILDERS = {
    "Ridge": build_ridge,
    "SVR (RBF)": lambda: build_svr_rbf(cv=3),
    "Random Forest": lambda: build_random_forest(cv=3, n_iter=8, random_state=0),
    "Gradient Boosting": lambda: build_gradient_boosting(cv=3, n_iter=8, random_state=0),
}
MODEL_NAMES = ["PLSR"] + list(MODEL_BUILDERS.keys())

# Les 7 pretraitements nommes (brut, centrage, SNV, MSC, variantes
# Savitzky-Golay) sont ceux que cette app propose dans le menu interactif.
# `Detrend`, le point de depart choisi dans les notebooks a partir du
# diagnostic spectral, a besoin d'un axe de longueurs d'onde et d'une borne
# de coupure plutot que d'un simple nom, donc il n'est pas dans ce menu ;
# "raw" sert de valeur par defaut a la place.
DEFAULT_PREPROCESSING = "raw"

TARGET_NAMES = ["Moisture", "Oil", "Protein", "Starch"]
INSTRUMENTS = ["m5", "mp5", "mp6"]

st.set_page_config(
    page_title="Corn NIR Calibration Transfer",
    page_icon=":material/grain:",
    layout="wide",
)


@st.cache_data
def get_dataset():
    return load_corn_mat(DATA_PATH)


@st.cache_data(show_spinner="Ajustement du PLSR pour l'interpretabilite...")
def fit_vip_model(target: str, prep_name: str, seed: int = 0):
    ds = get_dataset()
    X = ds.spectra["m5"]
    y = ds.targets[target].values
    prep = build_preprocessing(prep_name)
    X_t = prep.fit_transform(X) if prep is not None else X.copy()
    pls, best_n = fit_plsr_with_cv_components(X_t, y, max_components=10, inner_splits=5, random_state=seed)
    vip = compute_vip(pls, X_t)
    return best_n, vip


@st.cache_data(show_spinner="Ajustement du modele sur un split Kennard-Stone...")
def fit_interactive_model(target: str, prep_name: str, model_name: str, n_train: int = 60, seed: int = 0):
    ds = get_dataset()
    X = ds.spectra["m5"]
    y = ds.targets[target].values
    prep = build_preprocessing(prep_name)
    X_t = prep.fit_transform(X) if prep is not None else X.copy()

    train_idx, test_idx = kennard_stone_split(X_t, n_train=n_train)

    if model_name == "PLSR":
        model, _ = fit_plsr_with_cv_components(
            X_t[train_idx], y[train_idx], max_components=10, inner_splits=5, random_state=seed,
        )
    else:
        model = MODEL_BUILDERS[model_name]()
        model.fit(X_t[train_idx], y[train_idx])

    y_pred_test = model.predict(X_t[test_idx])
    metrics = regression_metrics(y[test_idx], y_pred_test)
    return y[test_idx], np.asarray(y_pred_test).ravel(), metrics


with st.sidebar:
    st.title(":material/eco: Corn NIR")
    st.caption("Calibration transfer across 3 spectrometers")
    target = st.segmented_control("Property", TARGET_NAMES, default=TARGET_NAMES[0])
    st.markdown(
        "**Dataset**: 80 corn samples, 3 NIR spectrometers "
        "(m5, mp5, mp6), 1100-2498 nm.\n\n"
        "Source: Cargill / Eigenvector Research."
    )
    st.caption("Laté Lawson  \nlatejeanjacques@gmail.com")

if target is None:
    target = TARGET_NAMES[0]

ds = get_dataset()

tab_about, tab_spectra, tab_interactive = st.tabs(
    ["About", "Spectra explorer", "Interactive model"],
)

# --- About ---------------------------------------------------------------
with tab_about:
    st.header("Corn NIR calibration transfer benchmark")
    st.markdown(
        """
Can we predict corn composition from NIR spectra, and keep that performance
when the spectrometer changes? This app explores the classic **Corn**
dataset: 80 corn samples measured on **three** NIR spectrometers (`m5`,
`mp5`, `mp6`), 1100-2498 nm at 2 nm steps (700 channels), with four
reference-lab properties (**Moisture, Oil, Protein, Starch**).
        """
    )
    with st.container(horizontal=True):
        st.badge("80 samples", icon=":material/database:", color="green")
        st.badge("3 spectrometers", icon=":material/science:", color="green")
        st.badge("4 properties", icon=":material/analytics:", color="green")
        st.badge("700 wavelengths", icon=":material/show_chart:", color="green")

    with st.container(border=True):
        st.subheader("Dataset provenance")
        st.markdown(
            """
Measured at **Cargill**, distributed with permission by **Eigenvector
Research** as a benchmark for inter-instrument calibration-transfer methods
(*EigenNews*, Vol. 1 No. 3, October 1999). Everything shown here is computed
directly from `data/raw/corn.mat`, using the same functions as the project's
notebooks.
            """
        )
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height="stretch"):
            st.subheader("What this app does")
            st.markdown(
                """
- Explore raw and preprocessed spectra across the 3 instruments
- Inspect VIP wavelength importance for the selected property
- Fit PLSR or an ML model (Ridge, SVR-RBF, Random Forest, Gradient
  Boosting) on a reproducible Kennard-Stone split and see the result
  immediately

The full methodology (preprocessing diagnosis, nested cross-validation,
calibration transfer across instruments) is developed in the project's
notebooks, see the README for the complete write-up.
                """
            )
    with col2:
        with st.container(border=True, height="stretch"):
            st.subheader("Key references")
            st.markdown(
                """
- Fatemi, Singh & Kamruzzaman (2022). *Food Chemistry*, 383, 132442.
- Samuel, Chinnu & Lakshmanan (2015). *Materials Today: Proceedings*, 2(3).

See the project README for the complete reference list.
                """
            )
    st.warning(
        "**Limitations**: n=80, every metric here should be read with its "
        "cross-validation spread, not as a point estimate. Deep learning was "
        "deliberately not used as a primary model on a dataset this small.",
        icon=":material/warning:",
    )

# --- Spectra explorer ------------------------------------------------------
with tab_spectra:
    st.header("Spectra explorer")
    col_controls, col_plot = st.columns([1, 3])
    with col_controls:
        shown_instruments = st.pills(
            "Instruments", INSTRUMENTS, default=INSTRUMENTS, selection_mode="multi",
        )
        prep_choice = st.selectbox(
            "Preprocessing", PREPROCESSING_NAMES,
            index=PREPROCESSING_NAMES.index(DEFAULT_PREPROCESSING),
        )
        show_colored = st.toggle(f"Color spectra by {target} value (jet)", value=False)
        show_vip = st.toggle(f"Overlay VIP > 1 wavelengths for {target} (on m5)", value=True)

    if not shown_instruments:
        st.info("Select at least one instrument.", icon=":material/info:")
    else:
        prep = build_preprocessing(prep_choice)
        spectra_t = {}
        for instrument in shown_instruments:
            X = ds.spectra[instrument]
            spectra_t[instrument] = prep.fit_transform(X) if prep is not None else X

        with col_plot:
            fig = plot_spectra_overlay(ds.wavelength_nm, spectra_t)
            st.pyplot(fig, width="stretch")

        if show_colored:
            st.caption(f"Each spectrum colored by its {target} value")
            colored_cols = st.columns(len(shown_instruments))
            for col, instrument in zip(colored_cols, shown_instruments):
                with col:
                    fig_colored = plot_spectra_colored_by_target(
                        ds.wavelength_nm, spectra_t[instrument], ds.targets[target].values,
                        target_name=target,
                    )
                    st.pyplot(fig_colored, width="stretch")

        if show_vip:
            best_n, vip = fit_vip_model(target, prep_choice)
            mask, selected_wl = select_wavelengths_by_vip(vip, ds.wavelength_nm, threshold=1.0)
            fig_vip = plot_vip(
                ds.wavelength_nm, vip, threshold=1.0,
                title=f"VIP score for {target} (m5, {prep_choice}, {best_n} components)",
            )
            st.pyplot(fig_vip, width="stretch")
            st.caption(f"{mask.sum()} / 700 wavelengths have VIP > 1 for {target}.")

# --- Interactive model --------------------------------------------------------
with tab_interactive:
    st.header("Interactive model")
    st.caption(
        "Fits the selected model on a reproducible 60/20 Kennard-Stone split of "
        "m5 (calibration/prediction), recomputed here from your choices, not read from a report."
    )
    col_controls, col_plot = st.columns([1, 2])
    with col_controls:
        model_name = st.selectbox("Model", MODEL_NAMES)
        prep_choice_interactive = st.selectbox(
            "Preprocessing ", PREPROCESSING_NAMES,
            index=PREPROCESSING_NAMES.index(DEFAULT_PREPROCESSING), key="prep_interactive",
        )

    y_test, y_pred, metrics = fit_interactive_model(target, prep_choice_interactive, model_name)
    with col_controls:
        st.metric("RMSE", f"{metrics['RMSE']:.3f}", border=True)
        st.metric("R2", f"{metrics['R2']:.3f}", border=True)
        st.metric("RPD", f"{metrics['RPD']:.2f}", border=True)
    with col_plot:
        fig = plot_parity(y_test, y_pred, title=f"{model_name} on {target} (m5, held-out 20 samples)")
        st.pyplot(fig, width="stretch")

from pathlib import Path

import pandas as pd
import pytest

from corn_nir.data import CornDataset, load_corn_mat
from corn_nir.experiments import (
    fit_m5_models,
    run_baseline_plsr_loo,
    run_calibration_transfer,
    run_cross_instrument_robustness,
    run_ml_comparison_loo,
    run_preprocessing_comparison_loo,
    run_variable_selection_loo,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "corn.mat"


@pytest.fixture(scope="module")
def ds():
    full = load_corn_mat(DATA_PATH)
    full.target_names = ["Moisture", "Protein"]
    full.targets = full.targets[["Moisture", "Protein"]]
    return full


@pytest.fixture(scope="module")
def small_ds(ds):
    # La LOO relance sa boucle interne une fois par echantillon : on reduit
    # le jeu de donnees pour les tests qui verifient juste la plomberie/les
    # dimensions, pas des chiffres realistes.
    n = 15
    return CornDataset(
        wavelength_nm=ds.wavelength_nm,
        spectra={"m5": ds.spectra["m5"][:n]},
        targets=ds.targets.iloc[:n].reset_index(drop=True),
        target_names=ds.target_names,
    )


@pytest.fixture(scope="module")
def best_prep():
    return {"Moisture": "sg_smooth", "Protein": "sg_deriv1"}


def test_run_baseline_plsr_loo_no_disk_io(ds, tmp_path):
    before = set(tmp_path.iterdir())
    loo_all, summary_df = run_baseline_plsr_loo(
        ds, max_components=5, inner_splits=3, verbose=False,
    )
    assert set(tmp_path.iterdir()) == before  # rien ecrit sur le disque
    assert set(summary_df["target"]) == {"Moisture", "Protein"}
    assert len(loo_all) == 2 * len(ds.targets)  # une ligne par (cible, echantillon)
    assert (summary_df["RMSE"] > 0).all()


def test_run_baseline_plsr_loo_pipeline_factory_override(ds):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import Pipeline

    from corn_nir.preprocessing import SNV

    def snv_factory(n_components):
        return Pipeline([
            ("prep", SNV()),
            ("plsr", PLSRegression(n_components=n_components, scale=False)),
        ])

    loo_all, summary_df = run_baseline_plsr_loo(
        ds, max_components=5, inner_splits=3, verbose=False, pipeline_factory=snv_factory,
    )
    assert set(summary_df["target"]) == {"Moisture", "Protein"}
    assert (summary_df["RMSE"] > 0).all()


def test_run_preprocessing_comparison_loo(small_ds):
    loo_all, summary_df, best = run_preprocessing_comparison_loo(
        small_ds, max_components=3, inner_splits=2, random_state=0, verbose=False,
        preprocessing_names=["raw", "sg_deriv1"],
    )
    assert len(summary_df) == 2 * 2  # 2 pretraitements x 2 cibles
    assert set(best["target"]) == {"Moisture", "Protein"}
    assert len(loo_all) == 2 * 2 * len(small_ds.targets)  # une ligne par (pretraitement, cible, echantillon)


def test_run_variable_selection_loo(small_ds, best_prep):
    result = run_variable_selection_loo(
        small_ds, best_prep, max_components=3, inner_splits=2, random_state=0, verbose=False,
        enet_l1_ratio=(0.9,), enet_n_alphas=3, enet_cv=2, enet_max_iter=2000, enet_tol=1e-1,
    )
    assert set(result["per_target"].keys()) == {"Moisture", "Protein"}
    for target, info in result["per_target"].items():
        assert info["vip"].shape == (700,)
    assert len(result["vip_summary"]) == 2
    assert len(result["enet_summary"]) == 2
    assert (result["enet_summary"]["RMSE"] > 0).all()


def test_run_ml_comparison_loo_with_plsr_reference(small_ds, best_prep):
    plsr_reference = pd.DataFrame([
        {"target": "Moisture", "RMSE": 0.01, "MAE": 0.008, "R2": 0.99, "RPD": 10.0},
        {"target": "Protein", "RMSE": 0.1, "MAE": 0.08, "R2": 0.9, "RPD": 5.0},
    ])
    loo_all, summary_df, best = run_ml_comparison_loo(
        small_ds, best_prep, inner_splits=2, random_state=0,
        verbose=False, model_names=["ridge"], plsr_reference=plsr_reference,
    )
    assert set(summary_df["model"]) == {"ridge", "plsr"}
    assert set(best["target"]) == {"Moisture", "Protein"}
    assert len(loo_all) == len(small_ds.targets) * 2  # 1 modele x 2 cibles x n echantillons


def test_fit_m5_models(ds, best_prep):
    models = fit_m5_models(ds, best_prep, max_components=8, inner_splits=2, random_state=0)
    assert set(models.keys()) == {"Moisture", "Protein"}
    for prep, model, best_n in models.values():
        assert 1 <= best_n <= 8


def test_cross_instrument_and_calibration_transfer_share_models(ds, best_prep):
    no_corr_df, models = run_cross_instrument_robustness(
        ds, best_prep, max_components=8, inner_splits=2, random_state=0, verbose=False,
    )
    assert set(no_corr_df["test_instrument"]) == {"mp5", "mp6"}
    assert "in_domain_RMSE_mean" not in no_corr_df.columns  # pas fourni ici, donc absent

    transfer_df, pivot, examples = run_calibration_transfer(
        ds, best_prep, max_components=8, inner_splits=2, n_transfer=25, random_state=0,
        models=models, verbose=False,
    )
    assert set(transfer_df["method"]) >= {"no_correction", "DS", "PDS", "SBC"}
    assert ("Moisture", "mp5") in examples
    y_eval, y_pred_no_corr, y_pred_ds, y_pred_pds, y_pred_sbc = examples[("Moisture", "mp5")]
    assert len(y_eval) == len(y_pred_no_corr) == len(y_pred_ds) == len(y_pred_pds) == len(y_pred_sbc)

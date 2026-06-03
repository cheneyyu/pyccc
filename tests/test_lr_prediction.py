import numpy as np
import pandas as pd
import pytest
from joblib import load

import pyccc as pc


def _fixture_training():
    interactions = pd.DataFrame(
        {
            "species": ["toy_a", "toy_a", "toy_b", "toy_b"],
            "clade": ["clade_a", "clade_a", "clade_b", "clade_b"],
            "ligand_gene": ["L1", "L2", "L3", "L4"],
            "receptor_gene": ["R1", "R2", "R3", "R4"],
            "resource": ["fixture_a", "fixture_a", "fixture_b", "fixture_b"],
            "ligand_family": ["lfam1", "lfam2", "lfam3", "lfam4"],
            "receptor_family": ["rfam1", "rfam2", "rfam3", "rfam4"],
            "ligand_role_score": [0.9, 0.8, 0.7, 0.6],
            "receptor_role_score": [0.9, 0.8, 0.7, 0.6],
            "evidence_type": ["curated_direct"] * 4,
            "is_positive_label": [True] * 4,
        }
    )
    embeddings = pd.DataFrame(
        {
            "gene_id": ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"],
            "embedding": [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.8, 0.2, 0.0], dtype=np.float32),
                np.array([0.2, 0.8, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.8, 0.2], dtype=np.float32),
                np.array([1.0, 0.0, 0.2], dtype=np.float32),
                np.array([0.7, 0.1, 0.2], dtype=np.float32),
            ],
            "sequence_length": [100, 120, 130, 140, 300, 330, 340, 350],
        }
    )
    return interactions, embeddings


def test_train_score_lr_link_predictor_sklearn_fixture(tmp_path):
    interactions, embeddings = _fixture_training()

    card = pc.train_lr_link_predictor(
        interactions,
        embeddings,
        model="sklearn",
        output_dir=tmp_path,
        negative_ratio=1,
        validation_splits=["leave_species_out", "leave_resource_out", "leave_family_out", "leave_clade_out"],
    )
    scores = pc.score_lr_candidates(pd.DataFrame({"ligand_gene": ["L1"], "receptor_gene": ["R1"]}), embeddings, model=tmp_path)

    assert (tmp_path / "lr_link_model.joblib").exists()
    assert (tmp_path / "model_card.md").exists()
    assert (tmp_path / "density_prior.tsv").exists()
    payload = load(tmp_path / "lr_link_model.joblib")
    assert "calibrator" in payload
    assert card["metrics"]["n_positive"] == 4
    assert card["final_model_training"] == "all_pairs_after_validation"
    assert card["validation_feature_encoder_fit"] == "train_split_only"
    assert card["calibration_method"] in {"isotonic", "skipped"}
    assert card["density_prior_groupby"] == "clade"
    assert "leave_species_out" in card["validation_report"]
    assert "leave_resource_out" in card["validation_report"]
    assert "leave_family_out" in card["validation_report"]
    assert card["validation_report"]["random_stratified"]["status"] == "ok"
    assert card["validation_report"]["leave_species_out"]["status"] == "ok"
    assert card["validation_report"]["leave_resource_out"]["status"] == "ok"
    assert card["validation_report"]["leave_family_out"]["status"] == "ok"
    assert card["validation_report"]["leave_clade_out"]["status"] == "ok"
    assert "degree_prior" in card["validation_report"]["random_stratified"]["baseline_pr_auc"]
    assert "embedding_cosine" in card["validation_report"]["leave_species_out"]["summary"]["mean_baseline_pr_auc"]
    assert "top_100" in card["validation_report"]["leave_species_out"]["summary"]["mean_top_k_precision"]
    assert "role_only" in card["validation_report"]["leave_species_out"]["summary"]["mean_baseline_top_k_precision"]
    gates = pc.evaluate_lr_model_quality_gates(card, required_splits=["leave_species_out"], min_pr_auc_delta=-1.0)
    assert gates.attrs["passed"]
    assert {"degree_prior", "embedding_cosine", "role_only", "random"}.issubset(set(gates["baseline"]))
    top_k_gates = pc.evaluate_lr_model_quality_gates(
        card,
        required_splits=["leave_species_out"],
        min_pr_auc_delta=-1.0,
        top_k="top_100",
        min_top_k_delta=-1.0,
    )
    assert top_k_gates.attrs["passed"]
    assert "delta_top_k_precision" in top_k_gates.columns
    strict_gates = pc.evaluate_lr_model_quality_gates(tmp_path, required_splits=["leave_species_out"], min_pr_auc_delta=2.0)
    assert not strict_gates.attrs["passed"]
    assert {"model_score", "calibrated_probability"}.issubset(scores.columns)
    assert {"nearest_reference_ligand", "nearest_reference_receptor", "nearest_reference_species", "nearest_reference_resource"}.issubset(scores.columns)
    assert scores.loc[0, "nearest_reference_ligand"] != ""
    assert np.isfinite(scores.loc[0, "nearest_reference_distance"])
    assert scores.loc[0, "model_score"] >= 0.0


@pytest.mark.predict
def test_lightgbm_predictor_import_path_when_available(tmp_path):
    pytest.importorskip("lightgbm")
    interactions, embeddings = _fixture_training()

    card = pc.train_lr_link_predictor(interactions, embeddings, model="lightgbm", output_dir=tmp_path, negative_ratio=1)

    assert card["model_type"] == "lightgbm"

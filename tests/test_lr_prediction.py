import numpy as np
import pandas as pd
import pytest

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
    assert card["metrics"]["n_positive"] == 4
    assert "leave_species_out" in card["validation_report"]
    assert "leave_resource_out" in card["validation_report"]
    assert "leave_family_out" in card["validation_report"]
    assert card["validation_report"]["random_stratified"]["status"] == "ok"
    assert card["validation_report"]["leave_species_out"]["status"] == "ok"
    assert card["validation_report"]["leave_resource_out"]["status"] == "ok"
    assert card["validation_report"]["leave_family_out"]["status"] == "ok"
    assert card["validation_report"]["leave_clade_out"]["status"] == "ok"
    assert scores.loc[0, "model_score"] >= 0.0


@pytest.mark.predict
def test_lightgbm_predictor_import_path_when_available(tmp_path):
    pytest.importorskip("lightgbm")
    interactions, embeddings = _fixture_training()

    card = pc.train_lr_link_predictor(interactions, embeddings, model="lightgbm", output_dir=tmp_path, negative_ratio=1)

    assert card["model_type"] == "lightgbm"

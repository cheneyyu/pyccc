import numpy as np
import pandas as pd
import pytest

import pyccc as pc


def _fixture_training():
    interactions = pd.DataFrame(
        {
            "species": ["toy"] * 2,
            "ligand_gene": ["L1", "L2"],
            "receptor_gene": ["R1", "R2"],
            "resource": ["fixture", "fixture"],
            "evidence_type": ["curated_direct", "curated_direct"],
            "is_positive_label": [True, True],
        }
    )
    embeddings = pd.DataFrame(
        {
            "gene_id": ["L1", "L2", "R1", "R2"],
            "embedding": [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.8, 0.2, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.8, 0.2], dtype=np.float32),
            ],
            "sequence_length": [100, 120, 300, 330],
        }
    )
    return interactions, embeddings


def test_train_score_lr_link_predictor_sklearn_fixture(tmp_path):
    interactions, embeddings = _fixture_training()

    card = pc.train_lr_link_predictor(interactions, embeddings, model="sklearn", output_dir=tmp_path, negative_ratio=1)
    scores = pc.score_lr_candidates(pd.DataFrame({"ligand_gene": ["L1"], "receptor_gene": ["R1"]}), embeddings, model=tmp_path)

    assert (tmp_path / "lr_link_model.joblib").exists()
    assert (tmp_path / "model_card.md").exists()
    assert card["metrics"]["n_positive"] == 2
    assert scores.loc[0, "model_score"] >= 0.0


@pytest.mark.predict
def test_lightgbm_predictor_import_path_when_available(tmp_path):
    pytest.importorskip("lightgbm")
    interactions, embeddings = _fixture_training()

    card = pc.train_lr_link_predictor(interactions, embeddings, model="lightgbm", output_dir=tmp_path, negative_ratio=1)

    assert card["model_type"] == "lightgbm"

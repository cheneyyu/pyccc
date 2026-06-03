import pandas as pd
import pytest

import pyccc as pc


def test_heuristic_and_trained_role_prediction(tmp_path):
    proteins = pd.DataFrame(
        {
            "gene_id": ["L1", "R1"],
            "protein_id": ["pL1", "pR1"],
            "protein_sequence": ["MCCCC", "MAVVVVVVVVVVVVV"],
        }
    )
    embeddings = pc.embed_proteins_esmc(proteins, backend="hash")
    heuristic = pc.predict_protein_roles(proteins, embeddings, model="heuristic")
    train = pd.DataFrame({"ligand_gene": ["L1"], "receptor_gene": ["R1"]})

    card = pc.train_protein_role_classifier(train, embeddings, model="sklearn", output_dir=tmp_path)
    trained = pc.predict_protein_roles(proteins, embeddings, model=tmp_path)

    assert heuristic.loc[0, "ligand_like_score"] > 0
    assert card["model_type"] == "protein_role_classifier"
    assert card["classifier"] == "sklearn"
    assert {"ligand_like_score", "receptor_like_score", "secreted_like_score", "membrane_like_score"}.issubset(trained.columns)


@pytest.mark.predict
def test_lightgbm_role_classifier_import_path_when_available(tmp_path):
    pytest.importorskip("lightgbm")
    proteins = pd.DataFrame(
        {
            "gene_id": ["L1", "L2", "R1", "R2"],
            "protein_id": ["pL1", "pL2", "pR1", "pR2"],
            "protein_sequence": ["MCCCC", "MCCCCC", "MAVVVVVVVVVVVVV", "MAVVVVVVVVVVVVVA"],
        }
    )
    embeddings = pc.embed_proteins_esmc(proteins, backend="hash")
    train = pd.DataFrame({"ligand_gene": ["L1", "L2"], "receptor_gene": ["R1", "R2"]})

    card = pc.train_protein_role_classifier(train, embeddings, model="lightgbm", output_dir=tmp_path)
    trained = pc.predict_protein_roles(proteins, embeddings, model=tmp_path)

    assert card["classifier"] == "lightgbm"
    assert trained["ligand_like_score"].between(0, 1).all()

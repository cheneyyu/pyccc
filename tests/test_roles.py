import pandas as pd
import pytest

import pyccc as pc


class _FirstDimRoleModel:
    def predict_proba(self, X):
        score = X[:, 0].astype(float)
        return pd.DataFrame({0: 1.0 - score, 1: score}).to_numpy()


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
    assert card["embedding_model"]["model_name"] == [pc.ESMC_300M_MODEL_NAME]
    assert card["embedding_model"]["embedding_backend"] == ["hash"]
    assert (tmp_path / "model_card.md").exists()
    assert card["metrics"]["ligand_like"]["validation_status"] == "skipped"
    assert {"ligand_like_score", "receptor_like_score", "secreted_like_score", "membrane_like_score"}.issubset(trained.columns)


def test_trained_role_prediction_aligns_embeddings_by_gene_id():
    proteins = pd.DataFrame(
        {
            "gene_id": ["A", "B"],
            "protein_id": ["pA", "pB"],
            "protein_sequence": ["MAAA", "MBBB"],
        }
    )
    embeddings = pd.DataFrame(
        {
            "gene_id": ["B", "A"],
            "embedding": [[0.1, 0.0], [0.9, 0.0]],
        }
    )
    payload = {"roles": ["ligand_like"], "models": {"ligand_like": _FirstDimRoleModel()}}

    roles = pc.predict_protein_roles(proteins, embeddings, model=payload)

    assert roles.loc[roles["gene_id"] == "A", "ligand_like_score"].iloc[0] == pytest.approx(0.9)
    assert roles.loc[roles["gene_id"] == "B", "ligand_like_score"].iloc[0] == pytest.approx(0.1)


def test_role_classifier_model_card_reports_holdout_metrics(tmp_path):
    genes = [f"L{i}" for i in range(4)] + [f"R{i}" for i in range(4)] + [f"B{i}" for i in range(4)]
    proteins = pd.DataFrame(
        {
            "gene_id": genes,
            "protein_id": [f"p{gene}" for gene in genes],
            "protein_sequence": ["M" + "A" * (i + 4) for i in range(len(genes))],
        }
    )
    embeddings = pc.embed_proteins_esmc(proteins, backend="hash")
    train = pd.DataFrame({"ligand_gene": [f"L{i}" for i in range(4)], "receptor_gene": [f"R{i}" for i in range(4)]})

    card = pc.train_protein_role_classifier(train, embeddings, model="sklearn", output_dir=tmp_path)

    ligand_metrics = card["metrics"]["ligand_like"]
    assert ligand_metrics["validation_status"] == "ok"
    assert "validation_pr_auc" in ligand_metrics
    assert "validation_top_k_precision" in ligand_metrics
    assert (tmp_path / "model_card.json").exists()
    assert (tmp_path / "model_card.md").read_text(encoding="utf-8").startswith("# pyccc Protein Role Classifier")


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

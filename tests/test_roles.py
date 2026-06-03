import pandas as pd

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

    card = pc.train_protein_role_classifier(train, embeddings, output_dir=tmp_path)
    trained = pc.predict_protein_roles(proteins, embeddings, model=tmp_path)

    assert heuristic.loc[0, "ligand_like_score"] > 0
    assert card["model_type"] == "protein_role_classifier"
    assert {"ligand_like_score", "receptor_like_score", "secreted_like_score", "membrane_like_score"}.issubset(trained.columns)

import numpy as np
import pandas as pd

import pyccc as pc


def test_hash_embeddings_are_cached_and_pair_features_are_deterministic(tmp_path):
    proteins = pd.DataFrame(
        {
            "gene_id": ["L1", "R1"],
            "protein_id": ["pL1", "pR1"],
            "protein_sequence": ["MCCCC", "MAVVVVVVVV"],
        }
    )
    first = pc.embed_proteins_esmc(proteins, backend="hash", cache_dir=tmp_path)
    second = pc.embed_proteins_esmc(proteins, backend="hash", cache_dir=tmp_path)
    pairs = pd.DataFrame({"ligand_gene": ["L1"], "receptor_gene": ["R1"], "ligand_role_score": [0.8], "receptor_role_score": [0.7]})

    features_a = pc.make_lr_pair_features(pairs, first)
    features_b = pc.make_lr_pair_features(pairs, second)

    np.testing.assert_allclose(first.loc[0, "embedding"], second.loc[0, "embedding"])
    np.testing.assert_allclose(features_a.X, features_b.X)
    assert "cosine" in features_a.feature_names

import pandas as pd

import pyccc as pc


def test_density_prior_and_predicted_lr_table_metadata():
    train = pd.DataFrame(
        {
            "species": ["human", "human", "mouse"],
            "clade": ["mammal", "mammal", "mammal"],
            "ligand_gene": ["L1", "L2", "L1"],
            "receptor_gene": ["R1", "R2", "R1"],
            "resource": ["A", "A", "B"],
        }
    )
    density = pc.estimate_lr_density_prior(train, groupby="clade", min_species_edges=1)
    scores = pd.DataFrame(
        {
            "ligand_gene": ["L1", "L2", "L3"],
            "receptor_gene": ["R1", "R2", "R3"],
            "model_score": [0.9, 0.8, 0.1],
            "ligand_role_score": [0.8, 0.7, 0.4],
            "receptor_role_score": [0.8, 0.7, 0.4],
        }
    )

    db = pc.build_predicted_lr_table(scores, density_prior=density, species_hint="mammal", min_score=0.5, max_pairs=10)

    assert db.interactions["evidence_type"].eq("embedding_link_prediction").all()
    assert {"nearest_reference_pathway", "nearest_reference_distance", "warning"}.issubset(db.interactions.columns)
    assert "prediction_summary" in db.metadata
    summary = db.metadata["prediction_summary"]
    assert summary.loc[0, "selected_pair_count"] == db.interactions.shape[0]
    assert summary.loc[0, "candidate_grid_size"] == 9
    assert summary.loc[0, "target_pair_count"] == 7
    assert summary.loc[0, "achieved_density"] == summary.loc[0, "selected_pair_count"] / 9

    gates = pc.evaluate_predicted_lr_density_prior(db, max_fold_error=10.0)
    assert gates.attrs["passed"]
    assert {"density_prior", "achieved_density", "density_fold_error", "passed"}.issubset(gates.columns)

    strict_gates = pc.evaluate_predicted_lr_density_prior(summary.assign(achieved_density=0.01), max_abs_delta=0.001, max_fold_error=1.1)
    assert not strict_gates.attrs["passed"]


def test_density_prior_table_falls_back_to_median_when_clade_is_unmatched():
    density = pd.DataFrame(
        {
            "clade": ["mammal", "plant"],
            "density_prior": [0.2, 0.6],
            "density_source_species": ["human;mouse", "rice"],
            "density_source_resources": ["CellChatDB;OmniPath", "PlantCellChat"],
        }
    )
    scores = pd.DataFrame(
        {
            "ligand_gene": [f"L{i}" for i in range(5) for _ in range(2)],
            "receptor_gene": ["R1", "R2"] * 5,
            "model_score": [1.0 - i * 0.05 for i in range(10)],
        }
    )

    db = pc.build_predicted_lr_table(scores, density_prior=density, species_hint="unknown", min_score=0.0, max_pairs=10)
    summary = db.metadata["prediction_summary"]

    assert summary.loc[0, "density_prior"] == 0.4
    assert summary.loc[0, "density_mode"] == "table_fallback_median"
    assert summary.loc[0, "target_pair_count"] == 4
    assert summary.loc[0, "selected_pair_count"] == 4
    assert summary.loc[0, "density_source_species"] == "human;mouse;rice"
    assert summary.loc[0, "density_source_resources"] == "CellChatDB;OmniPath;PlantCellChat"

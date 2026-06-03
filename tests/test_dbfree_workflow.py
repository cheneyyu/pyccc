import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc


def test_dbfree_toy_prediction_runs_through_compute_communication(tmp_path):
    fasta = tmp_path / "proteins.fa"
    fasta.write_text(">pL gene=L1\nMCCCCCC\n>pR gene=R1\nMAVVVVVVVVVVVVV\n>pX gene=X1\nMAAAAA\n", encoding="utf-8")
    adata = AnnData(
        np.array([[4, 0, 1], [5, 0, 1], [0, 3, 1], [0, 4, 1]], dtype=float),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame({"gene_id": ["L1", "R1", "X1"]}, index=["L1", "R1", "X1"]),
    )

    predicted = pc.predict_lr_dbfree(
        adata,
        protein_fasta=fasta,
        gene_id_key="gene_id",
        species_name="toy",
        model="heuristic",
        role_model="heuristic",
        embedding_backend="hash",
        allow_fixture_models=True,
        density_prior=1.0,
        min_score=0.0,
        max_pairs=3,
        max_pairs_per_ligand=1,
        max_pairs_per_receptor=1,
        allow_low_score_density_fill=True,
        expression_min_fraction=0.0,
    )
    result = pc.compute_communication(adata, "cell_type", predicted, gene_symbols_key="gene_id", min_pct=0.0, score_method="cellchat")

    assert not predicted.interactions.empty
    assert {"model_score", "density_rank", "evidence_type", "warning", "nearest_reference_ligand"}.issubset(predicted.interactions.columns)
    assert {"ligand_secreted_like_score", "receptor_membrane_like_score"}.issubset(predicted.interactions.columns)
    warnings = ";".join(predicted.interactions["warning"].astype(str))
    assert "heuristic_pair_ranker" in warnings
    assert "heuristic_role_model" in warnings
    assert "hash_embedding_backend" in warnings
    assert "prediction_summary" in predicted.metadata
    assert predicted.metadata["dbfree_model_stack"]["allow_fixture_models"]
    assert predicted.metadata["dbfree_model_stack"]["embedding_backend"] == "hash"
    summary = predicted.metadata["prediction_summary"]
    assert summary.loc[0, "model_stack"] == pc.DBFREE_STACK_NAME
    assert "heuristic_role_model" in str(summary.loc[0, "warning"])
    assert summary.loc[0, "min_score"] == 0.0
    assert summary.loc[0, "max_pairs_per_ligand"] == 1
    assert summary.loc[0, "max_pairs_per_receptor"] == 1
    assert bool(summary.loc[0, "allow_low_score_density_fill"])
    assert not result.interactions.empty


def test_dbfree_production_stack_rejects_implicit_fixture_models(tmp_path):
    fasta = tmp_path / "proteins.fa"
    fasta.write_text(">pL gene=L1\nMCCCCCC\n>pR gene=R1\nMAVVVVV\n", encoding="utf-8")
    adata = AnnData(
        np.ones((2, 2), dtype=float),
        obs=pd.DataFrame({"cell_type": ["A", "B"]}, index=["c1", "c2"]),
        var=pd.DataFrame({"gene_id": ["L1", "R1"]}, index=["L1", "R1"]),
    )

    try:
        pc.predict_lr_dbfree(
            adata,
            protein_fasta=fasta,
            gene_id_key="gene_id",
            model="heuristic",
            role_model="heuristic",
            embedding_backend="hash",
            density_prior=1.0,
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("fixture DB-free stack should require explicit opt-in")

    assert "ESMC-300M embeddings" in message
    assert "LightGBM protein role classifier" in message
    assert "LightGBM pair ranker" in message
    assert "clade-aware density" in message
    assert "allow_fixture_models=True" in message

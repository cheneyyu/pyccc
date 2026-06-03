from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc


def main() -> None:
    adata = AnnData(
        np.array([[4, 0, 1], [5, 0, 1], [0, 3, 1], [0, 4, 1]], dtype=float),
        obs=pd.DataFrame({"cell_type": ["sender", "sender", "receiver", "receiver"]}, index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame({"gene_id": ["L1", "R1", "X1"]}, index=["L1", "R1", "X1"]),
    )
    with tempfile.TemporaryDirectory() as tmp:
        fasta = Path(tmp) / "toy.protein.fa"
        fasta.write_text(">pL gene=L1\nMCCCCCC\n>pR gene=R1\nMAVVVVVVVVVVVVV\n>pX gene=X1\nMAAAAA\n", encoding="utf-8")
        # Toy dry run only; production prediction uses ESMC embeddings plus trained LightGBM role and pair models.
        predicted = pc.predict_lr_dbfree(
            adata,
            protein_fasta=fasta,
            gene_id_key="gene_id",
            species_name="toy_species",
            model="heuristic",
            role_model="heuristic",
            embedding_backend="hash",
            allow_fixture_models=True,
            density_prior=1.0,
            expression_min_fraction=0.0,
            max_pairs=3,
        )
    result = pc.compute_communication(adata, "cell_type", predicted, gene_symbols_key="gene_id", min_pct=0.0, score_method="cellchat")
    print(predicted.interactions[["ligand", "receptor", "model_score", "density_rank"]])
    print(result.interactions[["source", "target", "ligand", "receptor", "prob"]])


if __name__ == "__main__":
    main()

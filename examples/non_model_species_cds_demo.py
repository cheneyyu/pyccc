from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy.cds.fa"
        path.write_text(
            ">txL gene=L1 transcript=L1.1\nATGTGTTGTTGTTAA\n"
            ">txR gene=R1 transcript=R1.1\nATGGCTGTTGTTGTTGTTTAA\n",
            encoding="utf-8",
        )
        proteins = pc.load_cds_translations(path)
        adata = AnnData(
            np.array([[3, 0], [4, 0], [0, 2], [0, 3]], dtype=float),
            obs=pd.DataFrame({"cell_type": ["sender", "sender", "receiver", "receiver"]}, index=[f"cell{i}" for i in range(4)]),
            var=pd.DataFrame({"gene_id": ["L1", "R1"]}, index=["L1", "R1"]),
        )
        print(proteins[["gene_id", "protein_id", "valid_translation", "protein_length"]])
        print(pc.match_expression_genes(adata, proteins, gene_id_key="gene_id"))


if __name__ == "__main__":
    main()

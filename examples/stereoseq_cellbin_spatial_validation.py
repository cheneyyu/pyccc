from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc
import pyccc.plotting as cp


def build_toy_cellbin() -> tuple[AnnData, pc.CellChatDB]:
    adata = AnnData(
        np.array([[5, 0, 2, 1], [4, 0, 2, 1], [0, 3, 1, 2], [0, 4, 1, 2]], dtype=float),
        obs=pd.DataFrame(
            {
                "cell_type": ["sender", "sender", "receiver", "receiver"],
                "section_id": ["section_a", "section_b", "section_a", "section_b"],
                "cell_area": [1.0, 1.0, 1.0, 1.0],
            },
            index=[f"cell{i}" for i in range(4)],
        ),
        var=pd.DataFrame(index=["L1", "R1", "L2", "R2"]),
    )
    adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    predicted = pc.CellChatDB(
        pd.DataFrame(
            {
                "ligand": ["L1", "L2"],
                "receptor": ["R1", "R2"],
                "pathway": ["DB-free predicted", "DB-free predicted"],
                "model_score": [0.9, 0.4],
                "ligand_role_score": [0.8, 0.3],
                "receptor_role_score": [0.7, 0.2],
                "ligand_secreted_like_score": [0.9, 0.1],
                "ligand_membrane_like_score": [0.2, 0.1],
                "receptor_membrane_like_score": [0.8, 0.2],
            }
        ),
        name="toy_predicted",
    )
    return adata, predicted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a tiny Stereo-seq/cellbin-style pyccc spatial validation example.")
    parser.add_argument("--out-dir", default="spatial_validation_example", help="Directory for report tables and PNG plots.")
    parser.add_argument("--n-permutations", type=int, default=10)
    args = parser.parse_args(argv)

    adata, predicted = build_toy_cellbin()
    curated = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"], "pathway": ["toy_curated"]})
    report = pc.validate_spatial_lr_table(
        adata,
        predicted,
        groupby="cell_type",
        section_key="section_id",
        n_permutations=args.n_permutations,
        curated_lr_table=curated,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.summary.to_csv(out_dir / "spatial_validation_summary.tsv", sep="\t", index=False)
    report.celltype_pair_summary.to_csv(out_dir / "spatial_validation_celltype_pairs.tsv", sep="\t", index=False)
    report.null_distribution.to_csv(out_dir / "spatial_validation_null_distribution.tsv", sep="\t", index=False)
    report.distance_decay.to_csv(out_dir / "spatial_validation_distance_decay.tsv", sep="\t", index=False)
    if report.top_k_enrichment is not None:
        report.top_k_enrichment.to_csv(out_dir / "spatial_validation_top_k_enrichment.tsv", sep="\t", index=False)

    enrichment_ax = cp.spatial_validation_enrichment(report)
    cp.save_figure(enrichment_ax.figure, str(out_dir / "spatial_validation_enrichment.png"), also_svg=False)
    decay_ax = cp.spatial_validation_distance_decay(report)
    cp.save_figure(decay_ax.figure, str(out_dir / "spatial_validation_distance_decay.png"), also_svg=False)
    print(f"Wrote spatial validation report to {out_dir}")


if __name__ == "__main__":
    main()

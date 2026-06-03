import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc


def test_spatial_validation_reports_null_statistics():
    adata = AnnData(
        np.array([[5, 0], [4, 0], [0, 3], [0, 4]], dtype=float),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["L1", "R1"]),
    )
    adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    lr = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"], "pathway": ["P"], "model_score": [0.9]})

    report = pc.validate_spatial_lr_table(adata, lr, groupby="cell_type", n_permutations=3)

    assert not report.summary.empty
    assert not report.celltype_pair_summary.empty
    assert not report.distance_decay.empty
    assert set(report.null_distribution["null_model"]) == {"coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation"}
    assert {"ligand", "receptor", "kernel", "score_type", "score_value"}.issubset(report.null_distribution.columns)
    assert {"spatial_ccc_score", "model_weighted_spatial_ccc_score"}.issubset(set(report.null_distribution["score_type"]))
    assert "empirical_pvalue" in report.summary.columns
    assert "model_weighted_empirical_pvalue" in report.summary.columns
    assert "mean_spatial_ccc_score" in report.distance_decay.columns

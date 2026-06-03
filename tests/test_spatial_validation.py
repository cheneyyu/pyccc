import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import pyccc as pc


@pytest.mark.spatial
def test_spatial_validation_reports_null_statistics():
    adata = AnnData(
        np.array([[5, 0], [4, 0], [0, 3], [0, 4]], dtype=float),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"], "section": ["s1", "s2", "s1", "s2"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["L1", "R1"]),
    )
    adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    lr = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"], "pathway": ["P"], "model_score": [0.9]})

    report = pc.validate_spatial_lr_table(adata, lr, groupby="cell_type", n_permutations=3, section_key="section", section_top_k=1)

    assert not report.summary.empty
    assert not report.celltype_pair_summary.empty
    assert not report.distance_decay.empty
    assert not report.section_reproducibility.empty
    assert set(report.null_distribution["null_model"]) == {"coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation"}
    assert {"ligand", "receptor", "kernel", "score_type", "score_value"}.issubset(report.null_distribution.columns)
    assert {"spatial_ccc_score", "model_weighted_spatial_ccc_score"}.issubset(set(report.null_distribution["score_type"]))
    assert "empirical_pvalue" in report.summary.columns
    assert "model_weighted_empirical_pvalue" in report.summary.columns
    assert "mean_spatial_ccc_score" in report.distance_decay.columns
    assert {"n_sections", "top_k_section_fraction", "median_section_rank"}.issubset(report.section_reproducibility.columns)
    assert report.section_reproducibility["n_sections"].max() == 2
    assert report.metadata["section_key"] == "section"

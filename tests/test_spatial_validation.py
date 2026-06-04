import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyccc as pc
import pyccc.plotting as cp
from pyccc.spatial_validation import _distance_matrix_cache, _matched_random_lr, _permute_groups_for_celltype_null, _spatial_weight_tables


@pytest.mark.spatial
def test_spatial_validation_reports_null_statistics():
    adata = AnnData(
        np.array([[5, 0, 2, 1], [4, 0, 2, 1], [0, 3, 1, 2], [0, 4, 1, 2]], dtype=float),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"], "section": ["s1", "s2", "s1", "s2"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["L1", "R1", "L2", "R2"]),
    )
    adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    lr = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "pathway": ["P", "P"],
            "model_score": [0.9, 0.4],
            "ligand_role_score": [0.8, 0.3],
            "receptor_role_score": [0.7, 0.2],
            "ligand_secreted_like_score": [0.9, 0.1],
            "ligand_membrane_like_score": [0.2, 0.1],
            "receptor_membrane_like_score": [0.8, 0.2],
        }
    )
    curated_lr = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"], "pathway": ["curated"]})

    report = pc.validate_spatial_lr_table(
        adata,
        lr,
        groupby="cell_type",
        n_permutations=3,
        section_key="section",
        section_top_k=1,
        curated_lr_table=curated_lr,
    )

    assert not report.summary.empty
    assert not report.celltype_pair_summary.empty
    assert not report.distance_decay.empty
    assert not report.section_reproducibility.empty
    assert report.top_k_enrichment is not None
    assert not report.top_k_enrichment.empty
    assert report.role_kernel_enrichment is not None
    assert not report.role_kernel_enrichment.empty
    assert report.curated_overlap_enrichment is not None
    assert not report.curated_overlap_enrichment.empty
    assert {"role_class", "kernel", "role_kernel_enrichment", "n_role_pairs"}.issubset(report.role_kernel_enrichment.columns)
    assert {("secreted_like", "exp"), ("membrane_contact_like", "contact")}.issubset(
        set(zip(report.role_kernel_enrichment["role_class"], report.role_kernel_enrichment["kernel"], strict=True))
    )
    assert {"k", "score_type", "null_model", "observed_mean", "top_k_enrichment_z", "top_k_empirical_pvalue"}.issubset(report.top_k_enrichment.columns)
    assert set(report.top_k_enrichment["k"]) == {100, 500, 1000}
    assert {"pooled", "matched_random_lr"}.issubset(set(report.top_k_enrichment["null_model"].astype(str)))
    assert set(report.null_distribution["null_model"]) == {"coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation"}
    assert {"ligand", "receptor", "kernel", "score_type", "score_value"}.issubset(report.null_distribution.columns)
    assert {"spatial_ccc_score", "model_weighted_spatial_ccc_score"}.issubset(set(report.null_distribution["score_type"]))
    celltype_null = report.null_distribution[report.null_distribution["null_model"].astype(str) == "celltype_permutation"]
    assert set(celltype_null["celltype_permutation_scope"]) == {"section"}
    matched = report.null_distribution[report.null_distribution["null_model"].astype(str) == "matched_random_lr"]
    assert {"matched_ligand", "matched_receptor", "ligand_match_expression_delta", "ligand_match_role_delta", "ligand_match_degree_delta"}.issubset(matched.columns)
    assert set(matched["ligand"]).issubset({"L1", "L2"})
    assert matched["matched_ligand"].notna().any()
    assert "empirical_pvalue" in report.summary.columns
    assert "model_weighted_empirical_pvalue" in report.summary.columns
    assert "curated_overlap" in report.summary.columns
    assert report.summary.groupby(["ligand", "receptor"])["curated_overlap"].first().to_dict() == {("L1", "R1"): True, ("L2", "R2"): False}
    assert {"n_curated_overlap_pairs", "curated_overlap_enrichment", "curated_overlap_fraction"}.issubset(report.curated_overlap_enrichment.columns)
    assert "mean_spatial_ccc_score" in report.distance_decay.columns
    assert {"n_sections", "top_k_section_fraction", "median_section_rank"}.issubset(report.section_reproducibility.columns)
    assert report.section_reproducibility["n_sections"].max() == 2
    assert report.metadata["section_key"] == "section"
    assert report.metadata["celltype_permutation_scope"] == "section"
    assert report.metadata["curated_lr_table"] is True
    assert report.metadata["top_k"] == [100, 500, 1000]
    assert cp.spatial_validation_enrichment(report) is not None
    assert cp.spatial_validation_distance_decay(report) is not None
    plt.close("all")


def test_celltype_null_permutation_can_be_section_stratified():
    groups = np.array(["A", "A", "B", "C", "C", "D"])
    sections = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])

    permuted = _permute_groups_for_celltype_null(groups, sections, np.random.default_rng(1))

    for section in sorted(set(sections)):
        mask = sections == section
        assert sorted(permuted[mask].tolist()) == sorted(groups[mask].tolist())


def test_matched_random_lr_preserves_weighting_metadata():
    lr = pd.DataFrame(
        {
            "ligand": ["L1", "L2"],
            "receptor": ["R1", "R2"],
            "model_score": [0.25, 0.75],
            "confidence": [0.2, 0.7],
            "density_rank": [1, 2],
            "ligand_role_score": [0.8, 0.6],
            "receptor_role_score": [0.7, 0.5],
        }
    )
    expression = pd.Series({"L1": 1.0, "L2": 1.2, "R1": 0.8, "R2": 0.9, "G": 1.1})

    matched = _matched_random_lr(lr, expression, np.random.default_rng(0))

    assert {"model_score", "confidence", "density_rank", "ligand_role_score", "receptor_role_score"}.issubset(matched.columns)
    assert matched["model_score"].tolist() == [0.25, 0.75]
    assert matched["density_rank"].tolist() == [1, 2]


def test_spatial_weight_tables_match_distance_cache():
    coords = np.array([[0, 0], [0, 1], [2, 0], [2, 2], [4, 0]], dtype=float)
    groups = np.array(["A", "A", "B", "B", "C"])
    dist = _distance_matrix_cache(coords, max_cells=10)

    uncached = _spatial_weight_tables(coords, groups, radius=1.5, sigma=2.0, kernels=("contact", "exp"))
    cached = _spatial_weight_tables(coords, groups, radius=1.5, sigma=2.0, kernels=("contact", "exp"), distance_matrix=dist)

    for kernel in ("contact", "exp"):
        left = uncached[kernel].sort_values(["source", "target"]).reset_index(drop=True)
        right = cached[kernel].sort_values(["source", "target"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


def test_spatial_validation_can_skip_slow_diagnostics():
    adata = AnnData(
        np.array([[5, 0], [4, 0], [0, 3], [0, 4]], dtype=float),
        obs=pd.DataFrame({"cell_type": ["A", "A", "B", "B"], "section": ["s1", "s1", "s1", "s1"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=["L1", "R1"]),
    )
    adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    lr = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"], "model_score": [0.9]})

    report = pc.validate_spatial_lr_table(
        adata,
        lr,
        groupby="cell_type",
        n_permutations=1,
        section_key="section",
        compute_distance_decay=False,
        compute_section_reproducibility=False,
    )

    assert not report.summary.empty
    assert report.distance_decay.empty
    assert {"ligand", "receptor", "distance_min", "distance_max", "mean_distance", "mean_spatial_ccc_score"}.issubset(report.distance_decay.columns)
    assert report.section_reproducibility.empty
    assert {"ligand", "receptor", "kernel", "n_sections", "top_k_section_fraction"}.issubset(report.section_reproducibility.columns)
    assert report.metadata["compute_distance_decay"] is False
    assert report.metadata["compute_section_reproducibility"] is False


def test_spatial_validation_distance_decay_can_sample_cells():
    adata = AnnData(
        np.tile(np.array([[5, 0], [0, 4]], dtype=float), (5, 1)),
        obs=pd.DataFrame({"cell_type": ["A", "B"] * 5}, index=[f"c{i}" for i in range(10)]),
        var=pd.DataFrame(index=["L1", "R1"]),
    )
    adata.obsm["spatial"] = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
    lr = pd.DataFrame({"ligand": ["L1"], "receptor": ["R1"], "model_score": [0.9]})

    report = pc.validate_spatial_lr_table(
        adata,
        lr,
        groupby="cell_type",
        n_permutations=1,
        compute_section_reproducibility=False,
        distance_decay_max_cells=4,
    )

    assert report.metadata["distance_decay_max_cells"] == 4
    assert not report.distance_decay.empty
    assert report.distance_decay["mean_distance"].nunique() <= 5


def test_stereoseq_cellbin_example_writes_report_and_plots(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "spatial_example"

    subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "stereoseq_cellbin_spatial_validation.py"),
            "--out-dir",
            str(out_dir),
            "--n-permutations",
            "2",
        ],
        check=True,
        cwd=root,
    )

    expected = [
        "spatial_validation_summary.tsv",
        "spatial_validation_celltype_pairs.tsv",
        "spatial_validation_null_distribution.tsv",
        "spatial_validation_distance_decay.tsv",
        "spatial_validation_top_k_enrichment.tsv",
        "spatial_validation_enrichment.png",
        "spatial_validation_distance_decay.png",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists()
        assert path.stat().st_size > 0

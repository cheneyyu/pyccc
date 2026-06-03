import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData


ROOT = Path(__file__).resolve().parents[1]


def test_dbfree_validation_scripts_smoke(tmp_path):
    source = tmp_path / "source.h5ad"
    adata = AnnData(
        np.array(
            [
                [30, 0, 4],
                [20, 0, 3],
                [0, 25, 2],
                [0, 15, 2],
            ],
            dtype=float,
        ),
        obs=pd.DataFrame({"annotation": ["A", "A", "B", "B"]}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame({"gene_id": ["L1", "R1", "X1"]}, index=["L1", "R1", "X1"]),
    )
    adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [4, 0], [4, 1]], dtype=float)
    adata.write_h5ad(source)
    fasta = tmp_path / "proteins.fa"
    fasta.write_text(">pL gene=L1\nMCCCCCC\n>pR gene=R1\nMAVVVV\n>pX gene=X1\nMAAAA\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        json.dumps(
            {
                "name": "toy_validation",
                "species": "Toy species",
                "species_hint": "unknown",
                "clade": "toy",
                "technology": "toy spatial",
                "source_page": "https://example.org/toy",
                "protein_fasta": str(fasta),
                "annotation_priority": ["annotation"],
                "spatial_key": "spatial",
                "gene_id_key": "gene_id",
                "publishable_gene_match_min": 0.5,
                "sections": [
                    {
                        "name": "toy_section",
                        "h5ad_url": "https://example.org/toy.h5ad",
                        "local_path": str(source),
                        "file_name": "toy.h5ad",
                        "expected_size": source.stat().st_size,
                        "smoke": True,
                    }
                ],
                "spatial_validation": {
                    "distance_kernels": ["contact", "exp"],
                    "null_models": ["coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation"],
                    "top_k": [1, 2],
                    "development_permutations": 2,
                    "random_seed": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    data = tmp_path / "data"
    _run("scripts/download_dbfree_validation_data.py", "--manifest", manifest, "--data-dir", data, "--results-dir", results, "--smoke-only", "--dry-run")
    _run("scripts/prepare_dbfree_validation_data.py", "--manifest", manifest, "--data-dir", data, "--results-dir", results, "--smoke-only")
    _run("scripts/prepare_target_proteome.py", "--manifest", manifest, "--results-dir", results)
    dataset_dir = results / "toy_validation"
    predicted = pd.DataFrame(
        {
            "ligand": ["L1", "X1"],
            "receptor": ["R1", "R1"],
            "pathway": ["DB-free predicted", "DB-free predicted"],
            "annotation": ["Predicted LR", "Predicted LR"],
            "evidence": ["test", "test"],
            "model_score": [0.9, 0.2],
            "confidence": [0.9, 0.2],
            "ligand_role_score": [0.9, 0.4],
            "receptor_role_score": [0.8, 0.8],
            "embedding_cosine": [0.7, 0.1],
        }
    )
    predicted.to_csv(dataset_dir / "predicted_lr.tsv", sep="\t", index=False)
    _run(
        "scripts/run_dbfree_spatial_validation.py",
        "--manifest",
        manifest,
        "--results-dir",
        results,
        "--smoke-only",
        "--n-permutations",
        "2",
        "--baseline",
        "dbfree",
        "--baseline",
        "role_only",
    )
    figure_prefix = tmp_path / "figures" / "dbfree_spatial_validation_main"
    _run("scripts/make_dbfree_spatial_validation_figure.py", "--results-dir", results, "--output-prefix", figure_prefix)

    assert (results / "download_manifest.tsv").exists()
    assert (dataset_dir / "section_qc.tsv").exists()
    assert (dataset_dir / "gene_protein_match.tsv").exists()
    assert (dataset_dir / "spatial_validation_summary.tsv").exists()
    assert (results / "baseline_comparison.tsv").exists()
    for ext in ("png", "svg", "pdf"):
        assert figure_prefix.with_suffix(f".{ext}").exists()
    assert figure_prefix.with_name("dbfree_spatial_validation_main_legend.md").exists()
    assert figure_prefix.with_name("dbfree_spatial_validation_main_source_tables.tar.gz").exists()


def _run(script, *args):
    subprocess.run(
        [sys.executable, str(ROOT / script), *[str(arg) for arg in args]],
        cwd=ROOT,
        check=True,
    )

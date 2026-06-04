import importlib.util
import json
import subprocess
import sys
import tarfile
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
    normalized_fasta = tmp_path / "normalized" / "toy.longest_protein.fa"
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
                "protein_fasta": str(normalized_fasta),
                "protein_source": {
                    "url": "https://example.org/proteins.fa",
                    "local_path": str(fasta),
                    "gene_id_regex": r"gene=([^\s]+)",
                    "protein_id_regex": r"^([^\s]+)",
                    "select": "longest",
                },
                "annotation_priority": ["annotation"],
                "spatial_key": "spatial",
                "gene_id_key": "gene_id",
                "publishable_gene_match_min": 0.5,
                "prediction": {
                    "role_model": "universal_esmc300m_lgbm_role_classifiers_v0",
                    "pair_model": "universal_esmc300m_lgbm_pair_ranker_v0",
                    "density_prior": "auto",
                    "embedding_model": "biohub/esmc-300m-2024-12",
                    "embedding_revision": "main",
                },
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
    full_topk = pd.read_csv(dataset_dir / "spatial_validation_top_k_enrichment.tsv", sep="\t")
    _run(
        "scripts/run_dbfree_spatial_validation.py",
        "--manifest",
        manifest,
        "--results-dir",
        results,
        "--smoke-only",
        "--n-permutations",
        "2",
        "--top-k-only",
        "--baseline",
        "dbfree",
        "--baseline",
        "role_only",
    )
    topk = pd.read_csv(dataset_dir / "spatial_validation_top_k_enrichment.tsv", sep="\t")
    assert topk["top_k_enrichment_z"].notna().any()
    assert set(topk["null_model"].astype(str)).issuperset({"matched_random_lr", "score_permutation"})
    compare_cols = [
        "validation_strategy",
        "kernel",
        "score_type",
        "null_model",
        "k",
        "observed_mean",
        "null_mean",
        "null_sd",
        "top_k_enrichment_z",
        "top_k_empirical_pvalue",
    ]
    full_cmp = full_topk[compare_cols].sort_values(compare_cols[:5]).reset_index(drop=True)
    fast_cmp = topk[compare_cols].sort_values(compare_cols[:5]).reset_index(drop=True)
    pd.testing.assert_frame_equal(fast_cmp, full_cmp, check_dtype=False, atol=1e-12, rtol=1e-12)
    topk_before = (dataset_dir / "spatial_validation_top_k_enrichment.tsv").read_text(encoding="utf-8")
    _run(
        "scripts/run_dbfree_spatial_validation.py",
        "--manifest",
        manifest,
        "--results-dir",
        results,
        "--smoke-only",
        "--distance-decay-only",
        "--baseline",
        "dbfree",
        "--baseline",
        "role_only",
    )
    assert (dataset_dir / "spatial_validation_top_k_enrichment.tsv").read_text(encoding="utf-8") == topk_before
    distance_decay = pd.read_csv(dataset_dir / "spatial_validation_distance_decay.tsv", sep="\t")
    assert "model_weighted_mean_spatial_ccc_score" in distance_decay.columns
    figure_prefix = tmp_path / "figures" / "dbfree_spatial_validation_main"
    _run("scripts/make_dbfree_spatial_validation_figure.py", "--results-dir", results, "--output-prefix", figure_prefix)
    acceptance = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_dbfree_validation_acceptance.py"),
            "--manifest",
            str(manifest),
            "--results-dir",
            str(results),
            "--figures-dir",
            str(figure_prefix.parent),
        ],
        cwd=ROOT,
        check=False,
    )

    assert (results / "download_manifest.tsv").exists()
    assert (dataset_dir / "section_qc.tsv").exists()
    assert (dataset_dir / "gene_protein_match.tsv").exists()
    assert normalized_fasta.exists()
    assert "gene=L1" in normalized_fasta.read_text(encoding="utf-8")
    model_summary = pd.read_csv(dataset_dir / "validation_model_card.tsv", sep="\t")
    assert {"embedding_model_revision", "role_model_checksum16", "pair_model_checksum16", "density_prior_checksum16", "training_resources", "clades_included"}.issubset(model_summary.columns)
    assert model_summary["embedding_model_revision"].iloc[0] == "main"
    assert model_summary["role_model_manifest"].iloc[0] == "universal_esmc300m_lgbm_role_classifiers_v0"
    assert model_summary["pair_model_manifest"].iloc[0] == "universal_esmc300m_lgbm_pair_ranker_v0"
    assert model_summary["role_model_path"].str.contains("src/pyccc/models/universal_esmc300m_lgbm_role_classifiers_v0", regex=False).all()
    assert model_summary["pair_model_path"].str.contains("src/pyccc/models/universal_esmc300m_lgbm_pair_ranker_v0", regex=False).all()
    checksum_lengths = model_summary[["role_model_checksum16", "pair_model_checksum16", "density_prior_checksum16"]].fillna("").astype(str).apply(lambda col: col.str.len())
    assert checksum_lengths.ge(16).all().all()
    assert (dataset_dir / "spatial_validation_summary.tsv").exists()
    assert (results / "baseline_comparison.tsv").exists()
    for ext in ("png", "svg", "pdf"):
        assert figure_prefix.with_suffix(f".{ext}").exists()
    assert figure_prefix.with_name("dbfree_spatial_validation_main_legend.md").exists()
    assert figure_prefix.with_name("dbfree_spatial_validation_main_source_tables.tar.gz").exists()
    report = pd.read_csv(results / "acceptance_report.tsv", sep="\t")
    assert acceptance.returncode != 0
    assert {"data", "sequence", "model", "spatial", "figure", "reproducibility"}.issubset(set(report["category"]))
    model_gates = set(report.loc[report["category"] == "model", "gate"])
    assert {"role_model_file_exists", "pair_model_card_exists", "density_prior_table_exists", "validation_model_card_has_checksums"}.issubset(model_gates)
    model_gate_status = report.loc[report["category"] == "model"].set_index("gate")["passed"].astype(bool)
    assert model_gate_status.loc["role_model_file_exists"]
    assert model_gate_status.loc["pair_model_file_exists"]
    assert model_gate_status.loc["density_prior_table_exists"]
    assert model_gate_status.loc["validation_model_card_summary_complete"]
    assert not model_gate_status.loc["density_prior_has_manifest_clade"]


def test_artista_acceptance_requires_role_only_control(tmp_path):
    checker = _load_script("check_dbfree_validation_acceptance")
    results_dir = tmp_path / "artista"
    results_dir.mkdir()
    dbfree_rows = [
        _topk_row(section, strategy="dbfree", z=z, p=0.01, k=k)
        for section, z in [("Control_Juv", 8.0), ("5DPI_1", 7.0), ("30DPI", 1.0)]
        for k in (500, 1000)
    ]
    pd.DataFrame(dbfree_rows).to_csv(results_dir / "spatial_validation_top_k_enrichment.tsv", sep="\t", index=False)

    missing_control = checker._artista_spatial_gate("artista_axolotl", results_dir)
    assert not missing_control["passed"]
    assert "missing_role_only" in missing_control["evidence"]

    rows = dbfree_rows + [
        _topk_row(section, strategy="role_only", z=z, p=0.02, k=k)
        for section, z in [("Control_Juv", 3.0), ("5DPI_1", 2.5), ("30DPI", 0.5)]
        for k in (500, 1000)
    ]
    pd.DataFrame(rows).to_csv(results_dir / "spatial_validation_top_k_enrichment.tsv", sep="\t", index=False)
    passing = checker._artista_spatial_gate("artista_axolotl", results_dir)
    assert passing["passed"]
    assert "passing_sections=2" in passing["evidence"]


def test_acceptance_reproducibility_design_gates(tmp_path):
    checker = _load_script("check_dbfree_validation_acceptance")
    manifest = {
        "name": "toy_complete",
        "species": "Toy species",
        "protein_fasta": "toy.fa",
        "protein_source": {"url": "https://example.org/toy.fa"},
        "required_final_sections": ["S1", "S2"],
        "predicted_lr_gene_match_min": 0.70,
        "spatial_validation": {
            "distance_kernels": ["contact", "exp"],
            "null_models": ["coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation"],
            "top_k": [100, 500],
            "final_permutations": 1000,
        },
    }
    results_dir = tmp_path / "results" / "toy_complete"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir(parents=True)
    figures_dir.mkdir()

    pd.DataFrame(
        [
            _download_row("toy_complete", "S1", "spatial_h5ad"),
            _download_row("toy_complete", "S2", "spatial_h5ad"),
            _download_row("toy_complete", "proteome", "protein_fasta"),
        ]
    ).to_csv(results_dir / "download_manifest.tsv", sep="\t", index=False)
    pd.DataFrame({"ligand": ["L1", "L2"], "receptor": ["R1", "R2"]}).to_csv(results_dir / "predicted_lr.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "gene_id": ["L1", "L2", "R1", "R2", "missing"],
            "in_expression": [True, True, True, True, True],
            "in_proteins": [True, True, True, True, False],
        }
    ).to_csv(results_dir / "gene_protein_match.tsv", sep="\t", index=False)
    for name in checker.REQUIRED_SPATIAL_OUTPUTS:
        (results_dir / name).write_text("placeholder\n", encoding="utf-8")
    topk = pd.DataFrame(
        [
            _design_topk_row(kernel=kernel, null_model=null_model, strategy=strategy, k=k)
            for kernel in ("contact", "exp")
            for null_model in ("coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation")
            for strategy in ("dbfree", "role_only", "embedding_cosine", "expression_only")
            for k in (100, 500)
        ]
    )
    topk.to_csv(results_dir / "spatial_validation_top_k_enrichment.tsv", sep="\t", index=False)
    (tmp_path / "results" / "baseline_comparison.tsv").write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "results" / "baseline_topk_enrichment.tsv").write_text("placeholder\n", encoding="utf-8")
    legend = (
        "Predicted LR edges are computational candidates. Spatial validation is plausibility evidence "
        "with null models, cells, groups, LR pairs, and permutations."
    )
    (figures_dir / "dbfree_spatial_validation_main_legend.md").write_text(legend, encoding="utf-8")
    for name in (
        "dbfree_spatial_validation_main.png",
        "dbfree_spatial_validation_main.svg",
        "dbfree_spatial_validation_main.pdf",
        "dbfree_spatial_validation_main_source_tables.tar.gz",
        "dbfree_spatial_validation_source_tables.tar.gz",
    ):
        (figures_dir / name).write_text("placeholder\n", encoding="utf-8")
    with tarfile.open(figures_dir / "dbfree_spatial_validation_main_source_tables.tar.gz", "w:gz") as archive:
        for relative in (
            "baseline_comparison.tsv",
            "baseline_topk_enrichment.tsv",
            "toy_complete/spatial_validation_summary.tsv",
            "toy_complete/spatial_validation_top_k_enrichment.tsv",
            "toy_complete/spatial_validation_distance_decay.tsv",
            "toy_complete/validation_model_card.tsv",
        ):
            source = tmp_path / relative.replace("/", "_")
            source.write_text("placeholder\n", encoding="utf-8")
            archive.add(source, arcname=relative)

    assert checker._download_manifest_gate(manifest, results_dir)["passed"]
    assert checker._predicted_lr_gene_coverage_gate(manifest, results_dir)["passed"]
    assert checker._spatial_output_files_gate("toy_complete", results_dir)["passed"]
    assert checker._spatial_design_gate(manifest, results_dir)["passed"]
    global_rows = {row["gate"]: row for row in checker._global_rows(tmp_path / "results", figures_dir)}
    assert global_rows["main_figure_outputs_exist"]["passed"]
    assert global_rows["main_figure_legend_complete"]["passed"]
    assert global_rows["source_tables_tarball_complete"]["passed"]


def _topk_row(section: str, *, strategy: str, z: float, p: float, k: int) -> dict[str, object]:
    return {
        "section_id": section,
        "validation_strategy": strategy,
        "kernel": "exp",
        "score_type": "model_weighted_spatial_ccc_score",
        "null_model": "matched_random_lr",
        "k": k,
        "top_k_enrichment_z": z,
        "top_k_empirical_pvalue": p,
    }


def _download_row(dataset: str, section_id: str, asset_type: str) -> dict[str, object]:
    return {
        "asset_type": asset_type,
        "dataset": dataset,
        "section_id": section_id,
        "source_url": f"https://example.org/{section_id}",
        "local_path": f"data/{section_id}",
        "actual_bytes": 123,
        "sha256": "a" * 64,
        "status": "ok",
        "downloaded_at": "2026-06-04T00:00:00+00:00",
    }


def _design_topk_row(*, kernel: str, null_model: str, strategy: str, k: int) -> dict[str, object]:
    return {
        "kernel": kernel,
        "score_type": "model_weighted_spatial_ccc_score",
        "null_model": null_model,
        "k": k,
        "validation_strategy": strategy,
        "n_permutations": 1000,
        "random_seed": 0,
        "observed_mean": 1.0,
        "null_mean": 0.5,
        "null_sd": 0.1,
        "top_k_enrichment_z": 5.0,
        "top_k_empirical_pvalue": 0.001,
    }


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(script, *args):
    subprocess.run(
        [sys.executable, str(ROOT / script), *[str(arg) for arg in args]],
        cwd=ROOT,
        check=True,
    )

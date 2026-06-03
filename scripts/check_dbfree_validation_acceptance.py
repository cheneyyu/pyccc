from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dbfree_validation_utils import load_manifest, manifest_results_dir, write_tsv


FORBIDDEN_FINAL_WARNINGS = {
    "hash_embedding_backend",
    "heuristic_role_model",
    "heuristic_pair_ranker",
    "default_unknown_density",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DB-free real-data validation acceptance gates.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    rows = []
    for manifest_path in args.manifest:
        manifest = load_manifest(manifest_path)
        results_dir = manifest_results_dir(manifest, args.results_dir)
        rows.extend(_dataset_rows(manifest, results_dir))
    rows.extend(_global_rows(Path(args.results_dir or "results/dbfree_validation"), Path(args.figures_dir)))
    report = pd.DataFrame(rows)
    output_root = Path(args.results_dir or "results/dbfree_validation")
    write_tsv(report, output_root / "acceptance_report.tsv")
    if not bool(report["passed"].all()):
        raise SystemExit("DB-free validation acceptance gates are not all passing. See acceptance_report.tsv.")


def _dataset_rows(manifest: dict[str, object], results_dir: Path) -> list[dict[str, object]]:
    dataset = str(manifest["name"])
    rows = []
    rows.append(_gate(dataset, "data", "section_qc_exists", (results_dir / "section_qc.tsv").exists(), str(results_dir / "section_qc.tsv")))
    if (results_dir / "section_qc.tsv").exists():
        qc = pd.read_csv(results_dir / "section_qc.tsv", sep="\t")
        rows.append(_gate(dataset, "data", "sections_load_as_anndata", not qc.empty and qc["spatial_key_found"].astype(bool).all(), f"n_sections={len(qc)}"))
        rows.append(_gate(dataset, "data", "standard_columns_present", _qc_standard_columns_pass(qc), "pyccc_group, section_id, spatial, gene_id recorded by prepare script"))
    rows.append(_gene_gate(manifest, results_dir))
    rows.append(_model_warning_gate(dataset, results_dir))
    if dataset == "artista_axolotl":
        rows.append(_artista_spatial_gate(dataset, results_dir))
    elif dataset == "sota_soybean":
        rows.append(_sota_spatial_gate(dataset, results_dir))
    else:
        rows.append(_generic_spatial_gate(dataset, results_dir))
    return rows


def _qc_standard_columns_pass(qc: pd.DataFrame) -> bool:
    required = {"n_cells_or_bins", "n_genes", "n_groups", "spatial_key_found", "groupby_column_used", "n_expression_genes"}
    return required.issubset(qc.columns) and not qc.empty and (qc["n_groups"].astype(int) > 0).all()


def _gene_gate(manifest: dict[str, object], results_dir: Path) -> dict[str, object]:
    dataset = str(manifest["name"])
    path = results_dir / "gene_protein_match_summary.tsv"
    if not path.exists():
        return _gate(dataset, "sequence", "gene_match_summary_exists", False, str(path))
    summary = pd.read_csv(path, sep="\t")
    threshold = float(manifest.get("publishable_gene_match_min", 0.0))
    fraction = float(summary["matched_fraction"].iloc[0]) if not summary.empty and "matched_fraction" in summary else 0.0
    return _gate(dataset, "sequence", "gene_match_fraction", fraction >= threshold, f"matched_fraction={fraction:.3f}; threshold={threshold:.3f}")


def _model_warning_gate(dataset: str, results_dir: Path) -> dict[str, object]:
    path = results_dir / "prediction_summary.tsv"
    if not path.exists():
        return _gate(dataset, "model", "prediction_summary_exists", False, str(path))
    summary = pd.read_csv(path, sep="\t")
    warnings = ";".join(summary.get("warning", pd.Series(dtype=str)).fillna("").astype(str))
    bad = sorted(item for item in FORBIDDEN_FINAL_WARNINGS if item in warnings)
    return _gate(dataset, "model", "no_fixture_warnings", not bad, "forbidden=" + ",".join(bad))


def _artista_spatial_gate(dataset: str, results_dir: Path) -> dict[str, object]:
    topk = _topk(results_dir)
    if topk.empty:
        return _gate(dataset, "spatial", "artista_main_gate", False, "missing spatial_validation_top_k_enrichment.tsv")
    frame = _primary_topk(topk)
    frame = frame[(frame["validation_strategy"].astype(str) == "dbfree") & (frame["k"].astype(int).isin([500, 1000]))]
    if frame.empty:
        return _gate(dataset, "spatial", "artista_main_gate", False, "missing dbfree top-500/top-1000 rows")
    by_section = frame.groupby("section_id").agg(
        max_z=("top_k_enrichment_z", "max"),
        min_p=("top_k_empirical_pvalue", "min"),
    )
    passing = by_section[(by_section["max_z"] >= 2.0) & (by_section["min_p"] <= 0.05)]
    return _gate(dataset, "spatial", "artista_main_gate", len(passing) >= 2, f"passing_sections={len(passing)}")


def _sota_spatial_gate(dataset: str, results_dir: Path) -> dict[str, object]:
    topk = _topk(results_dir)
    if topk.empty:
        return _gate(dataset, "spatial", "sota_feasibility_gate", False, "missing spatial_validation_top_k_enrichment.tsv")
    frame = _primary_topk(topk)
    frame = frame[(frame["validation_strategy"].astype(str) == "dbfree") & (frame["k"].astype(int).isin([500, 1000]))]
    passing = frame[(frame["top_k_enrichment_z"].astype(float) > 0) & (frame["observed_mean"].astype(float) > frame["null_mean"].astype(float))]
    return _gate(dataset, "spatial", "sota_feasibility_gate", passing["section_id"].nunique() >= 1, f"passing_sections={passing['section_id'].nunique() if not passing.empty else 0}")


def _generic_spatial_gate(dataset: str, results_dir: Path) -> dict[str, object]:
    topk = _topk(results_dir)
    return _gate(dataset, "spatial", "topk_table_exists", not topk.empty, str(results_dir / "spatial_validation_top_k_enrichment.tsv"))


def _topk(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "spatial_validation_top_k_enrichment.tsv"
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()


def _primary_topk(topk: pd.DataFrame) -> pd.DataFrame:
    required = {"kernel", "score_type", "k", "top_k_enrichment_z", "top_k_empirical_pvalue", "validation_strategy"}
    if not required.issubset(topk.columns):
        return pd.DataFrame()
    return topk[
        (topk["kernel"].astype(str) == "exp")
        & (topk["score_type"].astype(str) == "model_weighted_spatial_ccc_score")
    ].copy()


def _global_rows(results_root: Path, figures_dir: Path) -> list[dict[str, object]]:
    required = [
        "dbfree_spatial_validation_main.png",
        "dbfree_spatial_validation_main.svg",
        "dbfree_spatial_validation_main.pdf",
        "dbfree_spatial_validation_main_legend.md",
        "dbfree_spatial_validation_main_source_tables.tar.gz",
    ]
    return [
        _gate("global", "figure", "main_figure_outputs_exist", all((figures_dir / name).exists() for name in required), str(figures_dir)),
        _gate("global", "reproducibility", "baseline_tables_exist", (results_root / "baseline_comparison.tsv").exists() and (results_root / "baseline_topk_enrichment.tsv").exists(), str(results_root)),
    ]


def _gate(dataset: str, category: str, gate: str, passed: bool, evidence: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "category": category,
        "gate": gate,
        "passed": bool(passed),
        "evidence": evidence,
    }


if __name__ == "__main__":
    main()

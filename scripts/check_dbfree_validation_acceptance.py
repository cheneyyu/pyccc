from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import pandas as pd

from dbfree_validation_utils import load_manifest, manifest_results_dir, write_tsv
from pyccc.model_resources import resolve_dbfree_model_path


FORBIDDEN_FINAL_WARNINGS = {
    "hash_embedding_backend",
    "heuristic_role_model",
    "heuristic_pair_ranker",
    "default_unknown_density",
}
REQUIRED_SPATIAL_OUTPUTS = (
    "spatial_validation_summary.tsv",
    "spatial_validation_celltype_pair_summary.tsv",
    "spatial_validation_null_distribution.tsv",
    "spatial_validation_top_k_enrichment.tsv",
    "spatial_validation_role_kernel_enrichment.tsv",
    "spatial_validation_distance_decay.tsv",
    "spatial_validation_section_reproducibility.tsv",
)
REQUIRED_VALIDATION_STRATEGIES = {"dbfree", "role_only", "embedding_cosine", "expression_only"}
REQUIRED_DOWNLOAD_COLUMNS = {"asset_type", "dataset", "section_id", "source_url", "local_path", "actual_bytes", "sha256", "status", "downloaded_at"}
REQUIRED_LEGEND_PHRASES = (
    "computational candidates",
    "plausibility evidence",
    "null models",
    "permutations",
    "cell",
    "groups",
    "lr pairs",
)
REQUIRED_MODEL_CARD_SUMMARY_COLUMNS = (
    "embedding_model_name",
    "embedding_model_revision",
    "role_model_path",
    "role_model_checksum16",
    "pair_model_path",
    "pair_model_checksum16",
    "density_prior_path",
    "density_prior_checksum16",
    "training_resources",
    "species_included",
    "clades_included",
    "validation_split_summary",
    "negative_sampling_strategy",
)


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
    rows.append(_download_manifest_gate(manifest, results_dir))
    rows.append(_gate(dataset, "data", "section_qc_exists", (results_dir / "section_qc.tsv").exists(), str(results_dir / "section_qc.tsv")))
    if (results_dir / "section_qc.tsv").exists():
        qc = pd.read_csv(results_dir / "section_qc.tsv", sep="\t")
        rows.append(_gate(dataset, "data", "sections_load_as_anndata", not qc.empty and qc["spatial_key_found"].astype(bool).all(), f"n_sections={len(qc)}"))
        rows.append(_required_sections_gate(manifest, qc))
        rows.append(_gate(dataset, "data", "standard_columns_present", _qc_standard_columns_pass(qc), "pyccc_group, section_id, spatial, gene_id recorded by prepare script"))
    rows.append(_gene_gate(manifest, results_dir))
    rows.append(_predicted_lr_gene_coverage_gate(manifest, results_dir))
    rows.extend(_model_artifact_gates(manifest, results_dir))
    rows.append(_model_warning_gate(dataset, results_dir))
    rows.append(_spatial_output_files_gate(dataset, results_dir))
    rows.append(_spatial_design_gate(manifest, results_dir))
    if dataset == "artista_axolotl":
        rows.append(_artista_spatial_gate(dataset, results_dir))
    elif dataset == "sota_soybean":
        rows.append(_sota_spatial_gate(dataset, results_dir))
    else:
        rows.append(_generic_spatial_gate(dataset, results_dir))
    return rows


def _download_manifest_gate(manifest: dict[str, object], results_dir: Path) -> dict[str, object]:
    dataset = str(manifest["name"])
    manifests = []
    for path in (results_dir / "download_manifest.tsv", results_dir.parent / "download_manifest.tsv"):
        if path.exists():
            manifests.append(pd.read_csv(path, sep="\t"))
    if not manifests:
        return _gate(dataset, "data", "download_manifest_complete", False, str(results_dir / "download_manifest.tsv"))
    downloads = pd.concat(manifests, ignore_index=True, sort=False)
    downloads = downloads[_text_column(downloads, "dataset") == dataset].copy()
    if downloads.empty:
        return _gate(dataset, "data", "download_manifest_complete", False, f"dataset={dataset}")
    missing_columns = sorted(REQUIRED_DOWNLOAD_COLUMNS - set(downloads.columns))
    required_sections = _required_download_sections(manifest)
    present_sections = set(_text_column(downloads[_text_column(downloads, "asset_type") == "spatial_h5ad"], "section_id"))
    missing_sections = sorted(required_sections - present_sections)
    needs_proteome = bool(manifest.get("protein_source") or manifest.get("protein_fasta"))
    proteome_present = "proteome" in set(_text_column(downloads, "section_id"))
    complete_rows = _download_rows_are_complete(downloads) if not missing_columns else False
    passed = not missing_columns and not missing_sections and (proteome_present or not needs_proteome) and complete_rows
    evidence = (
        f"rows={len(downloads)}; missing_columns={','.join(missing_columns)}; "
        f"missing_sections={','.join(missing_sections)}; proteome_present={proteome_present}; complete_rows={complete_rows}"
    )
    return _gate(dataset, "data", "download_manifest_complete", passed, evidence)


def _required_download_sections(manifest: dict[str, object]) -> set[str]:
    required = [str(item) for item in manifest.get("required_final_sections", [])]
    if required:
        return set(required)
    return {str(section.get("name")) for section in manifest.get("sections", []) if section.get("name")}


def _download_rows_are_complete(downloads: pd.DataFrame) -> bool:
    if downloads.empty or not REQUIRED_DOWNLOAD_COLUMNS.issubset(downloads.columns):
        return False
    frame = downloads.copy()
    status_ok = frame["status"].fillna("").astype(str).eq("ok").all()
    bytes_ok = pd.to_numeric(frame["actual_bytes"], errors="coerce").fillna(0).gt(0).all()
    sha_ok = frame["sha256"].fillna("").astype(str).str.len().ge(32).all()
    url_ok = frame["source_url"].fillna("").astype(str).str.len().gt(0).all()
    time_ok = frame["downloaded_at"].fillna("").astype(str).str.len().gt(0).all()
    return bool(status_ok and bytes_ok and sha_ok and url_ok and time_ok)


def _text_column(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame:
        return pd.Series([""] * len(frame), index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def _qc_standard_columns_pass(qc: pd.DataFrame) -> bool:
    required = {"n_cells_or_bins", "n_genes", "n_groups", "spatial_key_found", "groupby_column_used", "n_expression_genes"}
    return required.issubset(qc.columns) and not qc.empty and (qc["n_groups"].astype(int) > 0).all()


def _required_sections_gate(manifest: dict[str, object], qc: pd.DataFrame) -> dict[str, object]:
    dataset = str(manifest["name"])
    required = [str(item) for item in manifest.get("required_final_sections", [])]
    if not required:
        return _gate(dataset, "data", "required_final_sections_prepared", True, "no required_final_sections configured")
    present = set(qc.get("section_id", pd.Series(dtype=str)).astype(str))
    missing = [section for section in required if section not in present]
    return _gate(
        dataset,
        "data",
        "required_final_sections_prepared",
        not missing,
        f"required={','.join(required)}; missing={','.join(missing)}",
    )


def _gene_gate(manifest: dict[str, object], results_dir: Path) -> dict[str, object]:
    dataset = str(manifest["name"])
    path = results_dir / "gene_protein_match_summary.tsv"
    if not path.exists():
        return _gate(dataset, "sequence", "gene_match_summary_exists", False, str(path))
    summary = pd.read_csv(path, sep="\t")
    threshold = float(manifest.get("publishable_gene_match_min", 0.0))
    fraction = float(summary["matched_fraction"].iloc[0]) if not summary.empty and "matched_fraction" in summary else 0.0
    return _gate(dataset, "sequence", "gene_match_fraction", fraction >= threshold, f"matched_fraction={fraction:.3f}; threshold={threshold:.3f}")


def _predicted_lr_gene_coverage_gate(manifest: dict[str, object], results_dir: Path) -> dict[str, object]:
    dataset = str(manifest["name"])
    predicted_path = results_dir / "predicted_lr.tsv"
    match_path = results_dir / "gene_protein_match.tsv"
    if not predicted_path.exists() or not match_path.exists():
        return _gate(dataset, "sequence", "predicted_lr_gene_coverage", False, f"{predicted_path}; {match_path}")
    predicted = pd.read_csv(predicted_path, sep="\t")
    match = pd.read_csv(match_path, sep="\t")
    if predicted.empty or not {"ligand", "receptor"}.issubset(predicted.columns) or "gene_id" not in match:
        return _gate(dataset, "sequence", "predicted_lr_gene_coverage", False, "missing ligand/receptor or gene_id columns")
    predicted_genes = set(predicted["ligand"].astype(str)).union(set(predicted["receptor"].astype(str)))
    if "in_expression" in match:
        expression_genes = set(match.loc[match["in_expression"].astype(bool), "gene_id"].astype(str))
    else:
        expression_genes = set(match["gene_id"].astype(str))
    covered = len(predicted_genes & expression_genes)
    total = len(predicted_genes)
    fraction = covered / total if total else 0.0
    threshold = float(manifest.get("predicted_lr_gene_match_min", 0.70))
    return _gate(
        dataset,
        "sequence",
        "predicted_lr_gene_coverage",
        fraction >= threshold,
        f"covered={covered}; total={total}; fraction={fraction:.3f}; threshold={threshold:.3f}",
    )


def _model_warning_gate(dataset: str, results_dir: Path) -> dict[str, object]:
    path = results_dir / "prediction_summary.tsv"
    if not path.exists():
        return _gate(dataset, "model", "prediction_summary_exists", False, str(path))
    summary = pd.read_csv(path, sep="\t")
    warnings = ";".join(summary.get("warning", pd.Series(dtype=str)).fillna("").astype(str))
    bad = sorted(item for item in FORBIDDEN_FINAL_WARNINGS if item in warnings)
    return _gate(dataset, "model", "no_fixture_warnings", not bad, "forbidden=" + ",".join(bad))


def _model_artifact_gates(manifest: dict[str, object], results_dir: Path) -> list[dict[str, object]]:
    dataset = str(manifest["name"])
    cfg = dict(manifest.get("prediction", {}))
    role_dir = _resolve_optional_model_dir(cfg.get("role_model"), expected_file="role_model.joblib")
    pair_dir = _resolve_optional_model_dir(cfg.get("pair_model"), expected_file="lr_link_model.joblib")
    role_card = _read_json(_model_file(role_dir, "model_card.json"))
    pair_card = _read_json(_model_file(pair_dir, "model_card.json"))
    density = _read_density(_model_file(pair_dir, "density_prior.tsv"))
    rows = [
        _gate(dataset, "model", "role_model_file_exists", _model_file(role_dir, "role_model.joblib").exists(), _artifact_evidence(cfg.get("role_model"), role_dir, "role_model.joblib")),
        _gate(dataset, "model", "role_model_card_exists", role_card is not None, _artifact_evidence(cfg.get("role_model"), role_dir, "model_card.json")),
        _gate(dataset, "model", "role_model_is_lightgbm", _card_value(role_card, "classifier") == "lightgbm", f"classifier={_card_value(role_card, 'classifier')}"),
        _gate(dataset, "model", "role_model_uses_esmc300m", _embedding_model_is_esmc300m(role_card), _embedding_evidence(role_card)),
        _gate(dataset, "model", "pair_model_file_exists", _model_file(pair_dir, "lr_link_model.joblib").exists(), _artifact_evidence(cfg.get("pair_model"), pair_dir, "lr_link_model.joblib")),
        _gate(dataset, "model", "pair_model_card_exists", pair_card is not None, _artifact_evidence(cfg.get("pair_model"), pair_dir, "model_card.json")),
        _gate(dataset, "model", "pair_model_is_lightgbm", _card_value(pair_card, "model_type") == "lightgbm", f"model_type={_card_value(pair_card, 'model_type')}"),
        _gate(dataset, "model", "pair_model_uses_esmc300m", _embedding_model_is_esmc300m(pair_card), _embedding_evidence(pair_card)),
        _gate(dataset, "model", "pair_model_training_metadata_present", _pair_training_metadata_present(pair_card), _pair_training_evidence(pair_card)),
        _gate(dataset, "model", "density_prior_table_exists", density is not None and not density.empty, _artifact_evidence(cfg.get("pair_model"), pair_dir, "density_prior.tsv")),
        _gate(dataset, "model", "density_prior_has_manifest_clade", _density_has_clade(density, str(manifest.get("clade", ""))), f"clade={manifest.get('clade', '')}"),
        _gate(dataset, "model", "validation_model_card_summary_exists", (results_dir / "validation_model_card.tsv").exists(), str(results_dir / "validation_model_card.tsv")),
    ]
    if (results_dir / "validation_model_card.tsv").exists():
        summary = pd.read_csv(results_dir / "validation_model_card.tsv", sep="\t")
        rows.append(
            _gate(
                dataset,
                "model",
                "validation_model_card_has_checksums",
                _summary_checksums_present(summary),
                _summary_checksum_evidence(summary),
            )
        )
        rows.append(
            _gate(
                dataset,
                "model",
                "validation_model_card_summary_complete",
                _summary_model_card_complete(summary),
                _summary_model_card_evidence(summary),
            )
        )
    else:
        rows.append(_gate(dataset, "model", "validation_model_card_has_checksums", False, str(results_dir / "validation_model_card.tsv")))
        rows.append(_gate(dataset, "model", "validation_model_card_summary_complete", False, str(results_dir / "validation_model_card.tsv")))
    return rows


def _resolve_optional_model_dir(value: object, *, expected_file: str) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return resolve_dbfree_model_path(str(value), expected_file=expected_file)


def _model_file(path: Path | None, file_name: str) -> Path:
    return (path / file_name) if path is not None else Path(file_name)


def _artifact_evidence(manifest_value: object, resolved_dir: Path | None, file_name: str) -> str:
    manifest_text = str(manifest_value or "")
    resolved = str(_model_file(resolved_dir, file_name)) if resolved_dir is not None else ""
    if manifest_text and resolved and manifest_text != str(resolved_dir):
        return f"manifest={manifest_text}; resolved={resolved}"
    return resolved or manifest_text or file_name


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _read_density(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def _card_value(card: dict[str, object] | None, key: str) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get(key, ""))


def _embedding_model_is_esmc300m(card: dict[str, object] | None) -> bool:
    if not isinstance(card, dict):
        return False
    embedding = card.get("embedding_model", {})
    if not isinstance(embedding, dict):
        return False
    model_names = _as_list(embedding.get("model_name"))
    backends = {str(value).lower() for value in _as_list(embedding.get("embedding_backend"))}
    names_ok = any("esmc" in str(name).lower() and "300m" in str(name).lower() for name in model_names)
    backend_ok = "hash" not in backends
    return bool(names_ok and backend_ok)


def _embedding_evidence(card: dict[str, object] | None) -> str:
    if not isinstance(card, dict):
        return "missing model_card.json"
    embedding = card.get("embedding_model", {})
    if not isinstance(embedding, dict):
        return "missing embedding_model"
    return f"model_name={_join(_as_list(embedding.get('model_name')))}; backend={_join(_as_list(embedding.get('embedding_backend')))}"


def _pair_training_metadata_present(card: dict[str, object] | None) -> bool:
    if not isinstance(card, dict):
        return False
    required = ("training_resources", "species_included", "clades_included", "negative_strategy", "negative_sampling", "validation_report")
    return all(bool(card.get(key)) for key in required)


def _pair_training_evidence(card: dict[str, object] | None) -> str:
    if not isinstance(card, dict):
        return "missing model_card.json"
    return (
        f"resources={_join(_as_list(card.get('training_resources')))}; "
        f"species={_join(_as_list(card.get('species_included')))}; "
        f"clades={_join(_as_list(card.get('clades_included')))}; "
        f"negative_strategy={card.get('negative_strategy', '')}"
    )


def _density_has_clade(density: pd.DataFrame | None, clade: str) -> bool:
    if density is None or density.empty or not clade:
        return False
    if "clade" in density:
        return clade in set(density["clade"].astype(str))
    if "species_hint" in density:
        return clade in set(density["species_hint"].astype(str))
    return False


def _summary_checksums_present(summary: pd.DataFrame) -> bool:
    required = ("role_model_checksum16", "pair_model_checksum16", "density_prior_checksum16")
    return all(col in summary and summary[col].fillna("").astype(str).str.len().gt(0).all() for col in required)


def _summary_model_card_complete(summary: pd.DataFrame) -> bool:
    if summary.empty:
        return False
    missing = [col for col in REQUIRED_MODEL_CARD_SUMMARY_COLUMNS if col not in summary]
    if missing:
        return False
    return all(summary[col].fillna("").astype(str).str.len().gt(0).all() for col in REQUIRED_MODEL_CARD_SUMMARY_COLUMNS)


def _summary_model_card_evidence(summary: pd.DataFrame) -> str:
    missing = [col for col in REQUIRED_MODEL_CARD_SUMMARY_COLUMNS if col not in summary]
    empty = [
        col
        for col in REQUIRED_MODEL_CARD_SUMMARY_COLUMNS
        if col in summary and not summary[col].fillna("").astype(str).str.len().gt(0).all()
    ]
    return "missing=" + ",".join(missing) + "; empty=" + ",".join(empty)


def _summary_checksum_evidence(summary: pd.DataFrame) -> str:
    parts = []
    for col in ("role_model_checksum16", "pair_model_checksum16", "density_prior_checksum16"):
        value = summary[col].iloc[0] if col in summary and not summary.empty else ""
        parts.append(f"{col}={value}")
    return "; ".join(parts)


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _join(values: list[object]) -> str:
    return ",".join(str(value) for value in values if str(value))


def _artista_spatial_gate(dataset: str, results_dir: Path) -> dict[str, object]:
    topk = _topk(results_dir)
    if topk.empty:
        return _gate(dataset, "spatial", "artista_main_gate", False, "missing spatial_validation_top_k_enrichment.tsv")
    frame = _primary_topk(topk)
    frame = frame[frame["k"].astype(int).isin([500, 1000])].copy()
    if frame.empty:
        return _gate(dataset, "spatial", "artista_main_gate", False, "missing top-500/top-1000 rows")
    passing_sections = _artista_passing_sections(frame)
    role_sections = set(frame.loc[frame["validation_strategy"].astype(str) == "role_only", "section_id"].astype(str))
    missing_role = sorted(set(frame.loc[frame["validation_strategy"].astype(str) == "dbfree", "section_id"].astype(str)) - role_sections)
    evidence = f"passing_sections={len(passing_sections)}"
    if missing_role:
        evidence += f"; missing_role_only={','.join(missing_role)}"
    return _gate(dataset, "spatial", "artista_main_gate", len(passing_sections) >= 2, evidence)


def _artista_passing_sections(frame: pd.DataFrame) -> list[str]:
    passing = []
    for section, section_frame in frame.groupby("section_id", sort=False):
        dbfree = section_frame[section_frame["validation_strategy"].astype(str) == "dbfree"].copy()
        role = section_frame[section_frame["validation_strategy"].astype(str) == "role_only"].copy()
        if dbfree.empty or role.empty:
            continue
        dbfree["top_k_enrichment_z"] = pd.to_numeric(dbfree["top_k_enrichment_z"], errors="coerce")
        dbfree["top_k_empirical_pvalue"] = pd.to_numeric(dbfree["top_k_empirical_pvalue"], errors="coerce")
        role["top_k_enrichment_z"] = pd.to_numeric(role["top_k_enrichment_z"], errors="coerce")
        role["top_k_empirical_pvalue"] = pd.to_numeric(role["top_k_empirical_pvalue"], errors="coerce")
        dbfree_pass = (dbfree["top_k_enrichment_z"] >= 2.0) & (dbfree["top_k_empirical_pvalue"] <= 0.05)
        if not bool(dbfree_pass.any()):
            continue
        dbfree_best_z = float(dbfree["top_k_enrichment_z"].max())
        role_best_z = float(role["top_k_enrichment_z"].max())
        role_pass = (role["top_k_enrichment_z"] >= 2.0) & (role["top_k_empirical_pvalue"] <= 0.05)
        improves_role = dbfree_best_z > role_best_z
        more_stable_topk = int(dbfree_pass.sum()) > int(role_pass.sum())
        if improves_role or more_stable_topk:
            passing.append(str(section))
    return passing


def _sota_spatial_gate(dataset: str, results_dir: Path) -> dict[str, object]:
    topk = _topk(results_dir)
    if topk.empty:
        return _gate(dataset, "spatial", "sota_feasibility_gate", False, "missing spatial_validation_top_k_enrichment.tsv")
    frame = _primary_topk(topk)
    frame = frame[(frame["validation_strategy"].astype(str) == "dbfree") & (frame["k"].astype(int).isin([500, 1000]))]
    passing = frame[(frame["top_k_enrichment_z"].astype(float) > 0) & (frame["observed_mean"].astype(float) > frame["null_mean"].astype(float))]
    return _gate(dataset, "spatial", "sota_feasibility_gate", passing["section_id"].nunique() >= 1, f"passing_sections={passing['section_id'].nunique() if not passing.empty else 0}")


def _spatial_output_files_gate(dataset: str, results_dir: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_SPATIAL_OUTPUTS if not (results_dir / name).exists()]
    return _gate(dataset, "spatial", "required_spatial_output_files", not missing, "missing=" + ",".join(missing))


def _spatial_design_gate(manifest: dict[str, object], results_dir: Path) -> dict[str, object]:
    dataset = str(manifest["name"])
    topk = _topk(results_dir)
    if topk.empty:
        return _gate(dataset, "spatial", "topk_design_complete", False, str(results_dir / "spatial_validation_top_k_enrichment.tsv"))
    spatial_cfg = dict(manifest.get("spatial_validation", {}))
    required_columns = {"kernel", "score_type", "null_model", "k", "validation_strategy", "n_permutations", "random_seed", "observed_mean", "null_mean", "null_sd", "top_k_enrichment_z", "top_k_empirical_pvalue"}
    missing_columns = sorted(required_columns - set(topk.columns))
    kernels = set(topk.get("kernel", pd.Series(dtype=str)).astype(str))
    nulls = set(topk.get("null_model", pd.Series(dtype=str)).astype(str))
    strategies = set(topk.get("validation_strategy", pd.Series(dtype=str)).astype(str))
    topks = {int(value) for value in pd.to_numeric(topk.get("k", pd.Series(dtype=int)), errors="coerce").dropna().astype(int)}
    required_kernels = {str(item) for item in spatial_cfg.get("distance_kernels", ("contact", "exp"))}
    required_nulls = {str(item) for item in spatial_cfg.get("null_models", ("coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation"))}
    required_topks = {int(item) for item in spatial_cfg.get("top_k", (100, 500, 1000, 5000))}
    final_permutations = spatial_cfg.get("final_permutations")
    max_permutations = int(pd.to_numeric(topk.get("n_permutations", pd.Series([0])), errors="coerce").fillna(0).max()) if "n_permutations" in topk else 0
    permutation_ok = True if final_permutations is None else max_permutations >= int(final_permutations)
    missing_kernels = sorted(required_kernels - kernels)
    missing_nulls = sorted(required_nulls - nulls)
    missing_strategies = sorted(REQUIRED_VALIDATION_STRATEGIES - strategies)
    missing_topks = sorted(required_topks - topks)
    passed = not missing_columns and not missing_kernels and not missing_nulls and not missing_strategies and not missing_topks and permutation_ok
    evidence = (
        f"missing_columns={','.join(missing_columns)}; missing_kernels={','.join(missing_kernels)}; "
        f"missing_nulls={','.join(missing_nulls)}; missing_strategies={','.join(missing_strategies)}; "
        f"missing_topk={','.join(str(item) for item in missing_topks)}; max_permutations={max_permutations}; "
        f"final_permutations={final_permutations if final_permutations is not None else ''}"
    )
    return _gate(dataset, "spatial", "topk_design_complete", passed, evidence)


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
    frame = topk[
        (topk["kernel"].astype(str) == "exp")
        & (topk["score_type"].astype(str) == "model_weighted_spatial_ccc_score")
    ].copy()
    if "null_model" in frame.columns:
        matched = frame[frame["null_model"].astype(str) == "matched_random_lr"].copy()
        if not matched.empty:
            return matched
    return frame


def _global_rows(results_root: Path, figures_dir: Path) -> list[dict[str, object]]:
    required = [
        "dbfree_spatial_validation_main.png",
        "dbfree_spatial_validation_main.svg",
        "dbfree_spatial_validation_main.pdf",
        "dbfree_spatial_validation_main_legend.md",
        "dbfree_spatial_validation_main_source_tables.tar.gz",
        "dbfree_spatial_validation_source_tables.tar.gz",
    ]
    return [
        _gate("global", "figure", "main_figure_outputs_exist", all((figures_dir / name).exists() for name in required), str(figures_dir)),
        _figure_legend_gate(figures_dir),
        _source_tarball_gate(results_root, figures_dir),
        _gate("global", "reproducibility", "baseline_tables_exist", (results_root / "baseline_comparison.tsv").exists() and (results_root / "baseline_topk_enrichment.tsv").exists(), str(results_root)),
    ]


def _figure_legend_gate(figures_dir: Path) -> dict[str, object]:
    path = figures_dir / "dbfree_spatial_validation_main_legend.md"
    if not path.exists():
        return _gate("global", "figure", "main_figure_legend_complete", False, str(path))
    text = path.read_text(encoding="utf-8").lower()
    missing = [phrase for phrase in REQUIRED_LEGEND_PHRASES if phrase not in text]
    return _gate("global", "figure", "main_figure_legend_complete", not missing, "missing=" + ",".join(missing))


def _source_tarball_gate(results_root: Path, figures_dir: Path) -> dict[str, object]:
    path = figures_dir / "dbfree_spatial_validation_main_source_tables.tar.gz"
    if not path.exists():
        return _gate("global", "reproducibility", "source_tables_tarball_complete", False, str(path))
    try:
        with tarfile.open(path, "r:gz") as archive:
            names = set(archive.getnames())
    except tarfile.TarError as exc:
        return _gate("global", "reproducibility", "source_tables_tarball_complete", False, str(exc))
    required = {"baseline_comparison.tsv", "baseline_topk_enrichment.tsv"}
    for dataset_dir in _dataset_result_dirs(results_root):
        dataset = dataset_dir.name
        required.update(
            {
                f"{dataset}/spatial_validation_summary.tsv",
                f"{dataset}/spatial_validation_top_k_enrichment.tsv",
                f"{dataset}/spatial_validation_distance_decay.tsv",
                f"{dataset}/validation_model_card.tsv",
            }
        )
    missing = sorted(required - names)
    return _gate("global", "reproducibility", "source_tables_tarball_complete", not missing, "missing=" + ",".join(missing))


def _dataset_result_dirs(results_root: Path) -> list[Path]:
    return [path.parent for path in sorted(results_root.glob("*/spatial_validation_top_k_enrichment.tsv"))]


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

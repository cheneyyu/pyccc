from __future__ import annotations

import argparse
import json
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
        rows.append(_required_sections_gate(manifest, qc))
        rows.append(_gate(dataset, "data", "standard_columns_present", _qc_standard_columns_pass(qc), "pyccc_group, section_id, spatial, gene_id recorded by prepare script"))
    rows.append(_gene_gate(manifest, results_dir))
    rows.extend(_model_artifact_gates(manifest, results_dir))
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
    else:
        rows.append(_gate(dataset, "model", "validation_model_card_has_checksums", False, str(results_dir / "validation_model_card.tsv")))
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

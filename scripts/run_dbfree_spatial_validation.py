from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist

import pyccc as pc
from pyccc import spatial_validation as sv
from pyccc.database import normalize_lr_table
from pyccc.model_resources import resolve_dbfree_model_path
from dbfree_validation_utils import (
    checksum_short,
    load_manifest,
    manifest_results_dir,
    read_tsv,
    selected_sections,
    write_tsv,
)


BASELINES = ("dbfree", "role_only", "embedding_cosine", "expression_only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DB-free LR spatial validation from standardized real-data sections.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--predicted-lr", default=None)
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--baseline", action="append", choices=BASELINES, default=[])
    parser.add_argument("--null-model", action="append", default=[], help="Override manifest null models; may be repeated.")
    parser.add_argument("--top-k-only", action="store_true", help="Use a memory-light path that computes observed summaries and top-K null enrichment without writing per-pair null distributions.")
    parser.add_argument("--distance-decay-only", action="store_true", help="Only regenerate spatial_validation_distance_decay.tsv without overwriting null/enrichment tables.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    results_dir = manifest_results_dir(manifest, args.results_dir)
    sections = selected_sections(manifest, section_names=args.section, smoke_only=args.smoke_only)
    spatial_cfg = dict(manifest.get("spatial_validation", {}))
    n_permutations = int(args.n_permutations if args.n_permutations is not None else spatial_cfg.get("development_permutations", 100))
    if args.distance_decay_only:
        n_permutations = 0
    baselines = tuple(args.baseline or BASELINES)
    null_models = tuple(args.null_model or spatial_cfg.get("null_models", ("coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation")))
    top_k_values = tuple(int(k) for k in spatial_cfg.get("top_k", (100, 500, 1000, 5000)))
    compute_distance_decay = bool(spatial_cfg.get("compute_distance_decay", True))
    compute_section_reproducibility = bool(spatial_cfg.get("compute_section_reproducibility", True))
    write_celltype_pair_summary = bool(spatial_cfg.get("write_celltype_pair_summary", True))
    write_null_distribution = bool(spatial_cfg.get("write_null_distribution", True))
    distance_matrix_max_cells = spatial_cfg.get("distance_matrix_max_cells", 15000)
    distance_matrix_max_cells = None if distance_matrix_max_cells is None else int(distance_matrix_max_cells)
    distance_decay_max_cells = spatial_cfg.get("distance_decay_max_cells", 5000)
    distance_decay_max_cells = None if distance_decay_max_cells is None else int(distance_decay_max_cells)
    spatial_weight_max_cells = spatial_cfg.get("spatial_weight_max_cells", None)
    spatial_weight_max_cells = None if spatial_weight_max_cells is None else int(spatial_weight_max_cells)
    dbfree_score_cfg = dict(spatial_cfg.get("dbfree_score", {}))
    prepared = _prepared_paths(results_dir, sections)
    predicted_lr = _load_or_predict_lr(manifest, results_dir, args.predicted_lr, prepared)
    predicted_lr = _limit_lr_for_validation(predicted_lr, max(top_k_values) if top_k_values else 5000)
    _write_model_card_summary(manifest, results_dir)

    report_tables: dict[str, list[pd.DataFrame]] = {
        "summary": [],
        "celltype_pair_summary": [],
        "null_distribution": [],
        "top_k_enrichment": [],
        "role_kernel_enrichment": [],
        "distance_decay": [],
        "section_reproducibility": [],
    }
    for section in sections:
        section_id = str(section["name"])
        adata = sc.read_h5ad(prepared[section_id])
        for baseline in baselines:
            print(f"Running {manifest['name']} {section_id} {baseline}", flush=True)
            lr_table = _baseline_lr(predicted_lr, baseline=baseline, adata=adata, gene_id_key="gene_id", dbfree_score_cfg=dbfree_score_cfg)
            context = _context(manifest, section, adata, baseline=baseline, n_lr_pairs=len(lr_table), n_permutations=n_permutations, lr_table=lr_table)
            if args.distance_decay_only:
                decay = _fast_distance_decay(
                    adata,
                    lr_table,
                    groupby="pyccc_group",
                    spatial_key="spatial",
                    gene_id_key="gene_id",
                    max_cells=distance_decay_max_cells,
                    random_state=int(spatial_cfg.get("random_seed", 0)),
                )
                report_tables["distance_decay"].append(_with_context(decay, context))
                continue
            if args.top_k_only:
                fast_report = _fast_topk_validation(
                    adata,
                    lr_table,
                    groupby="pyccc_group",
                    spatial_key="spatial",
                    gene_id_key="gene_id",
                    section_key="section_id",
                    distance_kernels=tuple(spatial_cfg.get("distance_kernels", ("contact", "exp"))),
                    null_models=null_models,
                    top_k_values=top_k_values,
                    n_permutations=n_permutations,
                    random_state=int(spatial_cfg.get("random_seed", 0)),
                    distance_matrix_max_cells=distance_matrix_max_cells,
                    spatial_weight_max_cells=spatial_weight_max_cells,
                    distance_decay_max_cells=distance_decay_max_cells,
                    compute_distance_decay=compute_distance_decay,
                    compute_section_reproducibility=compute_section_reproducibility,
                )
                for key, frame in fast_report.items():
                    report_tables[key].append(_with_context(frame, context))
                continue
            report = pc.validate_spatial_lr_table(
                adata,
                lr_table=lr_table,
                groupby="pyccc_group",
                spatial_key="spatial",
                gene_symbols_key="gene_id",
                section_key="section_id",
                distance_kernels=tuple(spatial_cfg.get("distance_kernels", ("contact", "exp"))),
                null_models=() if args.distance_decay_only else null_models,
                n_permutations=n_permutations,
                random_state=int(spatial_cfg.get("random_seed", 0)),
                top_k=top_k_values,
                compute_distance_decay=True if args.distance_decay_only else compute_distance_decay,
                compute_section_reproducibility=False if args.distance_decay_only else compute_section_reproducibility,
                distance_matrix_max_cells=distance_matrix_max_cells,
                distance_decay_max_cells=distance_decay_max_cells,
            )
            report_tables["summary"].append(_with_context(report.summary, context))
            celltype_pairs = report.celltype_pair_summary if write_celltype_pair_summary else report.celltype_pair_summary.head(0)
            null_distribution = report.null_distribution if write_null_distribution else report.null_distribution.head(0)
            report_tables["celltype_pair_summary"].append(_with_context(celltype_pairs, context))
            report_tables["null_distribution"].append(_with_context(null_distribution, context))
            top_k_enrichment = report.top_k_enrichment if report.top_k_enrichment is not None else pd.DataFrame()
            role_kernel_enrichment = report.role_kernel_enrichment if report.role_kernel_enrichment is not None else pd.DataFrame()
            report_tables["top_k_enrichment"].append(_with_context(top_k_enrichment, context))
            report_tables["role_kernel_enrichment"].append(_with_context(role_kernel_enrichment, context))
            report_tables["distance_decay"].append(_with_context(report.distance_decay, context))
            report_tables["section_reproducibility"].append(_with_context(report.section_reproducibility, context))

    keys_to_write = ("distance_decay",) if args.distance_decay_only else tuple(report_tables)
    for key in keys_to_write:
        frames = report_tables[key]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        write_tsv(frame, results_dir / f"spatial_validation_{key}.tsv")
    if not args.distance_decay_only:
        _write_global_baseline_tables(results_dir.parent)


def _prepared_paths(results_dir: Path, sections: list[dict[str, object]]) -> dict[str, str]:
    paths = read_tsv(results_dir / "prepared_paths.tsv")
    mapping = {str(row.section_id): str(row.prepared_path) for row in paths.itertuples(index=False)}
    missing = [str(section["name"]) for section in sections if str(section["name"]) not in mapping]
    if missing:
        raise FileNotFoundError(f"Prepared h5ad paths are missing for sections: {missing}")
    return mapping


def _load_or_predict_lr(
    manifest: dict[str, object],
    results_dir: Path,
    predicted_lr_path: str | None,
    prepared: dict[str, str],
) -> pd.DataFrame:
    path = Path(predicted_lr_path) if predicted_lr_path else results_dir / "predicted_lr.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t")
    cfg = dict(manifest.get("prediction", {}))
    role_model = _resolve_optional_model_dir(cfg.get("role_model"), expected_file="role_model.joblib")
    pair_model = _resolve_optional_model_dir(cfg.get("pair_model"), expected_file="lr_link_model.joblib")
    protein_fasta = Path(str(manifest.get("protein_fasta", "")))
    missing = []
    if role_model is None or not (role_model / "role_model.joblib").exists():
        missing.append(str(cfg.get("role_model") or "prediction.role_model"))
    if pair_model is None or not (pair_model / "lr_link_model.joblib").exists():
        missing.append(str(cfg.get("pair_model") or "prediction.pair_model"))
    if not protein_fasta.exists():
        missing.append(str(manifest.get("protein_fasta") or "protein_fasta"))
    if missing:
        raise FileNotFoundError(
            "No predicted LR table was found and prediction inputs are missing: "
            + "; ".join(missing)
            + ". Provide --predicted-lr for validation-only runs."
        )
    prepared_adatas = [sc.read_h5ad(path) for path in prepared.values()]
    gene_lookup = _union_gene_lookup(prepared_adatas)
    union = ad.concat(prepared_adatas, join="outer", merge="same")
    union.var["gene_id"] = [gene_lookup.get(str(name), str(name)) for name in union.var_names.astype(str)]
    predicted = pc.predict_lr_dbfree(
        union,
        protein_fasta=str(manifest["protein_fasta"]),
        gene_id_key="gene_id",
        species_name=str(manifest["species"]).replace(" ", "_"),
        species_hint=str(manifest.get("clade", manifest.get("species_hint", "unknown"))),
        role_model=str(role_model),
        model=str(pair_model),
        density_prior=cfg.get("density_prior", "auto"),
        embedding_model_name=str(cfg.get("embedding_model", pc.ESMC_300M_MODEL_NAME)),
        embedding_model_revision=_prediction_embedding_revision(cfg),
        expression_min_fraction=float(cfg.get("expression_min_fraction", 0.01)),
        ligand_role_min=float(cfg.get("ligand_role_min", 0.2)),
        receptor_role_min=float(cfg.get("receptor_role_min", 0.2)),
        max_ligands=int(cfg.get("max_ligands", 2500)),
        max_receptors=int(cfg.get("max_receptors", 3500)),
        max_candidate_pairs=int(cfg.get("max_candidate_pairs", 2_000_000)),
        nearest_neighbor_pairs=int(cfg.get("nearest_neighbor_pairs", 500_000)),
        nearest_neighbors_per_ligand=int(cfg.get("nearest_neighbors_per_ligand", 50)),
        min_score=float(cfg.get("min_score", 0.5)),
        max_pairs=int(cfg.get("max_pairs", 50_000)),
        max_pairs_per_ligand=int(cfg.get("max_pairs_per_ligand", 200)),
        max_pairs_per_receptor=int(cfg.get("max_pairs_per_receptor", 200)),
        cache_dir="data/cache/pyccc_dbfree",
    )
    write_tsv(predicted.interactions, path)
    summary = predicted.metadata.get("prediction_summary")
    if isinstance(summary, pd.DataFrame):
        write_tsv(summary, results_dir / "prediction_summary.tsv")
    return predicted.interactions


def _limit_lr_for_validation(lr: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    if max_pairs <= 0 or len(lr) <= max_pairs:
        return lr.copy()
    out = lr.copy()
    if "density_rank" in out.columns:
        out["_density_rank_numeric"] = pd.to_numeric(out["density_rank"], errors="coerce")
        out = out.sort_values(["_density_rank_numeric", "model_score"], ascending=[True, False], na_position="last")
        out = out.drop(columns=["_density_rank_numeric"])
    elif "model_score" in out.columns:
        out = out.sort_values("model_score", ascending=False)
    return out.head(max_pairs).reset_index(drop=True)


def _union_gene_lookup(adatas: list) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for adata in adatas:
        genes = adata.var["gene_id"].astype(str) if "gene_id" in adata.var else pd.Series(adata.var_names.astype(str), index=adata.var_names)
        for var_name, gene_id in zip(adata.var_names.astype(str), genes, strict=True):
            lookup.setdefault(str(var_name), str(gene_id))
    return lookup


def _write_model_card_summary(manifest: dict[str, object], results_dir: Path) -> None:
    cfg = dict(manifest.get("prediction", {}))
    role_manifest = str(cfg.get("role_model", ""))
    pair_manifest = str(cfg.get("pair_model", ""))
    role_path = _resolve_optional_model_dir(role_manifest, expected_file="role_model.joblib")
    pair_path = _resolve_optional_model_dir(pair_manifest, expected_file="lr_link_model.joblib")
    role_card = _read_json(_model_file(role_path, "model_card.json"))
    pair_card = _read_json(_model_file(pair_path, "model_card.json"))
    rows = [
        {
            "dataset": manifest["name"],
            "embedding_model_name": cfg.get("embedding_model", pc.ESMC_300M_MODEL_NAME),
            "embedding_model_revision": _prediction_embedding_revision(cfg, role_card=role_card, pair_card=pair_card),
            "role_model_manifest": role_manifest,
            "role_model_path": _model_path_string(role_path),
            "role_model_checksum16": _checksum_model_file(role_path, "role_model.joblib"),
            "role_model_card_checksum16": _checksum_model_file(role_path, "model_card.json"),
            "role_model_stack": _card_string(role_card, "model_stack"),
            "role_model_classifier": _card_string(role_card, "classifier"),
            "role_embedding_model_name": _embedding_card_value(role_card, "model_name"),
            "role_embedding_model_revision": _embedding_card_value(role_card, "model_revision"),
            "role_embedding_backend": _embedding_card_value(role_card, "embedding_backend"),
            "pair_model_manifest": pair_manifest,
            "pair_model_path": _model_path_string(pair_path),
            "pair_model_checksum16": _checksum_model_file(pair_path, "lr_link_model.joblib"),
            "pair_model_card_checksum16": _checksum_model_file(pair_path, "model_card.json"),
            "pair_model_stack": _card_string(pair_card, "model_stack"),
            "pair_model_type": _card_string(pair_card, "model_type"),
            "pair_embedding_model_name": _embedding_card_value(pair_card, "model_name"),
            "pair_embedding_model_revision": _embedding_card_value(pair_card, "model_revision"),
            "pair_embedding_backend": _embedding_card_value(pair_card, "embedding_backend"),
            "density_prior_path": str(_model_file(pair_path, "density_prior.tsv")),
            "density_prior_checksum16": _checksum_model_file(pair_path, "density_prior.tsv"),
            "density_prior": cfg.get("density_prior", "auto"),
            "density_prior_groupby": _card_string(pair_card, "density_prior_groupby"),
            "negative_sampling_strategy": _card_string(pair_card, "negative_strategy"),
            "validation_split_summary": _validation_split_summary(pair_card),
            "training_resources": _join_card_list(pair_card, "training_resources"),
            "species_included": _join_card_list(pair_card, "species_included"),
            "clades_included": _join_card_list(pair_card, "clades_included"),
        }
    ]
    write_tsv(pd.DataFrame(rows), results_dir / "validation_model_card.tsv")


def _prediction_embedding_revision(
    cfg: dict[str, object],
    *,
    role_card: dict[str, object] | None = None,
    pair_card: dict[str, object] | None = None,
) -> str | None:
    for key in ("embedding_revision", "embedding_model_revision"):
        value = cfg.get(key)
        if value is not None and str(value).strip():
            return str(value)
    for card in (pair_card, role_card):
        if isinstance(card, dict):
            value = _embedding_card_value(card, "model_revision")
            if str(value).strip():
                return str(value)
    return None


def _resolve_optional_model_dir(value: object, *, expected_file: str) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return resolve_dbfree_model_path(str(value), expected_file=expected_file)


def _model_file(path: Path | None, file_name: str) -> Path:
    return (path / file_name) if path is not None else Path(file_name)


def _model_path_string(path: Path | None) -> str:
    return str(path) if path is not None else ""


def _checksum_model_file(path: Path | None, file_name: str) -> str:
    return checksum_short(path / file_name) if path is not None else ""


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _card_string(card: dict[str, object], key: str) -> str:
    return str(card.get(key, "")) if isinstance(card, dict) else ""


def _embedding_card_value(card: dict[str, object], key: str) -> str:
    embedding = card.get("embedding_model", {}) if isinstance(card, dict) else {}
    if not isinstance(embedding, dict):
        return ""
    return _join_values(embedding.get(key))


def _join_card_list(card: dict[str, object], key: str) -> str:
    return _join_values(card.get(key)) if isinstance(card, dict) else ""


def _join_values(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value if str(item))
    return str(value)


def _validation_split_summary(card: dict[str, object]) -> str:
    report = card.get("validation_report", {}) if isinstance(card, dict) else {}
    if not isinstance(report, dict):
        return ""
    parts = []
    for split, item in sorted(report.items()):
        if isinstance(item, dict):
            parts.append(f"{split}:{item.get('status', 'unknown')}")
    return ";".join(parts)


def _baseline_lr(
    lr: pd.DataFrame,
    *,
    baseline: str,
    adata,
    gene_id_key: str,
    dbfree_score_cfg: dict[str, object] | None = None,
) -> pd.DataFrame:
    out = lr.copy()
    if baseline == "dbfree":
        out = _apply_dbfree_validation_score(out, adata=adata, gene_id_key=gene_id_key, score_cfg=dbfree_score_cfg or {})
        out["validation_strategy"] = "dbfree"
        return out
    if baseline == "role_only":
        out["model_score"] = pd.to_numeric(out.get("ligand_role_score", 0.5), errors="coerce").fillna(0.5) * pd.to_numeric(out.get("receptor_role_score", 0.5), errors="coerce").fillna(0.5)
    elif baseline == "embedding_cosine":
        if "embedding_cosine" in out.columns:
            out["model_score"] = pd.to_numeric(out["embedding_cosine"], errors="coerce").fillna(0.0)
        elif "nearest_reference_distance" in out.columns:
            dist = pd.to_numeric(out["nearest_reference_distance"], errors="coerce").fillna(np.inf)
            out["model_score"] = 1.0 / (1.0 + dist)
        else:
            out["model_score"] = 0.0
    elif baseline == "expression_only":
        expression = _global_gene_expression(adata, gene_id_key=gene_id_key)
        out["model_score"] = [float(expression.get(str(row.ligand), 0.0) * expression.get(str(row.receptor), 0.0)) for row in out.itertuples(index=False)]
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    out["confidence"] = out["model_score"]
    out["validation_strategy"] = baseline
    out["validation_score_method"] = baseline
    return out.sort_values("model_score", ascending=False).reset_index(drop=True)


def _apply_dbfree_validation_score(
    lr: pd.DataFrame,
    *,
    adata,
    gene_id_key: str,
    score_cfg: dict[str, object],
) -> pd.DataFrame:
    out = lr.copy()
    method = str(score_cfg.get("method", "model"))
    out["base_model_score"] = pd.to_numeric(out.get("model_score", 1.0), errors="coerce").fillna(0.0)
    out["validation_score_method"] = method
    if method in {"", "model", "model_score"}:
        return out.sort_values("model_score", ascending=False).reset_index(drop=True)
    if method != "model_x_expression_potential":
        raise ValueError(f"Unsupported dbfree_score method: {method}")
    stat = str(score_cfg.get("potential_stat", "mean"))
    power = float(score_cfg.get("power", 0.5))
    potential = _nonspatial_expression_potential(out, adata=adata, gene_id_key=gene_id_key, stat=stat)
    ranks = pd.Series(potential).rank(method="average", pct=True).fillna(0.0).to_numpy(dtype=float)
    score = out["base_model_score"].to_numpy(dtype=float) * np.power(ranks, power)
    out["expression_potential_score"] = potential
    out["expression_potential_rank"] = ranks
    out["expression_potential_stat"] = stat
    out["expression_potential_power"] = power
    out["model_score"] = score
    out["confidence"] = score
    return out.sort_values("model_score", ascending=False).reset_index(drop=True)


def _nonspatial_expression_potential(lr: pd.DataFrame, *, adata, gene_id_key: str, stat: str) -> np.ndarray:
    genes = sorted(set(lr["ligand"].astype(str)).union(set(lr["receptor"].astype(str))))
    means = _group_expression_means_for_genes(adata, genes, gene_id_key=gene_id_key)
    ligands = lr["ligand"].astype(str).to_numpy()
    receptors = lr["receptor"].astype(str).to_numpy()
    ligand_expr = means.reindex(columns=ligands, fill_value=0.0).to_numpy(dtype=float)
    receptor_expr = means.reindex(columns=receptors, fill_value=0.0).to_numpy(dtype=float)
    values = (ligand_expr[:, None, :] * receptor_expr[None, :, :]).reshape(-1, len(lr))
    if stat == "mean":
        return values.mean(axis=0)
    if stat == "max":
        return values.max(axis=0)
    if stat == "q90":
        return np.quantile(values, 0.9, axis=0)
    raise ValueError(f"Unsupported expression potential statistic: {stat}")


def _fast_distance_decay(
    adata,
    lr: pd.DataFrame,
    *,
    groupby: str,
    spatial_key: str,
    gene_id_key: str,
    max_cells: int | None,
    random_state: int,
) -> pd.DataFrame:
    if spatial_key not in adata.obsm:
        raise KeyError(f"`{spatial_key}` is not present in adata.obsm.")
    if groupby not in adata.obs:
        raise KeyError(f"`{groupby}` is not present in adata.obs.")
    columns = ["ligand", "receptor", "distance_min", "distance_max", "mean_distance", "mean_spatial_ccc_score", "model_weighted_mean_spatial_ccc_score"]
    if lr.empty:
        return pd.DataFrame(columns=columns)

    coords = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
    groups = adata.obs[groupby].astype(str).to_numpy()
    if max_cells is not None and len(coords) > int(max_cells):
        rng = np.random.default_rng(random_state)
        keep = np.sort(rng.choice(len(coords), size=int(max_cells), replace=False))
        coords = coords[keep]
        groups = groups[keep]
    dist = cdist(coords, coords)
    bins = np.unique(np.quantile(dist[np.isfinite(dist)], np.linspace(0, 1, 6)))
    if len(bins) < 2:
        return pd.DataFrame(columns=columns)

    group_levels = np.asarray(sorted(set(groups)), dtype=object)
    codes = pd.Categorical(groups, categories=group_levels).codes
    eye = np.eye(len(group_levels), dtype=float)
    onehot = eye[codes]
    counts = np.bincount(codes, minlength=len(group_levels)).astype(float)
    denominators = np.maximum(np.outer(counts, counts), 1.0)

    genes = sorted(set(lr["ligand"].astype(str)).union(set(lr["receptor"].astype(str))))
    expr = _group_expression_means_for_genes(adata, genes, gene_id_key=gene_id_key).reindex(index=group_levels, fill_value=0.0)
    ligands = lr["ligand"].astype(str).to_numpy()
    receptors = lr["receptor"].astype(str).to_numpy()
    model_scores = pd.to_numeric(lr.get("model_score", pd.Series([1.0] * len(lr))), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    ligand_expr = expr.reindex(columns=ligands, fill_value=0.0).to_numpy(dtype=float)
    receptor_expr = expr.reindex(columns=receptors, fill_value=0.0).to_numpy(dtype=float)

    rows = []
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        in_bin = ((dist >= left) & (dist <= right)).astype(float)
        weights = (onehot.T @ (in_bin @ onehot)) / denominators
        scores = np.einsum("ij,ik,jk->k", weights, ligand_expr, receptor_expr, optimize=True) / max(weights.size, 1)
        rows.append(
            pd.DataFrame(
                {
                    "ligand": ligands,
                    "receptor": receptors,
                    "distance_min": float(left),
                    "distance_max": float(right),
                    "mean_distance": float((left + right) / 2),
                    "mean_spatial_ccc_score": scores.astype(float),
                    "model_weighted_mean_spatial_ccc_score": scores.astype(float) * model_scores,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _fast_topk_validation(
    adata,
    lr_table: pd.DataFrame,
    *,
    groupby: str,
    spatial_key: str,
    gene_id_key: str,
    section_key: str,
    distance_kernels: Sequence[str],
    null_models: Sequence[str],
    top_k_values: Sequence[int],
    n_permutations: int,
    random_state: int,
    distance_matrix_max_cells: int | None,
    spatial_weight_max_cells: int | None,
    distance_decay_max_cells: int | None,
    compute_distance_decay: bool,
    compute_section_reproducibility: bool,
) -> dict[str, pd.DataFrame]:
    del compute_section_reproducibility
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    groups = adata.obs[groupby].astype(str).to_numpy()
    sections = adata.obs[section_key].astype(str).to_numpy() if section_key in adata.obs else None
    weight_idx = _stratified_spatial_sample(groups, max_cells=spatial_weight_max_cells, random_state=random_state)
    weight_coords = coords[weight_idx]
    weight_groups = groups[weight_idx]
    weight_sections = sections[weight_idx] if sections is not None else None
    gene_names = sv._gene_names(adata, gene_symbols_key=gene_id_key)
    lr = normalize_lr_table(lr_table)
    lr = sv._filter_lr(lr, set(gene_names))
    if lr.empty:
        raise ValueError("No LR rows have ligand and receptor genes in the expression matrix.")
    radius = sv._resolve_radius(coords, adata, "auto")
    sigma = sv._resolve_sigma(coords, "auto", fallback=radius)
    genes = sorted(set(lr["ligand"].astype(str)).union(set(lr["receptor"].astype(str))))
    lr_expression = _expression_matrix_for_genes(adata, gene_names, genes)
    expr_means = _fast_group_expression_means_from_matrix(lr_expression, groups)
    gene_expression = sv._gene_expression_means(adata, gene_names)
    distance_matrix = sv._distance_matrix_cache(weight_coords, max_cells=distance_matrix_max_cells)
    kernel_matrices = _precomputed_kernel_matrices(weight_coords, radius=radius, sigma=sigma, kernels=distance_kernels, distance_matrix=distance_matrix)
    base_weights = _weight_tables_from_kernel_matrices(weight_groups, kernel_matrices)
    summary = _fast_score_lr_summary(lr, expr_means, base_weights)
    summary = _add_empty_null_stat_columns(summary)
    role_kernel = sv._role_kernel_enrichment(summary)
    top_k = _fast_top_k_enrichment(
        lr,
        adata,
        weight_coords,
        weight_groups,
        gene_names,
        expr_means,
        lr_expression,
        gene_expression,
        base_weights,
        summary,
        radius=radius,
        sigma=sigma,
        kernels=distance_kernels,
        null_models=null_models,
        top_k_values=top_k_values,
        n_permutations=n_permutations,
        random_state=random_state,
        sections=weight_sections,
        full_groups=groups,
        full_sections=sections,
        weight_idx=weight_idx,
        kernel_matrices=kernel_matrices,
    )
    distance_decay = (
        _fast_distance_decay(
            adata,
            lr,
            groupby=groupby,
            spatial_key=spatial_key,
            gene_id_key=gene_id_key,
            max_cells=distance_decay_max_cells,
            random_state=random_state,
        )
        if compute_distance_decay
        else pd.DataFrame(columns=["ligand", "receptor", "distance_min", "distance_max", "mean_distance", "mean_spatial_ccc_score", "model_weighted_mean_spatial_ccc_score"])
    )
    return {
        "summary": summary,
        "celltype_pair_summary": pd.DataFrame(),
        "null_distribution": pd.DataFrame(),
        "top_k_enrichment": top_k,
        "role_kernel_enrichment": role_kernel,
        "distance_decay": distance_decay,
        "section_reproducibility": pd.DataFrame(),
    }


def _fast_top_k_enrichment(
    lr: pd.DataFrame,
    adata,
    coords: np.ndarray,
    groups: np.ndarray,
    gene_names: pd.Index,
    expr_means: pd.DataFrame,
    lr_expression: tuple[list[str], object],
    gene_expression: pd.Series,
    base_weights: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    *,
    radius: float,
    sigma: float,
    kernels: Sequence[str],
    null_models: Sequence[str],
    top_k_values: Sequence[int],
    n_permutations: int,
    random_state: int,
    sections: np.ndarray | None,
    full_groups: np.ndarray,
    full_sections: np.ndarray | None,
    weight_idx: np.ndarray,
    kernel_matrices: dict[str, np.ndarray],
) -> pd.DataFrame:
    top_sets = _top_sets(summary, top_k_values=top_k_values)
    values: dict[tuple[str, str, str, int], list[float]] = {
        (kernel, score_col, null_model, k): []
        for kernel, score_col, k in top_sets
        for null_model in null_models
    }
    rng = np.random.default_rng(random_state)
    all_expr_means = None
    matched_pool = _matched_random_pool(lr, gene_expression) if "matched_random_lr" in set(map(str, null_models)) else None
    n = max(int(n_permutations), 0)
    for _ in range(n):
        for null_model in null_models:
            if null_model == "coordinate_permutation":
                perm = rng.permutation(len(coords))
                inverse = np.empty_like(perm)
                inverse[perm] = np.arange(len(perm))
                coord_groups = groups[inverse]
                weights = _weight_tables_from_kernel_matrices(coord_groups, kernel_matrices)
                null_summary = _fast_score_lr_summary(lr, expr_means, weights)
            elif null_model == "celltype_permutation":
                perm_groups = sv._permute_groups_for_celltype_null(full_groups, full_sections, rng)
                perm_means = _fast_group_expression_means_from_matrix(lr_expression, perm_groups)
                weights = _weight_tables_from_kernel_matrices(perm_groups[weight_idx], kernel_matrices)
                null_summary = _fast_score_lr_summary(lr, perm_means, weights)
            elif null_model == "matched_random_lr":
                random_lr = _sample_matched_random_lr(matched_pool, rng) if matched_pool is not None else sv._matched_random_lr(lr, gene_expression, rng)
                if all_expr_means is None:
                    all_expr_means = _fast_group_expression_means_from_matrix((list(pd.Index(gene_names).astype(str)), adata.X), full_groups)
                null_summary = _fast_score_lr_summary(random_lr, all_expr_means, base_weights)
            elif null_model == "score_permutation":
                permuted = lr.copy()
                if "model_score" in permuted.columns:
                    permuted["model_score"] = rng.permutation(permuted["model_score"].to_numpy())
                null_summary = _fast_score_lr_summary(permuted, expr_means, base_weights)
            else:
                raise ValueError(f"Unsupported null model: {null_model}")
            _append_top_null_values(values, top_sets, null_summary, null_model)
    return _top_k_rows(summary, top_sets, values, null_models)


def _matched_random_pool(lr: pd.DataFrame, gene_expression: pd.Series, *, pool_size: int = 10) -> dict[str, object]:
    table = sv._gene_match_table(lr, gene_expression).reset_index(drop=True)
    genes = table["gene"].astype(str).to_numpy() if "gene" in table else np.asarray([], dtype=str)
    if len(genes) == 0:
        return {"lr": lr.reset_index(drop=True), "ligand": [], "receptor": []}
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    expr_q = _numeric_pool_array(table, "expression_quantile", default=0.0)
    ligand_role = _numeric_pool_array(table, "ligand_role_score", default=0.5)
    receptor_role = _numeric_pool_array(table, "receptor_role_score", default=0.5)
    ligand_degree = _numeric_pool_array(table, "ligand_degree", default=0.0)
    receptor_degree = _numeric_pool_array(table, "receptor_degree", default=0.0)
    median_expr_q = float(np.nanmedian(expr_q)) if len(expr_q) else 0.0
    pools = {"ligand": [], "receptor": []}
    for row in lr.itertuples(index=False):
        for side in ("ligand", "receptor"):
            role = ligand_role if side == "ligand" else receptor_role
            degree = ligand_degree if side == "ligand" else receptor_degree
            gene = str(getattr(row, side))
            idx = gene_to_idx.get(gene)
            target_expr = float(expr_q[idx]) if idx is not None else median_expr_q
            target_role = float(getattr(row, f"{side}_role_score", role[idx] if idx is not None else 0.5))
            target_degree = float(degree[idx]) if idx is not None else 0.0
            distance = np.abs(expr_q - target_expr) + np.abs(role - target_role) + np.abs(np.log1p(degree) - np.log1p(target_degree))
            if idx is not None and len(distance) > 1:
                distance = distance.copy()
                distance[idx] = np.inf
            finite = np.flatnonzero(np.isfinite(distance))
            if len(finite) == 0:
                pool = np.asarray([gene], dtype=str)
            else:
                take = min(int(pool_size), len(finite))
                nearest = finite if take == len(finite) else finite[np.argpartition(distance[finite], take - 1)[:take]]
                pool = genes[nearest]
            pools[side].append(pool)
    return {"lr": lr.reset_index(drop=True), **pools}


def _numeric_pool_array(table: pd.DataFrame, col: str, *, default: float) -> np.ndarray:
    if col not in table:
        return np.full(len(table), float(default), dtype=float)
    return pd.to_numeric(table[col], errors="coerce").fillna(default).to_numpy(dtype=float)


def _sample_matched_random_lr(pool: dict[str, object], rng: np.random.Generator) -> pd.DataFrame:
    lr = pool["lr"].copy()
    ligand_pools = pool.get("ligand", [])
    receptor_pools = pool.get("receptor", [])
    ligands = []
    receptors = []
    for ligand_pool, receptor_pool in zip(ligand_pools, receptor_pools, strict=True):
        ligands.append(str(ligand_pool[int(rng.integers(0, len(ligand_pool)))]))
        receptors.append(str(receptor_pool[int(rng.integers(0, len(receptor_pool)))]))
    if len(ligands) != len(lr):
        ligands = lr["ligand"].astype(str).tolist()
        receptors = lr["receptor"].astype(str).tolist()
    out = lr.copy()
    out["original_ligand"] = lr["ligand"].astype(str).to_numpy()
    out["original_receptor"] = lr["receptor"].astype(str).to_numpy()
    out["ligand"] = ligands
    out["receptor"] = receptors
    return out


def _precomputed_kernel_matrices(
    coords: np.ndarray,
    *,
    radius: float,
    sigma: float,
    kernels: Sequence[str],
    distance_matrix: np.ndarray | None,
) -> dict[str, np.ndarray]:
    dist = distance_matrix if distance_matrix is not None else cdist(coords[:, :2], coords[:, :2])
    out = {}
    for kernel in kernels:
        kernel_name = str(kernel)
        if kernel_name == "contact":
            out[kernel_name] = (dist <= radius).astype(float)
        elif kernel_name == "exp":
            out[kernel_name] = np.exp(-dist / sigma)
        else:
            raise ValueError("Distance kernels must be `contact` or `exp`.")
    return out


def _weight_tables_from_kernel_matrices(groups: Sequence[str], kernel_matrices: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    group_values = np.asarray(groups).astype(str)
    levels = np.asarray(sorted(set(group_values)), dtype=object)
    if len(levels) == 0:
        return {kernel: pd.DataFrame(columns=["source", "target", "kernel", "spatial_weight"]) for kernel in kernel_matrices}
    codes = pd.Categorical(group_values, categories=levels).codes
    eye = np.eye(len(levels), dtype=float)
    onehot = eye[codes]
    counts = np.bincount(codes, minlength=len(levels)).astype(float)
    denominators = np.maximum(np.outer(counts, counts), 1.0)
    tables = {}
    for kernel, matrix in kernel_matrices.items():
        weights = (onehot.T @ (matrix @ onehot)) / denominators
        rows = [
            {"source": str(source), "target": str(target), "kernel": str(kernel), "spatial_weight": float(weights[i, j])}
            for i, source in enumerate(levels)
            for j, target in enumerate(levels)
        ]
        tables[str(kernel)] = pd.DataFrame(rows)
    return tables


def _stratified_spatial_sample(groups: np.ndarray, *, max_cells: int | None, random_state: int) -> np.ndarray:
    n = len(groups)
    if max_cells is None or n <= int(max_cells):
        return np.arange(n)
    max_cells = max(int(max_cells), 1)
    rng = np.random.default_rng(random_state)
    groups = np.asarray(groups)
    levels = sorted(set(groups.astype(str)))
    selected = []
    remaining = max_cells
    for i, level in enumerate(levels):
        idx = np.flatnonzero(groups.astype(str) == level)
        if len(idx) == 0:
            continue
        levels_left = len(levels) - i
        quota = max(1, int(round(max_cells * len(idx) / n)))
        quota = min(quota, len(idx), max(1, remaining - (levels_left - 1)))
        selected.extend(rng.choice(idx, size=quota, replace=False).tolist())
        remaining -= quota
        if remaining <= 0:
            break
    if len(selected) < max_cells:
        missing = np.setdiff1d(np.arange(n), np.asarray(selected, dtype=int), assume_unique=False)
        fill = min(max_cells - len(selected), len(missing))
        if fill > 0:
            selected.extend(rng.choice(missing, size=fill, replace=False).tolist())
    return np.asarray(sorted(selected[:max_cells]), dtype=int)


def _fast_score_lr_summary(lr: pd.DataFrame, expr_means: pd.DataFrame, weight_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    group_levels = pd.Index(expr_means.index.astype(str))
    ligands = lr["ligand"].astype(str).to_numpy()
    receptors = lr["receptor"].astype(str).to_numpy()
    out_ligands = lr["original_ligand"].astype(str).to_numpy() if "original_ligand" in lr.columns else ligands
    out_receptors = lr["original_receptor"].astype(str).to_numpy() if "original_receptor" in lr.columns else receptors
    model_scores = pd.to_numeric(lr.get("model_score", pd.Series([1.0] * len(lr))), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    ligand_expr = expr_means.reindex(index=group_levels, columns=ligands, fill_value=0.0).to_numpy(dtype=float)
    receptor_expr = expr_means.reindex(index=group_levels, columns=receptors, fill_value=0.0).to_numpy(dtype=float)
    for kernel, weights in weight_tables.items():
        matrix = (
            weights.pivot(index="source", columns="target", values="spatial_weight")
            .reindex(index=group_levels, columns=group_levels, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        spatial = np.einsum("ij,ik,jk->k", matrix, ligand_expr, receptor_expr, optimize=True) / max(matrix.size, 1)
        frame = pd.DataFrame(
            {
                "ligand": out_ligands,
                "receptor": out_receptors,
                "kernel": str(kernel),
                "spatial_ccc_score": spatial.astype(float),
                "max_spatial_ccc_score": spatial.astype(float),
                "model_weighted_spatial_ccc_score": spatial.astype(float) * model_scores,
            }
        )
        for col in (
            "model_score",
            "confidence",
            "density_rank",
            "ligand_secreted_like_score",
            "ligand_membrane_like_score",
            "receptor_secreted_like_score",
            "receptor_membrane_like_score",
        ):
            if col in lr.columns:
                frame[col] = lr[col].to_numpy()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _expression_matrix_for_genes(adata, gene_names: Sequence[str], genes: Sequence[str]) -> tuple[list[str], object]:
    gene_index = pd.Index(gene_names).astype(str)
    lookup = {gene: i for i, gene in enumerate(gene_index)}
    selected = [str(gene) for gene in genes if str(gene) in lookup]
    cols = [lookup[gene] for gene in selected]
    return selected, adata.X[:, cols]


def _fast_group_expression_means(adata, groups: Sequence[str], gene_names: Sequence[str], genes: Sequence[str]) -> pd.DataFrame:
    return _fast_group_expression_means_from_matrix(_expression_matrix_for_genes(adata, gene_names, genes), groups)


def _fast_group_expression_means_from_matrix(expression: tuple[list[str], object], groups: Sequence[str]) -> pd.DataFrame:
    selected, x = expression
    group_values = np.asarray(groups).astype(str)
    levels = np.asarray(sorted(set(group_values)), dtype=object)
    if len(selected) == 0 or len(levels) == 0:
        return pd.DataFrame(index=levels, columns=selected, dtype=float).fillna(0.0)
    codes = pd.Categorical(group_values, categories=levels).codes
    valid = codes >= 0
    indicator = sparse.csr_matrix(
        (np.ones(int(valid.sum()), dtype=float), (codes[valid], np.flatnonzero(valid))),
        shape=(len(levels), len(group_values)),
    )
    sums = indicator @ x
    sums = sums.toarray() if sparse.issparse(sums) else np.asarray(sums)
    counts = np.bincount(codes[valid], minlength=len(levels)).astype(float)
    means = sums / np.maximum(counts[:, None], 1.0)
    return pd.DataFrame(means, index=levels.astype(str), columns=selected).fillna(0.0)


def _add_empty_null_stat_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["spatial_enrichment_z"] = np.nan
    out["empirical_pvalue"] = np.nan
    out["model_weighted_spatial_enrichment_z"] = np.nan
    out["model_weighted_empirical_pvalue"] = np.nan
    return out


def _top_sets(summary: pd.DataFrame, *, top_k_values: Sequence[int]) -> dict[tuple[str, str, int], dict[str, object]]:
    out = {}
    for kernel, sub in summary.groupby("kernel", sort=False):
        sub = sub.reset_index(drop=True)
        pair_keys = list(zip(sub["ligand"].astype(str), sub["receptor"].astype(str), strict=True))
        for score_col in ("spatial_ccc_score", "model_weighted_spatial_ccc_score"):
            ranked = sv._rank_top_predicted_pairs(sub, score_col=score_col)
            for k in top_k_values:
                kk = min(max(int(k), 1), len(ranked))
                top = ranked.head(kk)
                keys = top[["ligand", "receptor"]].drop_duplicates()
                key_set = set(zip(keys["ligand"].astype(str), keys["receptor"].astype(str), strict=True))
                positions = np.asarray([idx for idx, key in enumerate(pair_keys) if key in key_set], dtype=int)
                out[(str(kernel), score_col, int(k))] = {
                    "keys": keys,
                    "positions": positions,
                    "n_pairs": int(kk),
                    "observed": float(top[score_col].astype(float).mean()) if not top.empty else np.nan,
                }
    return out


def _append_top_null_values(
    values: dict[tuple[str, str, str, int], list[float]],
    top_sets: dict[tuple[str, str, int], dict[str, object]],
    null_summary: pd.DataFrame,
    null_model: str,
) -> None:
    for (kernel, score_col, k), top in top_sets.items():
        positions = np.asarray(top["positions"], dtype=int)
        if len(positions) == 0:
            values[(kernel, score_col, null_model, k)].append(np.nan)
            continue
        sub = null_summary[null_summary["kernel"].astype(str) == kernel]
        scores = pd.to_numeric(sub[score_col], errors="coerce").to_numpy(dtype=float)
        if len(scores) <= int(positions.max(initial=-1)):
            values[(kernel, score_col, null_model, k)].append(np.nan)
            continue
        values[(kernel, score_col, null_model, k)].append(float(np.nanmean(scores[positions])))


def _top_k_rows(
    summary: pd.DataFrame,
    top_sets: dict[tuple[str, str, int], dict[str, object]],
    values: dict[tuple[str, str, str, int], list[float]],
    null_models: Sequence[str],
) -> pd.DataFrame:
    rows = []
    del summary
    for (kernel, score_col, k), top in top_sets.items():
        observed = float(top["observed"]) if pd.notna(top["observed"]) else np.nan
        for null_model in ("pooled", *[str(model) for model in null_models]):
            if null_model == "pooled":
                null_values = np.asarray(
                    [item for model in null_models for item in values.get((kernel, score_col, str(model), k), [])],
                    dtype=float,
                )
            else:
                null_values = np.asarray(values.get((kernel, score_col, null_model, k), []), dtype=float)
            null_values = null_values[np.isfinite(null_values)]
            null_mean = float(null_values.mean()) if len(null_values) else np.nan
            null_sd = float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0
            z = 0.0 if len(null_values) and null_sd == 0 else ((observed - null_mean) / null_sd if len(null_values) else np.nan)
            pvalue = float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1)) if len(null_values) else np.nan
            rows.append(
                {
                    "kernel": kernel,
                    "score_type": score_col,
                    "null_model": null_model,
                    "k": int(k),
                    "n_pairs": int(top["n_pairs"]),
                    "observed_mean": observed,
                    "null_mean": null_mean,
                    "null_sd": null_sd,
                    "top_k_enrichment_z": float(z) if pd.notna(z) else np.nan,
                    "top_k_empirical_pvalue": pvalue,
                }
            )
    return pd.DataFrame(rows)


def _group_expression_means_for_genes(adata, genes: Sequence[str], *, gene_id_key: str) -> pd.DataFrame:
    gene_names = adata.var[gene_id_key].astype(str) if gene_id_key in adata.var else pd.Index(adata.var_names.astype(str))
    groups = adata.obs["pyccc_group"].astype(str).to_numpy()
    return _fast_group_expression_means(adata, groups, gene_names, genes)


def _global_gene_expression(adata, *, gene_id_key: str) -> pd.Series:
    genes = adata.var[gene_id_key].astype(str) if gene_id_key in adata.var else pd.Index(adata.var_names.astype(str))
    values = np.asarray(adata.X.mean(axis=0)).ravel() if sparse.issparse(adata.X) else np.asarray(adata.X).mean(axis=0)
    return pd.Series(values.astype(float), index=genes)


def _context(
    manifest: dict[str, object],
    section: dict[str, object],
    adata,
    *,
    baseline: str,
    n_lr_pairs: int,
    n_permutations: int,
    lr_table: pd.DataFrame,
) -> dict[str, object]:
    return {
        "dataset": manifest["name"],
        "species": manifest.get("species", ""),
        "section_id": section["name"],
        "technology": manifest.get("technology", ""),
        "validation_strategy": baseline,
        "n_cells_or_bins": int(adata.n_obs),
        "n_groups": int(pd.Series(adata.obs["pyccc_group"].astype(str)).nunique()),
        "n_lr_pairs_in_table": int(n_lr_pairs),
        "random_seed": int(dict(manifest.get("spatial_validation", {})).get("random_seed", 0)),
        "n_permutations": int(n_permutations),
        "validation_score_method": _first_table_value(lr_table, "validation_score_method", baseline),
        "expression_potential_stat": _first_table_value(lr_table, "expression_potential_stat", ""),
        "expression_potential_power": _first_table_value(lr_table, "expression_potential_power", ""),
    }


def _first_table_value(frame: pd.DataFrame, col: str, default: object) -> object:
    if col not in frame or frame.empty:
        return default
    value = frame[col].iloc[0]
    return "" if pd.isna(value) else value


def _with_context(frame: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    out = frame.copy()
    for key, value in reversed(list(context.items())):
        out.insert(0, key, value)
    return out


def _baseline_comparison(topk: pd.DataFrame) -> pd.DataFrame:
    if topk.empty:
        return pd.DataFrame()
    key_cols = ["dataset", "species", "section_id", "kernel", "score_type", "k"]
    if "null_model" in topk.columns:
        key_cols.append("null_model")
    summary = topk.groupby([*key_cols, "validation_strategy"], as_index=False).agg(
        observed_mean=("observed_mean", "mean"),
        null_mean=("null_mean", "mean"),
        null_sd=("null_sd", "mean"),
        enrichment_z=("top_k_enrichment_z", "mean"),
        empirical_p=("top_k_empirical_pvalue", "mean"),
        n_permutations=("n_permutations", "max"),
    )
    dbfree = summary[summary["validation_strategy"] == "dbfree"][key_cols + ["enrichment_z"]].rename(columns={"enrichment_z": "dbfree_enrichment_z"})
    out = summary.merge(dbfree, on=key_cols, how="left")
    out["delta_z_vs_dbfree"] = out["enrichment_z"] - out["dbfree_enrichment_z"]
    return out


def _write_global_baseline_tables(results_root: Path) -> None:
    frames = []
    for path in sorted(results_root.glob("*/spatial_validation_top_k_enrichment.tsv")):
        try:
            frame = pd.read_csv(path, sep="\t")
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    topk = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_tsv(topk, results_root / "baseline_topk_enrichment.tsv")
    write_tsv(_baseline_comparison(topk), results_root / "baseline_comparison.tsv")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from .database import CellChatDB, normalize_lr_table


@dataclass(frozen=True)
class SpatialValidationReport:
    """Spatial enrichment summaries for a candidate LR table."""

    summary: pd.DataFrame
    celltype_pair_summary: pd.DataFrame
    null_distribution: pd.DataFrame
    distance_decay: pd.DataFrame
    section_reproducibility: pd.DataFrame
    metadata: dict[str, object]
    top_k_enrichment: pd.DataFrame | None = None
    role_kernel_enrichment: pd.DataFrame | None = None
    curated_overlap_enrichment: pd.DataFrame | None = None


def validate_spatial_lr_table(
    adata,
    lr_table: CellChatDB | pd.DataFrame,
    *,
    groupby: str,
    spatial_key: str = "spatial",
    mode: str = "cellbin",
    radius: str | float = "auto",
    sigma: str | float = "auto",
    distance_kernels: Sequence[str] = ("contact", "exp"),
    n_permutations: int = 1000,
    null_models: Sequence[str] = ("coordinate_permutation", "celltype_permutation", "matched_random_lr", "score_permutation"),
    random_state: int | None = 0,
    gene_symbols_key: str | None = None,
    section_key: str | None = None,
    section_top_k: int = 100,
    top_k: Sequence[int] = (100, 500, 1000),
    curated_lr_table: CellChatDB | pd.DataFrame | None = None,
    compute_distance_decay: bool = True,
    compute_section_reproducibility: bool = True,
    distance_matrix_max_cells: int | None = 15000,
) -> SpatialValidationReport:
    """Validate candidate LR scores against simple spatial null models."""

    if spatial_key not in adata.obsm:
        raise KeyError(f"`{spatial_key}` is not present in adata.obsm.")
    if groupby not in adata.obs:
        raise KeyError(f"`{groupby}` is not present in adata.obs.")
    if section_key is not None and section_key not in adata.obs:
        raise KeyError(f"`{section_key}` is not present in adata.obs.")
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must be a two-dimensional array with at least two columns.")
    lr = lr_table.interactions if isinstance(lr_table, CellChatDB) else normalize_lr_table(lr_table)
    gene_names = _gene_names(adata, gene_symbols_key=gene_symbols_key)
    lr = _filter_lr(lr, set(gene_names))
    if lr.empty:
        raise ValueError("No LR rows have ligand and receptor genes in the expression matrix.")
    groups = adata.obs[groupby].astype(str).to_numpy()
    sections = adata.obs[section_key].astype(str).to_numpy() if section_key is not None else None
    radius_value = _resolve_radius(coords, adata, radius)
    sigma_value = _resolve_sigma(coords, sigma, fallback=radius_value)
    expr_means = _group_expression_means(adata, groups, gene_names, sorted(set(lr["ligand"]).union(set(lr["receptor"]))))
    gene_expression = _gene_expression_means(adata, gene_names)
    distance_matrix = _distance_matrix_cache(coords, max_cells=distance_matrix_max_cells)
    weight_tables = _spatial_weight_tables(coords, groups, radius=radius_value, sigma=sigma_value, kernels=distance_kernels, distance_matrix=distance_matrix)
    observed = _score_lr_spatial(lr, expr_means, weight_tables)
    summary = _lr_summary(observed, lr)
    summary = _annotate_curated_overlap(summary, curated_lr_table, set(gene_names))
    null = _null_distribution(
        lr,
        adata,
        coords,
        groups,
        gene_names,
        expr_means,
        gene_expression,
        radius_value,
        sigma_value,
        distance_kernels,
        null_models,
        n_permutations,
        random_state,
        sections=sections,
        distance_matrix=distance_matrix,
    )
    summary = _attach_null_stats(summary, null)
    top_k_enrichment = _top_k_enrichment(summary, null, top_k_values=top_k)
    role_kernel_enrichment = _role_kernel_enrichment(summary)
    curated_overlap_enrichment = _curated_overlap_enrichment(summary, curated_lr_table)
    distance_decay = (
        _distance_decay(coords, groups, lr, expr_means)
        if compute_distance_decay
        else _empty_distance_decay()
    )
    section_reproducibility = (
        _section_reproducibility(
            adata,
            coords,
            groups,
            lr,
            gene_names,
            radius=radius_value,
            sigma=sigma_value,
            kernels=distance_kernels,
            section_key=section_key,
            section_top_k=section_top_k,
        )
        if compute_section_reproducibility
        else _empty_section_reproducibility()
    )
    metadata = {
        "mode": mode,
        "radius": radius_value,
        "sigma": sigma_value,
        "null_models": list(null_models),
        "n_permutations": n_permutations,
        "section_key": section_key,
        "section_top_k": int(section_top_k),
        "celltype_permutation_scope": "section" if section_key is not None else "global",
        "top_k": [int(k) for k in top_k],
        "curated_lr_table": curated_lr_table is not None,
        "compute_distance_decay": bool(compute_distance_decay),
        "compute_section_reproducibility": bool(compute_section_reproducibility),
        "distance_matrix_cached": distance_matrix is not None,
        "distance_matrix_max_cells": distance_matrix_max_cells,
    }
    return SpatialValidationReport(
        summary,
        observed,
        null,
        distance_decay,
        section_reproducibility,
        metadata,
        top_k_enrichment=top_k_enrichment,
        role_kernel_enrichment=role_kernel_enrichment,
        curated_overlap_enrichment=curated_overlap_enrichment,
    )


def _gene_names(adata, *, gene_symbols_key: str | None) -> pd.Index:
    if gene_symbols_key is None:
        return pd.Index(adata.var_names.astype(str))
    if gene_symbols_key not in adata.var:
        raise KeyError(f"`{gene_symbols_key}` is not present in adata.var.")
    return pd.Index(adata.var[gene_symbols_key].astype(str))


def _filter_lr(lr: pd.DataFrame, genes: set[str]) -> pd.DataFrame:
    mask = lr["ligand"].astype(str).isin(genes) & lr["receptor"].astype(str).isin(genes)
    return lr[mask].copy()


def _resolve_radius(coords: np.ndarray, adata, radius: str | float) -> float:
    if radius != "auto":
        return float(radius)
    for key in ("cell_area", "area"):
        if key in adata.obs:
            area = pd.to_numeric(adata.obs[key], errors="coerce").dropna()
            if not area.empty:
                return float(np.sqrt(area.median() / np.pi) * 2.0)
    if len(coords) < 2:
        return 0.0
    nearest, _ = cKDTree(coords[:, :2]).query(coords[:, :2], k=2)
    return float(np.median(nearest[:, 1]))


def _resolve_sigma(coords: np.ndarray, sigma: str | float, *, fallback: float) -> float:
    if sigma != "auto":
        return float(sigma)
    return float(max(fallback, 1e-9))


def _group_expression_means(
    adata,
    groups: np.ndarray,
    gene_names: pd.Index,
    genes: Sequence[str],
    *,
    cell_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    gene_lookup = {gene: i for i, gene in enumerate(gene_names)}
    cols = [gene_lookup[gene] for gene in genes if gene in gene_lookup]
    selected_genes = [gene for gene in genes if gene in gene_lookup]
    x = adata.X[:, cols] if cell_mask is None else adata.X[np.asarray(cell_mask, dtype=bool)][:, cols]
    rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        sub = x[mask]
        values = np.asarray(sub.mean(axis=0)).ravel() if sparse.issparse(sub) else np.asarray(sub).mean(axis=0)
        rows.append(pd.Series(values, index=selected_genes, name=group))
    return pd.DataFrame(rows).fillna(0.0)


def _gene_expression_means(adata, gene_names: pd.Index) -> pd.Series:
    x = adata.X
    values = np.asarray(x.mean(axis=0)).ravel() if sparse.issparse(x) else np.asarray(x).mean(axis=0)
    return pd.Series(values.astype(float), index=gene_names.astype(str))


def _distance_matrix_cache(coords: np.ndarray, *, max_cells: int | None) -> np.ndarray | None:
    if max_cells is not None and len(coords) > int(max_cells):
        return None
    xy = coords[:, :2]
    dist = np.empty((len(xy), len(xy)), dtype=np.float32)
    block_size = 2048
    for start in range(0, len(xy), block_size):
        stop = min(start + block_size, len(xy))
        dist[start:stop] = cdist(xy[start:stop], xy).astype(np.float32, copy=False)
    return dist


def _spatial_weight_tables(
    coords: np.ndarray,
    groups: np.ndarray,
    *,
    radius: float,
    sigma: float,
    kernels: Sequence[str],
    distance_matrix: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    group_levels = np.asarray(sorted(set(groups)), dtype=object)
    if len(group_levels) == 0:
        return {str(kernel): pd.DataFrame(columns=["source", "target", "kernel", "spatial_weight"]) for kernel in kernels}
    codes = pd.Categorical(groups, categories=group_levels).codes
    if (codes < 0).any():
        raise ValueError("Groups contain values outside the resolved category levels.")
    n_groups = len(group_levels)
    group_counts = np.bincount(codes, minlength=n_groups).astype(float)
    denominators = np.maximum(np.outer(group_counts, group_counts), 1.0)
    eye = np.eye(n_groups, dtype=float)
    target_onehot = eye[codes]
    sums = {str(kernel): np.zeros((n_groups, n_groups), dtype=float) for kernel in kernels}
    block_size = 2048
    xy = coords[:, :2]
    for start in range(0, len(xy), block_size):
        stop = min(start + block_size, len(xy))
        dist = distance_matrix[start:stop] if distance_matrix is not None else cdist(xy[start:stop], xy)
        source_onehot = eye[codes[start:stop]]
        for kernel in kernels:
            kernel_name = str(kernel)
            if kernel_name == "contact":
                weights = (dist <= radius).astype(float)
            elif kernel_name == "exp":
                weights = np.exp(-dist / sigma)
            else:
                raise ValueError("Distance kernels must be `contact` or `exp`.")
            sums[kernel_name] += source_onehot.T @ (weights @ target_onehot)
    tables = {}
    for kernel_name, weight_sum in sums.items():
        weights = weight_sum / denominators
        rows = [
            {"source": str(source), "target": str(target), "kernel": kernel_name, "spatial_weight": float(weights[i, j])}
            for i, source in enumerate(group_levels)
            for j, target in enumerate(group_levels)
        ]
        tables[kernel_name] = pd.DataFrame(rows)
    return tables


def _score_lr_spatial(lr: pd.DataFrame, expr_means: pd.DataFrame, weight_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metadata_cols = [
        "original_ligand",
        "original_receptor",
        "matched_ligand",
        "matched_receptor",
        "ligand_match_expression_delta",
        "receptor_match_expression_delta",
        "ligand_match_role_delta",
        "receptor_match_role_delta",
        "ligand_match_degree_delta",
        "receptor_match_degree_delta",
    ]
    frames = []
    ligands = lr["ligand"].astype(str).to_numpy()
    receptors = lr["receptor"].astype(str).to_numpy()
    model_scores = pd.to_numeric(lr.get("model_score", pd.Series([1.0] * len(lr))), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    n_lr = len(lr)
    for kernel, weights in weight_tables.items():
        if weights.empty or n_lr == 0:
            continue
        weights = weights.reset_index(drop=True)
        sources = weights["source"].astype(str).to_numpy()
        targets = weights["target"].astype(str).to_numpy()
        spatial_weights = pd.to_numeric(weights["spatial_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        source_expr = expr_means.reindex(index=sources, columns=ligands, fill_value=0.0).to_numpy(dtype=float)
        target_expr = expr_means.reindex(index=targets, columns=receptors, fill_value=0.0).to_numpy(dtype=float)
        expression_score = source_expr * target_expr
        spatial_score = expression_score * spatial_weights[:, None]
        n_weights = len(weights)
        frame = pd.DataFrame(
            {
                "ligand": np.tile(ligands, n_weights),
                "receptor": np.tile(receptors, n_weights),
                "source": np.repeat(sources, n_lr),
                "target": np.repeat(targets, n_lr),
                "kernel": str(kernel),
                "expression_score": expression_score.reshape(-1),
                "spatial_weight": np.repeat(spatial_weights, n_lr),
                "spatial_ccc_score": spatial_score.reshape(-1),
                "model_score": np.tile(model_scores, n_weights),
            }
        )
        frame["model_weighted_spatial_ccc_score"] = frame["spatial_ccc_score"].to_numpy(dtype=float) * frame["model_score"].to_numpy(dtype=float)
        for col in metadata_cols:
            if col in lr.columns:
                frame[col] = np.tile(lr[col].to_numpy(), n_weights)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _lr_summary(observed: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    summary = observed.groupby(["ligand", "receptor", "kernel"], as_index=False).agg(
        spatial_ccc_score=("spatial_ccc_score", "mean"),
        max_spatial_ccc_score=("spatial_ccc_score", "max"),
        model_weighted_spatial_ccc_score=("model_weighted_spatial_ccc_score", "mean"),
    )
    score_cols = [
        col
        for col in (
            "model_score",
            "confidence",
            "density_rank",
            "ligand_secreted_like_score",
            "ligand_membrane_like_score",
            "receptor_secreted_like_score",
            "receptor_membrane_like_score",
        )
        if col in lr.columns
    ]
    if score_cols:
        summary = summary.merge(lr[["ligand", "receptor", *score_cols]].drop_duplicates(["ligand", "receptor"]), on=["ligand", "receptor"], how="left")
    return summary


def _annotate_curated_overlap(
    summary: pd.DataFrame,
    curated_lr_table: CellChatDB | pd.DataFrame | None,
    genes: set[str],
) -> pd.DataFrame:
    if curated_lr_table is None:
        return summary
    curated = curated_lr_table.interactions if isinstance(curated_lr_table, CellChatDB) else normalize_lr_table(curated_lr_table)
    curated = _filter_lr(curated, genes)
    curated_pairs = set(zip(curated["ligand"].astype(str), curated["receptor"].astype(str), strict=True))
    out = summary.copy()
    out["curated_overlap"] = [
        (str(row.ligand), str(row.receptor)) in curated_pairs
        for row in out.itertuples(index=False)
    ]
    return out


def _null_distribution(
    lr: pd.DataFrame,
    adata,
    coords: np.ndarray,
    groups: np.ndarray,
    gene_names: pd.Index,
    expr_means: pd.DataFrame,
    gene_expression: pd.Series,
    radius: float,
    sigma: float,
    kernels: Sequence[str],
    null_models: Sequence[str],
    n_permutations: int,
    random_state: int | None,
    *,
    sections: np.ndarray | None = None,
    distance_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    n = max(int(n_permutations), 0)
    base_weights = _spatial_weight_tables(coords, groups, radius=radius, sigma=sigma, kernels=kernels, distance_matrix=distance_matrix)
    all_expr_means = None
    for i in range(n):
        for null_model in null_models:
            if null_model == "coordinate_permutation":
                perm = rng.permutation(len(coords))
                if distance_matrix is None:
                    weights = _spatial_weight_tables(coords[perm], groups, radius=radius, sigma=sigma, kernels=kernels)
                else:
                    inverse = np.empty_like(perm)
                    inverse[perm] = np.arange(len(perm))
                    coord_groups = groups[inverse]
                    weights = _spatial_weight_tables(coords, coord_groups, radius=radius, sigma=sigma, kernels=kernels, distance_matrix=distance_matrix)
                scored = _score_lr_spatial(lr, expr_means, weights)
            elif null_model == "celltype_permutation":
                perm_groups = _permute_groups_for_celltype_null(groups, sections, rng)
                perm_means = _group_expression_means(adata, perm_groups, gene_names, sorted(set(lr["ligand"]).union(set(lr["receptor"]))))
                weights = _spatial_weight_tables(coords, perm_groups, radius=radius, sigma=sigma, kernels=kernels, distance_matrix=distance_matrix)
                scored = _score_lr_spatial(lr, perm_means, weights)
            elif null_model == "matched_random_lr":
                random_lr = _matched_random_lr(lr, gene_expression, rng)
                if all_expr_means is None:
                    all_expr_means = _group_expression_means(adata, groups, gene_names, gene_names)
                scored = _score_lr_spatial(random_lr, all_expr_means, base_weights)
            elif null_model == "score_permutation":
                permuted = lr.copy()
                if "model_score" in permuted.columns:
                    permuted["model_score"] = rng.permutation(permuted["model_score"].to_numpy())
                scored = _score_lr_spatial(permuted, expr_means, base_weights)
            else:
                raise ValueError(f"Unsupported null model: {null_model}")
            score_cols = ["spatial_ccc_score", "model_weighted_spatial_ccc_score"]
            for score_col in score_cols:
                grouped = _group_null_scores(scored, score_col=score_col)
                for row in grouped.itertuples(index=False):
                    out = {
                        "iteration": i,
                        "null_model": null_model,
                        "ligand": str(row.ligand),
                        "receptor": str(row.receptor),
                        "kernel": row.kernel,
                        "score_type": score_col,
                        "score_value": float(getattr(row, score_col)),
                        "celltype_permutation_scope": _celltype_permutation_scope(null_model, sections),
                    }
                    for col in _MATCHED_NULL_COLUMNS:
                        if hasattr(row, col):
                            out[col] = getattr(row, col)
                    rows.append(out)
    return pd.DataFrame(
        rows,
        columns=[
            "iteration",
            "null_model",
            "ligand",
            "receptor",
            "kernel",
            "score_type",
            "score_value",
            "celltype_permutation_scope",
            *_MATCHED_NULL_COLUMNS,
        ],
    )


def _permute_groups_for_celltype_null(
    groups: np.ndarray,
    sections: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = np.asarray(groups)
    if sections is None:
        return groups[rng.permutation(len(groups))]
    sections = np.asarray(sections)
    if len(sections) != len(groups):
        raise ValueError("`sections` must have the same length as `groups`.")
    out = groups.copy()
    for section in sorted(set(sections)):
        idx = np.flatnonzero(sections == section)
        if len(idx) > 1:
            out[idx] = groups[idx][rng.permutation(len(idx))]
    return out


def _celltype_permutation_scope(null_model: str, sections: np.ndarray | None) -> str | float:
    if null_model != "celltype_permutation":
        return np.nan
    return "section" if sections is not None else "global"


_MATCHED_NULL_COLUMNS = [
    "matched_ligand",
    "matched_receptor",
    "ligand_match_expression_delta",
    "receptor_match_expression_delta",
    "ligand_match_role_delta",
    "receptor_match_role_delta",
    "ligand_match_degree_delta",
    "receptor_match_degree_delta",
]


def _group_null_scores(scored: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    scored = scored.copy()
    if "original_ligand" in scored.columns and "original_receptor" in scored.columns:
        scored["ligand"] = scored["original_ligand"].astype(str)
        scored["receptor"] = scored["original_receptor"].astype(str)
        aggregations = {score_col: (score_col, "mean")}
        for col in _MATCHED_NULL_COLUMNS:
            if col in scored.columns:
                aggregations[col] = (col, "first")
        return scored.groupby(["ligand", "receptor", "kernel"], as_index=False).agg(**aggregations)
    return scored.groupby(["ligand", "receptor", "kernel"], as_index=False)[score_col].mean()


def _matched_random_lr(lr: pd.DataFrame, gene_expression: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    match_table = _gene_match_table(lr, gene_expression)
    matcher = _GeneMatchSampler(match_table)
    rows = []
    for row in lr.itertuples(index=False):
        ligand_match = matcher.sample(row, side="ligand", rng=rng)
        receptor_match = matcher.sample(row, side="receptor", rng=rng)
        rows.append(
            {
                "ligand": ligand_match["gene"],
                "receptor": receptor_match["gene"],
                "original_ligand": str(row.ligand),
                "original_receptor": str(row.receptor),
                "matched_ligand": ligand_match["gene"],
                "matched_receptor": receptor_match["gene"],
                "ligand_match_expression_delta": ligand_match["expression_delta"],
                "receptor_match_expression_delta": receptor_match["expression_delta"],
                "ligand_match_role_delta": ligand_match["role_delta"],
                "receptor_match_role_delta": receptor_match["role_delta"],
                "ligand_match_degree_delta": ligand_match["degree_delta"],
                "receptor_match_degree_delta": receptor_match["degree_delta"],
            }
        )
    return pd.DataFrame(rows)


def _gene_match_table(lr: pd.DataFrame, gene_expression: pd.Series) -> pd.DataFrame:
    genes = pd.Index(gene_expression.index.astype(str)).drop_duplicates()
    frame = pd.DataFrame({"gene": genes.astype(str), "expression": gene_expression.reindex(genes).fillna(0.0).to_numpy(dtype=float)})
    if frame.empty:
        return frame
    q = min(10, len(frame))
    frame["expression_quantile"] = pd.qcut(frame["expression"].rank(method="first"), q=q, labels=False, duplicates="drop").astype(int)
    ligand_degree = lr["ligand"].astype(str).value_counts()
    receptor_degree = lr["receptor"].astype(str).value_counts()
    frame["ligand_degree"] = frame["gene"].map(ligand_degree).fillna(0).astype(float)
    frame["receptor_degree"] = frame["gene"].map(receptor_degree).fillna(0).astype(float)
    frame["ligand_role_score"] = _role_lookup(lr, gene_col="ligand", score_col="ligand_role_score", default=0.5).reindex(frame["gene"]).fillna(0.5).to_numpy(dtype=float)
    frame["receptor_role_score"] = _role_lookup(lr, gene_col="receptor", score_col="receptor_role_score", default=0.5).reindex(frame["gene"]).fillna(0.5).to_numpy(dtype=float)
    return frame


def _role_lookup(lr: pd.DataFrame, *, gene_col: str, score_col: str, default: float) -> pd.Series:
    if score_col not in lr.columns:
        return pd.Series(default, index=pd.Index([], dtype=str), dtype=float)
    values = pd.DataFrame({"gene": lr[gene_col].astype(str), "score": pd.to_numeric(lr[score_col], errors="coerce")})
    values = values.dropna()
    if values.empty:
        return pd.Series(default, index=pd.Index([], dtype=str), dtype=float)
    return values.groupby("gene")["score"].mean()


@dataclass
class _GeneMatchSampler:
    match_table: pd.DataFrame

    def __post_init__(self) -> None:
        table = self.match_table.reset_index(drop=True)
        self.genes = table["gene"].astype(str).to_numpy() if "gene" in table else np.asarray([], dtype=str)
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.genes)}
        self.expression_quantile = _numeric_array(table, "expression_quantile")
        self.ligand_role_score = _numeric_array(table, "ligand_role_score", default=0.5)
        self.receptor_role_score = _numeric_array(table, "receptor_role_score", default=0.5)
        self.ligand_degree = _numeric_array(table, "ligand_degree")
        self.receptor_degree = _numeric_array(table, "receptor_degree")
        self.median_expression_quantile = float(np.nanmedian(self.expression_quantile)) if len(self.expression_quantile) else np.nan

    def sample(self, row, *, side: str, rng: np.random.Generator) -> dict[str, object]:
        if len(self.genes) == 0:
            gene = str(getattr(row, side))
            return {"gene": gene, "expression_delta": np.nan, "role_delta": np.nan, "degree_delta": np.nan}
        gene = str(getattr(row, side))
        target = self._target_values(row, side=side)
        role_scores = self.ligand_role_score if side == "ligand" else self.receptor_role_score
        degrees = self.ligand_degree if side == "ligand" else self.receptor_degree
        expression_delta = np.abs(self.expression_quantile - target["expression_quantile"])
        role_delta = np.abs(role_scores - target["role_score"])
        degree_delta = np.abs(np.log1p(degrees) - np.log1p(target["degree"]))
        distance = expression_delta + role_delta + degree_delta
        if len(distance) > 1 and gene in self.gene_to_idx:
            distance = distance.copy()
            distance[self.gene_to_idx[gene]] = np.inf
        finite = np.isfinite(distance)
        if not finite.any():
            idx = self.gene_to_idx.get(gene, 0)
        else:
            finite_idx = np.flatnonzero(finite)
            pool_size = min(10, len(finite_idx))
            if pool_size < len(finite_idx):
                candidate_idx = finite_idx[np.argpartition(distance[finite_idx], pool_size - 1)[:pool_size]]
            else:
                candidate_idx = finite_idx
            idx = int(candidate_idx[int(rng.integers(0, len(candidate_idx)))])
        return {
            "gene": str(self.genes[idx]),
            "expression_delta": float(expression_delta[idx]),
            "role_delta": float(role_delta[idx]),
            "degree_delta": float(degree_delta[idx]),
        }

    def _target_values(self, row, *, side: str) -> dict[str, float]:
        gene = str(getattr(row, side))
        role_scores = self.ligand_role_score if side == "ligand" else self.receptor_role_score
        degrees = self.ligand_degree if side == "ligand" else self.receptor_degree
        idx = self.gene_to_idx.get(gene)
        if idx is None:
            return {
                "expression_quantile": self.median_expression_quantile,
                "role_score": float(getattr(row, f"{side}_role_score", 0.5)),
                "degree": 0.0,
            }
        return {
            "expression_quantile": float(self.expression_quantile[idx]),
            "role_score": float(getattr(row, f"{side}_role_score", role_scores[idx])),
            "degree": float(degrees[idx]),
        }


def _numeric_array(table: pd.DataFrame, col: str, *, default: float = 0.0) -> np.ndarray:
    if col not in table:
        return np.full(len(table), float(default), dtype=float)
    return pd.to_numeric(table[col], errors="coerce").fillna(default).to_numpy(dtype=float)


def _attach_null_stats(summary: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["spatial_enrichment_z"] = np.nan
    out["empirical_pvalue"] = np.nan
    out["model_weighted_spatial_enrichment_z"] = np.nan
    out["model_weighted_empirical_pvalue"] = np.nan
    if null.empty:
        return out
    out = _attach_one_null_stat(
        out,
        null[null["score_type"].astype(str) == "spatial_ccc_score"],
        observed_col="spatial_ccc_score",
        z_col="spatial_enrichment_z",
        p_col="empirical_pvalue",
    )
    return _attach_one_null_stat(
        out,
        null[null["score_type"].astype(str) == "model_weighted_spatial_ccc_score"],
        observed_col="model_weighted_spatial_ccc_score",
        z_col="model_weighted_spatial_enrichment_z",
        p_col="model_weighted_empirical_pvalue",
    )


def _attach_one_null_stat(
    out: pd.DataFrame,
    null: pd.DataFrame,
    *,
    observed_col: str,
    z_col: str,
    p_col: str,
) -> pd.DataFrame:
    if null.empty:
        return out
    null_by_key = {
        key: sub["score_value"].astype(float).to_numpy()
        for key, sub in null.groupby(["ligand", "receptor", "kernel"])
    }
    null_by_kernel = {kernel: sub["score_value"].astype(float).to_numpy() for kernel, sub in null.groupby("kernel")}
    for idx, row in out.iterrows():
        values = null_by_key.get((row["ligand"], row["receptor"], row["kernel"]), np.asarray([], dtype=float))
        if len(values) == 0:
            values = null_by_kernel.get(row["kernel"], np.asarray([], dtype=float))
        if len(values) == 0:
            continue
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        observed = float(row[observed_col])
        out.at[idx, z_col] = 0.0 if std == 0 else (observed - mean) / std
        out.at[idx, p_col] = (np.sum(values >= observed) + 1) / (len(values) + 1)
    return out


def _top_k_enrichment(summary: pd.DataFrame, null: pd.DataFrame, *, top_k_values: Sequence[int]) -> pd.DataFrame:
    columns = [
        "kernel",
        "score_type",
        "k",
        "n_pairs",
        "observed_mean",
        "null_mean",
        "null_sd",
        "top_k_enrichment_z",
        "top_k_empirical_pvalue",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for kernel, kernel_summary in summary.groupby("kernel", sort=False):
        for score_col in ("spatial_ccc_score", "model_weighted_spatial_ccc_score"):
            if score_col not in kernel_summary.columns:
                continue
            ranked = _rank_top_predicted_pairs(kernel_summary, score_col=score_col)
            for k in top_k_values:
                kk = min(max(int(k), 1), len(ranked))
                top = ranked.head(kk)
                observed = float(top[score_col].astype(float).mean())
                null_values = _top_k_null_values(null, top, kernel=str(kernel), score_type=score_col)
                null_mean = float(null_values.mean()) if len(null_values) else np.nan
                null_sd = float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0
                z = 0.0 if len(null_values) and null_sd == 0 else ((observed - null_mean) / null_sd if len(null_values) else np.nan)
                pvalue = float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1)) if len(null_values) else np.nan
                rows.append(
                    {
                        "kernel": str(kernel),
                        "score_type": score_col,
                        "k": int(k),
                        "n_pairs": int(kk),
                        "observed_mean": observed,
                        "null_mean": null_mean,
                        "null_sd": null_sd,
                        "top_k_enrichment_z": float(z),
                        "top_k_empirical_pvalue": pvalue,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _rank_top_predicted_pairs(summary: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    if "model_score" in summary.columns:
        return summary.sort_values(["model_score", score_col], ascending=[False, False])
    if "density_rank" in summary.columns:
        return summary.sort_values(["density_rank", score_col], ascending=[True, False])
    return summary.sort_values(score_col, ascending=False)


def _top_k_null_values(null: pd.DataFrame, top: pd.DataFrame, *, kernel: str, score_type: str) -> np.ndarray:
    if null.empty:
        return np.asarray([], dtype=float)
    keys = top[["ligand", "receptor"]].drop_duplicates().copy()
    sub = null[
        (null["kernel"].astype(str) == str(kernel))
        & (null["score_type"].astype(str) == str(score_type))
    ].copy()
    if sub.empty:
        return np.asarray([], dtype=float)
    sub = sub.merge(keys, on=["ligand", "receptor"], how="inner")
    if sub.empty:
        return np.asarray([], dtype=float)
    values = sub.groupby(["null_model", "iteration"], sort=False)["score_value"].mean()
    return values.astype(float).to_numpy()


def _role_kernel_enrichment(summary: pd.DataFrame, *, role_score_threshold: float = 0.5) -> pd.DataFrame:
    columns = [
        "role_class",
        "kernel",
        "score_type",
        "role_score_col",
        "role_score_threshold",
        "n_role_pairs",
        "n_background_pairs",
        "role_mean",
        "background_mean",
        "role_kernel_enrichment",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    frame = summary.copy()
    role_specs = []
    if "ligand_secreted_like_score" in frame.columns:
        role_specs.append(("secreted_like", "exp", "ligand_secreted_like_score"))
    membrane_cols = [col for col in ("ligand_membrane_like_score", "receptor_membrane_like_score") if col in frame.columns]
    if membrane_cols:
        frame["membrane_contact_like_score"] = frame[membrane_cols].astype(float).max(axis=1)
        role_specs.append(("membrane_contact_like", "contact", "membrane_contact_like_score"))
    rows = []
    for role_class, kernel, role_col in role_specs:
        sub = frame[frame["kernel"].astype(str) == kernel].copy()
        if sub.empty:
            continue
        role_score = pd.to_numeric(sub[role_col], errors="coerce").fillna(0.0)
        role_mask = role_score >= role_score_threshold
        background = sub[~role_mask].copy()
        if background.empty:
            background = sub
        for score_col in ("spatial_ccc_score", "model_weighted_spatial_ccc_score"):
            if score_col not in sub.columns:
                continue
            role_values = pd.to_numeric(sub.loc[role_mask, score_col], errors="coerce").dropna()
            background_values = pd.to_numeric(background[score_col], errors="coerce").dropna()
            role_mean = float(role_values.mean()) if not role_values.empty else np.nan
            background_mean = float(background_values.mean()) if not background_values.empty else np.nan
            enrichment = role_mean / background_mean if pd.notna(role_mean) and pd.notna(background_mean) and background_mean != 0 else np.nan
            rows.append(
                {
                    "role_class": role_class,
                    "kernel": kernel,
                    "score_type": score_col,
                    "role_score_col": role_col,
                    "role_score_threshold": float(role_score_threshold),
                    "n_role_pairs": int(role_mask.sum()),
                    "n_background_pairs": int(len(background)),
                    "role_mean": role_mean,
                    "background_mean": background_mean,
                    "role_kernel_enrichment": float(enrichment) if pd.notna(enrichment) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _curated_overlap_enrichment(summary: pd.DataFrame, curated_lr_table: CellChatDB | pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "kernel",
        "score_type",
        "n_curated_overlap_pairs",
        "n_background_pairs",
        "curated_overlap_mean",
        "background_mean",
        "curated_overlap_enrichment",
        "curated_overlap_fraction",
    ]
    if curated_lr_table is None or summary.empty or "curated_overlap" not in summary.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for kernel, sub in summary.groupby("kernel", sort=False):
        overlap_mask = sub["curated_overlap"].astype(bool)
        background = sub[~overlap_mask]
        for score_col in ("spatial_ccc_score", "model_weighted_spatial_ccc_score"):
            if score_col not in sub.columns:
                continue
            overlap_values = pd.to_numeric(sub.loc[overlap_mask, score_col], errors="coerce").dropna()
            background_values = pd.to_numeric(background[score_col], errors="coerce").dropna()
            overlap_mean = float(overlap_values.mean()) if not overlap_values.empty else np.nan
            background_mean = float(background_values.mean()) if not background_values.empty else np.nan
            enrichment = overlap_mean / background_mean if pd.notna(overlap_mean) and pd.notna(background_mean) and background_mean != 0 else np.nan
            rows.append(
                {
                    "kernel": str(kernel),
                    "score_type": score_col,
                    "n_curated_overlap_pairs": int(overlap_mask.sum()),
                    "n_background_pairs": int((~overlap_mask).sum()),
                    "curated_overlap_mean": overlap_mean,
                    "background_mean": background_mean,
                    "curated_overlap_enrichment": float(enrichment) if pd.notna(enrichment) else np.nan,
                    "curated_overlap_fraction": float(overlap_mask.mean()) if len(overlap_mask) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _distance_decay(coords: np.ndarray, groups: np.ndarray, lr: pd.DataFrame, expr_means: pd.DataFrame) -> pd.DataFrame:
    dist = cdist(coords[:, :2], coords[:, :2])
    bins = np.quantile(dist[np.isfinite(dist)], np.linspace(0, 1, 6))
    bins = np.unique(bins)
    rows = []
    group_levels = sorted(set(groups))
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        in_bin = (dist >= left) & (dist <= right)
        for lr_row in lr.itertuples(index=False):
            scores = []
            for source in group_levels:
                s_mask = groups == source
                for target in group_levels:
                    t_mask = groups == target
                    spatial_weight = float(in_bin[np.ix_(s_mask, t_mask)].mean())
                    expr_score = float(expr_means.loc[source, str(lr_row.ligand)] * expr_means.loc[target, str(lr_row.receptor)])
                    scores.append(expr_score * spatial_weight)
            rows.append(
                {
                    "ligand": str(lr_row.ligand),
                    "receptor": str(lr_row.receptor),
                    "distance_min": float(left),
                    "distance_max": float(right),
                    "mean_distance": float((left + right) / 2),
                    "mean_spatial_ccc_score": float(np.mean(scores)) if scores else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=_DISTANCE_DECAY_COLUMNS)


def _section_reproducibility(
    adata,
    coords: np.ndarray,
    groups: np.ndarray,
    lr: pd.DataFrame,
    gene_names: pd.Index,
    *,
    radius: float,
    sigma: float,
    kernels: Sequence[str],
    section_key: str | None,
    section_top_k: int,
) -> pd.DataFrame:
    if section_key is None:
        return _empty_section_reproducibility()
    sections = adata.obs[section_key].astype(str).to_numpy()
    genes = sorted(set(lr["ligand"]).union(set(lr["receptor"])))
    rows = []
    for section in sorted(set(sections)):
        mask = sections == section
        if int(mask.sum()) < 2:
            continue
        section_groups = groups[mask]
        if len(set(section_groups)) == 0:
            continue
        expr_means = _group_expression_means(adata, section_groups, gene_names, genes, cell_mask=mask)
        weights = _spatial_weight_tables(coords[mask], section_groups, radius=radius, sigma=sigma, kernels=kernels)
        observed = _score_lr_spatial(lr, expr_means, weights)
        if observed.empty:
            continue
        section_summary = observed.groupby(["ligand", "receptor", "kernel"], as_index=False).agg(
            spatial_ccc_score=("spatial_ccc_score", "mean"),
            model_weighted_spatial_ccc_score=("model_weighted_spatial_ccc_score", "mean"),
        )
        section_summary["section"] = str(section)
        section_summary["section_rank"] = section_summary.groupby("kernel")["model_weighted_spatial_ccc_score"].rank(method="min", ascending=False)
        section_summary["section_top_k"] = section_summary["section_rank"] <= max(int(section_top_k), 1)
        rows.append(section_summary)
    if not rows:
        return _empty_section_reproducibility()
    per_section = pd.concat(rows, ignore_index=True)
    out = per_section.groupby(["ligand", "receptor", "kernel"], as_index=False).agg(
        n_sections=("section", "nunique"),
        section_spatial_ccc_score_mean=("spatial_ccc_score", "mean"),
        section_spatial_ccc_score_sd=("spatial_ccc_score", "std"),
        section_model_weighted_spatial_ccc_score_mean=("model_weighted_spatial_ccc_score", "mean"),
        section_model_weighted_spatial_ccc_score_sd=("model_weighted_spatial_ccc_score", "std"),
        positive_section_fraction=("spatial_ccc_score", lambda values: float((values > 0).mean())),
        top_k_section_fraction=("section_top_k", "mean"),
        median_section_rank=("section_rank", "median"),
    )
    denom = out["section_spatial_ccc_score_mean"].abs().replace(0, np.nan)
    out["section_spatial_ccc_score_cv"] = out["section_spatial_ccc_score_sd"] / denom
    return out[_SECTION_REPRODUCIBILITY_COLUMNS]


_DISTANCE_DECAY_COLUMNS = [
    "ligand",
    "receptor",
    "distance_min",
    "distance_max",
    "mean_distance",
    "mean_spatial_ccc_score",
]


_SECTION_REPRODUCIBILITY_COLUMNS = [
    "ligand",
    "receptor",
    "kernel",
    "n_sections",
    "section_spatial_ccc_score_mean",
    "section_spatial_ccc_score_sd",
    "section_spatial_ccc_score_cv",
    "section_model_weighted_spatial_ccc_score_mean",
    "section_model_weighted_spatial_ccc_score_sd",
    "positive_section_fraction",
    "top_k_section_fraction",
    "median_section_rank",
]


def _empty_distance_decay() -> pd.DataFrame:
    return pd.DataFrame(columns=_DISTANCE_DECAY_COLUMNS)


def _empty_section_reproducibility() -> pd.DataFrame:
    return pd.DataFrame(columns=_SECTION_REPRODUCIBILITY_COLUMNS)

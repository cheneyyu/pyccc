from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist

from .database import CellChatDB, normalize_lr_table


@dataclass(frozen=True)
class SpatialValidationReport:
    """Spatial enrichment summaries for a candidate LR table."""

    summary: pd.DataFrame
    celltype_pair_summary: pd.DataFrame
    null_distribution: pd.DataFrame
    distance_decay: pd.DataFrame
    metadata: dict[str, object]


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
) -> SpatialValidationReport:
    """Validate candidate LR scores against simple spatial null models."""

    if spatial_key not in adata.obsm:
        raise KeyError(f"`{spatial_key}` is not present in adata.obsm.")
    if groupby not in adata.obs:
        raise KeyError(f"`{groupby}` is not present in adata.obs.")
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must be a two-dimensional array with at least two columns.")
    lr = lr_table.interactions if isinstance(lr_table, CellChatDB) else normalize_lr_table(lr_table)
    gene_names = _gene_names(adata, gene_symbols_key=gene_symbols_key)
    lr = _filter_lr(lr, set(gene_names))
    if lr.empty:
        raise ValueError("No LR rows have ligand and receptor genes in the expression matrix.")
    groups = adata.obs[groupby].astype(str).to_numpy()
    radius_value = _resolve_radius(coords, adata, radius)
    sigma_value = _resolve_sigma(coords, sigma, fallback=radius_value)
    expr_means = _group_expression_means(adata, groups, gene_names, sorted(set(lr["ligand"]).union(set(lr["receptor"]))))
    weight_tables = _spatial_weight_tables(coords, groups, radius=radius_value, sigma=sigma_value, kernels=distance_kernels)
    observed = _score_lr_spatial(lr, expr_means, weight_tables)
    summary = _lr_summary(observed, lr)
    null = _null_distribution(lr, adata, coords, groups, gene_names, expr_means, radius_value, sigma_value, distance_kernels, null_models, n_permutations, random_state)
    summary = _attach_null_stats(summary, null)
    distance_decay = _distance_decay(coords, groups, lr, expr_means)
    metadata = {"mode": mode, "radius": radius_value, "sigma": sigma_value, "null_models": list(null_models), "n_permutations": n_permutations}
    return SpatialValidationReport(summary, observed, null, distance_decay, metadata)


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
    dist = cdist(coords[:, :2], coords[:, :2])
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))


def _resolve_sigma(coords: np.ndarray, sigma: str | float, *, fallback: float) -> float:
    if sigma != "auto":
        return float(sigma)
    return float(max(fallback, 1e-9))


def _group_expression_means(adata, groups: np.ndarray, gene_names: pd.Index, genes: Sequence[str]) -> pd.DataFrame:
    gene_lookup = {gene: i for i, gene in enumerate(gene_names)}
    cols = [gene_lookup[gene] for gene in genes if gene in gene_lookup]
    selected_genes = [gene for gene in genes if gene in gene_lookup]
    x = adata.X[:, cols]
    rows = []
    for group in sorted(set(groups)):
        mask = groups == group
        sub = x[mask]
        values = np.asarray(sub.mean(axis=0)).ravel() if sparse.issparse(sub) else np.asarray(sub).mean(axis=0)
        rows.append(pd.Series(values, index=selected_genes, name=group))
    return pd.DataFrame(rows).fillna(0.0)


def _spatial_weight_tables(coords: np.ndarray, groups: np.ndarray, *, radius: float, sigma: float, kernels: Sequence[str]) -> dict[str, pd.DataFrame]:
    dist = cdist(coords[:, :2], coords[:, :2])
    tables = {}
    for kernel in kernels:
        if kernel == "contact":
            weights = (dist <= radius).astype(float)
        elif kernel == "exp":
            weights = np.exp(-dist / sigma)
        else:
            raise ValueError("Distance kernels must be `contact` or `exp`.")
        rows = []
        group_levels = sorted(set(groups))
        for source in group_levels:
            s_mask = groups == source
            for target in group_levels:
                t_mask = groups == target
                rows.append({"source": source, "target": target, "kernel": kernel, "spatial_weight": float(weights[np.ix_(s_mask, t_mask)].mean())})
        tables[kernel] = pd.DataFrame(rows)
    return tables


def _score_lr_spatial(lr: pd.DataFrame, expr_means: pd.DataFrame, weight_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for kernel, weights in weight_tables.items():
        for lr_row in lr.itertuples(index=False):
            ligand = str(lr_row.ligand)
            receptor = str(lr_row.receptor)
            for weight_row in weights.itertuples(index=False):
                source_expr = float(expr_means.loc[str(weight_row.source), ligand])
                target_expr = float(expr_means.loc[str(weight_row.target), receptor])
                expression_score = source_expr * target_expr
                rows.append(
                    {
                        "ligand": ligand,
                        "receptor": receptor,
                        "source": str(weight_row.source),
                        "target": str(weight_row.target),
                        "kernel": kernel,
                        "expression_score": expression_score,
                        "spatial_weight": float(weight_row.spatial_weight),
                        "spatial_ccc_score": expression_score * float(weight_row.spatial_weight),
                    }
                )
    return pd.DataFrame(rows)


def _lr_summary(observed: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    summary = observed.groupby(["ligand", "receptor", "kernel"], as_index=False).agg(spatial_ccc_score=("spatial_ccc_score", "mean"), max_spatial_ccc_score=("spatial_ccc_score", "max"))
    score_cols = [col for col in ("model_score", "confidence", "density_rank") if col in lr.columns]
    if score_cols:
        summary = summary.merge(lr[["ligand", "receptor", *score_cols]].drop_duplicates(["ligand", "receptor"]), on=["ligand", "receptor"], how="left")
    return summary


def _null_distribution(
    lr: pd.DataFrame,
    adata,
    coords: np.ndarray,
    groups: np.ndarray,
    gene_names: pd.Index,
    expr_means: pd.DataFrame,
    radius: float,
    sigma: float,
    kernels: Sequence[str],
    null_models: Sequence[str],
    n_permutations: int,
    random_state: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    n = max(int(n_permutations), 0)
    for i in range(n):
        for null_model in null_models:
            if null_model == "coordinate_permutation":
                perm_coords = coords[rng.permutation(len(coords))]
                weights = _spatial_weight_tables(perm_coords, groups, radius=radius, sigma=sigma, kernels=kernels)
                scored = _score_lr_spatial(lr, expr_means, weights)
            elif null_model == "celltype_permutation":
                perm_groups = groups[rng.permutation(len(groups))]
                perm_means = _group_expression_means(adata, perm_groups, gene_names, sorted(set(lr["ligand"]).union(set(lr["receptor"]))))
                weights = _spatial_weight_tables(coords, perm_groups, radius=radius, sigma=sigma, kernels=kernels)
                scored = _score_lr_spatial(lr, perm_means, weights)
            elif null_model == "matched_random_lr":
                random_lr = _matched_random_lr(lr, gene_names, rng)
                scored = _score_lr_spatial(random_lr, expr_means, _spatial_weight_tables(coords, groups, radius=radius, sigma=sigma, kernels=kernels))
            elif null_model == "score_permutation":
                scored = _score_lr_spatial(lr.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))), expr_means, _spatial_weight_tables(coords, groups, radius=radius, sigma=sigma, kernels=kernels))
            else:
                raise ValueError(f"Unsupported null model: {null_model}")
            grouped = scored.groupby("kernel", as_index=False)["spatial_ccc_score"].mean()
            for row in grouped.itertuples(index=False):
                rows.append({"iteration": i, "null_model": null_model, "kernel": row.kernel, "spatial_ccc_score": float(row.spatial_ccc_score)})
    return pd.DataFrame(rows, columns=["iteration", "null_model", "kernel", "spatial_ccc_score"])


def _matched_random_lr(lr: pd.DataFrame, gene_names: pd.Index, rng: np.random.Generator) -> pd.DataFrame:
    genes = np.asarray(gene_names.astype(str))
    return pd.DataFrame({"ligand": rng.choice(genes, size=len(lr), replace=True), "receptor": rng.choice(genes, size=len(lr), replace=True)})


def _attach_null_stats(summary: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["spatial_enrichment_z"] = np.nan
    out["empirical_pvalue"] = np.nan
    if null.empty:
        return out
    null_by_kernel = {kernel: sub["spatial_ccc_score"].astype(float).to_numpy() for kernel, sub in null.groupby("kernel")}
    for idx, row in out.iterrows():
        values = null_by_kernel.get(row["kernel"], np.asarray([], dtype=float))
        if len(values) == 0:
            continue
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        observed = float(row["spatial_ccc_score"])
        out.at[idx, "spatial_enrichment_z"] = 0.0 if std == 0 else (observed - mean) / std
        out.at[idx, "empirical_pvalue"] = (np.sum(values >= observed) + 1) / (len(values) + 1)
    return out


def _distance_decay(coords: np.ndarray, groups: np.ndarray, lr: pd.DataFrame, expr_means: pd.DataFrame) -> pd.DataFrame:
    dist = cdist(coords[:, :2], coords[:, :2])
    bins = np.quantile(dist[np.isfinite(dist)], np.linspace(0, 1, 6))
    bins = np.unique(bins)
    rows = []
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        rows.append({"distance_min": float(left), "distance_max": float(right), "mean_distance": float((left + right) / 2), "n_lr_pairs": int(len(lr))})
    return pd.DataFrame(rows)

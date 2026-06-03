from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .analysis import CCCResult, _complex_genes, _get_matrix, _group_expression_from_labels, _subset_matrix_columns
from .database import CellChatDB, normalize_lr_table

COFACTOR_GENE_COLUMNS = ("agonist_genes", "antagonist_genes", "co_A_receptor_genes", "co_I_receptor_genes")


def signaling_expression_frame(
    adata,
    groupby: str | None = None,
    *,
    result: CCCResult | None = None,
    genes: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    signaling: str | Sequence[str] | None = None,
    lr_table: CellChatDB | pd.DataFrame | None = None,
    enriched_only: bool = True,
    include_cofactors: bool = False,
    groups: Sequence[str] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    gene_symbols_key: str | None = None,
    aggregate: str = "mean",
    trim: float = 0.1,
    standard_scale: bool = True,
    scale_min: float = -2.5,
    scale_max: float = 2.5,
) -> pd.DataFrame:
    """Return a CellChat `plotGeneExpression`-style dot-plot data frame.

    The returned table contains one row per `(group, gene)` with average
    expression, fraction of expressing cells, and Seurat-like scaled average
    expression for dot plots.
    """

    groupby = _resolve_groupby(groupby, result)
    groups_resolved = _resolve_groups(adata, groupby, groups=groups, result=result)
    x, var_names = _get_matrix(adata, layer=layer, use_raw=use_raw, gene_symbols_key=gene_symbols_key)
    selected = _select_expression_genes(
        var_names,
        genes=genes,
        features=features,
        result=result,
        signaling=signaling,
        lr_table=lr_table,
        enriched_only=enriched_only,
        include_cofactors=include_cofactors,
    )
    x, var_names = _subset_matrix_columns(x, var_names, selected)
    labels = adata.obs[groupby].astype(str).to_numpy()
    means, pcts = _group_expression_from_labels(x, var_names, labels, groups_resolved, aggregate=aggregate, trim=trim)
    n_cells = pd.Series(labels).value_counts().reindex(groups_resolved, fill_value=0).astype(int)

    mean_long = means.reset_index(names="group").melt(id_vars="group", var_name="gene", value_name="mean_expression")
    pct_long = pcts.reset_index(names="group").melt(id_vars="group", var_name="gene", value_name="pct_expressed")
    frame = mean_long.merge(pct_long, on=["group", "gene"], how="left")
    frame["n_cells"] = frame["group"].map(n_cells).astype(int)
    frame["scaled_expression"] = _scale_expression(frame, standard_scale=standard_scale, scale_min=scale_min, scale_max=scale_max)
    frame["group"] = pd.Categorical(frame["group"].astype(str), categories=groups_resolved, ordered=True)
    frame["gene"] = pd.Categorical(frame["gene"].astype(str), categories=selected, ordered=True)
    return frame


def signaling_expression_values(
    adata,
    groupby: str | None = None,
    *,
    result: CCCResult | None = None,
    genes: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    signaling: str | Sequence[str] | None = None,
    lr_table: CellChatDB | pd.DataFrame | None = None,
    enriched_only: bool = True,
    include_cofactors: bool = False,
    groups: Sequence[str] | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    gene_symbols_key: str | None = None,
    max_cells_per_group: int | None = None,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Return long-form per-cell expression values for violin plots."""

    groupby = _resolve_groupby(groupby, result)
    groups_resolved = _resolve_groups(adata, groupby, groups=groups, result=result)
    x, var_names = _get_matrix(adata, layer=layer, use_raw=use_raw, gene_symbols_key=gene_symbols_key)
    selected = _select_expression_genes(
        var_names,
        genes=genes,
        features=features,
        result=result,
        signaling=signaling,
        lr_table=lr_table,
        enriched_only=enriched_only,
        include_cofactors=include_cofactors,
    )
    x, var_names = _subset_matrix_columns(x, var_names, selected)
    labels = adata.obs[groupby].astype(str).to_numpy()
    keep = np.isin(labels, groups_resolved)
    if max_cells_per_group is not None:
        if max_cells_per_group <= 0:
            raise ValueError("`max_cells_per_group` must be positive.")
        keep &= _sample_group_mask(labels, groups_resolved, max_cells_per_group=max_cells_per_group, random_state=random_state)
    x = x[keep]
    labels = labels[keep]
    obs_names = pd.Index(adata.obs_names.astype(str))[keep]
    arr = x.toarray() if sparse.issparse(x) else np.asarray(x)
    wide = pd.DataFrame(arr, columns=var_names.astype(str), index=obs_names)
    wide["cell"] = wide.index.astype(str)
    wide["group"] = labels.astype(str)
    long = wide.melt(id_vars=["cell", "group"], var_name="gene", value_name="expression")
    long["group"] = pd.Categorical(long["group"].astype(str), categories=groups_resolved, ordered=True)
    long["gene"] = pd.Categorical(long["gene"].astype(str), categories=selected, ordered=True)
    return long


def _resolve_groupby(groupby: str | None, result: CCCResult | None) -> str:
    if groupby is None:
        if result is None:
            raise ValueError("Provide `groupby` or a `result` with a groupby field.")
        return result.groupby
    return groupby


def _resolve_groups(adata, groupby: str, *, groups: Sequence[str] | None, result: CCCResult | None) -> list[str]:
    if groupby not in adata.obs:
        raise KeyError(f"`{groupby}` is not present in adata.obs.")
    available = set(adata.obs[groupby].astype(str))
    if groups is not None:
        resolved = [str(group) for group in groups]
    elif result is not None and result.groupby == groupby:
        resolved = [str(group) for group in result.groups]
    else:
        resolved = [str(x) for x in pd.Index(adata.obs[groupby].astype(str).unique()).sort_values()]
    missing = [group for group in resolved if group not in available]
    if missing:
        raise ValueError(f"Requested groups are not present in adata.obs[{groupby!r}]: {missing[:5]}")
    return resolved


def _select_expression_genes(
    var_names: pd.Index,
    *,
    genes: Sequence[str] | None,
    features: Sequence[str] | None,
    result: CCCResult | None,
    signaling: str | Sequence[str] | None,
    lr_table: CellChatDB | pd.DataFrame | None,
    enriched_only: bool,
    include_cofactors: bool,
) -> list[str]:
    if genes is not None and features is not None:
        raise ValueError("Use either `genes` or CellChat-compatible `features`, not both.")
    explicit = features if features is not None else genes
    if explicit is not None:
        return _resolve_matrix_genes(explicit, var_names, strict=True)

    lr = _expression_lr_table(result=result, signaling=signaling, lr_table=lr_table, enriched_only=enriched_only)
    candidates = _lr_gene_candidates(lr, include_cofactors=include_cofactors)
    selected = _resolve_matrix_genes(candidates, var_names, strict=False)
    if not selected:
        source = "LR table" if lr_table is not None and not enriched_only else "communication result"
        raise ValueError(f"No signaling genes from the selected {source} are present in the expression matrix.")
    return selected


def _expression_lr_table(
    *,
    result: CCCResult | None,
    signaling: str | Sequence[str] | None,
    lr_table: CellChatDB | pd.DataFrame | None,
    enriched_only: bool,
) -> pd.DataFrame:
    if not enriched_only and lr_table is not None:
        lr = lr_table.interactions.copy() if isinstance(lr_table, CellChatDB) else normalize_lr_table(lr_table)
    else:
        if result is None:
            raise ValueError("Provide `genes`/`features`, or provide `result` to derive signaling genes.")
        if enriched_only:
            lr = result.significant()
            if lr.empty and "pvalue" in result.interactions.columns and result.interactions["pvalue"].isna().all():
                lr = result.interactions.copy()
        else:
            lr = result.interactions.copy()
    if signaling is not None:
        signals = set(_as_list(signaling))
        lr = lr[lr["pathway"].astype(str).isin(signals)]
    if lr.empty:
        raise ValueError("No ligand-receptor rows match the selected signaling genes/pathways.")
    return lr


def _lr_gene_candidates(lr: pd.DataFrame, *, include_cofactors: bool) -> list[str]:
    cols = ["ligand", "receptor"]
    if include_cofactors:
        cols.extend(col for col in COFACTOR_GENE_COLUMNS if col in lr.columns)
    genes: list[str] = []
    for col in cols:
        if col not in lr.columns:
            continue
        for value in lr[col].fillna("").astype(str):
            genes.extend(_complex_genes(value))
    return list(dict.fromkeys(genes))


def _resolve_matrix_genes(candidates: Sequence[str], var_names: pd.Index, *, strict: bool) -> list[str]:
    ordered = list(dict.fromkeys(str(gene) for gene in candidates if str(gene).strip()))
    names = pd.Index(var_names.astype(str))
    exact = set(names)
    upper_lookup = {name.upper(): name for name in names}
    resolved: list[str] = []
    missing: list[str] = []
    for gene in ordered:
        if gene in exact:
            resolved.append(gene)
            continue
        match = upper_lookup.get(gene.upper())
        if match is not None:
            resolved.append(match)
        else:
            missing.append(gene)
    if strict and missing:
        raise KeyError(f"Requested genes are not present in the expression matrix: {missing[:5]}")
    return list(dict.fromkeys(resolved))


def _scale_expression(frame: pd.DataFrame, *, standard_scale: bool, scale_min: float, scale_max: float) -> pd.Series:
    values = pd.to_numeric(frame["mean_expression"], errors="coerce").fillna(0.0)
    if not standard_scale:
        return values
    scaled = values.groupby(frame["gene"], observed=True).transform(_zscore)
    if scale_min >= scale_max:
        raise ValueError("`scale_min` must be less than `scale_max`.")
    return scaled.clip(lower=scale_min, upper=scale_max)


def _zscore(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    sd = float(arr.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return pd.Series((arr - float(arr.mean())) / sd, index=values.index)


def _sample_group_mask(labels: np.ndarray, groups: Sequence[str], *, max_cells_per_group: int, random_state: int | None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    keep = np.zeros(labels.shape[0], dtype=bool)
    for group in groups:
        idx = np.flatnonzero(labels == group)
        if idx.size > max_cells_per_group:
            idx = rng.choice(idx, size=max_cells_per_group, replace=False)
        keep[idx] = True
    return keep


def _as_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]

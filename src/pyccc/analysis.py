from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats

from .database import CellChatDB, normalize_lr_table

VALID_AGGREGATES = {"mean", "clipped_mean", "gated_mean", "median", "tri_mean", "truncated_mean", "truncatedMean"}
VALID_SCORE_METHODS = {"sqrt", "cellchat"}
VALID_COMPLEX_AGGREGATES = {"auto", "min", "geometric_mean"}
VALID_DE_METHODS = {"mean", "wilcoxon"}
VALID_ARRAY_BACKENDS = {"cpu", "numpy"}


@dataclass
class CCCResult:
    """Result of CellChat-like communication inference."""

    interactions: pd.DataFrame
    groupby: str
    groups: list[str]
    condition: str | None = None
    lr_name: str = "custom"
    pvalue_cutoff: float = 0.05
    metadata: dict[str, object] = field(default_factory=dict)

    def significant(self, pvalue_cutoff: float | None = None) -> pd.DataFrame:
        cutoff = self.pvalue_cutoff if pvalue_cutoff is None else pvalue_cutoff
        if "pvalue" not in self.interactions.columns:
            return self.interactions.copy()
        return self.interactions[self.interactions["pvalue"] <= cutoff].copy()

    def network(
        self,
        *,
        value: str = "prob",
        pathway: str | Iterable[str] | None = None,
        significant_only: bool = False,
        pvalue_cutoff: float | None = None,
    ) -> pd.DataFrame:
        """Return a source-by-target network matrix."""

        df = self.significant(pvalue_cutoff) if significant_only else self.interactions
        if pathway is not None:
            pathways = {pathway} if isinstance(pathway, str) else set(pathway)
            df = df[df["pathway"].isin(pathways)]
        if value == "count":
            mat = pd.crosstab(df["source"], df["target"])
        else:
            if value not in df.columns:
                raise ValueError(f"`{value}` is not available in interactions.")
            mat = df.pivot_table(index="source", columns="target", values=value, aggfunc="sum", fill_value=0.0)
        return mat.reindex(index=self.groups, columns=self.groups, fill_value=0.0)

    def pathway_summary(self, *, significant_only: bool = False) -> pd.DataFrame:
        df = self.significant() if significant_only else self.interactions
        return (
            df.groupby("pathway", observed=True)
            .agg(prob=("prob", "sum"), count=("prob", "size"), mean_prob=("prob", "mean"))
            .sort_values(["prob", "count"], ascending=False)
            .reset_index()
        )

    def lr_summary(self, *, significant_only: bool = False) -> pd.DataFrame:
        df = self.significant() if significant_only else self.interactions
        return (
            df.groupby(["ligand", "receptor", "pathway"], observed=True)
            .agg(prob=("prob", "sum"), count=("prob", "size"), mean_prob=("prob", "mean"))
            .sort_values(["prob", "count"], ascending=False)
            .reset_index()
        )


def compute_communication(
    adata,
    groupby: str,
    lr_table: CellChatDB | pd.DataFrame,
    *,
    condition_key: str | None = None,
    condition: str | None = None,
    layer: str | None = None,
    use_raw: bool = False,
    gene_symbols_key: str | None = None,
    min_pct: float = 0.05,
    min_expr: float = 0.0,
    n_permutations: int = 0,
    random_state: int | None = 0,
    pvalue_cutoff: float = 0.05,
    aggregate: str = "tri_mean",
    trim: float = 0.1,
    clip_quantile: float = 0.99,
    population_size: bool = False,
    spatial_key: str | None = None,
    distance_decay: float | None = None,
    cofactor_adjust: bool = False,
    cofactor_kh: float = 0.5,
    cofactor_hill: float = 1.0,
    score_method: str = "sqrt",
    complex_aggregate: str = "auto",
    n_jobs: int = 1,
    array_backend: str | None = None,
    downsample_per_group: int | None = None,
    downsample_repeats: int = 1,
    de_gate: bool = False,
    overexpressed_genes: pd.DataFrame | dict[str, Iterable[str]] | None = None,
    de_method: str = "mean",
    de_min_pct: float = 0.1,
    de_min_logfc: float = 0.1,
    de_pvalue_cutoff: float | None = None,
) -> CCCResult:
    """Infer LR communication probabilities from an AnnData object.

    Communication probability is computed from sender ligand expression and
    receiver receptor expression. Multi-subunit complexes are represented by
    the weakest expressed subunit, matching CellChat's conservative intuition.
    """

    _validate_compute_options(
        layer=layer,
        use_raw=use_raw,
        min_pct=min_pct,
        min_expr=min_expr,
        n_permutations=n_permutations,
        pvalue_cutoff=pvalue_cutoff,
        aggregate=aggregate,
        trim=trim,
        clip_quantile=clip_quantile,
        distance_decay=distance_decay,
        cofactor_kh=cofactor_kh,
        cofactor_hill=cofactor_hill,
        score_method=score_method,
        complex_aggregate=complex_aggregate,
        n_jobs=n_jobs,
        array_backend=array_backend,
        downsample_per_group=downsample_per_group,
        downsample_repeats=downsample_repeats,
        de_method=de_method,
        de_min_pct=de_min_pct,
        de_min_logfc=de_min_logfc,
        de_pvalue_cutoff=de_pvalue_cutoff,
    )
    if groupby not in adata.obs:
        raise KeyError(f"`{groupby}` is not present in adata.obs.")

    obs_mask = np.ones(adata.n_obs, dtype=bool)
    if condition_key is not None:
        if condition_key not in adata.obs:
            raise KeyError(f"`{condition_key}` is not present in adata.obs.")
        if condition is not None:
            obs_mask = adata.obs[condition_key].astype(str).to_numpy() == str(condition)

    ad = adata[obs_mask].copy()
    groups = [str(x) for x in pd.Index(ad.obs[groupby].astype(str).unique()).sort_values()]
    if len(groups) < 2:
        raise ValueError("At least two cell groups are required for CCC inference.")

    lr_name = getattr(lr_table, "name", "custom")
    if isinstance(lr_table, CellChatDB):
        lr = lr_table.interactions
    else:
        lr = normalize_lr_table(lr_table)

    _, gene_names = _get_matrix(ad, layer=layer, use_raw=use_raw, gene_symbols_key=gene_symbols_key)
    gene_lookup = {g.upper(): g for g in gene_names.astype(str)}
    lr = _filter_lr_to_genes(lr, gene_lookup)
    if lr.empty:
        gene_source = f"adata.var[{gene_symbols_key!r}]" if gene_symbols_key is not None else "adata.var_names"
        raise ValueError(f"No ligand-receptor pairs have all required genes in {gene_source}.")

    expression_genes = _lr_expression_gene_candidates(lr, gene_lookup, include_cofactors=cofactor_adjust)
    complex_aggregate_resolved = _resolve_complex_aggregate(score_method, complex_aggregate)
    scale_expression_by_max = score_method == "cellchat"
    array_backend_resolved = _resolve_array_backend(array_backend)
    oe_sets = None
    if de_gate or overexpressed_genes is not None:
        if overexpressed_genes is None:
            overexpressed_genes = identify_overexpressed_genes(
                ad,
                groupby,
                layer=layer,
                use_raw=use_raw,
                gene_symbols_key=gene_symbols_key,
                method=de_method,
                min_pct=de_min_pct,
                min_logfc=de_min_logfc,
                pvalue_cutoff=de_pvalue_cutoff,
                candidate_genes=_lr_gene_candidates(lr, gene_lookup),
            )
        oe_sets = _overexpressed_gene_sets(overexpressed_genes, groups)
    if downsample_per_group is not None and downsample_repeats > 1:
        rows = _repeated_downsample_rows(
            ad,
            groupby,
            groups,
            lr,
            layer=layer,
            use_raw=use_raw,
            downsample_per_group=downsample_per_group,
            downsample_repeats=downsample_repeats,
            random_state=random_state,
            min_pct=min_pct,
            min_expr=min_expr,
            aggregate=aggregate,
            trim=trim,
            clip_quantile=clip_quantile,
            gene_symbols_key=gene_symbols_key,
            gene_lookup=gene_lookup,
            expression_genes=expression_genes,
            scale_expression_by_max=scale_expression_by_max,
            overexpressed_genes=oe_sets,
            population_size=population_size,
            spatial_key=spatial_key,
            distance_decay=distance_decay,
            cofactor_adjust=cofactor_adjust,
            cofactor_kh=cofactor_kh,
            cofactor_hill=cofactor_hill,
            score_method=score_method,
            complex_aggregate=complex_aggregate_resolved,
            n_jobs=n_jobs,
            array_backend=array_backend_resolved,
            n_permutations=n_permutations,
        )
    else:
        if downsample_per_group is not None:
            ad = _downsample_adata_by_group(ad, groupby, groups, n_per_group=downsample_per_group, rng=np.random.default_rng(random_state))
        rows = _compute_rows_for_adata(
            ad,
            groupby,
            groups,
            lr,
            layer=layer,
            use_raw=use_raw,
            min_pct=min_pct,
            min_expr=min_expr,
            aggregate=aggregate,
            trim=trim,
            clip_quantile=clip_quantile,
            gene_symbols_key=gene_symbols_key,
            gene_lookup=gene_lookup,
            expression_genes=expression_genes,
            scale_expression_by_max=scale_expression_by_max,
            overexpressed_genes=oe_sets,
            population_size=population_size,
            spatial_key=spatial_key,
            distance_decay=distance_decay,
            cofactor_adjust=cofactor_adjust,
            cofactor_kh=cofactor_kh,
            cofactor_hill=cofactor_hill,
            score_method=score_method,
            complex_aggregate=complex_aggregate_resolved,
            n_jobs=n_jobs,
            array_backend=array_backend_resolved,
            n_permutations=n_permutations,
            random_state=random_state,
        )

    rows["condition"] = condition if condition is not None else ""
    return CCCResult(
        rows.sort_values(["source", "target", "pathway", "prob"], ascending=[True, True, True, False]).reset_index(drop=True),
        groupby=groupby,
        groups=groups,
        condition=condition,
        lr_name=lr_name,
        pvalue_cutoff=pvalue_cutoff,
    )


def _validate_compute_options(
    *,
    layer: str | None,
    use_raw: bool,
    min_pct: float,
    min_expr: float,
    n_permutations: int,
    pvalue_cutoff: float,
    aggregate: str,
    trim: float,
    clip_quantile: float,
    distance_decay: float | None,
    cofactor_kh: float,
    cofactor_hill: float,
    score_method: str,
    complex_aggregate: str,
    n_jobs: int,
    array_backend: str | None,
    downsample_per_group: int | None,
    downsample_repeats: int,
    de_method: str,
    de_min_pct: float,
    de_min_logfc: float,
    de_pvalue_cutoff: float | None,
) -> None:
    if layer is not None and use_raw:
        raise ValueError("Pass either `layer` or `use_raw=True`, not both.")
    if not 0 <= min_pct <= 1:
        raise ValueError("`min_pct` must be between 0 and 1.")
    if min_expr < 0:
        raise ValueError("`min_expr` must be non-negative.")
    if n_permutations < 0:
        raise ValueError("`n_permutations` must be non-negative.")
    if not 0 <= pvalue_cutoff <= 1:
        raise ValueError("`pvalue_cutoff` must be between 0 and 1.")
    if aggregate not in VALID_AGGREGATES:
        allowed = ", ".join(sorted(VALID_AGGREGATES))
        raise ValueError(f"`aggregate` must be one of: {allowed}.")
    if not 0 <= trim < 0.5:
        raise ValueError("`trim` must be in [0, 0.5).")
    if not 0 < clip_quantile <= 1:
        raise ValueError("`clip_quantile` must be in (0, 1].")
    if distance_decay is not None and distance_decay <= 0:
        raise ValueError("`distance_decay` must be positive.")
    if cofactor_kh <= 0:
        raise ValueError("`cofactor_kh` must be positive.")
    if cofactor_hill <= 0:
        raise ValueError("`cofactor_hill` must be positive.")
    if score_method not in VALID_SCORE_METHODS:
        allowed = ", ".join(sorted(VALID_SCORE_METHODS))
        raise ValueError(f"`score_method` must be one of: {allowed}.")
    if complex_aggregate not in VALID_COMPLEX_AGGREGATES:
        allowed = ", ".join(sorted(VALID_COMPLEX_AGGREGATES))
        raise ValueError(f"`complex_aggregate` must be one of: {allowed}.")
    if array_backend is not None and str(array_backend).lower() not in VALID_ARRAY_BACKENDS:
        allowed = ", ".join(sorted(VALID_ARRAY_BACKENDS))
        raise ValueError(f"`array_backend` must be one of: {allowed}.")
    if n_jobs == 0:
        raise ValueError("`n_jobs` must be non-zero.")
    if downsample_per_group is not None and downsample_per_group < 1:
        raise ValueError("`downsample_per_group` must be positive.")
    if downsample_repeats < 1:
        raise ValueError("`downsample_repeats` must be positive.")
    if de_method not in VALID_DE_METHODS:
        allowed = ", ".join(sorted(VALID_DE_METHODS))
        raise ValueError(f"`de_method` must be one of: {allowed}.")
    if not 0 <= de_min_pct <= 1:
        raise ValueError("`de_min_pct` must be between 0 and 1.")
    if de_min_logfc < 0:
        raise ValueError("`de_min_logfc` must be non-negative.")
    if de_pvalue_cutoff is not None and not 0 <= de_pvalue_cutoff <= 1:
        raise ValueError("`de_pvalue_cutoff` must be between 0 and 1.")
    if de_pvalue_cutoff is not None and de_method != "wilcoxon":
        raise ValueError("`de_pvalue_cutoff` requires `de_method='wilcoxon'`.")


def compute_pathway_communication(result: CCCResult, *, significant_only: bool = False) -> pd.DataFrame:
    """Aggregate LR interactions into pathway-level source-target communication."""

    df = result.significant() if significant_only else result.interactions
    return (
        df.groupby(["source", "target", "pathway"], observed=True)
        .agg(prob=("prob", "sum"), count=("prob", "size"), mean_prob=("prob", "mean"))
        .sort_values("prob", ascending=False)
        .reset_index()
    )


def identify_overexpressed_genes(
    adata,
    groupby: str,
    *,
    layer: str | None = None,
    use_raw: bool = False,
    gene_symbols_key: str | None = None,
    method: str = "mean",
    min_pct: float = 0.1,
    min_logfc: float = 0.1,
    pvalue_cutoff: float | None = None,
    pseudocount: float = 1e-9,
    candidate_genes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Identify group-enriched genes for CellChat-style LR gating.

    The default `method="mean"` is an effect-size gate intended for fast CCC
    prefiltering. `method="wilcoxon"` adds one-vs-rest Mann-Whitney p-values
    and Benjamini-Hochberg adjusted p-values.
    """

    if groupby not in adata.obs:
        raise KeyError(f"`{groupby}` is not present in adata.obs.")
    if layer is not None and use_raw:
        raise ValueError("Pass either `layer` or `use_raw=True`, not both.")
    if method not in VALID_DE_METHODS:
        allowed = ", ".join(sorted(VALID_DE_METHODS))
        raise ValueError(f"`method` must be one of: {allowed}.")
    if not 0 <= min_pct <= 1:
        raise ValueError("`min_pct` must be between 0 and 1.")
    if min_logfc < 0:
        raise ValueError("`min_logfc` must be non-negative.")
    if pvalue_cutoff is not None and not 0 <= pvalue_cutoff <= 1:
        raise ValueError("`pvalue_cutoff` must be between 0 and 1.")
    if pvalue_cutoff is not None and method != "wilcoxon":
        raise ValueError("`pvalue_cutoff` requires `method='wilcoxon'`.")

    x, var_names = _get_matrix(adata, layer=layer, use_raw=use_raw, gene_symbols_key=gene_symbols_key)
    gene_lookup = {gene.upper(): gene for gene in var_names.astype(str)}
    if candidate_genes is None:
        genes = list(var_names.astype(str))
    else:
        genes = [gene_lookup[str(gene).upper()] for gene in candidate_genes if str(gene).upper() in gene_lookup]
        genes = list(dict.fromkeys(genes))
    if not genes:
        return _empty_markers()

    groups = [str(x) for x in pd.Index(adata.obs[groupby].astype(str).unique()).sort_values()]
    labels = adata.obs[groupby].astype(str).to_numpy()
    gene_indices = [var_names.get_loc(gene) for gene in genes]
    rows = []
    for group in groups:
        in_mask = labels == group
        out_mask = ~in_mask
        if not in_mask.any() or not out_mask.any():
            continue
        for gene, idx in zip(genes, gene_indices):
            in_values = _matrix_column(x, idx, in_mask)
            out_values = _matrix_column(x, idx, out_mask)
            mean_in = float(in_values.mean()) if in_values.size else 0.0
            mean_out = float(out_values.mean()) if out_values.size else 0.0
            pct_in = float(np.mean(in_values > 0)) if in_values.size else 0.0
            pct_out = float(np.mean(out_values > 0)) if out_values.size else 0.0
            logfc = float(np.log2((mean_in + pseudocount) / (mean_out + pseudocount)))
            pvalue = np.nan
            if method == "wilcoxon":
                pvalue = _mannwhitney_greater(in_values, out_values)
            rows.append(
                {
                    "group": group,
                    "gene": gene,
                    "mean_in": mean_in,
                    "mean_out": mean_out,
                    "pct_in": pct_in,
                    "pct_out": pct_out,
                    "logfc": logfc,
                    "pvalue": pvalue,
                }
            )

    markers = pd.DataFrame(rows)
    if markers.empty:
        return _empty_markers()
    if method == "wilcoxon":
        markers["padj"] = markers.groupby("group", observed=True)["pvalue"].transform(_bh_adjust)
    else:
        markers["padj"] = np.nan
    keep = (markers["pct_in"] >= min_pct) & (markers["logfc"] >= min_logfc) & (markers["mean_in"] > markers["mean_out"])
    if pvalue_cutoff is not None:
        p_col = "padj" if method == "wilcoxon" else "pvalue"
        keep &= markers[p_col].fillna(1.0) <= pvalue_cutoff
    markers["overexpressed"] = keep
    return markers.sort_values(["group", "overexpressed", "logfc"], ascending=[True, False, False]).reset_index(drop=True)


def identify_overexpressed_interactions(
    lr_table: CellChatDB | pd.DataFrame,
    markers: pd.DataFrame | dict[str, Iterable[str]],
    groups: Iterable[str],
) -> pd.DataFrame:
    """Return source-target LR pairs passing the overexpressed gene gate."""

    lr = lr_table.interactions if isinstance(lr_table, CellChatDB) else normalize_lr_table(lr_table)
    groups = [str(group) for group in groups]
    marker_sets = _overexpressed_gene_sets(markers, groups)
    rows = []
    for row in lr.itertuples(index=False):
        ligand = str(row.ligand)
        receptor = str(row.receptor)
        for source in groups:
            if not _complex_in_gene_set(ligand, marker_sets[source]):
                continue
            for target in groups:
                if _complex_in_gene_set(receptor, marker_sets[target]):
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "ligand": ligand,
                            "receptor": receptor,
                            "pathway": getattr(row, "pathway", "unknown"),
                            "annotation": getattr(row, "annotation", ""),
                        }
                    )
    return pd.DataFrame(rows, columns=["source", "target", "ligand", "receptor", "pathway", "annotation"])


def _empty_markers() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "group",
            "gene",
            "mean_in",
            "mean_out",
            "pct_in",
            "pct_out",
            "logfc",
            "pvalue",
            "padj",
            "overexpressed",
        ]
    )


def _matrix_column(x, idx: int, mask: np.ndarray) -> np.ndarray:
    col = x[mask, idx]
    if sparse.issparse(col):
        return np.asarray(col.toarray()).ravel()
    return np.asarray(col).ravel()


def _mannwhitney_greater(in_values: np.ndarray, out_values: np.ndarray) -> float:
    if in_values.size == 0 or out_values.size == 0:
        return 1.0
    if np.allclose(in_values, out_values[0]) and np.allclose(out_values, out_values[0]):
        return 1.0
    try:
        return float(stats.mannwhitneyu(in_values, out_values, alternative="greater").pvalue)
    except ValueError:
        return 1.0


def _bh_adjust(pvalues: pd.Series) -> np.ndarray:
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    finite_idx = np.flatnonzero(finite)
    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out[finite_idx[order]] = np.minimum(adjusted, 1.0)
    return out


def _compute_rows_for_adata(
    adata,
    groupby: str,
    groups: list[str],
    lr: pd.DataFrame,
    *,
    layer: str | None,
    use_raw: bool,
    min_pct: float,
    min_expr: float,
    aggregate: str,
    trim: float,
    clip_quantile: float,
    gene_symbols_key: str | None,
    gene_lookup: dict[str, str],
    expression_genes: Iterable[str] | None,
    scale_expression_by_max: bool,
    overexpressed_genes: dict[str, set[str]] | None,
    population_size: bool,
    spatial_key: str | None,
    distance_decay: float | None,
    cofactor_adjust: bool,
    cofactor_kh: float,
    cofactor_hill: float,
    score_method: str,
    complex_aggregate: str,
    n_jobs: int,
    array_backend: str,
    n_permutations: int,
    random_state: int | None,
) -> pd.DataFrame:
    expr_mean, expr_pct = _group_expression(
        adata,
        groupby,
        groups,
        layer=layer,
        use_raw=use_raw,
        gene_symbols_key=gene_symbols_key,
        aggregate=aggregate,
        trim=trim,
        clip_quantile=clip_quantile,
        genes=expression_genes,
        scale_by_max=scale_expression_by_max,
        array_backend=array_backend,
    )
    group_weights = _group_weights(adata, groupby, groups) if population_size else None
    pair_weights = _spatial_pair_weights(adata, groupby, groups, spatial_key=spatial_key, distance_decay=distance_decay)
    rows = _score_lr_table_vectorized(
        lr,
        expr_mean,
        expr_pct,
        groups,
        min_pct=min_pct,
        min_expr=min_expr,
        gene_lookup=gene_lookup,
        overexpressed_genes=overexpressed_genes,
        score_method=score_method,
        complex_aggregate=complex_aggregate,
        group_weights=group_weights,
        pair_weights=pair_weights,
        cofactor_adjust=cofactor_adjust,
        cofactor_kh=cofactor_kh,
        cofactor_hill=cofactor_hill,
        array_backend=array_backend,
    )
    if rows.empty:
        return _empty_interactions()
    if n_permutations > 0:
        rows["pvalue"] = _permutation_pvalues(
            adata,
            groupby,
            groups,
            lr,
            rows,
            layer=layer,
            use_raw=use_raw,
            min_pct=min_pct,
            min_expr=min_expr,
            aggregate=aggregate,
            trim=trim,
            clip_quantile=clip_quantile,
            gene_symbols_key=gene_symbols_key,
            gene_lookup=gene_lookup,
            expression_genes=expression_genes,
            scale_expression_by_max=scale_expression_by_max,
            overexpressed_genes=overexpressed_genes,
            population_size=population_size,
            spatial_key=spatial_key,
            distance_decay=distance_decay,
            cofactor_adjust=cofactor_adjust,
            cofactor_kh=cofactor_kh,
            cofactor_hill=cofactor_hill,
            score_method=score_method,
            complex_aggregate=complex_aggregate,
            array_backend=array_backend,
            n_jobs=n_jobs,
            n_permutations=n_permutations,
            random_state=random_state,
        )
    else:
        rows["pvalue"] = np.nan
    return rows


def _repeated_downsample_rows(
    adata,
    groupby: str,
    groups: list[str],
    lr: pd.DataFrame,
    *,
    layer: str | None,
    use_raw: bool,
    downsample_per_group: int,
    downsample_repeats: int,
    random_state: int | None,
    min_pct: float,
    min_expr: float,
    aggregate: str,
    trim: float,
    clip_quantile: float,
    gene_symbols_key: str | None,
    gene_lookup: dict[str, str],
    expression_genes: Iterable[str] | None,
    scale_expression_by_max: bool,
    overexpressed_genes: dict[str, set[str]] | None,
    population_size: bool,
    spatial_key: str | None,
    distance_decay: float | None,
    cofactor_adjust: bool,
    cofactor_kh: float,
    cofactor_hill: float,
    score_method: str,
    complex_aggregate: str,
    n_jobs: int,
    array_backend: str,
    n_permutations: int,
) -> pd.DataFrame:
    if aggregate == "mean" and n_permutations == 0 and spatial_key is None and not population_size:
        try:
            return _repeated_downsample_rows_cpu_batched(
                adata,
                groupby,
                groups,
                lr,
                layer=layer,
                use_raw=use_raw,
                downsample_per_group=downsample_per_group,
                downsample_repeats=downsample_repeats,
                random_state=random_state,
                min_pct=min_pct,
                min_expr=min_expr,
                aggregate=aggregate,
                clip_quantile=clip_quantile,
                gene_symbols_key=gene_symbols_key,
                gene_lookup=gene_lookup,
                expression_genes=expression_genes,
                scale_expression_by_max=scale_expression_by_max,
                overexpressed_genes=overexpressed_genes,
                cofactor_adjust=cofactor_adjust,
                cofactor_kh=cofactor_kh,
                cofactor_hill=cofactor_hill,
                score_method=score_method,
                complex_aggregate=complex_aggregate,
            )
        except Exception as exc:
            warnings.warn(f"CPU batched sketches failed; falling back to per-sketch loop. Reason: {exc}", RuntimeWarning, stacklevel=2)

    rng = np.random.default_rng(random_state)
    sampled = []
    for repeat in range(downsample_repeats):
        sketch = _downsample_adata_by_group(adata, groupby, groups, n_per_group=downsample_per_group, rng=rng)
        rows = _compute_rows_for_adata(
            sketch,
            groupby,
            groups,
            lr,
            layer=layer,
            use_raw=use_raw,
            min_pct=min_pct,
            min_expr=min_expr,
            aggregate=aggregate,
            trim=trim,
            clip_quantile=clip_quantile,
            gene_symbols_key=gene_symbols_key,
            gene_lookup=gene_lookup,
            expression_genes=expression_genes,
            scale_expression_by_max=scale_expression_by_max,
            overexpressed_genes=overexpressed_genes,
            population_size=population_size,
            spatial_key=spatial_key,
            distance_decay=distance_decay,
            cofactor_adjust=cofactor_adjust,
            cofactor_kh=cofactor_kh,
            cofactor_hill=cofactor_hill,
            score_method=score_method,
            complex_aggregate=complex_aggregate,
            n_jobs=n_jobs,
            array_backend=array_backend,
            n_permutations=n_permutations,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        if rows.empty:
            continue
        rows = rows.copy()
        rows["_repeat"] = repeat
        sampled.append(rows)

    if not sampled:
        out = _empty_interactions()
        out["prob_std"] = pd.Series(dtype=float)
        out["stability"] = pd.Series(dtype=float)
        out["sketch_repeats"] = pd.Series(dtype=int)
        out["sketch_present"] = pd.Series(dtype=int)
        return out
    frame = pd.concat(sampled, ignore_index=True)
    return _aggregate_downsample_repeats(frame, repeats=downsample_repeats, include_pvalue=n_permutations > 0)


def _downsample_adata_by_group(adata, groupby: str, groups: list[str], *, n_per_group: int, rng: np.random.Generator):
    labels = adata.obs[groupby].astype(str).to_numpy()
    keep = []
    for group in groups:
        idx = np.flatnonzero(labels == group)
        if idx.size > n_per_group:
            idx = rng.choice(idx, size=n_per_group, replace=False)
        keep.append(idx)
    indices = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)
    return adata[indices].copy()


def _repeated_downsample_rows_cpu_batched(
    adata,
    groupby: str,
    groups: list[str],
    lr: pd.DataFrame,
    *,
    layer: str | None,
    use_raw: bool,
    downsample_per_group: int,
    downsample_repeats: int,
    random_state: int | None,
    min_pct: float,
    min_expr: float,
    aggregate: str,
    clip_quantile: float,
    gene_symbols_key: str | None,
    gene_lookup: dict[str, str],
    expression_genes: Iterable[str] | None,
    scale_expression_by_max: bool,
    overexpressed_genes: dict[str, set[str]] | None,
    cofactor_adjust: bool,
    cofactor_kh: float,
    cofactor_hill: float,
    score_method: str,
    complex_aggregate: str,
) -> pd.DataFrame:
    if aggregate != "mean":  # pragma: no cover - guarded by caller
        raise NotImplementedError("CPU batched sketches currently support aggregate='mean'.")

    x, var_names = _get_matrix(adata, layer=layer, use_raw=use_raw, gene_symbols_key=gene_symbols_key)
    x, var_names = _subset_matrix_columns(x, var_names, expression_genes)
    if sparse.issparse(x):
        x_for_mean = x.tocsr().astype(float, copy=False)
        x_positive = x_for_mean.copy()
        x_positive.data = (x_positive.data > 0).astype(float, copy=False)
        x_for_max = x_for_mean
    else:
        x_for_mean = np.asarray(x, dtype=float)
        x_positive = (x_for_mean > 0).astype(float)
        x_for_max = x_for_mean

    rng = np.random.default_rng(random_state)
    labels = adata.obs[groupby].astype(str).to_numpy()
    n_obs = int(labels.size)
    n_groups = len(groups)
    row_chunks = []
    col_chunks = []
    sketch_scales = []
    for repeat in range(downsample_repeats):
        repeat_indices = []
        for group_idx, group in enumerate(groups):
            idx = np.flatnonzero(labels == group)
            if idx.size > downsample_per_group:
                idx = rng.choice(idx, size=downsample_per_group, replace=False)
            row_id = repeat * n_groups + group_idx
            row_chunks.append(np.full(idx.size, row_id, dtype=np.int32))
            col_chunks.append(idx.astype(np.int32, copy=False))
            repeat_indices.append(idx)
        if scale_expression_by_max:
            selected = np.concatenate(repeat_indices) if repeat_indices else np.array([], dtype=int)
            if selected.size:
                max_value = float(x_for_max[selected].max()) if sparse.issparse(x_for_max) else float(np.nanmax(x_for_max[selected]))
            else:
                max_value = 0.0
            sketch_scales.append(max_value if np.isfinite(max_value) and max_value > 0 else 1.0)
        else:
            sketch_scales.append(1.0)
        rng.integers(0, np.iinfo(np.int32).max)
    if not row_chunks:
        out = _empty_interactions()
        out["prob_std"] = pd.Series(dtype=float)
        out["stability"] = pd.Series(dtype=float)
        out["sketch_repeats"] = pd.Series(dtype=int)
        out["sketch_present"] = pd.Series(dtype=int)
        return out

    row_np = np.concatenate(row_chunks)
    col_np = np.concatenate(col_chunks)
    counts = np.bincount(row_np, minlength=downsample_repeats * n_groups).reshape(downsample_repeats, n_groups).astype(float)
    counts_safe = np.where(counts > 0, counts, 1.0)
    group_matrix = sparse.csr_matrix((np.ones(row_np.size, dtype=float), (row_np, col_np)), shape=(downsample_repeats * n_groups, n_obs))
    means_np = group_matrix @ x_for_mean
    pcts_np = group_matrix @ x_positive
    if sparse.issparse(means_np):
        means_np = means_np.toarray()
    else:
        means_np = np.asarray(means_np, dtype=float)
    if sparse.issparse(pcts_np):
        pcts_np = pcts_np.toarray()
    else:
        pcts_np = np.asarray(pcts_np, dtype=float)

    means_np = means_np.reshape(downsample_repeats, n_groups, len(var_names))
    pcts_np = pcts_np.reshape(downsample_repeats, n_groups, len(var_names))
    means_np = np.where(counts[:, :, None] > 0, means_np / counts_safe[:, :, None], 0.0)
    means_np = means_np / np.asarray(sketch_scales, dtype=float)[:, None, None]
    pcts_np = np.where(counts[:, :, None] > 0, pcts_np / counts_safe[:, :, None], 0.0)

    return _score_lr_batch_summary_frame_numpy(
        lr,
        means_np,
        pcts_np,
        groups,
        var_names=var_names,
        min_pct=min_pct,
        min_expr=min_expr,
        gene_lookup=gene_lookup,
        overexpressed_genes=overexpressed_genes,
        score_method=score_method,
        complex_aggregate=complex_aggregate,
        group_weights=None,
        cofactor_adjust=cofactor_adjust,
        cofactor_kh=cofactor_kh,
        cofactor_hill=cofactor_hill,
        repeats=downsample_repeats,
    )


def _aggregate_downsample_repeats(frame: pd.DataFrame, *, repeats: int, include_pvalue: bool) -> pd.DataFrame:
    key_cols = ["source", "target", "ligand", "receptor", "pathway"]
    group = frame.groupby(key_cols, observed=True, sort=False)
    base = group.agg(
        annotation=("annotation", "first"),
        ligand_expr=("ligand_expr", "mean"),
        receptor_expr=("receptor_expr", "mean"),
        ligand_pct=("ligand_pct", "mean"),
        receptor_pct=("receptor_pct", "mean"),
        prob_sum=("prob", "sum"),
        prob_sq_sum=("prob", lambda values: float(np.square(values).sum())),
        sketch_present=("_repeat", "nunique"),
    ).reset_index()
    base["prob"] = base["prob_sum"] / repeats
    if repeats > 1:
        variance = (base["prob_sq_sum"] - repeats * np.square(base["prob"])) / (repeats - 1)
        base["prob_std"] = np.sqrt(np.maximum(variance, 0.0))
    else:
        base["prob_std"] = 0.0
    base["stability"] = base["sketch_present"] / repeats
    base["sketch_repeats"] = repeats
    if include_pvalue and "pvalue" in frame.columns:
        pvalue_sum = group["pvalue"].sum(min_count=1).reset_index(name="pvalue_sum")
        base = base.merge(pvalue_sum, on=key_cols, how="left")
        base["pvalue"] = (base["pvalue_sum"].fillna(0.0) + (repeats - base["sketch_present"])) / repeats
    else:
        base["pvalue"] = np.nan
    base = base.drop(columns=[col for col in ("prob_sum", "prob_sq_sum", "pvalue_sum") if col in base.columns])
    for col in [col for col in frame.columns if col not in set(base.columns).union({"_repeat", "pvalue"})]:
        if col not in key_cols:
            base[col] = group[col].first().to_numpy()
    return base


def _get_matrix(adata, *, layer: str | None, use_raw: bool, gene_symbols_key: str | None = None):
    if use_raw:
        if adata.raw is None:
            raise ValueError("`use_raw=True` but adata.raw is None.")
        return adata.raw.X, _matrix_var_names(adata.raw.var, pd.Index(adata.raw.var_names), gene_symbols_key=gene_symbols_key)
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"`{layer}` is not present in adata.layers.")
        return adata.layers[layer], _matrix_var_names(adata.var, pd.Index(adata.var_names), gene_symbols_key=gene_symbols_key)
    return adata.X, _matrix_var_names(adata.var, pd.Index(adata.var_names), gene_symbols_key=gene_symbols_key)


def _subset_matrix_columns(x, var_names: pd.Index, genes: Iterable[str] | None):
    if genes is None:
        return x, var_names

    ordered = list(dict.fromkeys(str(gene) for gene in genes))
    if not ordered:
        return x[:, []], pd.Index([], dtype=str)

    name_strings = pd.Index(var_names.astype(str))
    if name_strings.is_unique:
        positions = name_strings.get_indexer(ordered)
    else:
        positions = np.array([np.flatnonzero(name_strings == gene)[0] for gene in ordered], dtype=int)
    if (positions < 0).any():
        missing = [gene for gene, pos in zip(ordered, positions) if pos < 0]
        raise KeyError(f"Requested genes are not present in the expression matrix: {missing[:5]}")
    if len(positions) == len(var_names) and np.array_equal(positions, np.arange(len(var_names))):
        return x, var_names
    return x[:, positions], pd.Index(ordered)


def _scale_matrix_by_max(x):
    if sparse.issparse(x):
        max_value = float(x.max()) if x.nnz else 0.0
        if not np.isfinite(max_value) or max_value <= 0:
            return x
        return x.multiply(1.0 / max_value)
    arr = np.asarray(x, dtype=float)
    max_value = float(np.nanmax(arr)) if arr.size else 0.0
    if not np.isfinite(max_value) or max_value <= 0:
        return arr
    return arr / max_value


def _matrix_var_names(var: pd.DataFrame, fallback: pd.Index, *, gene_symbols_key: str | None) -> pd.Index:
    if gene_symbols_key is None:
        return fallback
    if gene_symbols_key not in var:
        raise KeyError(f"`gene_symbols_key={gene_symbols_key!r}` is not present in adata.var.")
    raw_names = var[gene_symbols_key].astype(object)
    names = pd.Index(raw_names.where(pd.notna(raw_names), "").astype(str))
    if (names == "").any():
        raise ValueError(f"`adata.var[{gene_symbols_key!r}]` contains empty gene names.")
    duplicated = names[names.duplicated()].unique()
    if len(duplicated):
        preview = ", ".join(duplicated[:5].astype(str))
        raise ValueError(f"`adata.var[{gene_symbols_key!r}]` contains duplicated gene names, e.g. {preview}.")
    return names


def _group_expression(
    adata,
    groupby: str,
    groups: list[str],
    *,
    layer: str | None,
    use_raw: bool,
    gene_symbols_key: str | None,
    aggregate: str,
    trim: float,
    clip_quantile: float,
    genes: Iterable[str] | None = None,
    scale_by_max: bool = False,
    array_backend: str = "cpu",
):
    x, var_names = _get_matrix(adata, layer=layer, use_raw=use_raw, gene_symbols_key=gene_symbols_key)
    x, var_names = _subset_matrix_columns(x, var_names, genes)
    if scale_by_max:
        x = _scale_matrix_by_max(x)
    labels = adata.obs[groupby].astype(str).to_numpy()
    return _group_expression_from_labels(x, var_names, labels, groups, aggregate=aggregate, trim=trim, clip_quantile=clip_quantile, array_backend=array_backend)


def _group_expression_from_labels(
    x,
    var_names: pd.Index,
    labels: np.ndarray,
    groups: list[str],
    *,
    aggregate: str,
    trim: float,
    clip_quantile: float = 0.99,
    array_backend: str = "cpu",
):
    clip_caps = _positive_quantile_caps(x, clip_quantile) if aggregate in {"clipped_mean", "gated_mean"} else None
    means = pd.DataFrame(index=groups, columns=var_names, dtype=float)
    pcts = pd.DataFrame(index=groups, columns=var_names, dtype=float)
    for group in groups:
        mask = labels == group
        sub = x[mask]
        if sparse.issparse(sub):
            pct = np.asarray((sub > 0).mean(axis=0)).ravel()
            if aggregate == "mean":
                means.loc[group] = np.asarray(sub.mean(axis=0)).ravel()
                pcts.loc[group] = pct
                continue
            if aggregate == "tri_mean":
                means.loc[group] = _sparse_tri_mean(sub)
                pcts.loc[group] = pct
                continue
            if aggregate == "median":
                means.loc[group] = _sparse_quantiles(sub, [50.0])[0]
                pcts.loc[group] = pct
                continue
            if aggregate == "clipped_mean":
                means.loc[group] = _clipped_sparse_mean(sub, caps=clip_caps)
                pcts.loc[group] = pct
                continue
            if aggregate == "gated_mean":
                means.loc[group] = _gated_sparse_mean(sub, caps=clip_caps, pct=pct)
                pcts.loc[group] = pct
                continue
            arr = sub.toarray()
        else:
            arr = np.asarray(sub)
            pct = (arr > 0).mean(axis=0)
        if aggregate == "clipped_mean":
            means.loc[group] = np.minimum(np.asarray(arr, dtype=float), clip_caps[None, :]).mean(axis=0)
            pcts.loc[group] = pct
            continue
        if aggregate == "gated_mean":
            clipped = np.minimum(np.asarray(arr, dtype=float), clip_caps[None, :])
            means.loc[group] = _gated_mean_from_all_mean_and_pct(clipped.mean(axis=0), pct)
            pcts.loc[group] = pct
            continue
        means.loc[group] = _aggregate_matrix(arr, aggregate=aggregate, trim=trim, clip_quantile=clip_quantile)
        pcts.loc[group] = pct
    return means, pcts


def _clipped_sparse_mean(x, *, caps: np.ndarray) -> np.ndarray:
    x_csr = x.tocsr().astype(float, copy=False)
    if x_csr.shape[0] == 0:
        return np.zeros(x_csr.shape[1], dtype=float)
    clipped = x_csr.copy()
    if clipped.nnz:
        clipped.data = np.minimum(clipped.data, caps[clipped.indices])
    return np.asarray(clipped.mean(axis=0)).ravel()


def _gated_sparse_mean(x, *, caps: np.ndarray, pct: np.ndarray) -> np.ndarray:
    return _gated_mean_from_all_mean_and_pct(_clipped_sparse_mean(x, caps=caps), pct)


def _sparse_tri_mean(x) -> np.ndarray:
    q1, q2, q3 = _sparse_quantiles(x, [25.0, 50.0, 75.0])
    return np.asarray((q1 + 2 * q2 + q3) / 4).ravel()


def _sparse_quantiles(x, quantiles: Sequence[float]) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((len(quantiles), x.shape[1]), dtype=float)
    x_csc = x.tocsc().astype(float, copy=False)
    x_csc.eliminate_zeros()
    if x_csc.data.size and not np.isfinite(x_csc.data).all():
        return np.percentile(x.toarray(), quantiles, axis=0)
    out = np.zeros((len(quantiles), x_csc.shape[1]), dtype=float)
    n_rows = int(x_csc.shape[0])
    for col in range(x_csc.shape[1]):
        start, end = x_csc.indptr[col], x_csc.indptr[col + 1]
        values = np.sort(x_csc.data[start:end])
        zero_count = n_rows - len(values)
        negative_count = int(np.searchsorted(values, 0.0, side="left"))
        for row, quantile in enumerate(quantiles):
            position = (n_rows - 1) * float(quantile) / 100.0
            lower = int(np.floor(position))
            upper = int(np.ceil(position))
            lower_value = _sparse_sorted_value(values, lower, zero_count=zero_count, negative_count=negative_count)
            if lower == upper:
                out[row, col] = lower_value
            else:
                upper_value = _sparse_sorted_value(values, upper, zero_count=zero_count, negative_count=negative_count)
                out[row, col] = lower_value + (position - lower) * (upper_value - lower_value)
    return out


def _sparse_sorted_value(values: np.ndarray, index: int, *, zero_count: int, negative_count: int) -> float:
    if index < negative_count:
        return float(values[index])
    zero_end = negative_count + zero_count
    if index < zero_end:
        return 0.0
    return float(values[index - zero_count])


def _gated_mean_from_all_mean_and_pct(all_mean: np.ndarray, pct: np.ndarray) -> np.ndarray:
    all_mean = np.asarray(all_mean, dtype=float)
    pct = np.asarray(pct, dtype=float)
    positive_mean = np.divide(all_mean, pct, out=np.zeros_like(all_mean, dtype=float), where=pct > 0)
    return positive_mean * _tri_mean_detection_weight(pct)


def _tri_mean_detection_weight(pct: np.ndarray) -> np.ndarray:
    pct = np.asarray(pct, dtype=float)
    eps = 1e-12
    return np.select([pct > 0.75 + eps, pct > 0.5 + eps, pct > 0.25 + eps], [1.0, 0.75, 0.25], default=0.0)


def _positive_quantile_caps(x, clip_quantile: float) -> np.ndarray:
    if sparse.issparse(x):
        return _sparse_positive_quantile_caps(x.tocsr().astype(float, copy=False), clip_quantile)
    return _dense_positive_quantile_caps(np.asarray(x, dtype=float), clip_quantile)


def _sparse_positive_quantile_caps(x_csr, clip_quantile: float) -> np.ndarray:
    x_csc = x_csr.tocsc()
    caps = np.zeros(x_csc.shape[1], dtype=float)
    for col in range(x_csc.shape[1]):
        start, end = x_csc.indptr[col], x_csc.indptr[col + 1]
        values = x_csc.data[start:end]
        positive = values[values > 0]
        if positive.size:
            caps[col] = float(np.quantile(positive, clip_quantile))
    return caps


def _dense_positive_quantile_caps(arr: np.ndarray, clip_quantile: float) -> np.ndarray:
    caps = np.zeros(arr.shape[1], dtype=float)
    for col in range(arr.shape[1]):
        positive = arr[:, col][arr[:, col] > 0]
        if positive.size:
            caps[col] = float(np.quantile(positive, clip_quantile))
    return caps


def _group_weights(adata, groupby: str, groups: list[str]) -> dict[str, float]:
    return _group_weights_from_labels(adata.obs[groupby].astype(str).to_numpy(), groups)


def _group_weights_from_labels(labels: np.ndarray, groups: list[str]) -> dict[str, float]:
    counts = pd.Series(labels).value_counts(normalize=True)
    return {group: float(counts.get(group, 0.0)) for group in groups}


def _spatial_pair_weights(
    adata,
    groupby: str,
    groups: list[str],
    *,
    spatial_key: str | None,
    distance_decay: float | None,
) -> dict[tuple[str, str], float] | None:
    if spatial_key is None or distance_decay is None:
        return None
    if spatial_key not in adata.obsm:
        raise KeyError(f"`{spatial_key}` is not present in adata.obsm.")
    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"`adata.obsm[{spatial_key!r}]` must have at least two coordinate columns.")
    labels = adata.obs[groupby].astype(str).to_numpy()
    return _spatial_pair_weights_from_labels(coords, labels, groups, distance_decay=distance_decay)


def _spatial_pair_weights_from_labels(
    coords: np.ndarray,
    labels: np.ndarray,
    groups: list[str],
    *,
    distance_decay: float | None,
) -> dict[tuple[str, str], float] | None:
    if distance_decay is None:
        return None
    frame = pd.DataFrame({"group": labels, "x": coords[:, 0], "y": coords[:, 1]})
    centroids = frame.groupby("group", observed=True)[["x", "y"]].mean().reindex(groups)
    weights = {}
    for source in groups:
        for target in groups:
            if centroids.loc[[source, target]].isna().any().any():
                weights[(source, target)] = 0.0
                continue
            dist = float(np.linalg.norm(centroids.loc[source].to_numpy(dtype=float) - centroids.loc[target].to_numpy(dtype=float)))
            weights[(source, target)] = float(np.exp(-dist / distance_decay))
    return weights


def _aggregate_matrix(arr: np.ndarray, *, aggregate: str, trim: float, clip_quantile: float = 0.99) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(arr.shape[1], dtype=float)
    if aggregate == "mean":
        return np.asarray(arr.mean(axis=0)).ravel()
    if aggregate == "clipped_mean":
        caps = _dense_positive_quantile_caps(np.asarray(arr, dtype=float), clip_quantile)
        return np.minimum(np.asarray(arr, dtype=float), caps[None, :]).mean(axis=0)
    if aggregate == "gated_mean":
        arr = np.asarray(arr, dtype=float)
        caps = _dense_positive_quantile_caps(arr, clip_quantile)
        clipped = np.minimum(arr, caps[None, :])
        return _gated_mean_from_all_mean_and_pct(clipped.mean(axis=0), (arr > 0).mean(axis=0))
    if aggregate == "median":
        return np.asarray(np.median(arr, axis=0)).ravel()
    if aggregate == "tri_mean":
        q1 = np.percentile(arr, 25, axis=0)
        q2 = np.percentile(arr, 50, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        return np.asarray((q1 + 2 * q2 + q3) / 4).ravel()
    if aggregate in {"truncated_mean", "truncatedMean"}:
        return _truncated_mean(arr, trim=trim)
    allowed = ", ".join(sorted(VALID_AGGREGATES))
    raise ValueError(f"`aggregate` must be one of: {allowed}.")


def _truncated_mean(arr: np.ndarray, *, trim: float) -> np.ndarray:
    if trim <= 0:
        return np.asarray(arr.mean(axis=0)).ravel()
    n = arr.shape[0]
    cut = int(np.floor(n * trim))
    if cut == 0 or n - 2 * cut <= 0:
        return np.asarray(arr.mean(axis=0)).ravel()
    ordered = np.sort(np.asarray(arr, dtype=float), axis=0)
    return np.asarray(ordered[cut : n - cut].mean(axis=0)).ravel()


def _filter_lr_to_genes(lr: pd.DataFrame, gene_lookup: dict[str, str]) -> pd.DataFrame:
    keep = []
    for row in lr.itertuples(index=False):
        ligand_genes = _complex_genes(row.ligand)
        receptor_genes = _complex_genes(row.receptor)
        genes = ligand_genes + receptor_genes
        keep.append(bool(ligand_genes) and bool(receptor_genes) and all(g.upper() in gene_lookup for g in genes))
    return lr.loc[keep].reset_index(drop=True)


def _lr_gene_candidates(lr: pd.DataFrame, gene_lookup: dict[str, str]) -> list[str]:
    return _lr_expression_gene_candidates(lr, gene_lookup, include_cofactors=False)


def _lr_expression_gene_candidates(lr: pd.DataFrame, gene_lookup: dict[str, str], *, include_cofactors: bool) -> list[str]:
    genes: list[str] = []
    cofactor_cols = ("agonist_genes", "antagonist_genes", "co_A_receptor_genes", "co_I_receptor_genes")
    for row in lr.itertuples(index=False):
        complexes = [row.ligand, row.receptor]
        if include_cofactors:
            complexes.extend(getattr(row, col, "") for col in cofactor_cols)
        for complex_name in complexes:
            for gene in _complex_genes(complex_name):
                resolved = gene_lookup.get(gene.upper())
                if resolved is not None:
                    genes.append(resolved)
    return list(dict.fromkeys(genes))


@lru_cache(maxsize=4096)
def _complex_genes(name: str) -> list[str]:
    normalized = str(name).replace("+", "_").replace("&", "_").replace(":", "_")
    return [part.strip() for part in normalized.split("_") if part.strip()]


def _overexpressed_gene_sets(markers: pd.DataFrame | dict[str, Iterable[str]], groups: Iterable[str]) -> dict[str, set[str]]:
    groups = [str(group) for group in groups]
    if isinstance(markers, dict):
        return {group: {str(gene).upper() for gene in markers.get(group, [])} for group in groups}
    required = {"group", "gene"}
    missing = required - set(markers.columns)
    if missing:
        raise ValueError(f"`markers` is missing required columns: {sorted(missing)}")
    frame = markers.copy()
    if "overexpressed" in frame.columns:
        frame = frame[frame["overexpressed"].astype(bool)]
    out = {group: set() for group in groups}
    for group, sub in frame.groupby(frame["group"].astype(str), observed=True):
        if group in out:
            out[group] = {str(gene).upper() for gene in sub["gene"]}
    return out


def _complex_in_gene_set(complex_name: str, genes: set[str]) -> bool:
    subunits = _complex_genes(complex_name)
    return bool(subunits) and all(gene.upper() in genes for gene in subunits)


def _complex_overexpressed_matrix(complexes: pd.Series, groups: list[str], markers: dict[str, set[str]]) -> np.ndarray:
    out = np.zeros((len(complexes), len(groups)), dtype=bool)
    for i, complex_name in enumerate(complexes.astype(str)):
        for j, group in enumerate(groups):
            out[i, j] = _complex_in_gene_set(complex_name, markers.get(group, set()))
    return out


def _complex_value(values: pd.Series, complex_name: str, gene_lookup: dict[str, str]) -> float:
    genes = [gene_lookup[g.upper()] for g in _complex_genes(complex_name)]
    return float(values.loc[genes].min())


def _optional_complex_values(values: pd.Series, complex_name: str, gene_lookup: dict[str, str]) -> np.ndarray:
    genes = [gene_lookup[g.upper()] for g in _complex_genes(complex_name) if g.upper() in gene_lookup]
    if not genes:
        return np.array([], dtype=float)
    return values.loc[genes].to_numpy(dtype=float)


def _cofactor_multiplier(
    lr_row,
    source_values: pd.Series,
    target_values: pd.Series,
    gene_lookup: dict[str, str],
    *,
    kh: float,
    hill: float,
) -> float:
    agonist = _optional_complex_values(source_values, getattr(lr_row, "agonist_genes", ""), gene_lookup)
    antagonist = _optional_complex_values(source_values, getattr(lr_row, "antagonist_genes", ""), gene_lookup)
    co_activation = _optional_complex_values(target_values, getattr(lr_row, "co_A_receptor_genes", ""), gene_lookup)
    co_inhibition = _optional_complex_values(target_values, getattr(lr_row, "co_I_receptor_genes", ""), gene_lookup)
    kh_pow = kh**hill

    agonist_factor = float(np.prod(1.0 + np.power(agonist, hill) / (kh_pow + np.power(agonist, hill)))) if agonist.size else 1.0
    antagonist_factor = float(np.prod(kh_pow / (kh_pow + np.power(antagonist, hill)))) if antagonist.size else 1.0
    co_activation_factor = float(np.prod(1.0 + co_activation)) if co_activation.size else 1.0
    co_inhibition_factor = float(np.prod(1.0 + co_inhibition)) if co_inhibition.size else 1.0
    return agonist_factor * antagonist_factor * co_activation_factor / co_inhibition_factor


def _resolve_complex_aggregate(score_method: str, complex_aggregate: str) -> str:
    if complex_aggregate != "auto":
        return complex_aggregate
    return "geometric_mean" if score_method == "cellchat" else "min"


def _resolve_array_backend(array_backend: str | None) -> str:
    requested = str(array_backend or "cpu").lower()
    if requested not in VALID_ARRAY_BACKENDS:
        allowed = ", ".join(sorted(VALID_ARRAY_BACKENDS))
        raise ValueError(f"`array_backend` must be one of: {allowed}.")
    return "cpu"


def _score_lr_table_vectorized(
    lr: pd.DataFrame,
    expr_mean: pd.DataFrame,
    expr_pct: pd.DataFrame,
    groups: list[str],
    *,
    min_pct: float,
    min_expr: float,
    gene_lookup: dict[str, str],
    overexpressed_genes: dict[str, set[str]] | None = None,
    score_method: str,
    complex_aggregate: str,
    group_weights: dict[str, float] | None = None,
    pair_weights: dict[tuple[str, str], float] | None = None,
    cofactor_adjust: bool = False,
    cofactor_kh: float = 0.5,
    cofactor_hill: float = 1.0,
    array_backend: str = "cpu",
) -> pd.DataFrame:
    if lr.empty:
        return _empty_interactions()

    lig_expr = _complex_matrix(expr_mean, lr["ligand"], gene_lookup, aggregate=complex_aggregate)
    rec_expr = _complex_matrix(expr_mean, lr["receptor"], gene_lookup, aggregate=complex_aggregate)
    lig_pct = _complex_matrix(expr_pct, lr["ligand"], gene_lookup, aggregate="min")
    rec_pct = _complex_matrix(expr_pct, lr["receptor"], gene_lookup, aggregate="min")
    rec_expr_for_score = rec_expr

    if cofactor_adjust and score_method == "cellchat":
        co_activation = _cellchat_coreceptor_factor_matrix(lr, expr_mean, groups, gene_lookup, "co_A_receptor_genes")
        co_inhibition = _cellchat_coreceptor_factor_matrix(lr, expr_mean, groups, gene_lookup, "co_I_receptor_genes")
        rec_expr_for_score = rec_expr_for_score * co_activation / co_inhibition

    valid = (lig_expr[:, :, None] > min_expr) & (lig_pct[:, :, None] >= min_pct) & (rec_expr_for_score[:, None, :] > min_expr) & (rec_pct[:, None, :] >= min_pct)
    if overexpressed_genes is not None:
        lig_oe = _complex_overexpressed_matrix(lr["ligand"], groups, overexpressed_genes)
        rec_oe = _complex_overexpressed_matrix(lr["receptor"], groups, overexpressed_genes)
        valid &= lig_oe[:, :, None] & rec_oe[:, None, :]
    lr_signal = lig_expr[:, :, None] * rec_expr_for_score[:, None, :]
    if score_method == "sqrt":
        prob = np.sqrt(lr_signal)
    elif score_method == "cellchat":
        kh_pow = cofactor_kh**cofactor_hill
        signal_pow = np.power(lr_signal, cofactor_hill)
        prob = signal_pow / (kh_pow + signal_pow)
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"Unknown score_method: {score_method}")

    if cofactor_adjust and score_method == "cellchat":
        agonist = _cellchat_hill_cofactor_factor_matrix(lr, expr_mean, groups, gene_lookup, "agonist_genes", kh=cofactor_kh, hill=cofactor_hill, mode="agonist")
        antagonist = _cellchat_hill_cofactor_factor_matrix(lr, expr_mean, groups, gene_lookup, "antagonist_genes", kh=cofactor_kh, hill=cofactor_hill, mode="antagonist")
        prob = prob * agonist[:, :, None] * agonist[:, None, :] * antagonist[:, :, None] * antagonist[:, None, :]
    elif cofactor_adjust:
        prob = prob * _cofactor_matrix(lr, expr_mean, groups, gene_lookup, kh=cofactor_kh, hill=cofactor_hill)
    if group_weights is not None:
        weights = np.array([group_weights[group] for group in groups], dtype=float)
        pair_group_weights = weights[None, :, None] * weights[None, None, :]
        if score_method == "cellchat":
            prob = prob * pair_group_weights
        else:
            prob = prob * np.sqrt(pair_group_weights)
    if pair_weights is not None:
        pair = np.array([[pair_weights[(source, target)] for target in groups] for source in groups], dtype=float)
        prob = prob * pair[None, :, :]

    valid &= np.isfinite(prob) & (prob > 0)
    lr_idx, source_idx, target_idx = np.nonzero(valid)
    prob_np = prob
    if lr_idx.size == 0:
        return _empty_interactions()

    group_arr = np.asarray(groups, dtype=object)
    out = pd.DataFrame(
        {
            "source": group_arr[source_idx],
            "target": group_arr[target_idx],
            "ligand": lr["ligand"].to_numpy(dtype=object)[lr_idx],
            "receptor": lr["receptor"].to_numpy(dtype=object)[lr_idx],
            "pathway": lr["pathway"].to_numpy(dtype=object)[lr_idx],
            "annotation": lr["annotation"].to_numpy(dtype=object)[lr_idx] if "annotation" in lr.columns else "",
            "ligand_expr": lig_expr[lr_idx, source_idx],
            "receptor_expr": rec_expr[lr_idx, target_idx],
            "ligand_pct": lig_pct[lr_idx, source_idx],
            "receptor_pct": rec_pct[lr_idx, target_idx],
            "prob": prob_np[lr_idx, source_idx, target_idx],
        }
    )
    for col in _lr_metadata_columns(lr):
        out[col] = lr[col].to_numpy(dtype=object)[lr_idx]
    return out


def _score_lr_batch_summary_frame_numpy(
    lr: pd.DataFrame,
    means_np: np.ndarray,
    pcts_np: np.ndarray,
    groups: list[str],
    *,
    var_names: pd.Index,
    min_pct: float,
    min_expr: float,
    gene_lookup: dict[str, str],
    overexpressed_genes: dict[str, set[str]] | None,
    score_method: str,
    complex_aggregate: str,
    group_weights: dict[str, float] | None,
    cofactor_adjust: bool,
    cofactor_kh: float,
    cofactor_hill: float,
    repeats: int,
    lr_chunk_size: int = 256,
) -> pd.DataFrame:
    if overexpressed_genes is not None:
        raise NotImplementedError("CPU batched sketches do not yet support DE-gated interactions.")
    if cofactor_adjust:
        raise NotImplementedError("CPU batched sketches do not yet support cofactor adjustment.")

    frames = []
    for start in range(0, len(lr), lr_chunk_size):
        chunk = lr.iloc[start : start + lr_chunk_size]
        if chunk.empty:
            continue
        frame = _score_lr_batch_summary_chunk_numpy(
            chunk,
            means_np,
            pcts_np,
            groups,
            var_names=var_names,
            min_pct=min_pct,
            min_expr=min_expr,
            gene_lookup=gene_lookup,
            score_method=score_method,
            complex_aggregate=complex_aggregate,
            group_weights=group_weights,
            cofactor_kh=cofactor_kh,
            cofactor_hill=cofactor_hill,
            repeats=repeats,
        )
        if not frame.empty:
            frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    out = _empty_interactions()
    out["prob_std"] = pd.Series(dtype=float)
    out["stability"] = pd.Series(dtype=float)
    out["sketch_repeats"] = pd.Series(dtype=int)
    out["sketch_present"] = pd.Series(dtype=int)
    return out


def _score_lr_batch_summary_chunk_numpy(
    lr: pd.DataFrame,
    means_np: np.ndarray,
    pcts_np: np.ndarray,
    groups: list[str],
    *,
    var_names: pd.Index,
    min_pct: float,
    min_expr: float,
    gene_lookup: dict[str, str],
    score_method: str,
    complex_aggregate: str,
    group_weights: dict[str, float] | None,
    cofactor_kh: float,
    cofactor_hill: float,
    repeats: int,
) -> pd.DataFrame:
    lig_expr = _complex_batch_matrix_numpy(means_np, lr["ligand"], gene_lookup, var_names, aggregate=complex_aggregate)
    rec_expr = _complex_batch_matrix_numpy(means_np, lr["receptor"], gene_lookup, var_names, aggregate=complex_aggregate)
    lig_pct = _complex_batch_matrix_numpy(pcts_np, lr["ligand"], gene_lookup, var_names, aggregate="min")
    rec_pct = _complex_batch_matrix_numpy(pcts_np, lr["receptor"], gene_lookup, var_names, aggregate="min")

    valid = (
        (lig_expr[:, :, :, None] > min_expr)
        & (lig_pct[:, :, :, None] >= min_pct)
        & (rec_expr[:, :, None, :] > min_expr)
        & (rec_pct[:, :, None, :] >= min_pct)
    )
    lr_signal = lig_expr[:, :, :, None] * rec_expr[:, :, None, :]
    if score_method == "sqrt":
        prob = np.sqrt(lr_signal)
    elif score_method == "cellchat":
        kh_pow = cofactor_kh**cofactor_hill
        signal_pow = np.power(lr_signal, cofactor_hill)
        prob = signal_pow / (kh_pow + signal_pow)
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"Unknown score_method: {score_method}")

    if group_weights is not None:
        weights = np.array([group_weights[group] for group in groups], dtype=float)
        pair_group_weights = weights[None, None, :, None] * weights[None, None, None, :]
        prob = prob * pair_group_weights if score_method == "cellchat" else prob * np.sqrt(pair_group_weights)

    valid &= np.isfinite(prob) & (prob > 0)
    present = valid.sum(axis=0)
    keep = present > 0
    lr_idx, source_idx, target_idx = np.nonzero(keep)
    if lr_idx.size == 0:
        return _empty_interactions()

    valid_float = valid.astype(float)
    prob_present = np.where(valid, prob, 0.0)
    prob_mean = prob_present.sum(axis=0) / repeats
    prob_sq_sum = np.square(prob_present).sum(axis=0)
    if repeats > 1:
        variance = (prob_sq_sum - repeats * np.square(prob_mean)) / (repeats - 1)
        prob_std = np.sqrt(np.maximum(variance, 0.0))
    else:
        prob_std = np.zeros_like(prob_mean)

    present_safe = np.maximum(present.astype(float), 1.0)
    lig_expr_mean = (lig_expr[:, :, :, None] * valid_float).sum(axis=0) / present_safe
    rec_expr_mean = (rec_expr[:, :, None, :] * valid_float).sum(axis=0) / present_safe
    lig_pct_mean = (lig_pct[:, :, :, None] * valid_float).sum(axis=0) / present_safe
    rec_pct_mean = (rec_pct[:, :, None, :] * valid_float).sum(axis=0) / present_safe

    group_arr = np.asarray(groups, dtype=object)
    out = pd.DataFrame(
        {
            "source": group_arr[source_idx],
            "target": group_arr[target_idx],
            "ligand": lr["ligand"].to_numpy(dtype=object)[lr_idx],
            "receptor": lr["receptor"].to_numpy(dtype=object)[lr_idx],
            "pathway": lr["pathway"].to_numpy(dtype=object)[lr_idx],
            "annotation": lr["annotation"].to_numpy(dtype=object)[lr_idx] if "annotation" in lr.columns else "",
            "ligand_expr": lig_expr_mean[lr_idx, source_idx, target_idx],
            "receptor_expr": rec_expr_mean[lr_idx, source_idx, target_idx],
            "ligand_pct": lig_pct_mean[lr_idx, source_idx, target_idx],
            "receptor_pct": rec_pct_mean[lr_idx, source_idx, target_idx],
            "prob": prob_mean[lr_idx, source_idx, target_idx],
            "prob_std": prob_std[lr_idx, source_idx, target_idx],
            "stability": present[lr_idx, source_idx, target_idx] / repeats,
            "sketch_repeats": repeats,
            "sketch_present": present[lr_idx, source_idx, target_idx].astype(int),
            "pvalue": np.nan,
        }
    )
    for col in _lr_metadata_columns(lr):
        out[col] = lr[col].to_numpy(dtype=object)[lr_idx]
    return out


def _complex_batch_matrix_numpy(values_np: np.ndarray, complexes: pd.Series, gene_lookup: dict[str, str], var_names: pd.Index, *, aggregate: str) -> np.ndarray:
    var_index = {str(gene): idx for idx, gene in enumerate(var_names.astype(str))}
    unique_complexes = pd.Index(complexes.astype(str).unique())
    cache = {}
    for complex_name in unique_complexes:
        positions = [var_index[gene_lookup[gene.upper()]] for gene in _complex_genes(str(complex_name))]
        arr = values_np[:, :, positions]
        if len(positions) == 1:
            cache[str(complex_name)] = arr[:, :, 0]
        elif aggregate == "min":
            cache[str(complex_name)] = arr.min(axis=2)
        elif aggregate == "geometric_mean":
            positive = arr > 0
            keep = positive.all(axis=2)
            safe = np.where(positive, arr, 1.0)
            cache[str(complex_name)] = np.where(keep, np.exp(np.log(safe).mean(axis=2)), 0.0)
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"Unknown complex aggregate: {aggregate}")
    return np.stack([cache[str(name)] for name in complexes.astype(str)], axis=1)


def _complex_matrix(values: pd.DataFrame, complexes: pd.Series, gene_lookup: dict[str, str], *, aggregate: str) -> np.ndarray:
    unique_complexes = pd.Index(complexes.astype(str).unique())
    cache: dict[str, np.ndarray] = {}
    for complex_name in unique_complexes:
        genes = [gene_lookup[gene.upper()] for gene in _complex_genes(complex_name)]
        arr = values.loc[:, genes].to_numpy(dtype=float)
        if arr.shape[1] == 1:
            cache[complex_name] = arr[:, 0]
        elif aggregate == "min":
            cache[complex_name] = arr.min(axis=1)
        elif aggregate == "geometric_mean":
            positive = arr > 0
            geo = np.zeros(arr.shape[0], dtype=float)
            keep = positive.all(axis=1)
            if keep.any():
                geo[keep] = np.exp(np.log(arr[keep]).mean(axis=1))
            cache[complex_name] = geo
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"Unknown complex aggregate: {aggregate}")
    return np.vstack([cache[str(name)] for name in complexes.astype(str)])


def _cellchat_coreceptor_factor_matrix(lr: pd.DataFrame, expr_mean: pd.DataFrame, groups: list[str], gene_lookup: dict[str, str], column: str) -> np.ndarray:
    factors = np.ones((len(lr), len(groups)), dtype=float)
    if column not in lr.columns:
        return factors
    for i, complex_name in enumerate(lr[column].fillna("").astype(str)):
        for j, group in enumerate(groups):
            values = _optional_complex_values(expr_mean.loc[group], complex_name, gene_lookup)
            if values.size:
                factors[i, j] = float(np.prod(1.0 + values))
    return factors


def _cellchat_hill_cofactor_factor_matrix(
    lr: pd.DataFrame,
    expr_mean: pd.DataFrame,
    groups: list[str],
    gene_lookup: dict[str, str],
    column: str,
    *,
    kh: float,
    hill: float,
    mode: str,
) -> np.ndarray:
    factors = np.ones((len(lr), len(groups)), dtype=float)
    if column not in lr.columns:
        return factors
    kh_pow = kh**hill
    for i, complex_name in enumerate(lr[column].fillna("").astype(str)):
        for j, group in enumerate(groups):
            values = _optional_complex_values(expr_mean.loc[group], complex_name, gene_lookup)
            if not values.size:
                continue
            value_pow = np.power(values, hill)
            if mode == "agonist":
                factors[i, j] = float(np.prod(1.0 + value_pow / (kh_pow + value_pow)))
            elif mode == "antagonist":
                factors[i, j] = float(np.prod(kh_pow / (kh_pow + value_pow)))
            else:  # pragma: no cover - internal misuse
                raise ValueError("`mode` must be 'agonist' or 'antagonist'.")
    return factors


def _cofactor_matrix(lr: pd.DataFrame, expr_mean: pd.DataFrame, groups: list[str], gene_lookup: dict[str, str], *, kh: float, hill: float) -> np.ndarray:
    factors = np.ones((len(lr), len(groups), len(groups)), dtype=float)
    for i, lr_row in enumerate(lr.itertuples(index=False)):
        for s, source in enumerate(groups):
            source_values = expr_mean.loc[source]
            for t, target in enumerate(groups):
                factors[i, s, t] = _cofactor_multiplier(lr_row, source_values, expr_mean.loc[target], gene_lookup, kh=kh, hill=hill)
    return factors


def _lr_metadata_columns(lr: pd.DataFrame) -> list[str]:
    core = {"ligand", "receptor", "pathway", "annotation"}
    return [col for col in lr.columns if col not in core and pd.api.types.is_string_dtype(lr[col])]


def _score_lr_table(
    lr: pd.DataFrame,
    expr_mean: pd.DataFrame,
    expr_pct: pd.DataFrame,
    groups: list[str],
    *,
    min_pct: float,
    min_expr: float,
    gene_lookup: dict[str, str],
    group_weights: dict[str, float] | None = None,
    pair_weights: dict[tuple[str, str], float] | None = None,
    cofactor_adjust: bool = False,
    cofactor_kh: float = 0.5,
    cofactor_hill: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lr_row in lr.itertuples(index=False):
        for source in groups:
            ligand_expr = _complex_value(expr_mean.loc[source], lr_row.ligand, gene_lookup)
            ligand_pct = _complex_value(expr_pct.loc[source], lr_row.ligand, gene_lookup)
            if ligand_expr <= min_expr or ligand_pct < min_pct:
                continue
            for target in groups:
                receptor_expr = _complex_value(expr_mean.loc[target], lr_row.receptor, gene_lookup)
                receptor_pct = _complex_value(expr_pct.loc[target], lr_row.receptor, gene_lookup)
                if receptor_expr <= min_expr or receptor_pct < min_pct:
                    continue
                prob = float(np.sqrt(ligand_expr * receptor_expr))
                if cofactor_adjust:
                    prob *= _cofactor_multiplier(lr_row, expr_mean.loc[source], expr_mean.loc[target], gene_lookup, kh=cofactor_kh, hill=cofactor_hill)
                if group_weights is not None:
                    prob *= float(np.sqrt(group_weights[source] * group_weights[target]))
                if pair_weights is not None:
                    prob *= pair_weights[(source, target)]
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "ligand": lr_row.ligand,
                        "receptor": lr_row.receptor,
                        "pathway": lr_row.pathway,
                        "annotation": getattr(lr_row, "annotation", ""),
                        "ligand_expr": ligand_expr,
                        "receptor_expr": receptor_expr,
                        "ligand_pct": ligand_pct,
                        "receptor_pct": receptor_pct,
                        "prob": prob,
                    }
                )
    return pd.DataFrame(rows) if rows else _empty_interactions()


def _empty_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source",
            "target",
            "ligand",
            "receptor",
            "pathway",
            "annotation",
            "ligand_expr",
            "receptor_expr",
            "ligand_pct",
            "receptor_pct",
            "prob",
            "pvalue",
        ]
    )


def _permutation_pvalues(
    adata,
    groupby: str,
    groups: list[str],
    lr: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    layer: str | None,
    use_raw: bool,
    min_pct: float,
    min_expr: float,
    aggregate: str,
    trim: float,
    clip_quantile: float,
    gene_symbols_key: str | None,
    gene_lookup: dict[str, str],
    expression_genes: Iterable[str] | None,
    scale_expression_by_max: bool,
    overexpressed_genes: dict[str, set[str]] | None,
    population_size: bool,
    spatial_key: str | None,
    distance_decay: float | None,
    cofactor_adjust: bool,
    cofactor_kh: float,
    cofactor_hill: float,
    score_method: str,
    complex_aggregate: str,
    array_backend: str,
    n_jobs: int,
    n_permutations: int,
    random_state: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    obs_scores = observed["prob"].to_numpy()
    key_cols = ["source", "target", "ligand", "receptor", "pathway"]
    obs_keys = pd.MultiIndex.from_frame(observed[key_cols])
    original = adata.obs[groupby].astype(str).to_numpy()
    x, var_names = _get_matrix(adata, layer=layer, use_raw=use_raw, gene_symbols_key=gene_symbols_key)
    x, var_names = _subset_matrix_columns(x, var_names, expression_genes)
    if scale_expression_by_max:
        x = _scale_matrix_by_max(x)
    coords = None
    if spatial_key is not None and distance_decay is not None:
        coords = np.asarray(adata.obsm[spatial_key])
    static_group_weights = _group_weights_from_labels(original, groups) if population_size else None
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations, dtype=np.int64)

    worker_count = _resolve_n_jobs(n_jobs)
    if worker_count == 1 or n_permutations == 1:
        exceed = _permutation_exceed_chunk(
            seeds,
            x=x,
            var_names=var_names,
            original=original,
            groups=groups,
            lr=lr,
            obs_keys=obs_keys,
            obs_scores=obs_scores,
            aggregate=aggregate,
            trim=trim,
            clip_quantile=clip_quantile,
            min_pct=min_pct,
            min_expr=min_expr,
            gene_lookup=gene_lookup,
            overexpressed_genes=overexpressed_genes,
            group_weights=static_group_weights,
            coords=coords,
            distance_decay=distance_decay,
            cofactor_adjust=cofactor_adjust,
            cofactor_kh=cofactor_kh,
            cofactor_hill=cofactor_hill,
            score_method=score_method,
            complex_aggregate=complex_aggregate,
            array_backend=array_backend,
        )
    else:
        chunks = [chunk for chunk in np.array_split(seeds, min(worker_count, n_permutations)) if len(chunk)]
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            parts = executor.map(
                lambda chunk: _permutation_exceed_chunk(
                    chunk,
                    x=x,
                    var_names=var_names,
                    original=original,
                    groups=groups,
                    lr=lr,
                    obs_keys=obs_keys,
                    obs_scores=obs_scores,
                    aggregate=aggregate,
                    trim=trim,
                    clip_quantile=clip_quantile,
                    min_pct=min_pct,
                    min_expr=min_expr,
                    gene_lookup=gene_lookup,
                    overexpressed_genes=overexpressed_genes,
                    group_weights=static_group_weights,
                    coords=coords,
                    distance_decay=distance_decay,
                    cofactor_adjust=cofactor_adjust,
                    cofactor_kh=cofactor_kh,
                    cofactor_hill=cofactor_hill,
                    score_method=score_method,
                    complex_aggregate=complex_aggregate,
                    array_backend=array_backend,
                ),
                chunks,
            )
            exceed = np.sum(list(parts), axis=0)
    return (exceed + 1) / (n_permutations + 1)


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs < 0:
        try:
            import os

            return max((os.cpu_count() or 1) + 1 + n_jobs, 1)
        except Exception:  # pragma: no cover - defensive
            return 1
    return max(n_jobs, 1)


def _permutation_exceed_chunk(
    seeds: np.ndarray,
    *,
    x,
    var_names: pd.Index,
    original: np.ndarray,
    groups: list[str],
    lr: pd.DataFrame,
    obs_keys: pd.MultiIndex,
    obs_scores: np.ndarray,
    aggregate: str,
    trim: float,
    clip_quantile: float,
    min_pct: float,
    min_expr: float,
    gene_lookup: dict[str, str],
    overexpressed_genes: dict[str, set[str]] | None,
    group_weights: dict[str, float] | None,
    coords: np.ndarray | None,
    distance_decay: float | None,
    cofactor_adjust: bool,
    cofactor_kh: float,
    cofactor_hill: float,
    score_method: str,
    complex_aggregate: str,
    array_backend: str,
) -> np.ndarray:
    exceed = np.zeros(len(obs_scores), dtype=int)
    key_cols = ["source", "target", "ligand", "receptor", "pathway"]
    for seed in seeds:
        shuffled = np.random.default_rng(int(seed)).permutation(original)
        expr_mean, expr_pct = _group_expression_from_labels(
            x,
            var_names,
            shuffled,
            groups,
            aggregate=aggregate,
            trim=trim,
            clip_quantile=clip_quantile,
            array_backend=array_backend,
        )
        pair_weights = _spatial_pair_weights_from_labels(coords, shuffled, groups, distance_decay=distance_decay) if coords is not None else None
        perm = _score_lr_table_vectorized(
            lr,
            expr_mean,
            expr_pct,
            groups,
            min_pct=min_pct,
            min_expr=min_expr,
            gene_lookup=gene_lookup,
            overexpressed_genes=overexpressed_genes,
            score_method=score_method,
            complex_aggregate=complex_aggregate,
            group_weights=group_weights,
            pair_weights=pair_weights,
            cofactor_adjust=cofactor_adjust,
            cofactor_kh=cofactor_kh,
            cofactor_hill=cofactor_hill,
            array_backend=array_backend,
        )
        if perm.empty:
            continue
        perm_scores = perm.set_index(key_cols)["prob"]
        aligned = perm_scores.reindex(obs_keys, fill_value=0.0).to_numpy()
        exceed += aligned >= obs_scores
    return exceed

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage

from .analysis import CCCResult, compute_pathway_communication
from .diff import DifferentialCCC, pairwise_pathway_embedding, rank_pathway_similarity, signaling_changes
from .expression import signaling_expression_frame, signaling_expression_values
from .patterns import CommunicationPatterns, compute_pathway_clusters, compute_pathway_embedding

PROB_COLORS = ["#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142"]
EXPR_COLORS = ["#f7f7f7", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"]
DIFF_LOW = "#2b6cb0"
DIFF_MID = "#f7f7f7"
DIFF_HIGH = "#b8323b"
CELL_COLORS = ["#9a5b4f", "#2c7fb8", "#8dd3c7", "#fdb462", "#7b6bb1", "#4daf4a", "#e78ac3", "#a6cee3"]


def _pn():
    try:
        import plotnine as pn
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError("Install plotnine support with `uv sync --extra ggplot`.") from exc
    return pn


def bubble(
    result: CCCResult,
    *,
    sources: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    pathways: Sequence[str] | None = None,
    interactions: Sequence[str] | None = None,
    top_n: int | None = None,
    top_pairs: int | None = None,
    compact_pairs: bool = False,
    size_by: str = "prob",
    max_pvalue: float | None = None,
    significant_only: bool = False,
    facet_by: str | None = None,
    title: str = "LR bubble",
):
    """Plotnine version of the CellChat-style ligand-receptor bubble plot."""

    if size_by not in {"prob", "pvalue"}:
        raise ValueError("`size_by` must be 'prob' or 'pvalue'.")
    df = _filter_interactions(result, sources=sources, targets=targets, pathways=pathways, significant_only=significant_only).copy()
    df = _filter_by_pvalue(df, max_pvalue=max_pvalue)
    df["interaction"] = _interaction_labels(df)
    if interactions is not None:
        df = df[df["interaction"].isin(interactions)]
    if top_n is not None:
        keep = df.groupby("interaction", observed=True)["prob"].sum().sort_values(ascending=False).head(top_n).index
        df = df[df["interaction"].isin(keep)]
    df["pair"] = df["source"].astype(str) + " -> " + df["target"].astype(str)
    if top_pairs is not None:
        keep = df.groupby("pair", observed=True)["prob"].sum().sort_values(ascending=False).head(top_pairs).index
        df = df[df["pair"].isin(keep)]
    if compact_pairs:
        df["pair"] = df["pair"].map(_compact_pair)
    if size_by == "pvalue":
        df["size_value"] = _pvalue_size(df)
        size_label = "-log10(pvalue)"
    else:
        df["size_value"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)
        size_label = "probability"
    return _bubble_plot(df, color="prob", size="size_value", title=title, color_label="probability", size_label=size_label, facet_by=facet_by)


def diff_bubble(
    diff: DifferentialCCC,
    *,
    sources: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    pathways: Sequence[str] | None = None,
    top_n: int | None = 40,
    top_pairs: int | None = None,
    compact_pairs: bool = False,
    max_pvalue: float | None = None,
    facet_by: str | None = None,
    title: str = "Differential LR bubble",
):
    """Plotnine differential ligand-receptor bubble plot."""

    df = diff.interactions.copy()
    if sources is not None:
        df = df[df["source"].isin(sources)]
    if targets is not None:
        df = df[df["target"].isin(targets)]
    if pathways is not None:
        df = df[df["pathway"].isin(pathways)]
    df = _filter_by_pvalue(df, max_pvalue=max_pvalue)
    if top_n is not None:
        df = df.head(top_n)
    df["interaction"] = _interaction_labels(df)
    df["pair"] = df["source"].astype(str) + " -> " + df["target"].astype(str)
    if top_pairs is not None:
        keep = df.groupby("pair", observed=True)["abs_delta_prob"].sum().sort_values(ascending=False).head(top_pairs).index
        df = df[df["pair"].isin(keep)]
    if compact_pairs:
        df["pair"] = df["pair"].map(_compact_pair)
    return _bubble_plot(df, color="delta_prob", size="abs_delta_prob", title=title, color_label="delta probability", size_label="abs delta", diverging=True, facet_by=facet_by)


def diff_pathway_rank(
    diff: DifferentialCCC,
    *,
    top_n: int = 30,
    measure: str = "weight",
    title: str | None = None,
):
    """Plotnine CellChat `rankNet`-style pathway differential ranking."""

    df = _diff_summary_plot_frame(diff.pathway_changes, label_col="pathway", top_n=top_n, measure=measure)
    return _diff_rank_plot(
        df,
        label_col="pathway",
        value_col=_diff_measure_col(measure),
        title=title or f"Differential pathway rank: {diff.label_a} - {diff.label_b}",
        xlabel="Delta interaction weight" if measure in {"weight", "prob"} else "Delta interaction count",
    )


def diff_source_target_rank(
    diff: DifferentialCCC,
    *,
    top_n: int = 30,
    measure: str = "weight",
    title: str | None = None,
):
    """Plotnine source-target differential ranking."""

    changes = diff.source_target_changes.copy()
    if not changes.empty:
        changes["pair"] = changes["source"].astype(str) + " -> " + changes["target"].astype(str)
    df = _diff_summary_plot_frame(changes, label_col="pair", top_n=top_n, measure=measure)
    return _diff_rank_plot(
        df,
        label_col="pair",
        value_col=_diff_measure_col(measure),
        title=title or f"Differential source-target rank: {diff.label_a} - {diff.label_b}",
        xlabel="Delta interaction weight" if measure in {"weight", "prob"} else "Delta interaction count",
    )


def signaling_changes_scatter(
    diff: DifferentialCCC,
    group: str,
    *,
    pathways: Sequence[str] | None = None,
    exclude_pathways: Sequence[str] | None = None,
    top_label: float | int = 0.35,
    title: str | None = None,
):
    """Plotnine CellChat-style signaling-change scatter for one cell group."""

    pn = _pn()
    df = signaling_changes(diff, group, pathways=pathways, exclude_pathways=exclude_pathways)
    if df.empty:
        df = pd.DataFrame({"pathway": ["No changes"], "delta_outgoing": [0.0], "delta_incoming": [0.0], "abs_delta_total": [0.0], "direction": ["unchanged"]})
    labels = df.head(_top_label_count(len(df), top_label))
    labels = _label_positions(labels, x_col="delta_outgoing", y_col="delta_incoming")
    xlim = _zero_padded_limits(pd.concat([df["delta_outgoing"], labels["label_x"]], ignore_index=True))
    ylim = _zero_padded_limits(pd.concat([df["delta_incoming"], labels["label_y"]], ignore_index=True))
    return (
        pn.ggplot(df, pn.aes("delta_outgoing", "delta_incoming", color="direction", size="abs_delta_total"))
        + pn.geom_hline(yintercept=0, color="#9ca3af", linetype="dashed", size=0.3)
        + pn.geom_vline(xintercept=0, color="#9ca3af", linetype="dashed", size=0.3)
        + pn.geom_point(alpha=0.86, stroke=0.35)
        + pn.geom_text(labels, pn.aes("label_x", "label_y", label="pathway", ha="ha", va="va"), inherit_aes=False, size=8, color="#202124")
        + pn.scale_color_manual(values={diff.label_a: DIFF_HIGH, diff.label_b: DIFF_LOW, "unchanged": "#9ca3af"}, name="Higher in")
        + pn.scale_size_area(max_size=9, name="Abs delta")
        + pn.scale_x_continuous(limits=xlim, expand=(0, 0))
        + pn.scale_y_continuous(limits=ylim, expand=(0, 0))
        + pn.labs(
            title=title or f"Signaling changes of {group}",
            x=f"Delta outgoing strength ({diff.label_a} - {diff.label_b})",
            y=f"Delta incoming strength ({diff.label_a} - {diff.label_b})",
        )
        + _theme()
        + pn.theme(panel_grid_minor=pn.element_blank())
    )


def pathway_similarity_rank(
    diff: DifferentialCCC,
    *,
    similarity: str = "functional",
    top_n: int = 30,
    method: str = "auto",
    n_neighbors: int | None = None,
    min_dist: float = 0.3,
    random_state: int | None = 0,
    title: str | None = None,
):
    """Plotnine CellChat `rankSimilarity`-style pathway distance ranking."""

    pn = _pn()
    df = rank_pathway_similarity(diff, similarity=similarity, method=method, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state).head(top_n)
    if df.empty:
        df = pd.DataFrame({"pathway": ["No shared pathways"], "distance": [0.0], "delta_prob": [0.0]})
    df = df.sort_values("distance", ascending=True).copy()
    df["pathway"] = pd.Categorical(df["pathway"].astype(str), categories=df["pathway"].astype(str).tolist(), ordered=True)
    df["direction"] = np.where(df["delta_prob"].to_numpy(dtype=float) >= 0, diff.label_a, diff.label_b)
    return (
        pn.ggplot(df, pn.aes("pathway", "distance", fill="direction"))
        + pn.geom_col(width=0.72, alpha=0.88)
        + pn.coord_flip()
        + pn.scale_fill_manual(values={diff.label_a: DIFF_HIGH, diff.label_b: DIFF_LOW}, name="Higher flow")
        + pn.labs(title=title or f"Pathway distance rank ({similarity})", x="", y="Euclidean distance in joint manifold")
        + _theme()
        + pn.theme(axis_text_y=pn.element_text(size=_rank_axis_text_size(len(df))))
    )


def pathway_embedding_pairwise(
    diff: DifferentialCCC,
    *,
    similarity: str = "functional",
    significant_only: bool = False,
    min_prob: float = 0.0,
    thresh: float | None = None,
    method: str = "auto",
    n_neighbors: int | None = None,
    min_dist: float = 0.3,
    random_state: int | None = 0,
    remove_isolates: bool = True,
    top_label: float | int = 0.35,
    label_pathways: Sequence[str] | None = None,
    connect: bool = True,
    title: str | None = None,
):
    """Plotnine CellChat `netVisual_embeddingPairwise`-style joint embedding."""

    pn = _pn()
    df = pairwise_pathway_embedding(
        diff,
        similarity=similarity,
        significant_only=significant_only,
        min_prob=min_prob,
        thresh=thresh,
        method=method,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        remove_isolates=remove_isolates,
    )
    if df.empty:
        df = pd.DataFrame({"pathway": ["No pathways"], "condition": [diff.label_a], "dim1": [0.0], "dim2": [0.0], "prob": [0.0], "embedding_method": ["mds"]})
    df = df.copy()
    df["condition"] = pd.Categorical(df["condition"].astype(str), categories=[diff.label_a, diff.label_b], ordered=True)
    segments = _pairwise_embedding_segments(df, diff.label_a, diff.label_b)
    labels = _pairwise_embedding_labels(df, segments, top_label=top_label, label_pathways=label_pathways)
    labels = _embedding_label_positions(labels).copy()
    method_label = str(df["embedding_method"].iloc[0]) if "embedding_method" in df.columns and len(df) else method
    plot = (
        pn.ggplot(df, pn.aes("dim1", "dim2", color="condition", size="prob"))
        + pn.geom_point(alpha=0.88, stroke=0.35)
        + pn.geom_text(labels, pn.aes("label_x", "label_y", label="pathway", ha="ha", va="va"), inherit_aes=False, size=8, color="#202124")
        + pn.scale_color_manual(values={diff.label_a: DIFF_HIGH, diff.label_b: DIFF_LOW}, name="Condition")
        + pn.scale_size_area(max_size=9, name="pathway probability")
        + pn.scale_x_continuous(expand=(0.14, 0))
        + pn.scale_y_continuous(expand=(0.14, 0))
        + pn.labs(title=title or f"Pairwise pathway embedding ({similarity}, {method_label})", x="Dim 1", y="Dim 2")
        + _theme()
        + pn.theme(panel_grid_minor=pn.element_blank())
    )
    if connect and not segments.empty:
        plot += pn.geom_segment(segments, pn.aes(x="x", y="y", xend="xend", yend="yend"), inherit_aes=False, color="#9ca3af", alpha=0.55, size=0.35)
    return plot


def dotplot(
    result: CCCResult,
    *,
    level: str = "pathway",
    top_n: int = 30,
    title: str | None = None,
):
    """Plotnine dotplot ranking pathways or ligand-receptor pairs."""

    if level == "pathway":
        df = result.pathway_summary().head(top_n)
        y = "pathway"
    elif level in {"lr", "interaction"}:
        df = result.lr_summary().head(top_n)
        df["interaction"] = _interaction_labels(df)
        y = "interaction"
    else:
        raise ValueError("`level` must be 'pathway' or 'lr'.")
    df = df.copy()
    df["x"] = "all"
    return _bubble_plot(df, x="x", y=y, color="prob", size="count", title=title or f"{level} dotplot", color_label="probability", size_label="count")


def signaling_role_scatter(
    result: CCCResult,
    *,
    pathways: str | Sequence[str] | None = None,
    significant_only: bool = False,
    label: bool = True,
    title: str | None = None,
):
    """Plotnine CellChat-style outgoing-vs-incoming signaling role scatter."""

    pn = _pn()
    df = _signaling_role_scatter_frame(result, pathways=pathways, significant_only=significant_only)
    df = _scatter_label_positions(df)
    palette = {group: CELL_COLORS[i % len(CELL_COLORS)] for i, group in enumerate(df["group"])}
    size_breaks = sorted(value for value in df["link_count"].unique() if value > 0)
    plot = (
        pn.ggplot(df, pn.aes("outgoing", "incoming", color="group", size="link_count"))
        + pn.geom_point(alpha=0.84, stroke=0.35)
        + pn.scale_color_manual(values=palette, guide=None)
        + pn.scale_size_area(max_size=10, breaks=size_breaks or None, name="links")
        + pn.scale_x_continuous(expand=(0.12, 0))
        + pn.scale_y_continuous(expand=(0.12, 0))
        + pn.labs(title=title or _pathway_title("Signaling role", pathways, result.condition), x="Outgoing strength", y="Incoming strength")
        + _theme()
        + pn.theme(panel_grid_minor=pn.element_blank())
    )
    if label:
        plot += pn.geom_text(pn.aes("label_x", "label_y", label="group", ha="ha", va="va"), size=8, show_legend=False)
    return plot


def lr_contribution(
    result: CCCResult,
    pathway: str,
    *,
    top_n: int = 20,
    title: str | None = None,
):
    """Plotnine CellChat-style LR contribution plot within one pathway."""

    pn = _pn()
    summary = _lr_contribution_frame(result, pathways=[pathway], top_n=top_n)
    if summary.empty:
        summary = pd.DataFrame({"interaction": ["No interactions"], "contribution": [0.0]})
    else:
        summary = summary[summary["pathway"].astype(str) == str(pathway)].copy()
    summary = summary.sort_values("contribution", ascending=True)
    summary["interaction"] = pd.Categorical(summary["interaction"].astype(str), categories=summary["interaction"].astype(str).tolist(), ordered=True)
    return (
        pn.ggplot(summary, pn.aes("interaction", "contribution"))
        + pn.geom_col(fill="#6f63b6", width=0.72)
        + pn.coord_flip()
        + pn.scale_y_continuous(labels=lambda values: [f"{value:.0%}" for value in values])
        + pn.labs(title=title or f"{pathway} LR contribution", x="", y="Contribution")
        + _theme()
        + pn.theme(axis_text_y=pn.element_text(size=_rank_axis_text_size(len(summary))))
    )


def lr_contribution_multi(
    result: CCCResult,
    *,
    pathways: Sequence[str] | None = None,
    top_pathways: int = 6,
    top_n: int = 8,
    significant_only: bool = False,
    title: str | None = None,
):
    """Plotnine faceted CellChat `netAnalysis_contribution`-style LR contribution plot."""

    pn = _pn()
    if pathways is None:
        pathways = result.pathway_summary(significant_only=significant_only).head(top_pathways)["pathway"].astype(str).tolist()
    summary = _lr_contribution_frame(result, pathways=pathways, top_n=top_n, significant_only=significant_only)
    if summary.empty:
        summary = pd.DataFrame({"pathway": ["No pathways"], "interaction": ["No interactions"], "contribution": [0.0]})
    summary = summary.copy()
    pathway_levels = list(pathways) if pathways else list(dict.fromkeys(summary["pathway"].astype(str)))
    summary["pathway"] = pd.Categorical(summary["pathway"].astype(str), categories=pathway_levels, ordered=True)
    summary = _facet_interaction_order(summary, pathway_levels)
    return (
        pn.ggplot(summary, pn.aes("contribution", "interaction_facet"))
        + pn.geom_segment(pn.aes(x=0, xend="contribution", y="interaction_facet", yend="interaction_facet"), color="#6f63b6", size=5.2, lineend="butt")
        + pn.facet_grid("pathway ~ .", scales="free_y", space="free_y")
        + pn.scale_x_continuous(labels=lambda values: [f"{value:.0%}" for value in values], limits=(0, 1.0), expand=(0.015, 0))
        + pn.scale_y_discrete(labels=_strip_facet_interaction_label)
        + pn.labs(title=title or "LR contribution by pathway", x="Contribution", y="")
        + _theme()
        + pn.theme(axis_text_y=pn.element_text(size=7.0), panel_spacing=0.04)
    )


def pathway_embedding(
    result: CCCResult,
    *,
    similarity: str = "functional",
    significant_only: bool = False,
    min_prob: float = 0.0,
    thresh: float | None = None,
    method: str = "auto",
    n_neighbors: int | None = None,
    min_dist: float = 0.3,
    random_state: int | None = 0,
    remove_isolates: bool = True,
    cluster: bool = False,
    n_clusters: int | None = None,
    cluster_method: str = "spectral",
    top_label: float | int = 1.0,
    label_pathways: Sequence[str] | None = None,
    title: str | None = None,
):
    """Plotnine CellChat-style embedding of signaling pathway networks."""

    pn = _pn()
    df = compute_pathway_embedding(
        result,
        similarity=similarity,
        significant_only=significant_only,
        min_prob=min_prob,
        thresh=thresh,
        method=method,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        remove_isolates=remove_isolates,
    )
    if df.empty:
        df = pd.DataFrame({"pathway": ["No pathways"], "dim1": [0.0], "dim2": [0.0], "prob": [0.0], "count": [0], "embedding_method": ["mds"]})
    if cluster:
        clusters = compute_pathway_clusters(
            result,
            similarity=similarity,
            significant_only=significant_only,
            min_prob=min_prob,
            thresh=thresh,
            method=cluster_method,
            n_clusters=n_clusters,
            embedding=df,
            embedding_method=method,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            remove_isolates=remove_isolates,
        )[["pathway", "cluster"]]
        df = df.merge(clusters, on="pathway", how="left")
        df["cluster_label"] = "Group " + df["cluster"].fillna(0).astype(int).astype(str)
    if label_pathways is None:
        labels = df.head(_top_label_count(len(df), top_label))
    else:
        labels = df[df["pathway"].isin(label_pathways)]
    labels = _embedding_label_positions(labels).copy()
    size_col = "prob" if cluster else "count"
    color_col = "cluster_label" if cluster else "prob"
    plot = (
        pn.ggplot(df, pn.aes("dim1", "dim2", color=color_col, size=size_col))
        + pn.geom_point(alpha=0.86, stroke=0.35)
        + pn.geom_text(labels, pn.aes("label_x", "label_y", label="pathway", ha="ha", va="va"), inherit_aes=False, size=8, color="#202124")
        + pn.scale_size_area(max_size=9, name="pathway probability" if cluster else "interactions")
        + pn.scale_x_continuous(expand=(0.14, 0))
        + pn.scale_y_continuous(expand=(0.14, 0))
        + pn.labs(title=title or _embedding_title(similarity, df, cluster=cluster, cluster_method=cluster_method), x="Dim 1", y="Dim 2")
        + _theme()
        + pn.theme(panel_grid_minor=pn.element_blank())
    )
    if cluster:
        cluster_levels = sorted(df["cluster_label"].dropna().astype(str).unique())
        palette = {level: CELL_COLORS[i % len(CELL_COLORS)] for i, level in enumerate(cluster_levels)}
        plot += pn.scale_color_manual(values=palette, name="Signaling group")
    else:
        plot += pn.scale_color_gradientn(colors=PROB_COLORS, name="pathway probability")
    return plot


def signaling_gene_expression(
    adata,
    groupby: str | None = None,
    *,
    result: CCCResult | None = None,
    genes: Sequence[str] | None = None,
    features: Sequence[str] | None = None,
    signaling: str | Sequence[str] | None = None,
    lr_table=None,
    enriched_only: bool = True,
    kind: str = "dot",
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
    min_pct: float = 0.0,
    max_cells_per_group: int | None = None,
    random_state: int | None = 0,
    title: str | None = None,
):
    """Plotnine AnnData-native CellChat `plotGeneExpression` equivalent."""

    if not 0 <= min_pct <= 1:
        raise ValueError("`min_pct` must be between 0 and 1.")
    kind = kind.lower()
    if kind not in {"dot", "violin", "bar"}:
        raise ValueError("`kind` must be one of: 'dot', 'violin', or 'bar'.")
    pn = _pn()
    plot_title = title or _pathway_title("Signaling gene expression", signaling, result.condition if result is not None else None)
    if kind == "violin":
        values = signaling_expression_values(
            adata,
            groupby,
            result=result,
            genes=genes,
            features=features,
            signaling=signaling,
            lr_table=lr_table,
            enriched_only=enriched_only,
            include_cofactors=include_cofactors,
            groups=groups,
            layer=layer,
            use_raw=use_raw,
            gene_symbols_key=gene_symbols_key,
            max_cells_per_group=max_cells_per_group,
            random_state=random_state,
        )
        return _expression_violin_plot(values, title=plot_title)

    df = signaling_expression_frame(
        adata,
        groupby,
        result=result,
        genes=genes,
        features=features,
        signaling=signaling,
        lr_table=lr_table,
        enriched_only=enriched_only,
        include_cofactors=include_cofactors,
        groups=groups,
        layer=layer,
        use_raw=use_raw,
        gene_symbols_key=gene_symbols_key,
        aggregate=aggregate,
        trim=trim,
        standard_scale=standard_scale,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    if kind == "bar":
        return _expression_bar_plot(df, title=plot_title)
    return _expression_dot_plot(
        df,
        title=plot_title,
        standard_scale=standard_scale,
        scale_min=scale_min,
        scale_max=scale_max,
        min_pct=min_pct,
    )


def signaling_role_heatmap_compare(
    diff: DifferentialCCC,
    *,
    mode: str = "outgoing",
    pathways: Sequence[str] | None = None,
    row_scale: bool = True,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    title: str | None = None,
):
    """Plotnine side-by-side CellChat signaling-role heatmaps for two conditions."""

    pn = _pn()
    pathway_order = _compare_role_pathways(diff, pathways)
    frame = _signaling_role_compare_frame(diff, mode=mode, pathway_order=pathway_order, row_scale=row_scale, cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    if frame.empty:
        frame = pd.DataFrame({"condition": [diff.label_a], "pathway": ["No pathways"], "group": [""], "strength": [0.0]})
    frame = frame.copy()
    frame["strength_plot"] = pd.to_numeric(frame["strength"], errors="coerce").where(lambda values: values > 0)
    frame["condition"] = pd.Categorical(frame["condition"].astype(str), categories=[diff.label_a, diff.label_b], ordered=True)
    return (
        pn.ggplot(frame, pn.aes("group", "pathway", fill="strength_plot"))
        + pn.geom_tile(color="white", size=0.45)
        + pn.facet_grid(". ~ condition", scales="free_x", space="free_x")
        + pn.scale_fill_gradientn(colors=PROB_COLORS, name=f"{'relative ' if row_scale else ''}{mode} strength", na_value="white")
        + pn.labs(title=title or f"Compare signaling role heatmap ({mode})", x="Cell group", y="Pathway")
        + _theme()
        + pn.theme(axis_text_x=pn.element_text(rotation=90, ha="center", va="top", size=7.0 if frame["group"].nunique() > 10 else 8.0))
    )


def pathway_heatmap(
    result: CCCResult,
    *,
    pathways: Sequence[str] | None = None,
    source: str | None = None,
    target: str | None = None,
    top_pairs: int | None = None,
    compact_pairs: bool = False,
    significant_only: bool = False,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    title: str = "Pathway heatmap",
):
    """Plotnine pathway-level source-target heatmap."""

    pn = _pn()
    df = compute_pathway_communication(result, significant_only=significant_only)
    if source is not None:
        df = df[df["source"] == source]
    if target is not None:
        df = df[df["target"] == target]
    if pathways is not None:
        df = df[df["pathway"].isin(pathways)]
    df = df.copy()
    df["pair"] = df["source"].astype(str) + " -> " + df["target"].astype(str)
    if top_pairs is not None:
        keep = df.groupby("pair", observed=True)["prob"].sum().sort_values(ascending=False).head(top_pairs).index
        df = df[df["pair"].isin(keep)]
    if compact_pairs:
        df["pair"] = df["pair"].map(_compact_pair)
    if df.empty:
        df = pd.DataFrame({"pathway": ["No interactions"], "pair": [""], "prob": [0.0]})
    df = _order_heatmap_frame(df, row="pathway", col="pair", value="prob", cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    df["prob_plot"] = pd.to_numeric(df["prob"], errors="coerce").where(lambda values: values > 0)
    x_label_size = 7.0 if df["pair"].nunique() > 20 else 8.0
    return (
        pn.ggplot(df, pn.aes("pair", "pathway", fill="prob_plot"))
        + pn.geom_tile(color="white", size=0.45)
        + pn.scale_fill_gradientn(colors=PROB_COLORS, name="pathway probability", na_value="white")
        + pn.labs(title=title, x="Cell pair", y="Pathway")
        + _theme()
        + pn.theme(axis_text_x=pn.element_text(rotation=90, ha="center", va="top", size=x_label_size))
    )


def rank_signaling(
    result: CCCResult,
    *,
    level: str = "pathway",
    top_n: int = 30,
    title: str | None = None,
):
    """Plotnine horizontal rank plot for pathways or ligand-receptor pairs."""

    pn = _pn()
    if level == "pathway":
        df = result.pathway_summary().head(top_n).copy()
        y = "pathway"
    elif level in {"lr", "interaction"}:
        df = result.lr_summary().head(top_n).copy()
        df["interaction"] = _interaction_labels(df)
        y = "interaction"
    else:
        raise ValueError("`level` must be 'pathway' or 'lr'.")
    categories = df[y].astype(str).iloc[::-1].tolist()
    df[y] = pd.Categorical(df[y].astype(str), categories=categories, ordered=True)
    return (
        pn.ggplot(df, pn.aes("prob", y))
        + pn.geom_col(fill="#4f81bd", width=0.72)
        + pn.labs(title=title or f"Ranked {level}", x="Total probability", y="")
        + _theme()
        + pn.theme(axis_text_y=pn.element_text(size=_rank_axis_text_size(len(categories))))
    )


def rank_signaling_compare(
    diff: DifferentialCCC,
    *,
    level: str = "pathway",
    top_n: int = 30,
    stacked: bool = False,
    title: str | None = None,
):
    """Plotnine comparison of ranked pathway or LR information flow."""

    pn = _pn()
    if level == "pathway":
        a = diff.a.pathway_summary()[["pathway", "prob"]].rename(columns={"prob": diff.label_a})
        b = diff.b.pathway_summary()[["pathway", "prob"]].rename(columns={"prob": diff.label_b})
        key = "pathway"
    elif level in {"lr", "interaction"}:
        a = diff.a.lr_summary()
        b = diff.b.lr_summary()
        a["interaction"] = _interaction_labels(a)
        b["interaction"] = _interaction_labels(b)
        a = a[["interaction", "prob"]].rename(columns={"prob": diff.label_a})
        b = b[["interaction", "prob"]].rename(columns={"prob": diff.label_b})
        key = "interaction"
    else:
        raise ValueError("`level` must be 'pathway' or 'lr'.")
    merged = a.merge(b, on=key, how="outer").fillna(0.0)
    merged["total"] = merged[diff.label_a] + merged[diff.label_b]
    merged = merged.sort_values("total", ascending=False).head(top_n)
    categories = merged.sort_values("total", ascending=True)[key].astype(str).tolist()
    merged[key] = pd.Categorical(merged[key].astype(str), categories=categories, ordered=True)
    long = merged.melt(id_vars=key, value_vars=[diff.label_a, diff.label_b], var_name="condition", value_name="prob")
    long["condition"] = pd.Categorical(long["condition"].astype(str), categories=[diff.label_a, diff.label_b], ordered=True)
    bar_position = pn.position_stack(reverse=True) if stacked else "dodge"
    return (
        pn.ggplot(long, pn.aes(key, "prob", fill="condition"))
        + pn.geom_col(position=bar_position, width=0.72)
        + pn.coord_flip()
        + pn.scale_fill_manual(values={diff.label_a: DIFF_HIGH, diff.label_b: DIFF_LOW}, name="condition")
        + pn.labs(title=title or f"Compare ranked {level}{' (stacked)' if stacked else ''}", x="", y="Total probability")
        + _theme()
        + pn.theme(axis_text_y=pn.element_text(size=_rank_axis_text_size(len(categories))))
    )


def annotation_bar(result: CCCResult, *, value: str = "prob", significant_only: bool = False, title: str = "Annotation composition"):
    """Plotnine bar plot by CellChatDB annotation class."""

    pn = _pn()
    if value not in {"prob", "count"}:
        raise ValueError("`value` must be 'prob' or 'count'.")
    df = result.significant() if significant_only else result.interactions
    if "annotation" not in df.columns:
        summary = pd.DataFrame({"annotation": ["unknown"], "prob": [0.0], "count": [0]})
    else:
        summary = (
            df.assign(annotation=df["annotation"].replace("", "unknown").fillna("unknown"))
            .groupby("annotation", observed=True)
            .agg(prob=("prob", "sum"), count=("prob", "size"))
            .sort_values(value, ascending=False)
            .reset_index()
        )
    ylabel = "Total probability" if value == "prob" else "Interaction count"
    return (
        pn.ggplot(summary, pn.aes("annotation", value))
        + pn.geom_col(fill="#5ab4ac", width=0.72)
        + pn.coord_flip()
        + pn.labs(title=title, x="", y=ylabel)
        + _theme()
    )


def compare_interactions(diff: DifferentialCCC, *, value: str = "weight", title: str | None = None):
    """Plotnine comparison of global interaction count or weight."""

    pn = _pn()
    if value == "count":
        vals = {diff.label_a: float(diff.count_a.to_numpy().sum()), diff.label_b: float(diff.count_b.to_numpy().sum())}
        ylabel = "Interaction count"
    elif value in {"weight", "prob"}:
        vals = {diff.label_a: diff.network_a.to_numpy().sum(), diff.label_b: diff.network_b.to_numpy().sum()}
        ylabel = "Interaction weight"
    else:
        raise ValueError("`value` must be 'count' or 'weight'.")
    df = pd.DataFrame({"condition": list(vals), "value": list(vals.values())})
    return (
        pn.ggplot(df, pn.aes("condition", "value", fill="condition"))
        + pn.geom_col(width=0.68)
        + pn.scale_fill_manual(values={diff.label_a: DIFF_HIGH, diff.label_b: DIFF_LOW}, guide=None)
        + pn.labs(title=title or "Compare interactions", x="", y=ylabel)
        + _theme()
    )


def pathway_river(
    result: CCCResult,
    *,
    mode: str = "outgoing",
    pathways: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    top_n: int | None = 12,
    min_prob: float = 0.0,
    significant_only: bool = False,
    title: str | None = None,
):
    """Plotnine alluvial-style river from cell groups to signaling pathways."""

    pn = _pn()
    flows = _pathway_flow_table(
        result,
        mode=mode,
        pathways=pathways,
        groups=groups,
        top_n=top_n,
        min_prob=min_prob,
        significant_only=significant_only,
    )
    if flows.empty:
        flows = pd.DataFrame({"group": ["No interactions"], "pathway": [""], "prob": [0.0]})
    ribbons, left_bars, right_bars, labels = _river_frames(flows, result.groups)
    guides = _label_guide_frame(labels)
    headers = pd.DataFrame({"x": [0.13, 0.87], "y": [0.99, 0.99], "label": ["Cell groups", "Signaling pathways"]})
    palette = {group: CELL_COLORS[i % len(CELL_COLORS)] for i, group in enumerate(left_bars["label"])}
    label_size = _river_label_size(len(labels))
    return (
        pn.ggplot()
        + pn.geom_polygon(ribbons, pn.aes("x", "y", group="flow_id", fill="group"), alpha=0.34, color=None)
        + pn.geom_rect(left_bars, pn.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="label"), color="#202124", size=0.12, alpha=0.86)
        + pn.geom_rect(right_bars, pn.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"), fill="#6b7280", alpha=0.62, color="#202124", size=0.12)
        + pn.geom_segment(guides, pn.aes(x="x_anchor", xend="x", y="y_anchor", yend="y"), color="#8b96a5", size=0.18, alpha=0.48)
        + pn.geom_text(labels, pn.aes("x", "y", label="label", ha="ha"), size=label_size)
        + pn.geom_text(headers, pn.aes("x", "y", label="label"), size=8, color="#6b7280")
        + pn.scale_fill_manual(values=palette, guide=None)
        + pn.scale_x_continuous(limits=(-0.08, 1.10), expand=(0, 0))
        + pn.scale_y_continuous(limits=(0, 1.03), expand=(0.02, 0))
        + pn.labs(title=title or f"Pathway river ({mode})", x="", y="")
        + _theme()
        + pn.theme(axis_text=pn.element_blank(), axis_ticks=pn.element_blank(), panel_grid=pn.element_blank())
    )


def pattern_dot(
    patterns: CommunicationPatterns,
    *,
    kind: str = "pathway",
    top_n: int | None = 30,
    title: str | None = None,
):
    """CellChat-style pattern dot plot for pathway or cell-group loadings."""

    if kind == "pathway":
        df = patterns.pathway_pattern.copy()
        y = "pathway"
    elif kind == "group":
        df = patterns.group_pattern.copy()
        y = "group"
    else:
        raise ValueError("`kind` must be 'pathway' or 'group'.")
    if top_n is not None:
        keep = df.groupby(y, observed=True)["weight"].max().sort_values(ascending=False).head(top_n).index
        df = df[df[y].isin(keep)]
    return _bubble_plot(df, x="pattern", y=y, color="weight", size="weight", title=title or f"{kind.title()} communication patterns", color_label="loading", size_label="loading")


def pattern_river(
    patterns: CommunicationPatterns,
    *,
    top_groups: int | None = 12,
    top_pathways: int | None = 18,
    title: str | None = None,
):
    """Plotnine three-stage river: cell groups -> latent patterns -> pathways."""

    pn = _pn()
    ribbons, bars, labels = _pattern_river_frames(patterns, top_groups=top_groups, top_pathways=top_pathways)
    guides = _label_guide_frame(labels)
    headers = pd.DataFrame({"x": [0.12, 0.50, 0.88], "y": [0.99, 0.99, 0.99], "label": ["Cell groups", "Patterns", "Signaling pathways"]})
    group_palette = {group: CELL_COLORS[i % len(CELL_COLORS)] for i, group in enumerate(patterns.group_weights.index)}
    pattern_palette = {pattern: CELL_COLORS[(i + len(group_palette)) % len(CELL_COLORS)] for i, pattern in enumerate(patterns.group_weights.columns)}
    label_size = _river_label_size(len(labels))
    return (
        pn.ggplot()
        + pn.geom_polygon(ribbons, pn.aes("x", "y", group="flow_id", fill="pattern"), alpha=0.34, color=None)
        + pn.geom_rect(bars, pn.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax", fill="fill"), color="#202124", size=0.12, alpha=0.84)
        + pn.geom_segment(guides, pn.aes(x="x_anchor", xend="x", y="y_anchor", yend="y"), color="#8b96a5", size=0.18, alpha=0.48)
        + pn.geom_text(labels, pn.aes("x", "y", label="label", ha="ha"), size=label_size)
        + pn.geom_text(headers, pn.aes("x", "y", label="label"), size=8, color="#6b7280")
        + pn.scale_fill_manual(values={**group_palette, **pattern_palette, "terminal": "#6b7280"}, guide=None)
        + pn.scale_x_continuous(limits=(-0.08, 1.10), expand=(0, 0))
        + pn.scale_y_continuous(limits=(0, 1.03), expand=(0.02, 0))
        + pn.labs(title=title or f"Communication pattern river ({patterns.mode})", x="", y="")
        + _theme()
        + pn.theme(axis_text=pn.element_blank(), axis_ticks=pn.element_blank(), panel_grid=pn.element_blank())
    )


def pattern_number_plot(selection: pd.DataFrame, *, recommended_k: int | None = None, title: str = "Pattern number diagnostics"):
    """Plotnine selectK-style diagnostics for candidate NMF pattern counts."""

    pn = _pn()
    required = {"k", "mean_reconstruction_error", "explained", "stability"}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"`selection` is missing required columns: {sorted(missing)}")
    selection = selection.copy()
    selection["k"] = pd.to_numeric(selection["k"], errors="raise")
    selection = selection.sort_values("k")
    recommended_k = _recommended_pattern_k(selection) if recommended_k is None else int(recommended_k)
    long = selection.melt(
        id_vars="k",
        value_vars=["mean_reconstruction_error", "explained", "stability"],
        var_name="metric",
        value_name="value",
    )
    labels = {
        "mean_reconstruction_error": "reconstruction error",
        "explained": "explained fraction",
        "stability": "component stability",
    }
    long["metric"] = long["metric"].map(labels)
    metric_order = ["reconstruction error", "explained fraction", "component stability"]
    long["metric"] = pd.Categorical(long["metric"], categories=metric_order, ordered=True)
    recommended_points = long[long["k"] == recommended_k].copy()
    return (
        pn.ggplot(long, pn.aes("k", "value", color="metric"))
        + pn.geom_vline(xintercept=recommended_k, linetype="dashed", color="#3f3f46", size=0.35, alpha=0.62)
        + pn.geom_line(size=0.9)
        + pn.geom_point(size=2.4)
        + pn.geom_point(recommended_points, pn.aes("k", "value"), inherit_aes=False, size=3.6, color="#202124")
        + pn.geom_point(recommended_points, pn.aes("k", "value", color="metric"), inherit_aes=False, size=2.2)
        + pn.facet_wrap("~metric", scales="free_y", ncol=1)
        + pn.scale_color_manual(values={"reconstruction error": DIFF_HIGH, "explained fraction": "#1f78a8", "component stability": "#4daf4a"}, guide=None)
        + pn.labs(title=title, x="Number of patterns", y="")
        + _theme()
        + pn.theme(strip_background=pn.element_rect(fill="#f7f8fb", color="#c8cdd7", size=0.35), strip_text=pn.element_text(size=9.0))
    )


def _expression_dot_plot(df: pd.DataFrame, *, title: str, standard_scale: bool, scale_min: float, scale_max: float, min_pct: float):
    pn = _pn()
    if df.empty:
        df = pd.DataFrame({"group": [""], "gene": ["No genes"], "scaled_expression": [0.0], "mean_expression": [0.0], "pct_expressed": [0.0]})
    df = df.copy()
    groups = _category_levels(df["group"])
    genes = _category_levels(df["gene"])
    df["group"] = pd.Categorical(df["group"].astype(str), categories=groups, ordered=True)
    df["gene"] = pd.Categorical(df["gene"].astype(str), categories=genes[::-1], ordered=True)
    df["pct_size"] = np.where(df["pct_expressed"].to_numpy(dtype=float) >= min_pct, df["pct_expressed"].to_numpy(dtype=float) * 100.0, 0.0)
    color_col = "scaled_expression" if standard_scale else "mean_expression"
    plot = (
        pn.ggplot(df, pn.aes("group", "gene", color=color_col, size="pct_size"))
        + pn.geom_point(alpha=0.92, stroke=0.18)
        + pn.scale_size_area(max_size=8.5, name="Percent expressed")
        + pn.labs(title=title, x="Cell group", y="Signaling gene")
        + _theme()
        + pn.theme(axis_text_x=pn.element_text(rotation=45, ha="right"), panel_grid_major=pn.element_blank())
    )
    if len(groups) > 1:
        plot += pn.geom_vline(xintercept=np.arange(1.5, len(groups), 1), size=0.1, color="#e5e7eb")
    if len(genes) > 1:
        plot += pn.geom_hline(yintercept=np.arange(1.5, len(genes), 1), size=0.1, color="#e5e7eb")
    if standard_scale:
        plot += pn.scale_color_gradientn(colors=EXPR_COLORS, limits=(scale_min, scale_max), name="Scaled Expression", na_value="lightgrey")
    else:
        plot += pn.scale_color_gradientn(colors=EXPR_COLORS, name="Average expression", na_value="lightgrey")
    return plot


def _expression_violin_plot(values: pd.DataFrame, *, title: str):
    pn = _pn()
    if values.empty:
        values = pd.DataFrame({"group": [""], "gene": ["No genes"], "expression": [0.0]})
    values = values.copy()
    groups = _category_levels(values["group"])
    genes = _category_levels(values["gene"])
    values["group"] = pd.Categorical(values["group"].astype(str), categories=groups, ordered=True)
    values["gene"] = pd.Categorical(values["gene"].astype(str), categories=genes, ordered=True)
    palette = {gene: PROB_COLORS[i % len(PROB_COLORS)] for i, gene in enumerate(genes)}
    return (
        pn.ggplot(values, pn.aes("group", "expression", fill="gene"))
        + pn.geom_violin(position=pn.position_dodge(width=0.8), scale="width", trim=True, alpha=0.86, size=0.28)
        + pn.scale_fill_manual(values=palette, name="Signaling gene")
        + pn.labs(title=title, x="Cell group", y="Expression")
        + _theme()
        + pn.theme(axis_text_x=pn.element_text(rotation=45, ha="right"))
    )


def _expression_bar_plot(df: pd.DataFrame, *, title: str):
    pn = _pn()
    if df.empty:
        df = pd.DataFrame({"group": [""], "gene": ["No genes"], "mean_expression": [0.0]})
    df = df.copy()
    groups = _category_levels(df["group"])
    genes = _category_levels(df["gene"])
    df["group"] = pd.Categorical(df["group"].astype(str), categories=groups, ordered=True)
    df["gene"] = pd.Categorical(df["gene"].astype(str), categories=genes, ordered=True)
    palette = {gene: PROB_COLORS[i % len(PROB_COLORS)] for i, gene in enumerate(genes)}
    return (
        pn.ggplot(df, pn.aes("group", "mean_expression", fill="gene"))
        + pn.geom_col(position=pn.position_dodge(width=0.82), width=0.72, color="#2f3136", size=0.18, alpha=0.9)
        + pn.scale_fill_manual(values=palette, name="Signaling gene")
        + pn.labs(title=title, x="Cell group", y="Average expression")
        + _theme()
        + pn.theme(axis_text_x=pn.element_text(rotation=45, ha="right"))
    )


def _category_levels(values: pd.Series) -> list[str]:
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [str(value) for value in values.cat.categories]
    return list(dict.fromkeys(values.astype(str)))


def _diff_measure_col(measure: str) -> str:
    if measure in {"weight", "prob"}:
        return "delta_prob"
    if measure == "count":
        return "delta_count"
    raise ValueError("`measure` must be one of: 'weight', 'prob', or 'count'.")


def _rank_axis_text_size(n: int) -> float:
    if n > 24:
        return 6.5
    if n > 16:
        return 7.2
    if n > 10:
        return 8.0
    return 9.0


def _recommended_pattern_k(selection: pd.DataFrame) -> int:
    ordered = selection.sort_values("k").copy()
    if ordered.empty:
        raise ValueError("`selection` must contain at least one candidate pattern count.")
    if len(ordered) == 1:
        return int(ordered["k"].iloc[0])

    x = ordered["k"].to_numpy(dtype=float)
    error = pd.to_numeric(ordered["mean_reconstruction_error"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(error)
    if finite.sum() < 3:
        idx = int(np.nanargmin(error)) if np.isfinite(error).any() else len(ordered) - 1
        return int(ordered["k"].iloc[idx])

    x = x[finite]
    error = error[finite]
    k_values = ordered.loc[finite, "k"].to_numpy(dtype=int)
    x_span = x.max() - x.min()
    y_span = error.max() - error.min()
    if x_span == 0 or y_span == 0:
        return int(k_values[np.argmin(error)])

    x_norm = (x - x.min()) / x_span
    y_norm = (error - error.min()) / y_span
    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return int(k_values[np.argmin(error)])
    points = np.column_stack([x_norm, y_norm])
    distances = np.abs(line[0] * (start[1] - points[:, 1]) - line[1] * (start[0] - points[:, 0])) / line_norm

    stability = pd.to_numeric(ordered.loc[finite, "stability"], errors="coerce").to_numpy(dtype=float)
    stable = np.ones_like(distances, dtype=bool)
    if np.isfinite(stability).any():
        max_stability = np.nanmax(stability)
        stable = stability >= max_stability - 0.1
    if stable.any():
        distances = np.where(stable, distances, -np.inf)
    return int(k_values[int(np.argmax(distances))])


def _diff_summary_plot_frame(df: pd.DataFrame, *, label_col: str, top_n: int, measure: str) -> pd.DataFrame:
    value_col = _diff_measure_col(measure)
    if label_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame({label_col: ["No changes"], value_col: [0.0]})
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
    out = out.loc[out[value_col] != 0].copy()
    if out.empty:
        return pd.DataFrame({label_col: ["No changes"], value_col: [0.0]})
    out["_abs"] = out[value_col].abs()
    out = out.sort_values("_abs", ascending=False).head(top_n)
    out = out.sort_values(value_col, ascending=True).reset_index(drop=True)
    out[label_col] = pd.Categorical(out[label_col].astype(str), categories=out[label_col].astype(str).tolist(), ordered=True)
    out["_direction"] = np.where(out[value_col] >= 0, "increase", "decrease")
    return out


def _lr_contribution_frame(
    result: CCCResult,
    *,
    pathways: Sequence[str] | None,
    top_n: int,
    significant_only: bool = False,
) -> pd.DataFrame:
    df = result.significant() if significant_only else result.interactions
    if pathways is not None:
        df = df[df["pathway"].astype(str).isin(set(map(str, pathways)))]
    if df.empty:
        return pd.DataFrame(columns=["pathway", "interaction", "prob", "contribution"])
    df = df.copy()
    df["interaction"] = _interaction_labels(df)
    summary = (
        df.groupby(["pathway", "interaction"], observed=True)["prob"]
        .sum()
        .reset_index()
        .sort_values(["pathway", "prob"], ascending=[True, False])
    )
    pieces = []
    for pathway, sub in summary.groupby("pathway", observed=True, sort=False):
        sub = sub.sort_values("prob", ascending=False).head(top_n).copy()
        total = float(sub["prob"].sum()) or 1.0
        sub["contribution"] = sub["prob"] / total
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["pathway", "interaction", "prob", "contribution"])


def _facet_interaction_order(summary: pd.DataFrame, pathway_levels: Sequence[str]) -> pd.DataFrame:
    pieces = []
    for pathway in pathway_levels:
        sub = summary[summary["pathway"].astype(str) == str(pathway)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("contribution", ascending=True).reset_index(drop=True)
        sub["interaction"] = sub["interaction"].astype(str)
        sub["interaction_facet"] = [f"{pathway}|||{idx:03d}|||{label}" for idx, label in enumerate(sub["interaction"])]
        pieces.append(sub)
    if not pieces:
        out = summary.copy()
        out["interaction"] = out["interaction"].astype(str)
        out["interaction_facet"] = out["interaction"]
        return out
    out = pd.concat(pieces, ignore_index=True)
    out["interaction_facet"] = pd.Categorical(out["interaction_facet"], categories=out["interaction_facet"].tolist(), ordered=True)
    return out


def _strip_facet_interaction_label(labels):
    return [str(label).split("|||", 2)[-1] for label in labels]


def _signaling_role_matrix(
    result: CCCResult,
    *,
    mode: str,
    pathways: Sequence[str] | None,
    row_scale: bool,
) -> pd.DataFrame:
    if mode not in {"outgoing", "incoming", "all"}:
        raise ValueError("`mode` must be 'outgoing', 'incoming', or 'all'.")
    df = compute_pathway_communication(result)
    pathway_order = list(pathways) if pathways is not None else None
    if pathway_order is not None:
        df = df[df["pathway"].isin(pathway_order)]
    if mode == "all":
        outgoing = df.rename(columns={"source": "group"})
        incoming = df.rename(columns={"target": "group"})
        df = pd.concat([outgoing, incoming], ignore_index=True)
        group_col = "group"
    else:
        group_col = "source" if mode == "outgoing" else "target"
    mat = df.pivot_table(index="pathway", columns=group_col, values="prob", aggfunc="sum", fill_value=0.0)
    mat = mat.reindex(columns=result.groups, fill_value=0.0)
    if pathway_order is not None:
        mat = mat.reindex(pathway_order, fill_value=0.0)
    elif not mat.empty:
        mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    if row_scale:
        mat = mat.apply(lambda row: row / row.max() if row.max() > 0 else row, axis=1)
    return mat


def _compare_role_pathways(diff: DifferentialCCC, pathways: Sequence[str] | None) -> list[str]:
    if pathways is not None:
        return list(pathways)
    summary = (
        pd.concat(
            [
                diff.a.pathway_summary()[["pathway", "prob"]],
                diff.b.pathway_summary()[["pathway", "prob"]],
            ],
            ignore_index=True,
        )
        .groupby("pathway", observed=True)["prob"]
        .sum()
        .sort_values(ascending=False)
    )
    return summary.index.astype(str).tolist()


def _signaling_role_compare_frame(
    diff: DifferentialCCC,
    *,
    mode: str,
    pathway_order: Sequence[str],
    row_scale: bool,
    cluster_rows: bool,
    cluster_cols: bool,
) -> pd.DataFrame:
    left = _signaling_role_matrix(diff.a, mode=mode, pathways=pathway_order, row_scale=row_scale).reindex(pathway_order, fill_value=0.0)
    right = _signaling_role_matrix(diff.b, mode=mode, pathways=pathway_order, row_scale=row_scale).reindex(pathway_order, fill_value=0.0)
    if left.empty and right.empty:
        return pd.DataFrame(columns=["condition", "pathway", "group", "strength"])
    if cluster_rows and len(pathway_order) > 1:
        combined = pd.concat([left, right], axis=1)
        pathway_order = combined.index[_cluster_order(combined.to_numpy(dtype=float))].tolist()
        left = left.reindex(pathway_order, fill_value=0.0)
        right = right.reindex(pathway_order, fill_value=0.0)
    group_order = list(diff.groups)
    if cluster_cols and len(group_order) > 1:
        combined_cols = left.add(right, fill_value=0.0).reindex(columns=group_order, fill_value=0.0)
        group_order = combined_cols.columns[_cluster_order(combined_cols.T.to_numpy(dtype=float))].tolist()
        left = left.reindex(columns=group_order, fill_value=0.0)
        right = right.reindex(columns=group_order, fill_value=0.0)
    rows = []
    for condition, mat in ((diff.label_a, left), (diff.label_b, right)):
        long = mat.reset_index().melt(id_vars="pathway", var_name="group", value_name="strength")
        long["condition"] = condition
        rows.append(long)
    out = pd.concat(rows, ignore_index=True)
    out["pathway"] = pd.Categorical(out["pathway"].astype(str), categories=list(pathway_order)[::-1], ordered=True)
    out["group"] = pd.Categorical(out["group"].astype(str), categories=group_order, ordered=True)
    return out


def _pairwise_embedding_segments(frame: pd.DataFrame, label_a: str, label_b: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["pathway", "x", "y", "xend", "yend", "distance", "prob_a", "prob_b", "delta_prob"])
    wide = frame.pivot_table(index="pathway", columns="condition", values=["dim1", "dim2", "prob"], aggfunc="first", observed=True)
    rows = []
    for pathway in wide.index.astype(str):
        try:
            a_xy = np.array([wide.loc[pathway, ("dim1", label_a)], wide.loc[pathway, ("dim2", label_a)]], dtype=float)
            b_xy = np.array([wide.loc[pathway, ("dim1", label_b)], wide.loc[pathway, ("dim2", label_b)]], dtype=float)
        except KeyError:
            continue
        if not np.isfinite(a_xy).all() or not np.isfinite(b_xy).all():
            continue
        prob_a = float(wide.loc[pathway, ("prob", label_a)]) if ("prob", label_a) in wide.columns and pd.notna(wide.loc[pathway, ("prob", label_a)]) else 0.0
        prob_b = float(wide.loc[pathway, ("prob", label_b)]) if ("prob", label_b) in wide.columns and pd.notna(wide.loc[pathway, ("prob", label_b)]) else 0.0
        rows.append(
            {
                "pathway": pathway,
                "x": a_xy[0],
                "y": a_xy[1],
                "xend": b_xy[0],
                "yend": b_xy[1],
                "distance": float(np.linalg.norm(a_xy - b_xy)),
                "prob_a": prob_a,
                "prob_b": prob_b,
                "delta_prob": prob_a - prob_b,
            }
        )
    return pd.DataFrame(rows).sort_values("distance", ascending=False).reset_index(drop=True)


def _pairwise_embedding_labels(
    frame: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    top_label: float | int,
    label_pathways: Sequence[str] | None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["pathway", "dim1", "dim2"])
    if label_pathways is None:
        label_pathways = segments.head(_top_label_count(len(segments), top_label))["pathway"].astype(str).tolist()
    keep = set(map(str, label_pathways))
    if not keep:
        return pd.DataFrame(columns=["pathway", "dim1", "dim2"])
    labels = frame[frame["pathway"].astype(str).isin(keep)].groupby("pathway", observed=True).agg(dim1=("dim1", "mean"), dim2=("dim2", "mean")).reset_index()
    return labels.sort_values("pathway").reset_index(drop=True)


def _diff_rank_plot(df: pd.DataFrame, *, label_col: str, value_col: str, title: str, xlabel: str):
    pn = _pn()
    if "_direction" not in df.columns:
        df = df.copy()
        df["_direction"] = "unchanged"
    return (
        pn.ggplot(df, pn.aes(label_col, value_col, fill="_direction"))
        + pn.geom_col(width=0.72)
        + pn.geom_hline(yintercept=0, color="#3f3f46", size=0.3)
        + pn.coord_flip()
        + pn.scale_fill_manual(values={"increase": DIFF_HIGH, "decrease": DIFF_LOW, "unchanged": "#d4d4d8"}, guide=None)
        + pn.labs(title=title, x="", y=xlabel)
        + _theme()
    )


def _bubble_plot(
    df: pd.DataFrame,
    *,
    color: str,
    size: str,
    title: str,
    color_label: str,
    size_label: str,
    x: str = "pair",
    y: str = "interaction",
    diverging: bool = False,
    facet_by: str | None = None,
):
    pn = _pn()
    if df.empty:
        df = pd.DataFrame({x: [""], y: ["No interactions"], color: [0.0], size: [0.0]})
    if facet_by is not None and facet_by not in df.columns:
        raise ValueError(f"`facet_by={facet_by}` is not available in the plot data.")
    df = df.copy()
    x_levels = list(dict.fromkeys(df[x].astype(str)))
    y_levels = list(dict.fromkeys(df[y].astype(str)))
    if facet_by is None:
        df[x] = pd.Categorical(df[x].astype(str), categories=x_levels, ordered=True)
        df[y] = pd.Categorical(df[y].astype(str), categories=y_levels, ordered=True)
    else:
        df[x] = df[x].astype(str)
        df[y] = df[y].astype(str)
    plot = (
        pn.ggplot(df, pn.aes(x, y, color=color, size=size))
        + pn.geom_point(alpha=0.92, stroke=0.18)
        + pn.scale_size_area(max_size=8.5, name=size_label)
        + pn.labs(title=title, x="", y="")
        + _theme()
        + pn.theme(axis_text_x=pn.element_text(rotation=90, ha="center", va="top"), panel_grid_major=pn.element_blank())
    )
    if facet_by is None and len(x_levels) > 1:
        plot += pn.geom_vline(xintercept=np.arange(1.5, len(x_levels), 1), size=0.1, color="#e5e7eb")
    if facet_by is None and len(y_levels) > 1:
        plot += pn.geom_hline(yintercept=np.arange(1.5, len(y_levels), 1), size=0.1, color="#e5e7eb")
    if diverging:
        plot += pn.scale_color_gradient2(low=DIFF_LOW, mid=DIFF_MID, high=DIFF_HIGH, midpoint=0, name=color_label)
    else:
        plot += pn.scale_color_gradientn(colors=PROB_COLORS, name=color_label, na_value="white")
    if facet_by is not None:
        plot += pn.facet_wrap(f"~{facet_by}", scales="free")
    if color == size or color_label == size_label:
        plot += pn.guides(size=False)
    return plot


def _filter_interactions(
    result: CCCResult,
    *,
    sources: Sequence[str] | None,
    targets: Sequence[str] | None,
    pathways: Sequence[str] | None,
    significant_only: bool,
) -> pd.DataFrame:
    df = result.significant() if significant_only else result.interactions
    if sources is not None:
        df = df[df["source"].isin(sources)]
    if targets is not None:
        df = df[df["target"].isin(targets)]
    if pathways is not None:
        df = df[df["pathway"].isin(pathways)]
    return df


def _signaling_role_scatter_frame(result: CCCResult, *, pathways: str | Sequence[str] | None, significant_only: bool) -> pd.DataFrame:
    weight = result.network(value="prob", pathway=pathways, significant_only=significant_only)
    counts = result.network(value="count", pathway=pathways, significant_only=significant_only)
    outgoing = weight.sum(axis=1).reindex(result.groups).to_numpy(dtype=float)
    incoming = weight.sum(axis=0).reindex(result.groups).to_numpy(dtype=float)
    link_count = (counts > 0).sum(axis=1).add((counts > 0).sum(axis=0), fill_value=0).reindex(result.groups).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "group": result.groups,
            "outgoing": outgoing,
            "incoming": incoming,
            "link_count": link_count,
        }
    )


def _scatter_label_positions(df: pd.DataFrame) -> pd.DataFrame:
    return _label_positions(df, x_col="outgoing", y_col="incoming")


def _embedding_label_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = _label_positions(df, x_col="dim1", y_col="dim2")
    if len(out) <= 1:
        return out
    y = pd.to_numeric(out["dim2"], errors="coerce").fillna(0.0)
    y_span = float(y.max() - y.min())
    if y_span <= 0:
        return out
    lower = float(y.min() - 0.10 * y_span)
    upper = float(y.max() + 0.10 * y_span)
    out["label_y"] = _spread_label_positions(out["label_y"].to_numpy(dtype=float), min_gap=0.10 * y_span, lower=lower, upper=upper)
    return out


def _embedding_title(similarity: str, df: pd.DataFrame, *, cluster: bool, cluster_method: str) -> str:
    method = str(df["embedding_method"].iloc[0]) if "embedding_method" in df.columns and len(df) else "embedding"
    suffix = f"{similarity}, {method}"
    if cluster:
        suffix += f", {cluster_method}"
    return f"Pathway network embedding ({suffix})"


def _label_positions(df: pd.DataFrame, *, x_col: str, y_col: str) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out[x_col], errors="coerce").fillna(0.0)
    y = pd.to_numeric(out[y_col], errors="coerce").fillna(0.0)
    x_raw_span = float(x.max() - x.min()) if len(x) else 0.0
    y_raw_span = float(y.max() - y.min()) if len(y) else 0.0
    x_span = x_raw_span or 1.0
    y_span = y_raw_span or 1.0
    dx = 0.025 * x_span
    dy = 0.025 * y_span if y_raw_span > 0 else 0.0
    right_edge = x > 0 if x_raw_span == 0 else x >= x.min() + 0.82 * x_span
    top_edge = y > 0 if y_raw_span == 0 else y >= y.min() + 0.82 * y_span
    out["label_x"] = np.where(right_edge, x - dx, x + dx)
    out["label_y"] = np.where(top_edge, y - dy, y + dy)
    out["ha"] = np.where(right_edge, "right", "left")
    out["va"] = "center" if y_raw_span == 0 else np.where(top_edge, "top", "bottom")
    return out


def _top_label_count(n: int, top_label: float | int) -> int:
    if n <= 0:
        return 0
    if isinstance(top_label, float) and 0 < top_label <= 1:
        return max(1, int(np.ceil(n * top_label)))
    return min(n, max(0, int(top_label)))


def _zero_padded_limits(values: pd.Series | np.ndarray | Sequence[float], *, pad_fraction: float = 0.12, min_span: float = 0.10) -> tuple[float, float]:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-min_span / 2, min_span / 2)
    low = min(float(vals.min()), 0.0)
    high = max(float(vals.max()), 0.0)
    span = max(high - low, min_span)
    pad = max(span * pad_fraction, min_span * 0.5)
    return (low - pad, high + pad)


def _pathway_title(base: str, pathways: str | Sequence[str] | None, condition: str | None) -> str:
    bits = [base]
    if pathways is not None:
        if isinstance(pathways, str):
            bits.append(pathways)
        else:
            bits.append(", ".join(map(str, pathways)))
    if condition:
        bits.append(str(condition))
    return " | ".join(bits)


def _filter_by_pvalue(df: pd.DataFrame, *, max_pvalue: float | None) -> pd.DataFrame:
    if max_pvalue is None:
        return df
    if not 0 <= max_pvalue <= 1:
        raise ValueError("`max_pvalue` must be between 0 and 1.")
    pvalue_cols = [col for col in ("pvalue", "pvalue_a", "pvalue_b") if col in df.columns]
    if not pvalue_cols:
        return df
    pvalues = df[pvalue_cols].apply(pd.to_numeric, errors="coerce")
    return df[pvalues.min(axis=1) <= max_pvalue]


def _pathway_flow_table(
    result: CCCResult,
    *,
    mode: str,
    pathways: Sequence[str] | None,
    groups: Sequence[str] | None,
    top_n: int | None,
    min_prob: float,
    significant_only: bool,
) -> pd.DataFrame:
    if mode not in {"outgoing", "incoming", "all"}:
        raise ValueError("`mode` must be 'outgoing', 'incoming', or 'all'.")
    df = compute_pathway_communication(result, significant_only=significant_only)
    if pathways is not None:
        df = df[df["pathway"].isin(pathways)]
    if mode == "outgoing":
        df = df.rename(columns={"source": "group"})
    elif mode == "incoming":
        df = df.rename(columns={"target": "group"})
    else:
        df = pd.concat([df.rename(columns={"source": "group"}), df.rename(columns={"target": "group"})], ignore_index=True)
    if groups is not None:
        df = df[df["group"].isin(groups)]
    flows = df.groupby(["group", "pathway"], observed=True)["prob"].sum().reset_index()
    flows = flows[flows["prob"] > min_prob]
    if top_n is not None:
        keep = flows.groupby("pathway", observed=True)["prob"].sum().sort_values(ascending=False).head(top_n).index
        flows = flows[flows["pathway"].isin(keep)]
    return flows.sort_values("prob", ascending=False).reset_index(drop=True)


def _order_heatmap_frame(
    df: pd.DataFrame,
    *,
    row: str,
    col: str,
    value: str,
    cluster_rows: bool,
    cluster_cols: bool,
) -> pd.DataFrame:
    mat = df.pivot_table(index=row, columns=col, values=value, aggfunc="sum", fill_value=0.0)
    row_levels = list(mat.index)
    col_levels = list(mat.columns)
    if cluster_rows:
        row_levels = [row_levels[i] for i in _cluster_order(mat.to_numpy(dtype=float))]
    if cluster_cols:
        col_levels = [col_levels[i] for i in _cluster_order(mat.T.to_numpy(dtype=float))]
    out = df.copy()
    out[row] = pd.Categorical(out[row], categories=row_levels, ordered=True)
    out[col] = pd.Categorical(out[col], categories=col_levels, ordered=True)
    return out


def _cluster_order(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.arange(values.shape[0])
    if np.allclose(values, values[0]):
        return np.arange(values.shape[0])
    try:
        return leaves_list(linkage(values, method="average", metric="euclidean"))
    except ValueError:
        return np.arange(values.shape[0])


def _river_frames(flows: pd.DataFrame, result_groups: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_order = [group for group in result_groups if group in set(flows["group"])]
    cell_order += [group for group in flows["group"].drop_duplicates() if group not in set(cell_order)]
    pathway_order = flows.groupby("pathway", observed=True)["prob"].sum().sort_values(ascending=False).index.tolist()
    left = _stack_layout(flows.groupby("group", observed=True)["prob"].sum().reindex(cell_order))
    right = _stack_layout(flows.groupby("pathway", observed=True)["prob"].sum().reindex(pathway_order))
    group_totals = flows.groupby("group", observed=True)["prob"].sum()
    pathway_totals = flows.groupby("pathway", observed=True)["prob"].sum()
    left_cursor = {key: span[0] for key, span in left.items()}
    right_cursor = {key: span[0] for key, span in right.items()}
    rows = []
    for idx, row in enumerate(flows.sort_values(["group", "pathway"]).itertuples(index=False)):
        left_span = left[row.group]
        right_span = right[row.pathway]
        left_height = _local_flow_height(row.prob, group_totals.loc[row.group], left_span)
        right_height = _local_flow_height(row.prob, pathway_totals.loc[row.pathway], right_span)
        y0_bottom = left_cursor[row.group]
        y0_top = y0_bottom + left_height
        y1_bottom = right_cursor[row.pathway]
        y1_top = y1_bottom + right_height
        for point_id, (x, y) in enumerate(_ribbon_polygon_points(0.22, 0.78, y0_bottom, y0_top, y1_bottom, y1_top)):
            rows.append({"flow_id": idx, "point_id": point_id, "x": x, "y": y, "group": row.group, "pathway": row.pathway, "prob": row.prob})
        left_cursor[row.group] = y0_top
        right_cursor[row.pathway] = y1_top
    left_bars = _bar_frame(left, 0.08, 0.18)
    right_bars = _bar_frame(right, 0.82, 0.92)
    labels = pd.concat(
        [
            _label_frame(left, x=0.055, ha="right", x_anchor=0.08),
            _label_frame(right, x=0.945, ha="left", x_anchor=0.92),
        ],
        ignore_index=True,
    )
    return pd.DataFrame(rows), left_bars, right_bars, labels


def _pattern_river_frames(patterns: CommunicationPatterns, *, top_groups: int | None, top_pathways: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = patterns.group_weights.sum(axis=1).sort_values(ascending=False)
    pathways = patterns.pathway_weights.sum(axis=0).sort_values(ascending=False)
    if top_groups is not None:
        groups = groups.head(top_groups)
    if top_pathways is not None:
        pathways = pathways.head(top_pathways)
    group_weights = patterns.group_weights.loc[groups.index]
    pathway_weights = patterns.pathway_weights.loc[:, pathways.index]
    pattern_totals = group_weights.sum(axis=0).add(pathway_weights.sum(axis=1), fill_value=0.0)

    left = _stack_layout(groups)
    middle = _stack_layout(pattern_totals)
    right = _stack_layout(pathways)
    rows = []
    flow_id = 0
    left_cursor = {key: span[0] for key, span in left.items()}
    middle_left_cursor = {key: span[0] for key, span in middle.items()}
    middle_right_cursor = {key: span[0] for key, span in middle.items()}
    right_cursor = {key: span[0] for key, span in right.items()}
    group_totals = group_weights.sum(axis=1)
    pattern_group_totals = group_weights.sum(axis=0)
    pathway_totals = pathway_weights.sum(axis=0)
    pattern_pathway_totals = pathway_weights.sum(axis=1)
    for group in left:
        for pattern, weight in group_weights.loc[group].items():
            if weight <= 0:
                continue
            left_height = _local_flow_height(weight, group_totals.loc[group], left[group])
            middle_height = _local_flow_height(weight, pattern_group_totals.loc[pattern], middle[pattern])
            y0_bottom = left_cursor[group]
            y0_top = y0_bottom + left_height
            y1_bottom = middle_left_cursor[pattern]
            y1_top = y1_bottom + middle_height
            for point_id, (x, y) in enumerate(_ribbon_polygon_points(0.20, 0.50, y0_bottom, y0_top, y1_bottom, y1_top)):
                rows.append({"flow_id": flow_id, "point_id": point_id, "x": x, "y": y, "pattern": pattern, "weight": weight})
            left_cursor[group] = y0_top
            middle_left_cursor[pattern] = y1_top
            flow_id += 1
    for pattern in pathway_weights.index:
        for pathway, weight in pathway_weights.loc[pattern].items():
            if weight <= 0:
                continue
            middle_height = _local_flow_height(weight, pattern_pathway_totals.loc[pattern], middle[pattern])
            right_height = _local_flow_height(weight, pathway_totals.loc[pathway], right[pathway])
            y0_bottom = middle_right_cursor[pattern]
            y0_top = y0_bottom + middle_height
            y1_bottom = right_cursor[pathway]
            y1_top = y1_bottom + right_height
            for point_id, (x, y) in enumerate(_ribbon_polygon_points(0.50, 0.80, y0_bottom, y0_top, y1_bottom, y1_top)):
                rows.append({"flow_id": flow_id, "point_id": point_id, "x": x, "y": y, "pattern": pattern, "weight": weight})
            middle_right_cursor[pattern] = y0_top
            right_cursor[pathway] = y1_top
            flow_id += 1

    left_bars = _bar_frame(left, 0.08, 0.16)
    left_bars["fill"] = left_bars["label"]
    middle_bars = _bar_frame(middle, 0.46, 0.54)
    middle_bars["fill"] = middle_bars["label"]
    right_bars = _bar_frame(right, 0.84, 0.92)
    right_bars["fill"] = "terminal"
    bars = pd.concat([left_bars, middle_bars, right_bars], ignore_index=True)
    labels = pd.concat(
        [
            _label_frame(left, x=0.055, ha="right", x_anchor=0.08),
            _label_frame(middle, x=0.50, ha="center"),
            _label_frame(right, x=0.945, ha="left", x_anchor=0.92),
        ],
        ignore_index=True,
    )
    return pd.DataFrame(rows), bars, labels


def _stack_available(n: int, *, y0: float = 0.05, y1: float = 0.95, gap: float = 0.014) -> float:
    return max((y1 - y0) - gap * max(n - 1, 0), 0.0)


def _stack_layout(totals: pd.Series, *, y0: float = 0.05, y1: float = 0.95, gap: float = 0.014) -> dict[str, tuple[float, float]]:
    totals = totals.fillna(0.0).astype(float)
    total = float(totals.sum())
    if total <= 0:
        return {str(key): (y0, y0) for key in totals.index}
    available = _stack_available(len(totals), y0=y0, y1=y1, gap=gap)
    cursor = y1
    spans = {}
    for key, value in totals.items():
        height = available * float(value) / total
        top = cursor
        bottom = top - height
        spans[str(key)] = (bottom, top)
        cursor = bottom - gap
    return spans


def _bar_frame(spans: dict[str, tuple[float, float]], xmin: float, xmax: float) -> pd.DataFrame:
    return pd.DataFrame([{"label": label, "xmin": xmin, "xmax": xmax, "ymin": bottom, "ymax": top} for label, (bottom, top) in spans.items()])


def _local_flow_height(value: float, total: float, span: tuple[float, float]) -> float:
    if total <= 0:
        return 0.0
    return (span[1] - span[0]) * float(value) / float(total)


def _label_frame(spans: dict[str, tuple[float, float]], *, x: float, ha: str, x_anchor: float | None = None) -> pd.DataFrame:
    rows = [
        {
            "label": label,
            "x": x,
            "x_anchor": x if x_anchor is None else x_anchor,
            "y": (bottom + top) / 2,
            "y_anchor": (bottom + top) / 2,
            "ha": ha,
        }
        for label, (bottom, top) in spans.items()
    ]
    if len(rows) > 1:
        for row, y in zip(rows, _spread_label_positions([row["y"] for row in rows])):
            row["y"] = y
    return pd.DataFrame(rows)


def _label_guide_frame(labels: pd.DataFrame) -> pd.DataFrame:
    return labels.loc[labels["ha"] != "center", ["x", "x_anchor", "y", "y_anchor"]].copy()


def _river_label_size(n: int) -> float:
    if n > 28:
        return 6.0
    if n > 22:
        return 6.5
    if n > 16:
        return 7.0
    return 8.0


def _spread_label_positions(values: Sequence[float], *, min_gap: float = 0.032, lower: float = 0.04, upper: float = 0.96) -> list[float]:
    y = np.asarray(values, dtype=float)
    if y.size <= 1:
        return y.tolist()

    order = np.argsort(-y)
    placed = y[order].copy()
    span = upper - lower
    if min_gap * (len(placed) - 1) > span:
        min_gap = span / max(len(placed) - 1, 1)

    placed[0] = min(placed[0], upper)
    for i in range(1, len(placed)):
        placed[i] = min(placed[i], placed[i - 1] - min_gap)
    if placed[-1] < lower:
        placed += lower - placed[-1]
    if placed[0] > upper:
        placed -= placed[0] - upper

    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


def _bezier_points(x0: float, x1: float, y0: float, y1: float, n: int = 32) -> list[tuple[float, float]]:
    t = np.linspace(0, 1, n)
    cx0 = x0 + (x1 - x0) * 0.45
    cx1 = x0 + (x1 - x0) * 0.55
    x = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * cx0 + 3 * (1 - t) * t**2 * cx1 + t**3 * x1
    y = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * y0 + 3 * (1 - t) * t**2 * y1 + t**3 * y1
    return list(zip(x, y))


def _ribbon_polygon_points(
    x0: float,
    x1: float,
    y0_bottom: float,
    y0_top: float,
    y1_bottom: float,
    y1_top: float,
    n: int = 32,
) -> list[tuple[float, float]]:
    top = _bezier_points(x0, x1, y0_top, y1_top, n=n)
    bottom = _bezier_points(x0, x1, y0_bottom, y1_bottom, n=n)
    return top + bottom[::-1]


def _interaction_labels(df: pd.DataFrame) -> pd.Series:
    if "interaction_name_2" in df.columns:
        labels = df["interaction_name_2"].fillna("").astype(str).str.strip()
        fallback = df["ligand"].astype(str) + " - " + df["receptor"].astype(str)
        return labels.where(labels != "", fallback)
    return df["ligand"].astype(str) + " - " + df["receptor"].astype(str)


def _pvalue_size(df: pd.DataFrame) -> pd.Series:
    if "pvalue" not in df.columns:
        return pd.Series(0.0, index=df.index)
    pvalues = pd.to_numeric(df["pvalue"], errors="coerce")
    pvalues = pvalues.where(pvalues > 0)
    return -np.log10(pvalues).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _compact_pair(pair: str) -> str:
    if " -> " not in pair:
        return pair
    source, target = pair.split(" -> ", 1)
    return f"{source}\n{target}"


def _theme():
    pn = _pn()
    return (
        pn.theme_bw(base_size=10)
        + pn.theme(
            panel_background=pn.element_rect(fill="#fbfbfd", color="none"),
            plot_background=pn.element_rect(fill="white", color="white"),
            panel_border=pn.element_rect(fill=None, color="#b8bcc6", size=0.45),
            panel_grid_major=pn.element_line(color="#e5e7eb", size=0.35),
            panel_grid_minor=pn.element_blank(),
            axis_ticks=pn.element_blank(),
            axis_text=pn.element_text(color="#202124", size=8.5),
            axis_title=pn.element_text(color="#202124", size=9.5),
            plot_title=pn.element_text(color="#202124", size=11, weight="regular"),
            strip_background=pn.element_rect(fill="#eef0f4", color="#b8bcc6", size=0.45),
            strip_text=pn.element_text(color="#202124", size=9.5),
            legend_title=pn.element_text(color="#202124", size=9.5),
            legend_text=pn.element_text(color="#202124", size=8.5),
            figure_size=(7, 5),
            legend_background=pn.element_rect(fill="white", color="white"),
        )
    )

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch, PathPatch
from scipy.cluster.hierarchy import leaves_list, linkage

from .analysis import CCCResult, compute_pathway_communication
from .diff import DifferentialCCC, pairwise_pathway_embedding, rank_pathway_similarity, signaling_changes
from .expression import signaling_expression_frame, signaling_expression_values
from .patterns import compute_pathway_clusters, compute_pathway_embedding

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional at runtime
    adjust_text = None

CELLCHAT_RED = "#d95f5f"
CELLCHAT_BLUE = "#4f81bd"
CELLCHAT_TEAL = "#5ab4ac"
CELLCHAT_GREY = "#4d4d4d"
INK = "#202124"
MUTED = "#6b7280"
PANEL_BG = "#fbfbfd"
CELLCHAT_PALETTE = ["#9a5b4f", "#2c7fb8", "#8dd3c7", "#fdb462", "#7b6bb1", "#4daf4a", "#e78ac3", "#a6cee3"]
CELLCHAT_PROB_COLORS = ["#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142"]
CELLCHAT_EXPR_COLORS = ["#f7f7f7", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"]
PROB_CMAP = LinearSegmentedColormap.from_list("pyccc_prob", CELLCHAT_PROB_COLORS)
DIFF_CMAP = LinearSegmentedColormap.from_list("pyccc_diff", ["#2b6cb0", "#f7f7f7", "#b8323b"])
EXPR_CMAP = LinearSegmentedColormap.from_list("pyccc_expr", CELLCHAT_EXPR_COLORS)
CMAP_REGISTRY = {"pyccc_prob": PROB_CMAP, "pyccc_diff": DIFF_CMAP, "pyccc_expr": EXPR_CMAP}


def set_theme(context: str = "notebook", font_scale: float = 1.0, style: str = "nature") -> None:
    """Set a compact publication-style Matplotlib/Seaborn theme."""

    sns.set_theme(context=context, style="white", font_scale=font_scale, palette=CELLCHAT_PALETTE)
    font_family = "DejaVu Sans"
    if style not in {"nature", "cellchat", "minimal"}:
        raise ValueError("`style` must be one of: nature, cellchat, minimal.")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#d4d4d8",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "regular",
            "axes.titlesize": 11,
            "axes.titlepad": 6,
            "axes.labelsize": 10,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "savefig.facecolor": "white",
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.55,
            "legend.frameon": False,
        }
    )


def save_figure(fig, path: str, *, dpi: int = 360, transparent: bool = False, also_svg: bool = True) -> None:
    """Save a figure as high-resolution PNG and optionally matching SVG."""

    fig.savefig(path, dpi=dpi, transparent=transparent, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    if also_svg and not path.lower().endswith(".svg"):
        stem = path.rsplit(".", 1)[0]
        fig.savefig(f"{stem}.svg", transparent=transparent, facecolor="white", bbox_inches="tight", pad_inches=0.04)


def net_circle(
    result: CCCResult,
    *,
    weight: str = "prob",
    pathway: str | Sequence[str] | None = None,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_cells",
    edge_cmap: str = "pyccc_prob",
    title: str | None = None,
):
    """CellChat `netVisual_circle` style circular weighted network."""

    mat = result.network(value=weight, pathway=pathway, significant_only=significant_only)
    ax = _circle_ax(ax)
    _panel_face(ax)
    groups = list(mat.index)
    coords = _circle_positions(groups)
    colors = _palette(groups, cmap)
    values = mat.to_numpy(dtype=float)
    vmax = values.max() if values.size else 0.0
    edge_norm = plt.Normalize(0, vmax if vmax > 0 else 1)
    edge_colors = _cmap(edge_cmap)
    node_sizes = mat.sum(axis=1).add(mat.sum(axis=0), fill_value=0).reindex(groups).to_numpy()
    node_sizes = 360 + 1150 * node_sizes / (node_sizes.max() if node_sizes.max() > 0 else 1)
    node_shrinks = _node_shrink_map(groups, node_sizes)

    for source in groups:
        for target in groups:
            val = float(mat.loc[source, target])
            if val <= 0:
                continue
            _draw_curved_edge(
                ax,
                coords[source],
                coords[target],
                color=edge_colors(edge_norm(val)),
                lw=0.5 + 5 * val / (vmax or 1),
                alpha=0.75,
                start_shrink=node_shrinks[source],
                end_shrink=node_shrinks[target],
            )

    for i, group in enumerate(groups):
        x, y = coords[group]
        ax.scatter([x], [y], s=node_sizes[i], color=colors[group], edgecolor="white", linewidth=2.0, zorder=4)
        _draw_circle_label(ax, x, y, group, scale=1.18)
    _finish_circle_ax(ax)
    ax.set_title(title or _title("Network circle", pathway, result.condition))
    return ax


def net_chord(
    result: CCCResult,
    *,
    weight: str = "prob",
    pathway: str | Sequence[str] | None = None,
    top_n: int | None = 60,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_cells",
    title: str | None = None,
):
    """Chord-like circular diagram for source-target communication."""

    mat = result.network(value=weight, pathway=pathway, significant_only=significant_only)
    edges = _matrix_edges(mat)
    if top_n is not None:
        edges = edges.sort_values("weight", ascending=False).head(top_n)
    ax = _circle_ax(ax)
    _panel_face(ax)
    groups = list(mat.index)
    coords = _circle_positions(groups)
    colors = _palette(groups, cmap)
    max_w = edges["weight"].max() if len(edges) else 1

    for group in groups:
        x, y = coords[group]
        ax.scatter([x], [y], s=430, color=colors[group], edgecolor="white", linewidth=1.0, zorder=3)
        _draw_circle_label(ax, x, y, group, scale=1.16)

    for row in edges.itertuples(index=False):
        _draw_curved_edge(
            ax,
            coords[row.source],
            coords[row.target],
            color=colors[row.source],
            lw=0.4 + 4.0 * row.weight / (max_w or 1),
            alpha=0.45,
            curvature=0.15,
            start_shrink=10.4,
            end_shrink=10.8,
        )
    _finish_circle_ax(ax)
    ax.set_title(title or _title("Chord", pathway, result.condition))
    return ax


def net_chord_gene(
    result: CCCResult,
    *,
    sources: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    pathways: Sequence[str] | None = None,
    interactions: Sequence[str] | None = None,
    level: str = "lr",
    top_n: int | None = 30,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_cells",
    title: str | None = None,
):
    """CellChat `netVisual_chord_gene` style LR/pathway-mediated chord graph."""

    frame = _gene_chord_frame(
        result,
        sources=sources,
        targets=targets,
        pathways=pathways,
        interactions=interactions,
        level=level,
        top_n=top_n,
        significant_only=significant_only,
    )
    ax = _circle_ax(ax)
    _panel_face(ax)
    if frame.empty:
        return _empty_plot(ax, title or _title("Gene chord", pathways, result.condition))

    cell_nodes = [group for group in result.groups if group in set(frame["source"]).union(set(frame["target"]))]
    feature_nodes = frame.groupby("feature", observed=True)["prob"].sum().sort_values(ascending=False).index.astype(str).tolist()
    coords = _mediated_chord_positions(cell_nodes, feature_nodes)
    cell_colors = _palette(cell_nodes, cmap)
    feature_color = "#6b7280"

    max_w = float(frame["prob"].max()) or 1.0
    cell_totals = (
        frame.groupby("source", observed=True)["prob"].sum().add(frame.groupby("target", observed=True)["prob"].sum(), fill_value=0.0).reindex(cell_nodes).fillna(0.0)
    )
    feature_totals = frame.groupby("feature", observed=True)["prob"].sum().reindex(feature_nodes).fillna(0.0)
    cell_sizes = 320 + 860 * cell_totals.to_numpy(dtype=float) / (float(cell_totals.max()) or 1.0)
    feature_sizes = 120 + 520 * feature_totals.to_numpy(dtype=float) / (float(feature_totals.max()) or 1.0)
    node_sizes = {node: size for node, size in zip(cell_nodes, cell_sizes)}
    node_sizes.update({node: size for node, size in zip(feature_nodes, feature_sizes)})
    node_shrinks = _node_shrink_map(list(node_sizes), np.fromiter(node_sizes.values(), dtype=float), linewidth=1.2)

    for row in frame.itertuples(index=False):
        lw = 0.35 + 3.7 * float(row.prob) / max_w
        color = cell_colors.get(row.source, CELLCHAT_BLUE)
        _draw_curved_edge(
            ax,
            coords[row.source],
            coords[row.feature],
            color=color,
            lw=lw,
            alpha=0.36,
            curvature=0.10,
            start_shrink=node_shrinks[row.source],
            end_shrink=node_shrinks[row.feature],
        )
        _draw_curved_edge(
            ax,
            coords[row.feature],
            coords[row.target],
            color=color,
            lw=max(lw * 0.78, 0.35),
            alpha=0.24,
            curvature=-0.10,
            start_shrink=node_shrinks[row.feature],
            end_shrink=node_shrinks[row.target],
        )

    for group, size in zip(cell_nodes, cell_sizes):
        x, y = coords[group]
        ax.scatter([x], [y], s=size, color=cell_colors[group], edgecolor="white", linewidth=1.4, zorder=4)
        _draw_circle_label(ax, x, y, group, scale=1.17)
    for feature, size in zip(feature_nodes, feature_sizes):
        x, y = coords[feature]
        ax.scatter([x], [y], s=size, color=feature_color, edgecolor="white", linewidth=1.1, alpha=0.88, zorder=4)
        _draw_circle_label(ax, x, y, feature, scale=1.15)

    _finish_circle_ax(ax)
    ax.set_title(title or _title(f"{level.upper()} chord", pathways, result.condition))
    return ax


def net_individual(
    result: CCCResult,
    *,
    pathway: str | Sequence[str] | None = None,
    interaction: str | Sequence[str] | None = None,
    ligand: str | Sequence[str] | None = None,
    receptor: str | Sequence[str] | None = None,
    layout: str = "circle",
    vertex_receiver: Sequence[str | int] | None = None,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_cells",
    title: str | None = None,
):
    """CellChat `netVisual_individual` style network for one ligand-receptor pair."""

    if layout not in {"circle", "hierarchy", "chord"}:
        raise ValueError("`layout` must be one of: 'circle', 'hierarchy', or 'chord'.")
    frame, label = _individual_interaction_frame(
        result,
        pathway=pathway,
        interaction=interaction,
        ligand=ligand,
        receptor=receptor,
        significant_only=significant_only,
    )
    ax = _circle_ax(ax) if layout in {"circle", "chord"} else _plain_ax(ax)
    if frame.empty:
        return _empty_plot(ax, title or "Individual LR network")

    individual = replace(result, interactions=frame)
    plot_title = title or f"Individual LR {layout}: {label}"
    if layout == "circle":
        return net_circle(individual, ax=ax, cmap=cmap, title=plot_title)
    if layout == "chord":
        return net_chord(individual, ax=ax, cmap=cmap, title=plot_title)
    targets = _resolve_vertex_groups(result.groups, vertex_receiver) if vertex_receiver is not None else None
    sources = [group for group in result.groups if group not in set(targets or [])] if targets is not None else None
    return net_hierarchy(individual, sources=sources, targets=targets, ax=ax, title=plot_title)


def net_hierarchy(
    result: CCCResult,
    *,
    sources: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    pathway: str | Sequence[str] | None = None,
    weight: str = "prob",
    significant_only: bool = False,
    ax=None,
    title: str | None = None,
):
    """Sender-to-receiver hierarchy diagram similar to CellChat hierarchy plots."""

    mat = result.network(value=weight, pathway=pathway, significant_only=significant_only)
    sources = list(sources or mat.index)
    targets = list(targets or mat.columns)
    ax = _plain_ax(ax)
    left_y = np.linspace(0.9, 0.1, max(len(sources), 1))
    right_y = np.linspace(0.9, 0.1, max(len(targets), 1))
    src_pos = {g: (0.05, y) for g, y in zip(sources, left_y)}
    tgt_pos = {g: (0.95, y) for g, y in zip(targets, right_y)}
    max_w = mat.loc[sources, targets].to_numpy().max() if sources and targets else 1

    for source in sources:
        ax.text(0.0, src_pos[source][1], source, ha="left", va="center", fontsize=10, weight="bold")
    for target in targets:
        ax.text(1.0, tgt_pos[target][1], target, ha="right", va="center", fontsize=10, weight="bold")
    for source in sources:
        for target in targets:
            val = float(mat.loc[source, target])
            if val <= 0:
                continue
            _draw_bezier(ax, src_pos[source], tgt_pos[target], color=CELLCHAT_BLUE, lw=0.5 + 5 * val / (max_w or 1), alpha=0.55)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title(title or _title("Hierarchy", pathway, result.condition))
    return ax


def net_heatmap(
    result: CCCResult,
    *,
    value: str = "prob",
    pathway: str | Sequence[str] | None = None,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_prob",
    annot: bool = False,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    title: str | None = None,
):
    """Source-target interaction heatmap."""

    mat = result.network(value=value, pathway=pathway, significant_only=significant_only)
    mat = _cluster_matrix(mat, cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    ax = _plain_ax(ax)
    sns.heatmap(
        mat,
        cmap=_heatmap_cmap(cmap),
        mask=_zero_mask(mat),
        ax=ax,
        square=True,
        vmin=0,
        linewidths=0.65,
        linecolor="#efefef",
        annot=annot,
        cbar_kws={"label": value, "shrink": 0.72},
    )
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(title or _title(f"Network {value}", pathway, result.condition))
    _polish_matrix_ax(ax)
    return ax


def pathway_heatmap(
    result: CCCResult,
    *,
    pathways: Sequence[str] | None = None,
    source: str | None = None,
    target: str | None = None,
    top_pairs: int | None = None,
    compact_pairs: bool = False,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_prob",
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    title: str | None = None,
):
    """Heatmap of pathway activity across source-target pairs."""

    df = compute_pathway_communication(result, significant_only=significant_only)
    if source is not None:
        df = df[df["source"] == source]
    if target is not None:
        df = df[df["target"] == target]
    if pathways is not None:
        df = df[df["pathway"].isin(pathways)]
    df["pair"] = df["source"].astype(str) + " -> " + df["target"].astype(str)
    if top_pairs is not None:
        keep = df.groupby("pair", observed=True)["prob"].sum().sort_values(ascending=False).head(top_pairs).index
        df = df[df["pair"].isin(keep)]
    if compact_pairs:
        df["pair"] = [_compact_pair(pair) for pair in df["pair"]]
    mat = df.pivot_table(index="pathway", columns="pair", values="prob", aggfunc="sum", fill_value=0.0)
    ax = _plain_ax(ax)
    if mat.empty:
        return _empty_plot(ax, title or _title("Pathway heatmap", None, result.condition))
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    mat = _cluster_matrix(mat, cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    sns.heatmap(
        mat,
        cmap=_heatmap_cmap(cmap),
        mask=_zero_mask(mat),
        ax=ax,
        vmin=0,
        linewidths=0.65,
        linecolor="#efefef",
        cbar_kws={"label": "pathway probability", "shrink": 0.72},
    )
    ax.set_xlabel("Cell pair")
    ax.set_ylabel("Pathway")
    ax.set_title(title or _title("Pathway heatmap", None, result.condition))
    _polish_matrix_ax(ax)
    return ax


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
    show_size_legend: bool = True,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_prob",
    title: str | None = None,
):
    """Ligand-receptor bubble plot across source-target cell pairs."""

    if size_by not in {"prob", "pvalue"}:
        raise ValueError("`size_by` must be 'prob' or 'pvalue'.")
    df = _filter_interactions(result, sources=sources, targets=targets, pathways=pathways, significant_only=significant_only)
    df = df.copy()
    df = _filter_by_pvalue(df, max_pvalue=max_pvalue)
    df["interaction"] = _interaction_labels(df)
    if interactions is not None:
        df = df[df["interaction"].isin(interactions)]
    if top_n is not None:
        keep = df.groupby("interaction")["prob"].sum().sort_values(ascending=False).head(top_n).index
        df = df[df["interaction"].isin(keep)]
    df["pair"] = df["source"].astype(str) + " -> " + df["target"].astype(str)
    if top_pairs is not None:
        keep = df.groupby("pair", observed=True)["prob"].sum().sort_values(ascending=False).head(top_pairs).index
        df = df[df["pair"].isin(keep)]
    if compact_pairs:
        df["pair"] = [_compact_pair(pair) for pair in df["pair"]]
    size_col = "prob"
    size_label = "probability"
    if size_by == "pvalue":
        df["_significance"] = _pvalue_size(df)
        size_col = "_significance"
        size_label = "-log10(pvalue)"
    return _bubble_frame(
        df,
        x="pair",
        y="interaction",
        color="prob",
        size=size_col,
        ax=ax,
        cmap=cmap,
        title=title or "LR bubble",
        colorbar_label="probability",
        size_label=size_label,
        show_size_legend=show_size_legend,
    )


def dotplot(
    result: CCCResult,
    *,
    level: str = "pathway",
    top_n: int = 30,
    ax=None,
    cmap: str = "pyccc_prob",
    title: str | None = None,
    show_size_legend: bool = True,
):
    """Dotplot ranking pathways or ligand-receptor pairs by activity."""

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
    return _bubble_frame(df, x="x", y=y, color="prob", size="count", ax=ax, cmap=cmap, title=title or f"{level} dotplot", show_size_legend=show_size_legend)


def signaling_role_scatter(
    result: CCCResult,
    *,
    pathways: str | Sequence[str] | None = None,
    significant_only: bool = False,
    ax=None,
    label: bool = True,
    cmap: str = "pyccc_cells",
    size_range: tuple[float, float] = (70, 260),
    title: str | None = None,
):
    """Incoming vs outgoing signaling role scatter."""

    weight = result.network(value="prob", pathway=pathways, significant_only=significant_only)
    counts = result.network(value="count", pathway=pathways, significant_only=significant_only)
    out = weight.sum(axis=1)
    inc = weight.sum(axis=0)
    link_count = (counts > 0).sum(axis=1).add((counts > 0).sum(axis=0), fill_value=0).reindex(result.groups).to_numpy(dtype=float)
    size_min, size_max = size_range
    sizes = size_min + (size_max - size_min) * link_count / (link_count.max() if link_count.max() > 0 else 1)
    df = pd.DataFrame({"group": result.groups, "outgoing": out.reindex(result.groups).to_numpy(), "incoming": inc.reindex(result.groups).to_numpy(), "size": sizes})
    ax = _plain_ax(ax)
    colors = _palette(result.groups, cmap)
    for row in df.itertuples(index=False):
        ax.scatter(row.outgoing, row.incoming, s=row.size, color=colors[row.group], edgecolor="white", linewidth=1.4, alpha=0.82)
    if label:
        texts = []
        for row in df.itertuples(index=False):
            texts.append(ax.text(row.outgoing, row.incoming, f" {row.group}", va="center", fontsize=9, color=INK))
        if adjust_text is not None and texts:
            adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "#c7c7c7", "lw": 0.6})
    ax.set_xlabel("Outgoing strength")
    ax.set_ylabel("Incoming strength")
    ax.set_title(title or _title("Signaling role", pathways, result.condition))
    _polish_plain_ax(ax)
    return ax


def rank_signaling(
    result: CCCResult,
    *,
    level: str = "pathway",
    top_n: int = 30,
    ax=None,
    color: str = CELLCHAT_BLUE,
    title: str | None = None,
):
    """Rank pathways or LR pairs by total communication probability."""

    if level == "pathway":
        df = result.pathway_summary().head(top_n)
        y = "pathway"
    elif level in {"lr", "interaction"}:
        df = result.lr_summary().head(top_n)
        df["interaction"] = _interaction_labels(df)
        y = "interaction"
    else:
        raise ValueError("`level` must be 'pathway' or 'lr'.")
    ax = _plain_ax(ax)
    sns.barplot(df, x="prob", y=y, ax=ax, color=color)
    ax.set_xlabel("Total probability")
    ax.set_ylabel("")
    ax.set_title(title or f"Ranked {level}")
    _polish_plain_ax(ax)
    return ax


def annotation_bar(
    result: CCCResult,
    *,
    value: str = "prob",
    significant_only: bool = False,
    ax=None,
    title: str | None = None,
):
    """Bar plot of communication burden by CellChatDB annotation class."""

    if value not in {"prob", "count"}:
        raise ValueError("`value` must be 'prob' or 'count'.")
    df = result.significant() if significant_only else result.interactions
    if "annotation" not in df.columns:
        ax = _plain_ax(ax)
        return _empty_plot(ax, title or "Annotation composition")
    summary = (
        df.assign(annotation=df["annotation"].replace("", "unknown").fillna("unknown"))
        .groupby("annotation", observed=True)
        .agg(prob=("prob", "sum"), count=("prob", "size"))
        .sort_values(value, ascending=False)
        .reset_index()
    )
    ax = _plain_ax(ax)
    if summary.empty:
        return _empty_plot(ax, title or "Annotation composition")
    sns.barplot(summary, x=value, y="annotation", ax=ax, color=CELLCHAT_TEAL)
    ax.set_xlabel("Total probability" if value == "prob" else "Interaction count")
    ax.set_ylabel("")
    ax.set_title(title or "Annotation composition")
    _polish_plain_ax(ax)
    return ax


def lr_contribution(
    result: CCCResult,
    pathway: str,
    *,
    top_n: int = 20,
    ax=None,
    color: str = "#6f63b6",
    title: str | None = None,
):
    """Contribution of ligand-receptor pairs within one signaling pathway."""

    summary = _lr_contribution_frame(result, pathways=[pathway], top_n=top_n)
    if summary.empty:
        ax = _plain_ax(ax)
        ax.text(0.5, 0.5, "No interactions", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title(title or f"{pathway} LR contribution")
        return ax
    summary = summary[summary["pathway"].astype(str) == str(pathway)].sort_values("contribution", ascending=False)
    ax = _plain_ax(ax)
    sns.barplot(summary, x="contribution", y="interaction", ax=ax, color=color)
    ax.set_xlabel("Contribution")
    ax.set_ylabel("")
    ax.set_title(title or f"{pathway} LR contribution")
    _polish_plain_ax(ax)
    return ax


def lr_contribution_multi(
    result: CCCResult,
    *,
    pathways: Sequence[str] | None = None,
    top_pathways: int = 6,
    top_n: int = 8,
    ncols: int = 2,
    significant_only: bool = False,
    color: str = "#6f63b6",
    title: str | None = None,
):
    """Multi-panel CellChat `netAnalysis_contribution`-style LR contribution plot."""

    if pathways is None:
        pathways = result.pathway_summary(significant_only=significant_only).head(top_pathways)["pathway"].astype(str).tolist()
    frame = _lr_contribution_frame(result, pathways=pathways, top_n=top_n, significant_only=significant_only)
    n_panels = max(len(pathways), 1)
    ncols = max(1, min(int(ncols), n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, max(2.8 * nrows, 3.2)), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, pathway in zip(axes_arr, pathways):
        sub = frame[frame["pathway"].astype(str) == str(pathway)].sort_values("contribution", ascending=False)
        if sub.empty:
            _empty_plot(_plain_ax(ax), f"{pathway} LR contribution")
            continue
        sns.barplot(sub, x="contribution", y="interaction", ax=ax, color=color)
        ax.set_xlabel("Contribution")
        ax.set_ylabel("")
        ax.set_title(f"{pathway} LR contribution")
        _polish_plain_ax(ax)
    for ax in axes_arr[len(pathways) :]:
        ax.set_axis_off()
    if title:
        fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=12)
    return fig, axes_arr[: len(pathways)]


def signaling_role_heatmap(
    result: CCCResult,
    *,
    mode: str = "outgoing",
    pathways: Sequence[str] | None = None,
    row_scale: bool = True,
    ax=None,
    cmap: str = "pyccc_prob",
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    title: str | None = None,
):
    """CellChat-style signaling role heatmap by pathway and cell group."""

    mat = _signaling_role_matrix(result, mode=mode, pathways=pathways, row_scale=row_scale)
    ax = _plain_ax(ax)
    if mat.empty:
        return _empty_plot(ax, title or f"Signaling role heatmap ({mode})")
    heat = _cluster_matrix(mat, cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    label = f"relative {mode} strength" if row_scale else f"{mode} strength"
    sns.heatmap(heat, cmap=_heatmap_cmap(cmap), mask=_zero_mask(heat), ax=ax, vmin=0, linewidths=0.65, linecolor="#efefef", cbar_kws={"label": label, "shrink": 0.72})
    ax.set_xlabel("Cell group")
    ax.set_ylabel("Pathway")
    ax.set_title(title or f"Signaling role heatmap ({mode})")
    _polish_matrix_ax(ax)
    return ax


def signaling_role_heatmap_compare(
    diff: DifferentialCCC,
    *,
    mode: str = "outgoing",
    pathways: Sequence[str] | None = None,
    row_scale: bool = True,
    axes=None,
    cmap: str = "pyccc_prob",
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    title: str | None = None,
):
    """Side-by-side CellChat signaling-role heatmaps for two conditions."""

    pathway_order = _compare_role_pathways(diff, pathways)
    left = _signaling_role_matrix(diff.a, mode=mode, pathways=pathway_order, row_scale=row_scale).reindex(pathway_order, fill_value=0.0)
    right = _signaling_role_matrix(diff.b, mode=mode, pathways=pathway_order, row_scale=row_scale).reindex(pathway_order, fill_value=0.0)
    if left.empty and right.empty:
        if axes is None:
            fig, axes_arr = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        else:
            axes_arr = np.asarray(axes).ravel()
            fig = axes_arr[0].figure
        for ax, label in zip(axes_arr[:2], [diff.label_a, diff.label_b]):
            _empty_plot(_plain_ax(ax), f"{label}: {mode} roles")
        return fig, axes_arr[:2]

    if cluster_rows and len(pathway_order) > 1:
        combined = pd.concat([left, right], axis=1)
        pathway_order = combined.index[_cluster_order(combined.to_numpy(dtype=float))].tolist()
        left = left.reindex(pathway_order, fill_value=0.0)
        right = right.reindex(pathway_order, fill_value=0.0)
    if cluster_cols and len(diff.groups) > 1:
        combined_cols = left.add(right, fill_value=0.0).reindex(columns=diff.groups, fill_value=0.0)
        group_order = combined_cols.columns[_cluster_order(combined_cols.T.to_numpy(dtype=float))].tolist()
        left = left.reindex(columns=group_order, fill_value=0.0)
        right = right.reindex(columns=group_order, fill_value=0.0)

    if axes is None:
        fig, axes_arr = plt.subplots(1, 2, figsize=(12, max(4.8, 0.34 * max(len(pathway_order), 1))), constrained_layout=True)
    else:
        axes_arr = np.asarray(axes).ravel()
        if len(axes_arr) < 2:
            raise ValueError("`axes` must contain two Matplotlib axes.")
        fig = axes_arr[0].figure

    vmax = max(float(left.to_numpy(dtype=float).max()) if left.size else 0.0, float(right.to_numpy(dtype=float).max()) if right.size else 0.0, 1.0)
    label = f"relative {mode} strength" if row_scale else f"{mode} strength"
    for i, (ax, mat, condition) in enumerate(zip(axes_arr[:2], [left, right], [diff.label_a, diff.label_b])):
        sns.heatmap(
            mat,
            cmap=_heatmap_cmap(cmap),
            mask=_zero_mask(mat),
            ax=ax,
            vmin=0,
            vmax=vmax,
            linewidths=0.65,
            linecolor="#efefef",
            cbar=i == 1,
            cbar_kws={"label": label, "shrink": 0.72},
        )
        ax.set_xlabel("Cell group")
        ax.set_ylabel("Pathway" if i == 0 else "")
        ax.set_title(str(condition))
        _polish_matrix_ax(ax)
    if title:
        fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=12)
    return fig, axes_arr[:2]


def pathway_river(
    result: CCCResult,
    *,
    mode: str = "outgoing",
    pathways: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    top_n: int | None = 12,
    min_prob: float = 0.0,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_cells",
    title: str | None = None,
):
    """Alluvial river plot linking cell groups to signaling pathways.

    This is a deterministic CellChat `netAnalysis_river`-style summary: outgoing
    mode links sender groups to pathways, incoming mode links receiver groups to
    pathways, and all mode uses both sender and receiver contributions.
    """

    if mode not in {"outgoing", "incoming", "all"}:
        raise ValueError("`mode` must be 'outgoing', 'incoming', or 'all'.")

    flows = _pathway_flow_table(
        result,
        mode=mode,
        pathways=pathways,
        groups=groups,
        top_n=top_n,
        min_prob=min_prob,
        significant_only=significant_only,
    )
    ax = _plain_ax(ax)
    if flows.empty:
        return _empty_plot(ax, title or f"Pathway river ({mode})")

    cell_order = [g for g in result.groups if g in set(flows["group"])]
    cell_order += [g for g in flows["group"].drop_duplicates() if g not in set(cell_order)]
    pathway_order = flows.groupby("pathway", observed=True)["prob"].sum().sort_values(ascending=False).index.tolist()
    cell_totals = flows.groupby("group", observed=True)["prob"].sum().reindex(cell_order)
    pathway_totals = flows.groupby("pathway", observed=True)["prob"].sum().reindex(pathway_order)
    left = _stack_layout(cell_totals)
    right = _stack_layout(pathway_totals)
    left_cursor = {key: span[0] for key, span in left.items()}
    right_cursor = {key: span[0] for key, span in right.items()}
    colors = _palette(cell_order, cmap)
    total = float(flows["prob"].sum())

    for row in flows.sort_values(["group", "pathway"]).itertuples(index=False):
        height = (row.prob / total) * _stack_available(len(left))
        y0a = left_cursor[row.group]
        y0b = y0a + height
        y1a = right_cursor[row.pathway]
        y1b = y1a + height
        _draw_ribbon(ax, 0.24, 0.76, y0a, y0b, y1a, y1b, color=colors[row.group], alpha=0.42)
        left_cursor[row.group] = y0b
        right_cursor[row.pathway] = y1b

    for group, (bottom, top) in left.items():
        ax.fill_between([0.08, 0.22], bottom, top, color=colors[group], alpha=0.86, edgecolor=INK, linewidth=0.35)
        ax.text(0.06, (bottom + top) / 2, group, ha="right", va="center", fontsize=9, color=INK)
    for pathway, (bottom, top) in right.items():
        ax.fill_between([0.78, 0.92], bottom, top, color=CELLCHAT_GREY, alpha=0.62, edgecolor=INK, linewidth=0.35)
        ax.text(0.94, (bottom + top) / 2, pathway, ha="left", va="center", fontsize=9, color=INK)

    ax.text(0.15, 0.99, "Cell group", ha="center", va="bottom", fontsize=10, color=MUTED)
    ax.text(0.85, 0.99, "Signaling pathway", ha="center", va="bottom", fontsize=10, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.set_axis_off()
    ax.set_title(title or f"Pathway river ({mode})")
    return ax


def rank_signaling_compare(
    diff: DifferentialCCC,
    *,
    level: str = "pathway",
    top_n: int = 30,
    stacked: bool = False,
    ax=None,
    title: str | None = None,
):
    """Compare ranked pathway or LR information flow between two conditions."""

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
    ax = _plain_ax(ax)
    if stacked:
        display = merged.sort_values("total", ascending=True)
        labels = display[key].astype(str).to_numpy()
        left = display[diff.label_a].astype(float).to_numpy()
        right = display[diff.label_b].astype(float).to_numpy()
        ax.barh(labels, left, color=CELLCHAT_RED, alpha=0.88, label=diff.label_a)
        ax.barh(labels, right, left=left, color=CELLCHAT_BLUE, alpha=0.88, label=diff.label_b)
        ax.legend(title="condition", frameon=False, loc="best")
    else:
        long = merged.melt(id_vars=key, value_vars=[diff.label_a, diff.label_b], var_name="condition", value_name="prob")
        sns.barplot(long, x="prob", y=key, hue="condition", ax=ax, palette=[CELLCHAT_RED, CELLCHAT_BLUE])
    ax.set_xlabel("Total probability")
    ax.set_ylabel("")
    ax.set_title(title or f"Compare ranked {level}{' (stacked)' if stacked else ''}")
    _polish_plain_ax(ax)
    return ax


def compare_interactions(diff: DifferentialCCC, *, value: str = "weight", ax=None, title: str | None = None):
    """Bar plot comparing total count or communication weight between conditions."""

    if value == "count":
        vals = {diff.label_a: float(diff.count_a.to_numpy().sum()), diff.label_b: float(diff.count_b.to_numpy().sum())}
        ylabel = "Interaction count"
    elif value in {"weight", "prob"}:
        vals = {diff.label_a: diff.network_a.to_numpy().sum(), diff.label_b: diff.network_b.to_numpy().sum()}
        ylabel = "Interaction weight"
    else:
        raise ValueError("`value` must be 'count' or 'weight'.")
    df = pd.DataFrame({"condition": list(vals), "value": list(vals.values())})
    ax = _plain_ax(ax)
    sns.barplot(df, x="condition", y="value", ax=ax, palette=[CELLCHAT_RED, CELLCHAT_BLUE], hue="condition", legend=False)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Compare interactions")
    _polish_plain_ax(ax)
    return ax


def diff_network_circle(
    diff: DifferentialCCC,
    *,
    measure: str = "weight",
    ax=None,
    title: str | None = None,
    pos_color: str = CELLCHAT_RED,
    neg_color: str = CELLCHAT_BLUE,
):
    """Differential source-target network circle."""

    mat = diff.differential_network(measure=measure)
    ax = _circle_ax(ax)
    _panel_face(ax)
    groups = list(mat.index)
    coords = _circle_positions(groups)
    absmax = np.abs(mat.to_numpy()).max() if mat.size else 1
    for source in groups:
        for target in groups:
            val = float(mat.loc[source, target])
            if val == 0:
                continue
            _draw_curved_edge(
                ax,
                coords[source],
                coords[target],
                color=pos_color if val > 0 else neg_color,
                lw=0.5 + 5 * abs(val) / (absmax or 1),
                alpha=0.7,
                start_shrink=10.2,
                end_shrink=10.6,
            )
    for group in groups:
        x, y = coords[group]
        ax.scatter([x], [y], s=420, color="#f4f4f4", edgecolor="#333333", linewidth=1.0, zorder=3)
        _draw_circle_label(ax, x, y, group, scale=1.16)
    _finish_circle_ax(ax)
    ax.set_title(title or f"Differential {measure} network: {diff.label_a} - {diff.label_b}")
    return ax


def diff_heatmap(
    diff: DifferentialCCC,
    *,
    measure: str = "weight",
    ax=None,
    cmap: str = "pyccc_diff",
    annot: bool = False,
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    title: str | None = None,
):
    """Heatmap of source-target differential communication."""

    mat = _cluster_matrix(diff.differential_network(measure=measure), cluster_rows=cluster_rows, cluster_cols=cluster_cols)
    ax = _plain_ax(ax)
    absmax = _finite_absmax(mat.to_numpy(), default=1.0)
    sns.heatmap(
        mat,
        cmap=_cmap(cmap),
        center=0,
        norm=TwoSlopeNorm(vcenter=0, vmin=-absmax, vmax=absmax),
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot=annot,
        cbar_kws={"label": f"delta {measure}", "shrink": 0.72},
    )
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    ax.set_title(title or f"{diff.label_a} - {diff.label_b} ({measure})")
    _polish_matrix_ax(ax)
    return ax


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
    show_size_legend: bool = True,
    ax=None,
    cmap: str = "pyccc_diff",
    title: str | None = None,
):
    """Differential ligand-receptor bubble plot."""

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
        df["pair"] = [_compact_pair(pair) for pair in df["pair"]]
    return _bubble_frame(
        df,
        x="pair",
        y="interaction",
        color="delta_prob",
        size="abs_delta_prob",
        ax=ax,
        cmap=cmap,
        center=0,
        title=title or "Differential LR bubble",
        colorbar_label="delta probability",
        show_size_legend=show_size_legend,
    )


def diff_pathway_rank(
    diff: DifferentialCCC,
    *,
    top_n: int = 30,
    measure: str = "weight",
    ax=None,
    title: str | None = None,
    pos_color: str = CELLCHAT_RED,
    neg_color: str = CELLCHAT_BLUE,
):
    """CellChat `rankNet`-style pathway differential ranking."""

    df = _diff_summary_plot_frame(diff.pathway_changes, label_col="pathway", top_n=top_n, measure=measure)
    return _diff_rank_bar(
        df,
        label_col="pathway",
        value_col=_diff_measure_col(measure),
        ax=ax,
        title=title or f"Differential pathway rank: {diff.label_a} - {diff.label_b}",
        xlabel="Delta interaction weight" if measure in {"weight", "prob"} else "Delta interaction count",
        pos_color=pos_color,
        neg_color=neg_color,
    )


def diff_source_target_rank(
    diff: DifferentialCCC,
    *,
    top_n: int = 30,
    measure: str = "weight",
    ax=None,
    title: str | None = None,
    pos_color: str = CELLCHAT_RED,
    neg_color: str = CELLCHAT_BLUE,
):
    """Rank source-target pairs by differential communication."""

    changes = diff.source_target_changes.copy()
    if not changes.empty:
        changes["pair"] = changes["source"].astype(str) + " -> " + changes["target"].astype(str)
    df = _diff_summary_plot_frame(changes, label_col="pair", top_n=top_n, measure=measure)
    return _diff_rank_bar(
        df,
        label_col="pair",
        value_col=_diff_measure_col(measure),
        ax=ax,
        title=title or f"Differential source-target rank: {diff.label_a} - {diff.label_b}",
        xlabel="Delta interaction weight" if measure in {"weight", "prob"} else "Delta interaction count",
        pos_color=pos_color,
        neg_color=neg_color,
    )


def signaling_changes_scatter(
    diff: DifferentialCCC,
    group: str,
    *,
    pathways: Sequence[str] | None = None,
    exclude_pathways: Sequence[str] | None = None,
    top_label: float | int = 0.35,
    ax=None,
    title: str | None = None,
):
    """CellChat `netAnalysis_signalingChanges_scatter` style pathway-change scatter."""

    frame = signaling_changes(diff, group, pathways=pathways, exclude_pathways=exclude_pathways)
    ax = _plain_ax(ax)
    if frame.empty:
        return _empty_plot(ax, title or f"Signaling changes: {group}")
    colors = frame["direction"].map({diff.label_a: CELLCHAT_RED, diff.label_b: CELLCHAT_BLUE, "unchanged": "#9ca3af"}).fillna("#9ca3af")
    sizes_raw = frame["abs_delta_total"].to_numpy(dtype=float)
    sizes = 42 + 360 * sizes_raw / (sizes_raw.max() if sizes_raw.max() > 0 else 1)
    ax.scatter(frame["delta_outgoing"], frame["delta_incoming"], s=sizes, c=colors, edgecolor="white", linewidth=0.9, alpha=0.88)
    ax.axhline(0, color="#9ca3af", linewidth=0.65, linestyle="--")
    ax.axvline(0, color="#9ca3af", linewidth=0.65, linestyle="--")
    xlim = _zero_padded_limits(frame["delta_outgoing"])
    ylim = _zero_padded_limits(frame["delta_incoming"])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    label_frame = frame.head(_top_label_count(len(frame), top_label))
    x_mid = xlim[0] + 0.78 * (xlim[1] - xlim[0])
    texts = []
    for row in label_frame.itertuples(index=False):
        right_edge = float(row.delta_outgoing) >= x_mid
        texts.append(
            ax.annotate(
                str(row.pathway),
                (row.delta_outgoing, row.delta_incoming),
                xytext=(-4 if right_edge else 4, 0),
                textcoords="offset points",
                ha="right" if right_edge else "left",
                va="center",
                fontsize=8.5,
                color=INK,
            )
        )
    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "#c7c7c7", "lw": 0.55})

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CELLCHAT_RED, markeredgecolor="white", markersize=7, label=diff.label_a),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CELLCHAT_BLUE, markeredgecolor="white", markersize=7, label=diff.label_b),
    ]
    ax.legend(handles=handles, title="Higher in", frameon=False, loc="best")
    ax.set_xlabel(f"Delta outgoing strength ({diff.label_a} - {diff.label_b})")
    ax.set_ylabel(f"Delta incoming strength ({diff.label_a} - {diff.label_b})")
    ax.set_title(title or f"Signaling changes of {group}")
    _polish_plain_ax(ax)
    return ax


def pathway_similarity_rank(
    diff: DifferentialCCC,
    *,
    similarity: str = "functional",
    top_n: int = 30,
    method: str = "auto",
    n_neighbors: int | None = None,
    min_dist: float = 0.3,
    random_state: int | None = 0,
    ax=None,
    title: str | None = None,
):
    """CellChat `rankSimilarity` style ranking of pathway distances."""

    frame = rank_pathway_similarity(diff, similarity=similarity, method=method, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state).head(top_n)
    ax = _plain_ax(ax)
    if frame.empty:
        return _empty_plot(ax, title or f"Pathway distance rank ({similarity})")
    frame = frame.sort_values("distance", ascending=True)
    colors = np.where(frame["delta_prob"].to_numpy(dtype=float) >= 0, CELLCHAT_RED, CELLCHAT_BLUE)
    ax.barh(frame["pathway"].astype(str), frame["distance"].astype(float), color=colors, alpha=0.86)
    ax.set_xlabel("Euclidean distance in joint manifold")
    ax.set_ylabel("")
    ax.set_title(title or f"Pathway distance rank ({similarity})")
    _polish_plain_ax(ax)
    return ax


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
    ax=None,
    title: str | None = None,
):
    """CellChat `netVisual_embeddingPairwise` style joint pathway embedding."""

    frame = pairwise_pathway_embedding(
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
    ax = _plain_ax(ax)
    if frame.empty:
        return _empty_plot(ax, title or f"Pairwise pathway embedding ({similarity})")

    segments = _pairwise_embedding_segments(frame, diff.label_a, diff.label_b)
    if connect and not segments.empty:
        for row in segments.itertuples(index=False):
            ax.plot([row.x, row.xend], [row.y, row.yend], color="#9ca3af", linewidth=0.75, alpha=0.58, zorder=1)

    palette = {diff.label_a: CELLCHAT_RED, diff.label_b: CELLCHAT_BLUE}
    colors = frame["condition"].astype(str).map(palette).fillna(CELLCHAT_GREY)
    raw_sizes = frame["prob"].to_numpy(dtype=float)
    sizes = 55 + 250 * raw_sizes / (raw_sizes.max() if raw_sizes.max() > 0 else 1)
    ax.scatter(frame["dim1"], frame["dim2"], s=sizes, c=colors, edgecolor="white", linewidth=1.05, alpha=0.88, zorder=2)

    label_frame = _pairwise_embedding_labels(frame, segments, top_label=top_label, label_pathways=label_pathways)
    texts = [ax.text(row.dim1, row.dim2, f" {row.pathway}", fontsize=8.5, color=INK, va="center") for row in label_frame.itertuples(index=False)]
    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "#c7c7c7", "lw": 0.55})

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=palette[diff.label_a], markeredgecolor="white", markersize=7, label=diff.label_a),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=palette[diff.label_b], markeredgecolor="white", markersize=7, label=diff.label_b),
    ]
    condition_legend = ax.legend(handles=handles, title="Condition", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.add_artist(condition_legend)
    _size_legend(ax, raw_sizes, title="Pathway probability")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    method_label = str(frame["embedding_method"].iloc[0]) if "embedding_method" in frame.columns and len(frame) else method
    ax.set_title(title or f"Pairwise pathway embedding ({similarity}, {method_label})")
    _polish_plain_ax(ax)
    return ax


def centrality_network(result: CCCResult) -> pd.DataFrame:
    """Return basic outgoing/incoming/degree centrality values."""

    mat = result.network(value="prob")
    graph = nx.from_pandas_adjacency(mat, create_using=nx.DiGraph)
    hubs, authorities = _hits_scores(graph, result.groups)
    return pd.DataFrame(
        {
            "group": result.groups,
            "outgoing": mat.sum(axis=1).reindex(result.groups).to_numpy(),
            "incoming": mat.sum(axis=0).reindex(result.groups).to_numpy(),
            "mediator": pd.Series(nx.betweenness_centrality(graph, weight="weight", normalized=True)).reindex(result.groups).fillna(0.0).to_numpy(),
            "influencer": pd.Series(hubs).reindex(result.groups).fillna(0.0).to_numpy(),
            "authority": pd.Series(authorities).reindex(result.groups).fillna(0.0).to_numpy(),
            "pagerank": pd.Series(nx.pagerank(graph, weight="weight")).reindex(result.groups).to_numpy(),
        }
    )


def signaling_role_network(
    result: CCCResult,
    *,
    pathway: str | Sequence[str] | None = None,
    roles: Sequence[str] = ("sender", "receiver", "mediator", "influencer"),
    normalize: bool = True,
    significant_only: bool = False,
    ax=None,
    cmap: str = "pyccc_prob",
    annot: bool = False,
    title: str | None = None,
):
    """CellChat-style heatmap of sender/receiver/mediator/influencer roles."""

    mat = result.network(value="prob", pathway=pathway, significant_only=significant_only)
    graph = nx.from_pandas_adjacency(mat, create_using=nx.DiGraph)
    hubs, _ = _hits_scores(graph, result.groups)
    role_values = {
        "sender": mat.sum(axis=1).reindex(result.groups).to_numpy(dtype=float),
        "receiver": mat.sum(axis=0).reindex(result.groups).to_numpy(dtype=float),
        "mediator": pd.Series(nx.betweenness_centrality(graph, weight="weight", normalized=True)).reindex(result.groups).fillna(0.0).to_numpy(dtype=float),
        "influencer": pd.Series(hubs).reindex(result.groups).fillna(0.0).to_numpy(dtype=float),
    }
    missing = [role for role in roles if role not in role_values]
    if missing:
        raise ValueError(f"Unknown roles: {missing}.")
    frame = pd.DataFrame({role.title(): role_values[role] for role in roles}, index=result.groups).T
    if normalize:
        frame = frame.apply(lambda row: row / row.max() if row.max() > 0 else row, axis=1)

    ax = _plain_ax(ax)
    sns.heatmap(
        frame,
        cmap=_heatmap_cmap(cmap),
        mask=_zero_mask(frame),
        ax=ax,
        vmin=0,
        linewidths=0.65,
        linecolor="#efefef",
        annot=annot,
        cbar_kws={"label": "relative role score" if normalize else "role score", "shrink": 0.72},
    )
    ax.set_xlabel("Cell group")
    ax.set_ylabel("Network role")
    ax.set_title(title or _title("Signaling role network", pathway, result.condition))
    _polish_matrix_ax(ax)
    return ax


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
    ax=None,
    cmap: str = "pyccc_prob",
    title: str | None = None,
):
    """CellChat-style 2D embedding of signaling pathways by network similarity."""

    frame = compute_pathway_embedding(
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
    ax = _plain_ax(ax)
    if frame.empty:
        return _empty_plot(ax, title or f"Pathway embedding ({similarity})")

    if cluster:
        clusters = compute_pathway_clusters(
            result,
            similarity=similarity,
            significant_only=significant_only,
            min_prob=min_prob,
            thresh=thresh,
            method=cluster_method,
            n_clusters=n_clusters,
            embedding=frame,
            embedding_method=method,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            remove_isolates=remove_isolates,
        )[["pathway", "cluster"]]
        frame = frame.merge(clusters, on="pathway", how="left")
        raw_sizes = frame["prob"].to_numpy(dtype=float)
        sizes = 65 + 260 * raw_sizes / (raw_sizes.max() if raw_sizes.max() > 0 else 1)
        cluster_values = sorted(frame["cluster"].dropna().astype(int).unique())
        palette = _palette([str(value) for value in cluster_values], "pyccc_cells")
        colors = frame["cluster"].fillna(0).astype(int).astype(str).map(palette).fillna(CELLCHAT_GREY).to_list()
        ax.scatter(frame["dim1"], frame["dim2"], s=sizes, c=colors, edgecolor="white", linewidth=1.1, alpha=0.86)
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=palette[str(value)], markeredgecolor="white", markersize=7, label=f"Group {value}")
            for value in cluster_values
        ]
        if handles:
            cluster_legend = ax.legend(handles=handles, title="Signaling group", frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
            ax.add_artist(cluster_legend)
        _size_legend(ax, raw_sizes, title="Pathway probability")
    else:
        counts = frame["count"].to_numpy(dtype=float)
        sizes = 60 + 220 * counts / (counts.max() if counts.max() > 0 else 1)
        scatter = ax.scatter(
            frame["dim1"],
            frame["dim2"],
            s=sizes,
            c=frame["prob"],
            cmap=_heatmap_cmap(cmap),
            edgecolor="white",
            linewidth=1.1,
            alpha=0.86,
        )
        cbar = ax.figure.colorbar(scatter, ax=ax, shrink=0.72, pad=0.025)
        cbar.set_label("pathway probability")

    if label_pathways is None:
        label_frame = frame.head(_top_label_count(len(frame), top_label))
    else:
        label_frame = frame[frame["pathway"].isin(label_pathways)]
    texts = [ax.text(row.dim1, row.dim2, f" {row.pathway}", fontsize=8.5, color=INK, va="center") for row in label_frame.itertuples(index=False)]
    if adjust_text is not None and texts:
        adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "#c7c7c7", "lw": 0.55})

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    method_label = str(frame["embedding_method"].iloc[0]) if "embedding_method" in frame.columns and len(frame) else method
    suffix = f"{similarity}, {method_label}"
    if cluster:
        suffix += f", {cluster_method}"
    ax.set_title(title or f"Pathway network embedding ({suffix})")
    _polish_plain_ax(ax)
    return ax


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
    ax=None,
    cmap: str = "pyccc_expr",
    title: str | None = None,
):
    """AnnData-native CellChat `plotGeneExpression`-style expression plot."""

    if not 0 <= min_pct <= 1:
        raise ValueError("`min_pct` must be between 0 and 1.")
    kind = kind.lower()
    if kind not in {"dot", "violin", "bar"}:
        raise ValueError("`kind` must be one of: 'dot', 'violin', or 'bar'.")
    plot_title = title or _title("Signaling gene expression", signaling, result.condition if result is not None else None)
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
        return _signaling_expression_violin(values, ax=ax, title=plot_title)

    frame = signaling_expression_frame(
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
        return _signaling_expression_bar(frame, ax=ax, title=plot_title)
    return _signaling_expression_dot(
        frame,
        ax=ax,
        cmap=cmap,
        title=plot_title,
        standard_scale=standard_scale,
        scale_min=scale_min,
        scale_max=scale_max,
        min_pct=min_pct,
    )


def spatial_network(
    adata,
    result: CCCResult,
    *,
    groupby: str | None = None,
    basis: str = "spatial",
    pathway: str | Sequence[str] | None = None,
    weight: str = "prob",
    significant_only: bool = False,
    show_cells: bool = True,
    ax=None,
    cmap: str = "pyccc_cells",
    title: str | None = None,
):
    """Spatial cell-cell communication network over group centroids."""

    groupby = groupby or result.groupby
    if basis not in adata.obsm:
        raise KeyError(f"`{basis}` is not present in adata.obsm.")
    if groupby not in adata.obs:
        raise KeyError(f"`{groupby}` is not present in adata.obs.")
    coords = np.asarray(adata.obsm[basis])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"`adata.obsm[{basis!r}]` must have at least two coordinate columns.")
    frame = pd.DataFrame({"group": adata.obs[groupby].astype(str).to_numpy(), "x": coords[:, 0], "y": coords[:, 1]})
    centroids = frame.groupby("group", observed=True)[["x", "y"]].mean().reindex(result.groups).dropna()
    mat = result.network(value=weight, pathway=pathway, significant_only=significant_only).reindex(index=centroids.index, columns=centroids.index, fill_value=0.0)
    colors = _palette(list(centroids.index), cmap)
    ax = _plain_ax(ax)
    if show_cells:
        for group, sub in frame.groupby("group", observed=True):
            if group in colors:
                ax.scatter(sub["x"], sub["y"], s=12, color=colors[group], alpha=0.18, linewidth=0, zorder=1)

    vmax = _finite_absmax(mat.to_numpy(), default=1.0)
    for source in mat.index:
        for target in mat.columns:
            val = float(mat.loc[source, target])
            if val <= 0:
                continue
            _draw_spatial_edge(
                ax,
                tuple(centroids.loc[source]),
                tuple(centroids.loc[target]),
                color=colors[source],
                lw=0.6 + 5.0 * val / vmax,
                alpha=0.45,
            )

    totals = mat.sum(axis=1).add(mat.sum(axis=0), fill_value=0).reindex(centroids.index).to_numpy(dtype=float)
    sizes = 180 + 760 * totals / (totals.max() if totals.max() > 0 else 1)
    for i, group in enumerate(centroids.index):
        x, y = centroids.loc[group]
        ax.scatter([x], [y], s=sizes[i], color=colors[group], edgecolor="white", linewidth=1.7, zorder=4)
        ax.text(x, y, group, ha="center", va="center", fontsize=9, color=INK, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("spatial 1")
    ax.set_ylabel("spatial 2")
    ax.set_title(title or _title("Spatial communication", pathway, result.condition))
    _polish_plain_ax(ax)
    return ax


def key_plot_gallery(result: CCCResult, diff: DifferentialCCC | None = None, *, figsize: tuple[float, float] | None = None):
    """Create a CellChat-style gallery of the main pyccc visual summaries."""

    if diff is None:
        figsize = figsize or (14, 10)
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        net_circle(result, ax=axes[0, 0])
        net_heatmap(result, ax=axes[0, 1])
        bubble(result, ax=axes[0, 2], top_n=25)
        pathway_heatmap(result, ax=axes[1, 0])
        signaling_role_network(result, ax=axes[1, 1])
        pathway_river(result, ax=axes[1, 2], top_n=8)
    else:
        figsize = figsize or (16, 12)
        fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=True)
        net_circle(result, ax=axes[0, 0])
        net_heatmap(result, ax=axes[0, 1])
        bubble(result, ax=axes[0, 2], top_n=25)
        pathway_heatmap(result, ax=axes[1, 0])
        signaling_role_network(result, ax=axes[1, 1])
        pathway_river(result, ax=axes[1, 2], top_n=8)
        compare_interactions(diff, ax=axes[2, 0])
        diff_heatmap(diff, ax=axes[2, 1])
        diff_bubble(diff, ax=axes[2, 2], top_n=25)
    return fig, axes


def masterpiece_gallery(
    result: CCCResult,
    diff: DifferentialCCC | None = None,
    *,
    title: str = "Cell-cell communication atlas",
    subtitle: str | None = None,
    outfile: str | None = None,
    dpi: int = 360,
):
    """Create a polished multi-panel figure intended for publication/export."""

    set_theme(context="paper", font_scale=1.05, style="nature")
    if diff is None:
        fig = plt.figure(figsize=(16, 10.8), constrained_layout=False)
        gs = fig.add_gridspec(
            2,
            6,
            left=0.055,
            right=0.985,
            bottom=0.075,
            top=0.88,
            wspace=0.9,
            hspace=0.42,
            height_ratios=[1.0, 1.12],
        )
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 0:3]),
            fig.add_subplot(gs[1, 3:6]),
        ]
        plotters = [
            lambda ax: net_circle(result, ax=ax, title="Global communication network"),
            lambda ax: net_heatmap(result, ax=ax, title="Source-target strength"),
            lambda ax: signaling_role_network(result, ax=ax, title="Network role scores"),
            lambda ax: bubble(result, ax=ax, top_n=14, top_pairs=10, compact_pairs=True, show_size_legend=False, title="Top ligand-receptor programs"),
            lambda ax: pathway_river(result, ax=ax, top_n=10, title="Pathway river"),
        ]
    else:
        fig = plt.figure(figsize=(18, 13.2), constrained_layout=False)
        gs = fig.add_gridspec(
            3,
            6,
            left=0.055,
            right=0.985,
            bottom=0.07,
            top=0.865,
            wspace=1.15,
            hspace=0.58,
            height_ratios=[0.98, 1.08, 1.0],
        )
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 0:3]),
            fig.add_subplot(gs[1, 3:6]),
            fig.add_subplot(gs[2, 0:2]),
            fig.add_subplot(gs[2, 2:4]),
            fig.add_subplot(gs[2, 4:6]),
        ]
        plotters = [
            lambda ax: net_circle(result, ax=ax, title="Global communication network"),
            lambda ax: diff_network_circle(diff, ax=ax, title=f"Differential network: {diff.label_a} vs {diff.label_b}"),
            lambda ax: compare_interactions(diff, ax=ax, title="Global interaction shift"),
            lambda ax: bubble(result, ax=ax, top_n=12, top_pairs=8, compact_pairs=True, show_size_legend=False, title="Top ligand-receptor programs"),
            lambda ax: pathway_river(result, ax=ax, top_n=10, title="Pathway river"),
            lambda ax: diff_heatmap(diff, ax=ax, title="Differential source-target strength"),
            lambda ax: diff_bubble(diff, ax=ax, top_n=14, top_pairs=8, compact_pairs=True, show_size_legend=False, title="Differential ligand-receptor programs"),
            lambda ax: rank_signaling_compare(diff, ax=ax, top_n=10, title="Ranked pathway shift"),
        ]

    for i, (ax, plotter) in enumerate(zip(axes, plotters), start=1):
        plotter(ax)
        _panel_label(ax, chr(64 + i))

    fig.suptitle(title, x=0.055, y=0.965, ha="left", fontsize=18, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.055, 0.93, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
    if outfile:
        save_figure(fig, outfile, dpi=dpi, also_svg=True)
    return fig, axes


def _signaling_expression_dot(
    frame: pd.DataFrame,
    *,
    ax,
    cmap: str,
    title: str,
    standard_scale: bool,
    scale_min: float,
    scale_max: float,
    min_pct: float,
):
    ax = _plain_ax(ax)
    if frame.empty:
        return _empty_plot(ax, title)

    data = frame.copy()
    groups = _category_levels(data["group"])
    genes = _category_levels(data["gene"])
    y_levels = genes[::-1]
    data["_x"] = pd.Categorical(data["group"].astype(str), categories=groups, ordered=True).codes
    data["_y"] = pd.Categorical(data["gene"].astype(str), categories=y_levels, ordered=True).codes
    data["pct_size"] = np.where(data["pct_expressed"].to_numpy(dtype=float) >= min_pct, data["pct_expressed"].to_numpy(dtype=float) * 100.0, 0.0)
    raw_sizes = data["pct_size"].to_numpy(dtype=float)
    sizes = 30 + 420 * raw_sizes / (raw_sizes.max() if raw_sizes.max() > 0 else 1)
    colors = data["scaled_expression"].to_numpy(dtype=float)
    if standard_scale:
        norm = plt.Normalize(scale_min, scale_max)
        colorbar_label = "Scaled expression"
    else:
        finite = colors[np.isfinite(colors)]
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin, vmax)
        colorbar_label = "Average expression"

    scatter = ax.scatter(data["_x"], data["_y"], s=sizes, c=colors, cmap=_cmap(cmap), norm=norm, edgecolor="#2f3136", linewidth=0.22, alpha=0.92)
    ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right", va="top", fontsize=_categorical_fontsize(groups))
    ax.set_yticks(range(len(y_levels)), y_levels, fontsize=_categorical_fontsize(y_levels))
    ax.set_xlim(-0.5, max(len(groups) - 0.5, 0.5))
    ax.set_ylim(-0.5, max(len(y_levels) - 0.5, 0.5))
    _bubble_grid(ax, len(groups), len(y_levels))
    ax.set_axisbelow(True)
    ax.set_xlabel("Cell group")
    ax.set_ylabel("Signaling gene")
    ax.set_title(title)
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.032, pad=0.028, aspect=32)
    cbar.set_label(colorbar_label)
    cbar.outline.set_edgecolor("#d4d4d8")
    cbar.outline.set_linewidth(0.6)
    _size_legend(ax, raw_sizes, title="Percent expressed")
    return ax


def _signaling_expression_violin(values: pd.DataFrame, *, ax, title: str):
    ax = _plain_ax(ax)
    if values.empty:
        return _empty_plot(ax, title)
    groups = _category_levels(values["group"])
    genes = _category_levels(values["gene"])
    palette = {gene: _cmap("pyccc_prob")(i / max(len(genes) - 1, 1)) for i, gene in enumerate(genes)}
    sns.violinplot(
        data=values,
        x="group",
        y="expression",
        hue="gene",
        order=groups,
        hue_order=genes,
        cut=0,
        inner="quartile",
        linewidth=0.45,
        dodge=True,
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Cell group")
    ax.set_ylabel("Expression")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    if len(genes) <= 1 and ax.get_legend() is not None:
        ax.get_legend().remove()
    elif ax.get_legend() is not None:
        ax.legend(title="Signaling gene", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    _polish_plain_ax(ax)
    return ax


def _signaling_expression_bar(frame: pd.DataFrame, *, ax, title: str):
    ax = _plain_ax(ax)
    if frame.empty:
        return _empty_plot(ax, title)
    groups = _category_levels(frame["group"])
    genes = _category_levels(frame["gene"])
    palette = {gene: _cmap("pyccc_prob")(i / max(len(genes) - 1, 1)) for i, gene in enumerate(genes)}
    sns.barplot(
        data=frame,
        x="group",
        y="mean_expression",
        hue="gene",
        order=groups,
        hue_order=genes,
        errorbar=None,
        linewidth=0.35,
        edgecolor="#2f3136",
        palette=palette,
        ax=ax,
    )
    ax.set_xlabel("Cell group")
    ax.set_ylabel("Average expression")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    if len(genes) <= 1 and ax.get_legend() is not None:
        ax.get_legend().remove()
    elif ax.get_legend() is not None:
        ax.legend(title="Signaling gene", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    _polish_plain_ax(ax)
    return ax


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


def _gene_chord_frame(
    result: CCCResult,
    *,
    sources: Sequence[str] | None,
    targets: Sequence[str] | None,
    pathways: Sequence[str] | None,
    interactions: Sequence[str] | None,
    level: str,
    top_n: int | None,
    significant_only: bool,
) -> pd.DataFrame:
    if level not in {"lr", "interaction", "pathway"}:
        raise ValueError("`level` must be one of: 'lr', 'interaction', or 'pathway'.")
    df = _filter_interactions(result, sources=sources, targets=targets, pathways=pathways, significant_only=significant_only).copy()
    if df.empty:
        return pd.DataFrame(columns=["source", "target", "feature", "prob", "count"])
    if level == "pathway":
        df["feature"] = df["pathway"].astype(str)
    else:
        df["feature"] = _interaction_labels(df)
    if interactions is not None:
        df = df[df["feature"].isin(interactions)]
    if df.empty:
        return pd.DataFrame(columns=["source", "target", "feature", "prob", "count"])
    if top_n is not None:
        keep = df.groupby("feature", observed=True)["prob"].sum().sort_values(ascending=False).head(top_n).index
        df = df[df["feature"].isin(keep)]
    frame = (
        df.groupby(["source", "target", "feature"], observed=True)
        .agg(prob=("prob", "sum"), count=("prob", "size"))
        .reset_index()
        .sort_values("prob", ascending=False)
        .reset_index(drop=True)
    )
    return frame


def _individual_interaction_frame(
    result: CCCResult,
    *,
    pathway: str | Sequence[str] | None,
    interaction: str | Sequence[str] | None,
    ligand: str | Sequence[str] | None,
    receptor: str | Sequence[str] | None,
    significant_only: bool,
) -> tuple[pd.DataFrame, str]:
    pathways = _as_list(pathway)
    df = _filter_interactions(result, sources=None, targets=None, pathways=pathways, significant_only=significant_only).copy()
    if df.empty:
        return df, "No interaction"
    df["_interaction_label"] = _interaction_labels(df)
    if interaction is not None:
        keep = set(_as_list(interaction) or [])
        df = df[df["_interaction_label"].isin(keep)]
    if ligand is not None:
        df = df[df["ligand"].astype(str).isin(set(_as_list(ligand) or []))]
    if receptor is not None:
        df = df[df["receptor"].astype(str).isin(set(_as_list(receptor) or []))]
    if df.empty:
        return df.drop(columns=["_interaction_label"], errors="ignore"), "No interaction"

    summary = (
        df.groupby(["ligand", "receptor", "pathway", "_interaction_label"], observed=True)["prob"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    chosen = summary.iloc[0]
    selected = df[
        df["ligand"].astype(str).eq(str(chosen["ligand"]))
        & df["receptor"].astype(str).eq(str(chosen["receptor"]))
        & df["pathway"].astype(str).eq(str(chosen["pathway"]))
    ].drop(columns=["_interaction_label"], errors="ignore")
    label = str(chosen["_interaction_label"])
    return selected.reset_index(drop=True), label


def _as_list(value) -> list | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _resolve_vertex_groups(groups: Sequence[str], selected: Sequence[str | int] | None) -> list[str]:
    if selected is None:
        return list(groups)
    out = []
    for item in selected:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if 1 <= idx <= len(groups):
                out.append(str(groups[idx - 1]))
            elif 0 <= idx < len(groups):
                out.append(str(groups[idx]))
            else:
                raise ValueError(f"`vertex_receiver` index {idx} is outside the cell-group range.")
        else:
            if item not in groups:
                raise ValueError(f"`vertex_receiver` group {item!r} is not present in result.groups.")
            out.append(str(item))
    return out


def _mediated_chord_positions(cell_nodes: Sequence[str], feature_nodes: Sequence[str]) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    cell_angles = _arc_angles(len(cell_nodes), start=130, end=230)
    feature_angles = _arc_angles(len(feature_nodes), start=-55, end=55)
    for node, angle in zip(cell_nodes, cell_angles):
        coords[node] = (float(np.cos(angle)), float(np.sin(angle)))
    for node, angle in zip(feature_nodes, feature_angles):
        coords[node] = (float(np.cos(angle)), float(np.sin(angle)))
    return coords


def _arc_angles(n: int, *, start: float, end: float) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.deg2rad(np.array([(start + end) / 2.0], dtype=float))
    return np.deg2rad(np.linspace(start, end, n))


def _bubble_frame(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    size: str,
    ax=None,
    cmap: str,
    center: float | None = None,
    title: str,
    colorbar_label: str | None = None,
    size_label: str | None = None,
    show_size_legend: bool = True,
):
    ax = _plain_ax(ax)
    if df.empty:
        return _empty_plot(ax, title)

    frame = df.copy()
    x_levels = list(dict.fromkeys(frame[x].astype(str)))
    y_levels = list(dict.fromkeys(frame[y].astype(str)))
    frame["_x"] = pd.Categorical(frame[x].astype(str), categories=x_levels, ordered=True).codes
    frame["_y"] = pd.Categorical(frame[y].astype(str), categories=y_levels, ordered=True).codes
    raw_sizes = pd.to_numeric(frame[size], errors="coerce").fillna(0.0).abs().to_numpy()
    sizes = 30 + 420 * raw_sizes / (raw_sizes.max() if raw_sizes.max() > 0 else 1)
    colors = pd.to_numeric(frame[color], errors="coerce").fillna(0.0).to_numpy()
    if center is None:
        vmin = float(np.min(colors)) if colors.size else 0.0
        vmax = float(np.max(colors)) if colors.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin, vmax)
    else:
        vmax = _finite_absmax(colors - center, default=1.0)
        norm = TwoSlopeNorm(vcenter=center, vmin=center - vmax, vmax=center + vmax)
    scatter = ax.scatter(frame["_x"], frame["_y"], s=sizes, c=colors, cmap=_cmap(cmap), norm=norm, edgecolor="#2f3136", linewidth=0.22, alpha=0.9)
    ax.set_xticks(range(len(x_levels)), x_levels, rotation=90, ha="center", va="top", fontsize=_categorical_fontsize(x_levels))
    ax.set_yticks(range(len(y_levels)), y_levels, fontsize=_categorical_fontsize(y_levels))
    ax.set_xlim(-0.5, max(len(x_levels) - 0.5, 0.5))
    ax.set_ylim(-0.5, max(len(y_levels) - 0.5, 0.5))
    _bubble_grid(ax, len(x_levels), len(y_levels))
    ax.set_axisbelow(True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.032, pad=0.028, aspect=32)
    cbar.set_label(colorbar_label or color)
    cbar.outline.set_edgecolor("#d4d4d8")
    cbar.outline.set_linewidth(0.6)
    if show_size_legend:
        _size_legend(ax, raw_sizes, title=size_label or size)
    return ax


def _diff_measure_col(measure: str) -> str:
    if measure in {"weight", "prob"}:
        return "delta_prob"
    if measure == "count":
        return "delta_count"
    raise ValueError("`measure` must be one of: 'weight', 'prob', or 'count'.")


def _diff_summary_plot_frame(df: pd.DataFrame, *, label_col: str, top_n: int, measure: str) -> pd.DataFrame:
    value_col = _diff_measure_col(measure)
    if label_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=[label_col, value_col])
    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
    out = out.loc[out[value_col] != 0].copy()
    out["_abs"] = out[value_col].abs()
    out = out.sort_values("_abs", ascending=False).head(top_n)
    return out.sort_values(value_col, ascending=True).reset_index(drop=True)


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


def _diff_rank_bar(
    df: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    ax,
    title: str,
    xlabel: str,
    pos_color: str,
    neg_color: str,
):
    ax = _plain_ax(ax)
    if df.empty:
        return _empty_plot(ax, title)
    colors = np.where(df[value_col].to_numpy(dtype=float) >= 0, pos_color, neg_color)
    ax.barh(df[label_col].astype(str), df[value_col].astype(float), color=colors, alpha=0.88)
    ax.axvline(0, color="#3f3f46", linewidth=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)
    _polish_plain_ax(ax)
    return ax


def _matrix_edges(mat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in mat.index:
        for target in mat.columns:
            weight = float(mat.loc[source, target])
            if weight > 0:
                rows.append({"source": source, "target": target, "weight": weight})
    return pd.DataFrame(rows, columns=["source", "target", "weight"])


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


def _cluster_matrix(mat: pd.DataFrame, *, cluster_rows: bool, cluster_cols: bool) -> pd.DataFrame:
    out = mat.copy()
    if cluster_rows:
        out = out.iloc[_cluster_order(out.to_numpy(dtype=float)), :]
    if cluster_cols:
        out = out.iloc[:, _cluster_order(out.T.to_numpy(dtype=float))]
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
    df = compute_pathway_communication(result, significant_only=significant_only)
    if pathways is not None:
        df = df[df["pathway"].isin(pathways)]
    if mode == "outgoing":
        df = df.rename(columns={"source": "group"})
        if groups is not None:
            df = df[df["group"].isin(groups)]
    elif mode == "incoming":
        df = df.rename(columns={"target": "group"})
        if groups is not None:
            df = df[df["group"].isin(groups)]
    else:
        left = df.rename(columns={"source": "group"}).copy()
        right = df.rename(columns={"target": "group"}).copy()
        df = pd.concat([left, right], ignore_index=True)
        if groups is not None:
            df = df[df["group"].isin(groups)]
    flows = df.groupby(["group", "pathway"], observed=True)["prob"].sum().reset_index()
    flows = flows[flows["prob"] > min_prob]
    if top_n is not None:
        keep = flows.groupby("pathway", observed=True)["prob"].sum().sort_values(ascending=False).head(top_n).index
        flows = flows[flows["pathway"].isin(keep)]
    return flows.sort_values("prob", ascending=False).reset_index(drop=True)


def _stack_available(n: int, *, y0: float = 0.05, y1: float = 0.95, gap: float = 0.014) -> float:
    return max((y1 - y0) - gap * max(n - 1, 0), 0.0)


def _stack_layout(totals: pd.Series, *, y0: float = 0.05, y1: float = 0.95, gap: float = 0.014) -> dict[str, tuple[float, float]]:
    totals = totals.astype(float)
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
        spans[key] = (bottom, top)
        cursor = bottom - gap
    return spans


def _draw_ribbon(ax, x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float, *, color, alpha: float):
    mid = (x0 + x1) / 2
    verts = [
        (x0, y0a),
        (mid, y0a),
        (mid, y1a),
        (x1, y1a),
        (x1, y1b),
        (mid, y1b),
        (mid, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def _hits_scores(graph, groups: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    if graph.number_of_edges() == 0:
        zeros = pd.Series(0.0, index=groups)
        return zeros, zeros
    try:
        hubs, authorities = nx.hits(graph, max_iter=1000, normalized=True)
    except nx.PowerIterationFailedConvergence:
        zeros = pd.Series(0.0, index=groups)
        return zeros, zeros
    return pd.Series(hubs), pd.Series(authorities)


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


def _finite_absmax(values, *, default: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    vmax = float(np.abs(arr).max())
    return vmax if vmax > 0 else default


def _empty_plot(ax, title: str):
    ax.text(0.5, 0.5, "No interactions", ha="center", va="center", color=MUTED)
    ax.set_axis_off()
    ax.set_title(title)
    return ax


def _compact_pair(pair: str) -> str:
    if " -> " not in pair:
        return pair
    source, target = pair.split(" -> ", 1)
    return f"{source}\n{target}"


def _circle_positions(groups: Sequence[str]) -> dict[str, tuple[float, float]]:
    angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False)
    return {group: (float(np.cos(angle)), float(np.sin(angle))) for group, angle in zip(groups, angles)}


def _palette(groups: Sequence[str], cmap: str) -> dict[str, object]:
    if cmap == "pyccc_cells":
        colors = CELLCHAT_PALETTE
        if len(groups) > len(colors):
            colors = list(plt.get_cmap("tab20")(np.linspace(0, 1, len(groups))))
    else:
        colors = _cmap(cmap)(np.linspace(0, 1, max(len(groups), 1)))
    return {group: colors[i] for i, group in enumerate(groups)}


def _cmap(cmap: str):
    if cmap in CMAP_REGISTRY:
        return CMAP_REGISTRY[cmap]
    return plt.get_cmap(cmap)


def _heatmap_cmap(cmap: str):
    cmap_obj = _cmap(cmap).copy()
    cmap_obj.set_bad("white")
    return cmap_obj


def _zero_mask(mat: pd.DataFrame) -> pd.DataFrame:
    values = mat.to_numpy(dtype=float)
    if np.count_nonzero(np.isfinite(values) & (np.abs(values) > 0)) <= 1:
        return pd.DataFrame(False, index=mat.index, columns=mat.columns)
    return pd.DataFrame(np.isclose(values, 0.0), index=mat.index, columns=mat.columns)


def _node_shrink_map(groups: Sequence[str], node_sizes: np.ndarray, *, linewidth: float = 2.0) -> dict[str, float]:
    sizes = np.asarray(node_sizes, dtype=float)
    # Matplotlib marker sizes are in points squared; FancyArrowPatch shrink is in points.
    shrinks = np.sqrt(np.maximum(sizes, 1.0)) / 2.0 + linewidth * 0.5
    return {group: float(shrink) for group, shrink in zip(groups, shrinks)}


def _draw_curved_edge(
    ax,
    start,
    end,
    *,
    color,
    lw: float,
    alpha: float,
    curvature: float = 0.0,
    start_shrink: float = 10.0,
    end_shrink: float = 10.5,
):
    if start == end:
        x, y = start
        circle = plt.Circle((x * 0.88, y * 0.88), 0.135, fill=False, color=color, linewidth=max(lw * 0.72, 0.45), alpha=alpha)
        ax.add_patch(circle)
        return
    rad = 0.25 if curvature == 0.0 else curvature
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=4.2 + min(lw, 4.0) * 0.58,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=lw,
        alpha=alpha,
        shrinkA=start_shrink,
        shrinkB=end_shrink,
        joinstyle="miter",
        capstyle="round",
        zorder=2,
    )
    ax.add_patch(arrow)


def _draw_circle_label(ax, x: float, y: float, label: str, *, scale: float = 1.16) -> None:
    ha = "left" if x > 0.18 else "right" if x < -0.18 else "center"
    va = "bottom" if y > 0.18 else "top" if y < -0.18 else "center"
    ax.text(
        x * scale,
        y * scale,
        label,
        ha=ha,
        va=va,
        fontsize=9,
        clip_on=False,
        color=INK,
        zorder=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.45},
    )


def _draw_bezier(ax, start, end, *, control=None, color, lw: float, alpha: float):
    if control is None:
        control = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.08)
    path = Path([start, control, end], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha, capstyle="round"))


def _draw_spatial_edge(ax, start, end, *, color, lw: float, alpha: float):
    if start == end:
        ax.scatter([start[0]], [start[1]], s=lw * 35, facecolor="none", edgecolor=color, linewidth=lw, alpha=alpha, zorder=2)
        return
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8 + lw,
        connectionstyle="arc3,rad=0.08",
        color=color,
        linewidth=lw,
        alpha=alpha,
        shrinkA=18,
        shrinkB=18,
        zorder=2,
    )
    ax.add_patch(arrow)


def _plain_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    _panel_face(ax)
    return ax


def _circle_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    _panel_face(ax)
    return ax


def _finish_circle_ax(ax):
    ax.set_xlim(-1.58, 1.58)
    ax.set_ylim(-1.58, 1.58)
    ax.set_axis_off()
    ax.set_aspect("equal")


def _panel_face(ax):
    ax.set_facecolor(PANEL_BG)


def _polish_plain_ax(ax):
    ax.grid(True, axis="x", color="#e5e7eb", linewidth=0.7)
    ax.grid(False, axis="y")
    ax.spines["left"].set_color("#d4d4d8")
    ax.spines["bottom"].set_color("#d4d4d8")
    ax.tick_params(length=0)


def _polish_matrix_ax(ax):
    ax.tick_params(length=0, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("center")
        label.set_va("top")
    for label in ax.get_yticklabels():
        label.set_rotation(0)


def _bubble_grid(ax, n_x: int, n_y: int) -> None:
    ax.grid(False)
    if n_x > 1:
        ax.vlines(np.arange(0.5, n_x - 0.5, 1), -0.5, n_y - 0.5, colors="#e5e7eb", linewidth=0.32, zorder=0)
    if n_y > 1:
        ax.hlines(np.arange(0.5, n_y - 0.5, 1), -0.5, n_x - 0.5, colors="#e5e7eb", linewidth=0.32, zorder=0)


def _categorical_fontsize(levels: Sequence[str]) -> float:
    if len(levels) <= 8:
        return 9.0
    if len(levels) <= 16:
        return 8.0
    return 7.0


def _category_levels(values: pd.Series) -> list[str]:
    if isinstance(values.dtype, pd.CategoricalDtype):
        return [str(value) for value in values.cat.categories]
    return list(dict.fromkeys(values.astype(str)))


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


def _panel_label(ax, label: str):
    bbox = ax.get_position()
    ax.figure.text(
        max(bbox.x0 - 0.034, 0.002),
        min(bbox.y1 + 0.018, 0.985),
        label,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=INK,
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.4},
    )


def _size_legend(ax, values: np.ndarray, *, title: str):
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size == 0:
        return
    refs = np.unique(np.quantile(values, [0.25, 0.5, 0.9]).round(3))
    max_v = values.max()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=CELLCHAT_GREY,
            markersize=np.sqrt(30 + 420 * ref / (max_v if max_v > 0 else 1)) / 1.5,
            label=f"{ref:g}",
        )
        for ref in refs
    ]
    if handles:
        ax.legend(handles=handles, title=title, frameon=False, loc="center left", bbox_to_anchor=(1.14, 0.5), borderaxespad=0)


def _title(base: str, pathway, condition: str | None) -> str:
    bits = [base]
    if pathway is not None:
        bits.append(str(pathway))
    if condition:
        bits.append(str(condition))
    return " | ".join(bits)

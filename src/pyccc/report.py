from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .analysis import CCCResult
from .diff import DifferentialCCC
from . import plotting as cp


def save_cellchat_report(
    result: CCCResult,
    outfile: str | Path,
    *,
    diff: DifferentialCCC | None = None,
    title: str = "pyccc CellChat-style report",
    dpi: int = 180,
) -> Path:
    """Save a multi-page CellChat-style visual report as PDF."""

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(outfile) as pdf:
        _save_overview_page(pdf, result, diff=diff, title=title, dpi=dpi)
        _save_network_page(pdf, result, title="Network summaries", dpi=dpi)
        _save_pathway_page(pdf, result, title="Pathway summaries", dpi=dpi)
        _save_table_page(pdf, result, diff=diff, title="Summary tables", dpi=dpi)
        if diff is not None:
            _save_diff_page(pdf, diff, title="Differential communication", dpi=dpi)
    return outfile


def _save_overview_page(pdf: PdfPages, result: CCCResult, *, diff: DifferentialCCC | None, title: str, dpi: int) -> None:
    fig, axes = cp.key_plot_gallery(result, diff)
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=16, fontweight="bold")
    pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_network_page(pdf: PdfPages, result: CCCResult, *, title: str, dpi: int) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(21, 10), constrained_layout=True)
    cp.net_circle(result, ax=axes[0, 0], title="Global network")
    cp.net_chord(result, ax=axes[0, 1], title="Chord network")
    cp.net_chord_gene(result, ax=axes[0, 2], top_n=12, title="LR-mediated chord")
    cp.net_individual(result, ax=axes[0, 3], layout="circle", title="Individual LR circle")
    cp.net_heatmap(result, ax=axes[1, 0], cluster_rows=True, cluster_cols=True, title="Network heatmap")
    cp.net_hierarchy(result, ax=axes[1, 1], title="Hierarchy network")
    cp.signaling_role_network(result, ax=axes[1, 2], title="Network roles")
    cp.pathway_embedding(result, ax=axes[1, 3], cluster=True, title="Pathway embedding groups")
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=15, fontweight="bold")
    pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_pathway_page(pdf: PdfPages, result: CCCResult, *, title: str, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    cp.pathway_heatmap(result, ax=axes[0, 0], top_pairs=12, compact_pairs=True, cluster_rows=True, title="Pathway heatmap")
    cp.pathway_river(result, ax=axes[0, 1], top_n=10, title="Pathway river")
    cp.rank_signaling(result, ax=axes[0, 2], top_n=15, title="Ranked pathways")
    cp.bubble(result, ax=axes[1, 0], top_n=18, top_pairs=10, compact_pairs=True, show_size_legend=False, title="Top LR programs")
    top_pathways = result.pathway_summary().head(2)["pathway"].astype(str).tolist()
    for ax, pathway in zip(axes[1, 1:], top_pathways):
        cp.lr_contribution(result, pathway, top_n=6, ax=ax, title=f"{pathway} LR contribution")
    for ax in axes[1, 1 + len(top_pathways) :]:
        ax.set_axis_off()
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=15, fontweight="bold")
    pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_diff_page(pdf: PdfPages, diff: DifferentialCCC, *, title: str, dpi: int) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(21, 14), constrained_layout=True)
    cp.compare_interactions(diff, ax=axes[0, 0], title="Interaction shift")
    cp.diff_network_circle(diff, ax=axes[0, 1], title="Differential network")
    cp.diff_pathway_rank(diff, ax=axes[0, 2], top_n=14, title="Differential pathway rank")
    cp.rank_signaling_compare(diff, ax=axes[0, 3], top_n=14, title="Condition pathway rank")
    cp.diff_heatmap(diff, ax=axes[1, 0], cluster_rows=True, cluster_cols=True, title="Differential heatmap")
    cp.diff_bubble(diff, ax=axes[1, 1], top_n=18, compact_pairs=True, show_size_legend=False, title="Differential LR programs")
    cp.diff_source_target_rank(diff, ax=axes[1, 2], top_n=14, title="Differential source-target rank")
    cp.pathway_embedding_pairwise(diff, ax=axes[1, 3], top_label=3, title="Pairwise pathway embedding")
    cp.signaling_changes_scatter(diff, diff.groups[0], ax=axes[2, 0], title=f"Signaling changes: {diff.groups[0]}")
    cp.pathway_similarity_rank(diff, ax=axes[2, 1], top_n=14, title="Pathway distance rank")
    cp.signaling_role_heatmap_compare(diff, mode="outgoing", axes=[axes[2, 2], axes[2, 3]], title="Outgoing role heatmaps")
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=15, fontweight="bold")
    pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_table_page(pdf: PdfPages, result: CCCResult, *, diff: DifferentialCCC | None, title: str, dpi: int) -> None:
    nrows = 3 if diff is None else 4
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 3.6 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    _draw_table(axes[0], result.pathway_summary().head(10), title="Top pathways")
    _draw_table(axes[1], result.lr_summary().head(10), title="Top ligand-receptor pairs")
    _draw_table(axes[2], cp.centrality_network(result).head(10), title="Network centrality")
    if diff is not None:
        cols = ["source", "target", "ligand", "receptor", "pathway", "delta_prob", "log2fc"]
        _draw_table(axes[3], diff.interactions[[col for col in cols if col in diff.interactions.columns]].head(10), title="Top differential ligand-receptor pairs")
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=15, fontweight="bold")
    pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _draw_table(ax, df, *, title: str) -> None:
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    frame = df.copy()
    for col in frame.columns:
        if frame[col].dtype.kind in {"f", "c"}:
            frame[col] = frame[col].map(lambda value: f"{value:.3g}")
    table = ax.table(cellText=frame.astype(str).to_numpy(), colLabels=list(frame.columns), loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.18)

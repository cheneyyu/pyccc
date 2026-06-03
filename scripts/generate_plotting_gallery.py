from __future__ import annotations

import ast
import inspect
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc
import pyccc.ggplot as cg
import pyccc.interactive as ci
import pyccc.plotting as cp


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
STATIC = DOCS / "_static" / "plotting"
INTERACTIVE = DOCS / "_static" / "interactive"
GALLERY = DOCS / "plotting_gallery.md"


@dataclass
class Example:
    name: str
    func: Callable[[], object]
    code: str
    width: float = 6.4
    height: float = 4.8


def make_demo_adata(seed: int = 7) -> AnnData:
    rng = np.random.default_rng(seed)
    cell_types = ["T cell", "Myeloid", "Fibroblast", "Endothelial", "Tumor"]
    conditions = ["control", "treated"]
    genes = [
        "CXCL9",
        "CXCL10",
        "CXCL12",
        "CCL2",
        "TGFB1",
        "VEGFA",
        "MIF",
        "CD74",
        "COL1A1",
        "LAMA1",
        "TNF",
        "IL6",
        "IFNG",
        "PDGFA",
        "JAG1",
        "CXCR3",
        "CXCR4",
        "ACKR3",
        "CCR2",
        "TGFBR1",
        "TGFBR2",
        "KDR",
        "ITGA6",
        "ITGB1",
        "TNFRSF1A",
        "IL6R",
        "IFNGR1",
        "PDGFRA",
        "NOTCH1",
    ]
    rows = []
    obs = []
    for condition in conditions:
        for cell_type in cell_types:
            for _ in range(10):
                rows.append(rng.gamma(0.9, 0.25, len(genes)))
                obs.append({"cell_type": cell_type, "condition": condition})
    x = np.vstack(rows)
    obs_df = pd.DataFrame(obs)
    gene_idx = {gene: i for i, gene in enumerate(genes)}

    def boost(mask: np.ndarray, names: list[str], amount: float) -> None:
        for name in names:
            x[mask, gene_idx[name]] += rng.gamma(amount, 0.55, int(mask.sum()))

    cell_type_values = obs_df["cell_type"].to_numpy()
    for condition in conditions:
        cond = obs_df["condition"].to_numpy() == condition
        treated = condition == "treated"
        boost(cond & (cell_type_values == "T cell"), ["IFNG", "CXCL10", "TNF", "CXCR3"], 4.6 if treated else 2.4)
        boost(cond & (cell_type_values == "Myeloid"), ["CCL2", "MIF", "IL6", "TGFB1", "CCR2", "CD74"], 4.0 if treated else 2.7)
        boost(cond & (cell_type_values == "Fibroblast"), ["CXCL12", "COL1A1", "LAMA1", "TGFB1", "PDGFA", "JAG1"], 4.6 if treated else 3.1)
        boost(cond & (cell_type_values == "Endothelial"), ["VEGFA", "KDR", "ACKR3", "ITGA6", "ITGB1"], 3.6 if treated else 2.5)
        boost(cond & (cell_type_values == "Tumor"), ["CXCL9", "VEGFA", "MIF", "CD74", "CXCR4", "TGFBR1", "TGFBR2"], 4.3 if treated else 3.0)
        boost(cond & (cell_type_values == "Fibroblast"), ["TGFBR1", "TGFBR2", "PDGFRA", "NOTCH1"], 3.2)
        boost(cond & (cell_type_values == "Endothelial"), ["CXCR4", "ACKR3"], 2.8)

    obs_df.index = [f"cell{i}" for i in range(len(obs_df))]
    adata = AnnData(x, obs=obs_df, var=pd.DataFrame(index=genes))
    coords = []
    centers = {
        "T cell": (0.0, 0.0),
        "Myeloid": (1.2, 0.2),
        "Fibroblast": (0.6, 1.0),
        "Endothelial": (1.8, 1.0),
        "Tumor": (1.0, 1.8),
    }
    for cell_type in obs_df["cell_type"]:
        cx, cy = centers[cell_type]
        coords.append([cx + rng.normal(0, 0.08), cy + rng.normal(0, 0.08)])
    adata.obsm["spatial"] = np.asarray(coords)
    return adata


def make_lr_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ligand": [
                "CXCL9",
                "CXCL10",
                "CXCL12",
                "CXCL12",
                "CCL2",
                "TGFB1",
                "VEGFA",
                "MIF",
                "MIF",
                "COL1A1",
                "LAMA1",
                "TNF",
                "IL6",
                "IFNG",
                "PDGFA",
                "JAG1",
            ],
            "receptor": [
                "CXCR3",
                "CXCR3",
                "CXCR4",
                "ACKR3",
                "CCR2",
                "TGFBR1_TGFBR2",
                "KDR",
                "CD74_CXCR4",
                "CD74",
                "ITGA6_ITGB1",
                "ITGA6_ITGB1",
                "TNFRSF1A",
                "IL6R",
                "IFNGR1",
                "PDGFRA",
                "NOTCH1",
            ],
            "pathway": [
                "CXCL",
                "CXCL",
                "CXCL",
                "CXCL",
                "CCL",
                "TGFb",
                "VEGF",
                "MIF",
                "MIF",
                "COLLAGEN",
                "LAMININ",
                "TNF",
                "IL6",
                "IFN-II",
                "PDGF",
                "NOTCH",
            ],
        }
    )


def public_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")}


def table_image(frame: pd.DataFrame, title: str):
    shown = frame.head(8).copy()
    for col in shown.select_dtypes(include=[float]).columns:
        shown[col] = shown[col].map(lambda value: f"{value:.3g}")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title, loc="left")
    table = ax.table(cellText=shown.astype(str).values, colLabels=shown.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    return fig


def html_card(title: str, html_path: Path, details: str):
    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    ax.axis("off")
    ax.text(0.02, 0.78, title, fontsize=14, weight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.53, details, fontsize=10, transform=ax.transAxes)
    ax.text(0.02, 0.31, f"HTML: {html_path.name}", fontsize=9, color="#4f81bd", transform=ax.transAxes)
    ax.text(0.02, 0.15, "Open the linked HTML file for the interactive Plotly render.", fontsize=8.5, color="#6b7280", transform=ax.transAxes)
    return fig


def save_matplotlib_example(example: Example) -> str:
    plt.close("all")
    result = example.func()
    out = STATIC / f"{example.name}.png"
    if hasattr(result, "figure"):
        fig = result.figure
    elif isinstance(result, tuple) and hasattr(result[0], "savefig"):
        fig = result[0]
    elif hasattr(result, "savefig"):
        fig = result
    else:
        raise TypeError(f"{example.name} did not return a Matplotlib figure or axes.")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"_static/plotting/{out.name}"


def save_plotnine_example(example: Example) -> str:
    plot = example.func()
    out = STATIC / f"{example.name}.png"
    plot.save(out, width=example.width, height=example.height, dpi=150, verbose=False)
    plt.close("all")
    return f"_static/plotting/{out.name}"


def build_examples():
    adata = make_demo_adata()
    lr = make_lr_table()
    kwargs = dict(min_pct=0.05, aggregate="tri_mean", score_method="cellchat")
    control = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="control", **kwargs)
    treated = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="treated", **kwargs)
    diff = pc.compare_communication(treated, control, label_a="treated", label_b="control")
    patterns = pc.compute_communication_patterns(treated, n_patterns=3, random_state=3)
    selection = pc.select_communication_pattern_number(treated, k_range=range(1, 5), n_runs=3, random_state=2)

    mpl = [
        Example("plotting_set_theme", lambda: (cp.set_theme(font_scale=0.9), cp.net_circle(treated, title="set_theme + net_circle"))[1], 'cp.set_theme(font_scale=0.9)\ncp.net_circle(result)'),
        Example("plotting_save_figure", lambda: save_figure_demo(treated), 'fig, ax = plt.subplots()\ncp.net_circle(result, ax=ax)\ncp.save_figure(fig, "network.png")'),
        Example("plotting_net_circle", lambda: cp.net_circle(treated), 'cp.net_circle(result)'),
        Example("plotting_net_chord", lambda: cp.net_chord(treated, top_n=30), 'cp.net_chord(result, top_n=30)'),
        Example("plotting_net_chord_gene", lambda: cp.net_chord_gene(treated, pathways=["CXCL", "MIF"], top_n=20), 'cp.net_chord_gene(result, pathways=["CXCL", "MIF"], top_n=20)'),
        Example("plotting_net_individual", lambda: cp.net_individual(treated, pathway="CXCL", layout="circle"), 'cp.net_individual(result, pathway="CXCL", layout="circle")'),
        Example("plotting_net_hierarchy", lambda: cp.net_hierarchy(treated, pathway="CXCL", targets=["Tumor", "Myeloid"]), 'cp.net_hierarchy(result, pathway="CXCL", targets=["Tumor", "Myeloid"])'),
        Example("plotting_net_heatmap", lambda: cp.net_heatmap(treated, cluster_rows=True, cluster_cols=True), 'cp.net_heatmap(result, cluster_rows=True, cluster_cols=True)'),
        Example("plotting_pathway_heatmap", lambda: cp.pathway_heatmap(treated, top_pairs=14, compact_pairs=True), 'cp.pathway_heatmap(result, top_pairs=14, compact_pairs=True)'),
        Example("plotting_bubble", lambda: cp.bubble(treated, top_n=16, top_pairs=10, compact_pairs=True), 'cp.bubble(result, top_n=16, top_pairs=10, compact_pairs=True)'),
        Example("plotting_dotplot", lambda: cp.dotplot(treated, top_n=16), 'cp.dotplot(result, top_n=16)'),
        Example("plotting_signaling_role_scatter", lambda: cp.signaling_role_scatter(treated), 'cp.signaling_role_scatter(result)'),
        Example("plotting_rank_signaling", lambda: cp.rank_signaling(treated, top_n=12), 'cp.rank_signaling(result, top_n=12)'),
        Example("plotting_annotation_bar", lambda: cp.annotation_bar(treated), 'cp.annotation_bar(result)'),
        Example("plotting_lr_contribution", lambda: cp.lr_contribution(treated, "CXCL", top_n=8), 'cp.lr_contribution(result, "CXCL", top_n=8)'),
        Example("plotting_lr_contribution_multi", lambda: cp.lr_contribution_multi(treated, pathways=["CXCL", "MIF", "TGFb", "VEGF"], top_n=4, ncols=2), 'cp.lr_contribution_multi(result, pathways=["CXCL", "MIF", "TGFb", "VEGF"])'),
        Example("plotting_signaling_role_heatmap", lambda: cp.signaling_role_heatmap(treated, mode="outgoing", cluster_rows=True), 'cp.signaling_role_heatmap(result, mode="outgoing", cluster_rows=True)'),
        Example("plotting_signaling_role_heatmap_compare", lambda: cp.signaling_role_heatmap_compare(diff, mode="outgoing"), 'cp.signaling_role_heatmap_compare(diff, mode="outgoing")'),
        Example("plotting_pathway_river", lambda: cp.pathway_river(treated, top_n=12), 'cp.pathway_river(result, top_n=12)'),
        Example("plotting_rank_signaling_compare", lambda: cp.rank_signaling_compare(diff, top_n=12), 'cp.rank_signaling_compare(diff, top_n=12)'),
        Example("plotting_compare_interactions", lambda: cp.compare_interactions(diff), 'cp.compare_interactions(diff)'),
        Example("plotting_diff_network_circle", lambda: cp.diff_network_circle(diff), 'cp.diff_network_circle(diff)'),
        Example("plotting_diff_heatmap", lambda: cp.diff_heatmap(diff, cluster_rows=True, cluster_cols=True), 'cp.diff_heatmap(diff, cluster_rows=True, cluster_cols=True)'),
        Example("plotting_diff_bubble", lambda: cp.diff_bubble(diff, top_n=16, compact_pairs=True), 'cp.diff_bubble(diff, top_n=16, compact_pairs=True)'),
        Example("plotting_diff_pathway_rank", lambda: cp.diff_pathway_rank(diff, top_n=12), 'cp.diff_pathway_rank(diff, top_n=12)'),
        Example("plotting_diff_source_target_rank", lambda: cp.diff_source_target_rank(diff, top_n=12), 'cp.diff_source_target_rank(diff, top_n=12)'),
        Example("plotting_signaling_changes_scatter", lambda: cp.signaling_changes_scatter(diff, "Tumor"), 'cp.signaling_changes_scatter(diff, "Tumor")'),
        Example("plotting_pathway_similarity_rank", lambda: cp.pathway_similarity_rank(diff, top_n=12, method="mds"), 'cp.pathway_similarity_rank(diff, top_n=12, method="mds")'),
        Example("plotting_pathway_embedding_pairwise", lambda: cp.pathway_embedding_pairwise(diff, method="mds", top_label=4), 'cp.pathway_embedding_pairwise(diff, method="mds", top_label=4)'),
        Example("plotting_centrality_network", lambda: table_image(cp.centrality_network(treated), "centrality_network(result)"), 'cp.centrality_network(result).head()'),
        Example("plotting_signaling_role_network", lambda: cp.signaling_role_network(treated, annot=True), 'cp.signaling_role_network(result, annot=True)'),
        Example("plotting_pathway_embedding", lambda: cp.pathway_embedding(treated, cluster=True, n_clusters=3, method="mds", top_label=5), 'cp.pathway_embedding(result, cluster=True, n_clusters=3, method="mds")'),
        Example("plotting_signaling_gene_expression", lambda: cp.signaling_gene_expression(adata, result=treated, signaling="CXCL"), 'cp.signaling_gene_expression(adata, result=result, signaling="CXCL")'),
        Example("plotting_spatial_network", lambda: cp.spatial_network(adata, treated, pathway="CXCL"), 'cp.spatial_network(adata, result, pathway="CXCL")'),
        Example("plotting_key_plot_gallery", lambda: cp.key_plot_gallery(treated, diff), 'cp.key_plot_gallery(result, diff)'),
        Example("plotting_masterpiece_gallery", lambda: cp.masterpiece_gallery(treated, diff, title="pyccc gallery", subtitle="compact synthetic tissue atlas"), 'cp.masterpiece_gallery(result, diff)'),
    ]

    gg = [
        Example("ggplot_bubble", lambda: cg.bubble(treated, top_n=16), 'cg.bubble(result, top_n=16)', 7.0, 4.8),
        Example("ggplot_diff_bubble", lambda: cg.diff_bubble(diff, top_n=16), 'cg.diff_bubble(diff, top_n=16)', 7.0, 4.8),
        Example("ggplot_diff_pathway_rank", lambda: cg.diff_pathway_rank(diff, top_n=12), 'cg.diff_pathway_rank(diff, top_n=12)', 6.8, 4.4),
        Example("ggplot_diff_source_target_rank", lambda: cg.diff_source_target_rank(diff, top_n=12), 'cg.diff_source_target_rank(diff, top_n=12)', 6.8, 4.4),
        Example("ggplot_signaling_changes_scatter", lambda: cg.signaling_changes_scatter(diff, "Tumor"), 'cg.signaling_changes_scatter(diff, "Tumor")', 6.2, 4.8),
        Example("ggplot_pathway_similarity_rank", lambda: cg.pathway_similarity_rank(diff, top_n=12, method="mds"), 'cg.pathway_similarity_rank(diff, top_n=12, method="mds")', 6.2, 4.2),
        Example("ggplot_pathway_embedding_pairwise", lambda: cg.pathway_embedding_pairwise(diff, method="mds", top_label=4), 'cg.pathway_embedding_pairwise(diff, method="mds", top_label=4)', 6.2, 4.8),
        Example("ggplot_dotplot", lambda: cg.dotplot(treated, top_n=16), 'cg.dotplot(result, top_n=16)', 7.0, 4.8),
        Example("ggplot_signaling_role_scatter", lambda: cg.signaling_role_scatter(treated), 'cg.signaling_role_scatter(result)', 6.2, 4.8),
        Example("ggplot_lr_contribution", lambda: cg.lr_contribution(treated, "CXCL", top_n=8), 'cg.lr_contribution(result, "CXCL", top_n=8)', 6.4, 4.2),
        Example("ggplot_lr_contribution_multi", lambda: cg.lr_contribution_multi(treated, pathways=["CXCL", "MIF", "TGFb", "VEGF"], top_n=4), 'cg.lr_contribution_multi(result, pathways=["CXCL", "MIF", "TGFb", "VEGF"])', 7.5, 3.2),
        Example("ggplot_pathway_embedding", lambda: cg.pathway_embedding(treated, cluster=True, n_clusters=3, method="mds", top_label=5), 'cg.pathway_embedding(result, cluster=True, n_clusters=3, method="mds")', 6.2, 4.8),
        Example("ggplot_signaling_gene_expression", lambda: cg.signaling_gene_expression(adata, result=treated, signaling="CXCL"), 'cg.signaling_gene_expression(adata, result=result, signaling="CXCL")', 6.8, 4.8),
        Example("ggplot_signaling_role_heatmap_compare", lambda: cg.signaling_role_heatmap_compare(diff, mode="outgoing"), 'cg.signaling_role_heatmap_compare(diff, mode="outgoing")', 7.5, 4.8),
        Example("ggplot_pathway_heatmap", lambda: cg.pathway_heatmap(treated, top_pairs=14, compact_pairs=True), 'cg.pathway_heatmap(result, top_pairs=14, compact_pairs=True)', 7.0, 4.8),
        Example("ggplot_rank_signaling", lambda: cg.rank_signaling(treated, top_n=12), 'cg.rank_signaling(result, top_n=12)', 6.2, 4.4),
        Example("ggplot_rank_signaling_compare", lambda: cg.rank_signaling_compare(diff, top_n=12), 'cg.rank_signaling_compare(diff, top_n=12)', 6.8, 4.4),
        Example("ggplot_annotation_bar", lambda: cg.annotation_bar(treated), 'cg.annotation_bar(result)', 6.2, 4.2),
        Example("ggplot_compare_interactions", lambda: cg.compare_interactions(diff), 'cg.compare_interactions(diff)', 5.5, 4.0),
        Example("ggplot_pathway_river", lambda: cg.pathway_river(treated, top_n=12), 'cg.pathway_river(result, top_n=12)', 7.0, 4.8),
        Example("ggplot_pattern_dot", lambda: cg.pattern_dot(patterns), 'cg.pattern_dot(patterns)', 7.0, 4.8),
        Example("ggplot_pattern_river", lambda: cg.pattern_river(patterns), 'cg.pattern_river(patterns)', 7.0, 4.8),
        Example("ggplot_pattern_number_plot", lambda: cg.pattern_number_plot(selection), 'cg.pattern_number_plot(selection)', 5.8, 6.4),
    ]

    interactive = make_interactive_examples(treated, patterns)
    report = make_report_example(treated, diff)
    return mpl, gg, interactive, report


def save_figure_demo(result):
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    cp.net_circle(result, ax=ax, title="Saved with save_figure")
    tmp = STATIC / "plotting_save_figure.png"
    cp.save_figure(fig, str(tmp), dpi=150, also_svg=False)
    return fig


def make_interactive_examples(result, patterns):
    INTERACTIVE.mkdir(parents=True, exist_ok=True)
    pathway_fig = ci.interactive_pathway_river(result, top_n=12)
    pattern_fig = ci.interactive_pattern_river(patterns)
    pathway_html = ci.save_interactive_html(pathway_fig, INTERACTIVE / "interactive_pathway_river.html")
    pattern_html = ci.save_interactive_html(pattern_fig, INTERACTIVE / "interactive_pattern_river.html")
    save_html = ci.save_interactive_html(pathway_fig, INTERACTIVE / "save_interactive_html.html")
    examples = [
        Example(
            "interactive_pathway_river",
            lambda: html_card("interactive_pathway_river", pathway_html, f"nodes={len(pathway_fig.data[0].node.label)}, links={len(pathway_fig.data[0].link.value)}"),
            "fig = pc.interactive_pathway_river(result, top_n=12)",
        ),
        Example(
            "interactive_pattern_river",
            lambda: html_card("interactive_pattern_river", pattern_html, f"nodes={len(pattern_fig.data[0].node.label)}, links={len(pattern_fig.data[0].link.value)}"),
            "fig = pc.interactive_pattern_river(patterns)",
        ),
        Example(
            "save_interactive_html",
            lambda: html_card("save_interactive_html", save_html, "writes a Plotly figure to a standalone HTML file"),
            'pc.save_interactive_html(fig, "pathway_river.html")',
        ),
    ]
    return examples


def make_report_example(result, diff):
    report_dir = DOCS / "_static" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = pc.save_cellchat_report(result, report_dir / "cellchat_report.pdf", diff=diff)
    return Example(
        "report_save_cellchat_report",
        lambda: (cp.key_plot_gallery(result, diff)[0]),
        f'pc.save_cellchat_report(result, "{path.name}", diff=diff)',
    )


def check_coverage(mpl: list[Example], gg: list[Example], interactive: list[Example], report: Example) -> None:
    plotting_defined = public_functions(ROOT / "src" / "pyccc" / "plotting.py")
    ggplot_defined = public_functions(ROOT / "src" / "pyccc" / "ggplot.py")
    interactive_defined = public_functions(ROOT / "src" / "pyccc" / "interactive.py")
    report_defined = public_functions(ROOT / "src" / "pyccc" / "report.py")
    covered_plotting = {example.name.removeprefix("plotting_") for example in mpl}
    covered_ggplot = {example.name.removeprefix("ggplot_") for example in gg}
    covered_interactive = {example.name for example in interactive}
    covered_report = {report.name.removeprefix("report_")}
    missing = {
        "plotting": sorted(plotting_defined - covered_plotting),
        "ggplot": sorted(ggplot_defined - covered_ggplot),
        "interactive": sorted(interactive_defined - covered_interactive),
        "report": sorted(report_defined - covered_report),
    }
    missing = {module: funcs for module, funcs in missing.items() if funcs}
    if missing:
        raise RuntimeError(f"Gallery examples missing public functions: {missing}")


def function_doc(module, name: str) -> str:
    func = getattr(module, name)
    signature = str(inspect.signature(func))
    doc = inspect.getdoc(func) or ""
    first = doc.splitlines()[0] if doc else ""
    return f"`{name}{signature}`\n\n{first}"


def write_markdown(mpl_rows, gg_rows, interactive_rows, report_row) -> None:
    lines = [
        "# Plotting gallery",
        "",
        "This page is generated by `scripts/generate_plotting_gallery.py`. Each entry shows a minimal call and the rendered artifact created from the same synthetic AnnData example.",
        "",
        "The static Matplotlib and plotnine APIs render PNG files. Plotly helpers render standalone HTML files and a small generated preview card because the documentation build does not depend on a browser screenshot runtime.",
        "",
        "## Setup used by all examples",
        "",
        "```python",
        "import pyccc as pc",
        "import pyccc.plotting as cp",
        "import pyccc.ggplot as cg",
        "",
        "# result: CCCResult for the treated sample",
        "# diff: DifferentialCCC comparing treated against control",
        "# adata: the source AnnData used for expression and spatial examples",
        "# patterns: communication patterns computed from result",
        "```",
        "",
        "## Matplotlib API",
        "",
    ]
    for example, image in mpl_rows:
        name = example.name.removeprefix("plotting_")
        lines.extend(section(name, function_doc(cp, name), example.code, image))
    lines.extend(["## Plotnine API", ""])
    for example, image in gg_rows:
        name = example.name.removeprefix("ggplot_")
        lines.extend(section(name, function_doc(cg, name), example.code, image))
    lines.extend(["## Interactive and Report Helpers", ""])
    for example, image in interactive_rows:
        name = example.name
        lines.extend(section(name, function_doc(ci, name), example.code, image))
        html_name = image.rsplit("/", 1)[-1].replace(".png", ".html")
        if (INTERACTIVE / html_name).exists():
            lines.extend([f"[Open generated HTML](_static/interactive/{html_name})", ""])
    lines.extend(section("save_cellchat_report", function_doc(pc, "save_cellchat_report"), report_row[0].code, report_row[1]))
    lines.extend(["[Open generated report PDF](_static/reports/cellchat_report.pdf)", ""])
    GALLERY.write_text("\n".join(lines), encoding="utf-8")


def section(name: str, doc: str, code: str, image: str) -> list[str]:
    return [
        f"### `{name}`",
        "",
        doc,
        "",
        "```python",
        textwrap.dedent(code).strip(),
        "```",
        "",
        f"![{name}]({image})",
        "",
    ]


def main() -> None:
    for path in [STATIC, INTERACTIVE]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    cp.set_theme(font_scale=0.82)
    mpl, gg, interactive, report = build_examples()
    check_coverage(mpl, gg, interactive, report)

    mpl_rows = [(example, save_matplotlib_example(example)) for example in mpl]
    gg_rows = [(example, save_plotnine_example(example)) for example in gg]
    interactive_rows = [(example, save_matplotlib_example(example)) for example in interactive]
    report_row = (report, save_matplotlib_example(report))
    write_markdown(mpl_rows, gg_rows, interactive_rows, report_row)
    print(f"Wrote {GALLERY.relative_to(ROOT)}")
    print(f"Wrote {len(mpl_rows) + len(gg_rows) + len(interactive_rows) + 1} gallery artifacts")


if __name__ == "__main__":
    main()

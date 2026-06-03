from __future__ import annotations

import os
import warnings
import importlib.util
import subprocess
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse, stats

import pyccc as pc
import pyccc.plotting as cp

try:
    from cellchat_official_plotnine_demo import HUMAN_SKIN_RDA_URL, _download_if_missing, read_official_human_skin
except ModuleNotFoundError:  # pragma: no cover - used when imported from tests
    _official_demo_path = Path(__file__).with_name("cellchat_official_plotnine_demo.py")
    _spec = importlib.util.spec_from_file_location("cellchat_official_plotnine_demo", _official_demo_path)
    if _spec is None or _spec.loader is None:
        raise
    _official_demo = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_official_demo)
    HUMAN_SKIN_RDA_URL = _official_demo.HUMAN_SKIN_RDA_URL
    _download_if_missing = _official_demo._download_if_missing
    read_official_human_skin = _official_demo.read_official_human_skin


A4_LANDSCAPE = (11.69, 8.27)
KEY_COLS = ["source", "target", "ligand", "receptor", "pathway"]
CELLCHAT_R_RUNNER = r"""
suppressPackageStartupMessages(library(CellChat))

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[[1]]
lr_path <- args[[2]]
out_path <- args[[3]]
condition <- args[[4]]

load(data_path)
meta <- data_humanSkin$meta
keep <- meta$condition == condition
data.input <- data_humanSkin$data[, rownames(meta)[keep], drop = FALSE]
meta.use <- droplevels(meta[keep, , drop = FALSE])
meta.use$cell_type <- droplevels(meta.use$labels)

cellchat <- createCellChat(object = data.input, meta = meta.use, group.by = "cell_type", datatype = "RNA")
db.use <- CellChatDB.human
pairLR.use <- read.delim(lr_path, check.names = FALSE, stringsAsFactors = FALSE)
rownames(pairLR.use) <- pairLR.use$interaction_name
db.use$interaction <- pairLR.use
cellchat@DB <- db.use
cellchat <- subsetData(cellchat)
cellchat <- computeCommunProb(
  cellchat,
  type = "triMean",
  LR.use = pairLR.use,
  raw.use = TRUE,
  population.size = FALSE,
  nboot = 1,
  Kh = 0.5,
  n = 1
)

prob <- cellchat@net$prob
pval <- cellchat@net$pval
idx <- which(prob > 0, arr.ind = TRUE)
if (nrow(idx) == 0) {
  out <- data.frame(source = character(), target = character(), interaction_name = character(), prob = numeric(), pvalue = numeric())
} else {
  out <- data.frame(
    source = dimnames(prob)[[1]][idx[, 1]],
    target = dimnames(prob)[[2]][idx[, 2]],
    interaction_name = dimnames(prob)[[3]][idx[, 3]],
    prob = prob[idx],
    pvalue = pval[idx],
    stringsAsFactors = FALSE
  )
}
write.table(out, out_path, sep = "\t", row.names = FALSE, quote = FALSE)
"""


def main() -> None:
    cache_dir = Path(os.environ.get("PYCCC_CACHE_DIR", Path.home() / ".cache" / "pyccc")) / "cellchat_official"
    out_dir = Path(os.environ.get("PYCCC_OUTPUT_DIR", "cellchat_reference_comparison"))
    proxy = os.environ.get("PYCCC_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = cache_dir / "data_humanSkin_CellChat.rda"
    _download_if_missing(HUMAN_SKIN_RDA_URL, data_path, proxy=proxy)
    adata = read_official_human_skin(data_path)
    db = pc.load_cellchatdb("human", cache_dir=cache_dir, force_download=False, proxy=proxy)
    lr_db = pc.CellChatDB(
        db.interactions[db.interactions["annotation"].eq("Secreted Signaling")].copy(),
        name=f"{db.name}_secreted",
        metadata=db.metadata,
    )

    condition = "LS"
    kwargs = dict(
        condition_key="condition",
        condition=condition,
        min_pct=0.0,
        min_expr=0.0,
        aggregate="tri_mean",
        score_method="cellchat",
        cofactor_adjust=True,
        cofactor_kh=0.5,
        cofactor_hill=1.0,
        population_size=False,
    )
    pyccc_result = pc.compute_communication(adata, "cell_type", lr_db, **kwargs)
    reference_result = compute_cellchat_formula_reference(
        adata,
        "cell_type",
        lr_db,
        condition_key="condition",
        condition=condition,
        aggregate="tri_mean",
        kh=0.5,
        hill=1.0,
        cofactor_adjust=True,
        population_size=False,
    )

    metrics = comparison_metrics(pyccc_result, reference_result)
    metrics_path = out_dir / "pyccc_vs_cellchat_reference_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    pdf_path = out_dir / "pyccc_vs_cellchat_reference_a4.pdf"
    save_comparison_pdf(pyccc_result, reference_result, metrics, pdf_path, adata=adata, lr_db=lr_db, reference_label="CellChat formula reference")
    notes_path = out_dir / "pyccc_vs_cellchat_reference_notes.md"
    write_notes(notes_path, pyccc_result, reference_result, metrics, adata=adata, lr_db=lr_db, reference_label="CellChat formula reference")

    r_result = None
    r_metrics = None
    try:
        r_result = compute_cellchat_r_package_reference(
            adata,
            "cell_type",
            lr_db,
            data_path=data_path,
            out_dir=out_dir / "cellchat_r_reference",
            condition_key="condition",
            condition=condition,
        )
        r_metrics = comparison_metrics(pyccc_result, r_result)
        r_metrics_path = out_dir / "pyccc_vs_cellchat_r_metrics.tsv"
        r_metrics.to_csv(r_metrics_path, sep="\t", index=False)
        r_pdf_path = out_dir / "pyccc_vs_cellchat_r_a4.pdf"
        save_comparison_pdf(pyccc_result, r_result, r_metrics, r_pdf_path, adata=adata, lr_db=lr_db, reference_label="CellChat R package")
        r_notes_path = out_dir / "pyccc_vs_cellchat_r_notes.md"
        write_notes(r_notes_path, pyccc_result, r_result, r_metrics, adata=adata, lr_db=lr_db, reference_label="CellChat R package")
    except Exception as exc:  # pragma: no cover - depends on local R/CellChat availability
        fallback_path = out_dir / "pyccc_vs_cellchat_r_blocker.txt"
        fallback_path.write_text(str(exc) + "\n", encoding="utf-8")

    print(f"Dataset: official CellChat human skin data ({adata.n_obs} cells x {adata.n_vars} genes)")
    print(f"Condition: {condition}; LR table: {len(lr_db.interactions)} Secreted Signaling interactions")
    if r_result is not None and r_metrics is not None:
        print("\npyccc vs real CellChat R package:")
        print(r_metrics.to_string(index=False))
        print(f"Saved A4 CellChat R comparison PDF: {r_pdf_path.resolve()}")
        print(f"Saved CellChat R metrics: {r_metrics_path.resolve()}")
        print(f"Saved CellChat R notes: {r_notes_path.resolve()}")
    else:
        print("\nCellChat R package run was not available; formula-reference fallback was saved.")
    print(metrics.to_string(index=False))
    print(f"Saved A4 comparison PDF: {pdf_path.resolve()}")
    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved notes: {notes_path.resolve()}")


def compute_cellchat_formula_reference(
    adata: AnnData,
    groupby: str,
    lr_table: pc.CellChatDB | pd.DataFrame,
    *,
    condition_key: str | None = None,
    condition: str | None = None,
    aggregate: str = "tri_mean",
    kh: float = 0.5,
    hill: float = 1.0,
    cofactor_adjust: bool = True,
    population_size: bool = False,
) -> pc.CCCResult:
    """Independent CellChat `computeCommunProb` formula reference.

    This mirrors the RNA branch of CellChat's `computeCommunProb`: signaling
    expression is scaled by its global maximum, group averages are computed,
    LR products are passed through the Hill function, receptor cofactors are
    applied before Hill, and agonist/antagonist factors are source-target outer
    products.
    """

    if condition_key is not None and condition is not None:
        mask = adata.obs[condition_key].astype(str).to_numpy() == str(condition)
        ad = adata[mask].copy()
    else:
        ad = adata.copy()
    groups = [str(x) for x in pd.Index(ad.obs[groupby].astype(str).unique()).sort_values()]
    if isinstance(lr_table, pc.CellChatDB):
        lr = lr_table.interactions.copy()
        lr_name = lr_table.name
    else:
        lr = pc.CellChatDB(lr_table).interactions
        lr_name = "custom"

    gene_lookup = {str(gene).upper(): str(gene) for gene in ad.var_names}
    lr = _filter_lr_to_available_genes(lr, gene_lookup)
    genes = _required_expression_genes(lr, gene_lookup, include_cofactors=cofactor_adjust)
    expr = _scaled_expression_frame(ad, genes)
    labels = ad.obs[groupby].astype(str).to_numpy()
    expr_avg, expr_pct = _group_expression(expr, labels, groups, aggregate=aggregate)

    lig_expr = _complex_matrix(expr_avg, lr["ligand"], gene_lookup)
    rec_expr = _complex_matrix(expr_avg, lr["receptor"], gene_lookup)
    lig_pct = _complex_matrix(expr_pct, lr["ligand"], gene_lookup, mode="min")
    rec_pct = _complex_matrix(expr_pct, lr["receptor"], gene_lookup, mode="min")
    rec_for_score = rec_expr.copy()
    if cofactor_adjust:
        co_a = _coreceptor_factor_matrix(lr, expr_avg, groups, gene_lookup, "co_A_receptor_genes")
        co_i = _coreceptor_factor_matrix(lr, expr_avg, groups, gene_lookup, "co_I_receptor_genes")
        rec_for_score = rec_for_score * co_a / co_i

    signal = lig_expr[:, :, None] * rec_for_score[:, None, :]
    signal_pow = np.power(signal, hill)
    kh_pow = kh**hill
    prob = signal_pow / (kh_pow + signal_pow)
    if cofactor_adjust:
        agonist = _hill_factor_matrix(lr, expr_avg, groups, gene_lookup, "agonist_genes", kh=kh, hill=hill, mode="agonist")
        antagonist = _hill_factor_matrix(lr, expr_avg, groups, gene_lookup, "antagonist_genes", kh=kh, hill=hill, mode="antagonist")
        prob *= agonist[:, :, None] * agonist[:, None, :] * antagonist[:, :, None] * antagonist[:, None, :]
    if population_size:
        counts = pd.Series(labels).value_counts(normalize=True)
        weights = np.array([float(counts.get(group, 0.0)) for group in groups])
        prob *= weights[None, :, None] * weights[None, None, :]

    valid = np.isfinite(prob) & (prob > 0) & (lig_pct[:, :, None] >= 0) & (rec_pct[:, None, :] >= 0)
    lr_idx, source_idx, target_idx = np.nonzero(valid)
    if lr_idx.size == 0:
        rows = pd.DataFrame(columns=["source", "target", "ligand", "receptor", "pathway", "annotation", "ligand_expr", "receptor_expr", "ligand_pct", "receptor_pct", "prob", "pvalue"])
    else:
        group_arr = np.asarray(groups, dtype=object)
        rows = pd.DataFrame(
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
                "prob": prob[lr_idx, source_idx, target_idx],
                "pvalue": np.nan,
            }
        )
        for col in _lr_metadata_columns(lr):
            rows[col] = lr[col].to_numpy(dtype=object)[lr_idx]
    rows = rows.sort_values(["source", "target", "pathway", "prob"], ascending=[True, True, True, False]).reset_index(drop=True)
    return pc.CCCResult(rows, groupby=groupby, groups=groups, condition=condition, lr_name=f"{lr_name}_cellchat_formula_reference")


def compute_cellchat_r_package_reference(
    adata: AnnData,
    groupby: str,
    lr_table: pc.CellChatDB,
    *,
    data_path: Path,
    out_dir: Path,
    condition_key: str = "condition",
    condition: str = "LS",
) -> pc.CCCResult:
    """Run the installed CellChat R package and import `object@net$prob`.

    This helper is intentionally scoped to the official CellChat human-skin
    example so the R side can load the original `.rda` object without a lossy
    cross-language sparse-matrix export.
    """

    if groupby != "cell_type" or condition_key != "condition":
        raise ValueError("CellChat R package reference currently supports the official `cell_type`/`condition` example only.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(f"CellChat official data file does not exist: {data_path}")

    ad = adata[adata.obs[condition_key].astype(str).to_numpy() == str(condition)].copy()
    groups = [str(x) for x in pd.Index(ad.obs[groupby].astype(str).unique()).sort_values()]
    gene_lookup = {str(gene).upper(): str(gene) for gene in ad.var_names}
    lr = _filter_lr_to_available_genes(lr_table.interactions.copy(), gene_lookup)
    r_lr = _cellchat_r_lr_use(lr)

    lr_path = out_dir / "cellchat_lr_use.tsv"
    script_path = out_dir / "run_cellchat_reference.R"
    prob_path = out_dir / "cellchat_r_prob.tsv"
    log_path = out_dir / "cellchat_r_run.log"
    r_lr.to_csv(lr_path, sep="\t", index=False)
    script_path.write_text(textwrap.dedent(CELLCHAT_R_RUNNER).strip() + "\n", encoding="utf-8")

    env = os.environ.copy()
    local_r_lib = Path(".r-lib").resolve()
    if "R_LIBS_USER" not in env and local_r_lib.exists():
        env["R_LIBS_USER"] = str(local_r_lib)
    completed = subprocess.run(
        ["Rscript", str(script_path), str(data_path), str(lr_path), str(prob_path), str(condition)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CellChat R package run failed with exit code {completed.returncode}. See {log_path}.")
    if not prob_path.exists():
        raise RuntimeError(f"CellChat R package run did not write probabilities. See {log_path}.")

    imported = pd.read_csv(prob_path, sep="\t")
    meta_cols = ["interaction_name", "ligand", "receptor", "pathway", "annotation"] + _lr_metadata_columns(lr)
    meta_cols = list(dict.fromkeys(col for col in meta_cols if col in lr.columns))
    lr_meta = lr[meta_cols].copy()
    rows = imported.merge(lr_meta, on="interaction_name", how="left")
    if rows[["ligand", "receptor", "pathway"]].isna().any().any():
        missing = rows.loc[rows["ligand"].isna() | rows["receptor"].isna() | rows["pathway"].isna(), "interaction_name"].unique()
        raise RuntimeError(f"CellChat R output contained unknown LR names: {missing[:5]}")
    rows["ligand_expr"] = np.nan
    rows["receptor_expr"] = np.nan
    rows["ligand_pct"] = np.nan
    rows["receptor_pct"] = np.nan
    rows = rows.sort_values(["source", "target", "pathway", "prob"], ascending=[True, True, True, False]).reset_index(drop=True)
    return pc.CCCResult(rows, groupby=groupby, groups=groups, condition=condition, lr_name=f"{lr_table.name}_cellchat_r_package")


def comparison_metrics(pyccc_result: pc.CCCResult, reference_result: pc.CCCResult) -> pd.DataFrame:
    rows = []
    rows.append(_vector_metrics("lr_source_target_prob", _aligned_probabilities(pyccc_result.interactions, reference_result.interactions, _probability_key_cols(pyccc_result.interactions, reference_result.interactions))))
    rows.append(_vector_metrics("global_network_weight", _aligned_matrices(pyccc_result.network(), reference_result.network())))
    rows.append(_vector_metrics("pathway_information_flow", _aligned_summary(pyccc_result.pathway_summary(), reference_result.pathway_summary(), "pathway", "prob")))
    rows.append(_vector_metrics("lr_information_flow", _aligned_lr_summary(pyccc_result.lr_summary(), reference_result.lr_summary())))
    top_pathway = str(pyccc_result.pathway_summary().iloc[0]["pathway"]) if not pyccc_result.pathway_summary().empty else ""
    rows.append(_vector_metrics(f"{top_pathway}_lr_contribution", _aligned_contribution(pyccc_result, reference_result, top_pathway)))
    return pd.DataFrame(rows)


def save_comparison_pdf(
    pyccc_result: pc.CCCResult,
    reference_result: pc.CCCResult,
    metrics: pd.DataFrame,
    path: Path,
    *,
    adata: AnnData,
    lr_db: pc.CellChatDB,
    reference_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        _summary_page(pdf, pyccc_result, reference_result, metrics, adata=adata, lr_db=lr_db, reference_label=reference_label)
        _probability_page(pdf, pyccc_result, reference_result, metrics, reference_label=reference_label)
        _network_page(pdf, pyccc_result, reference_result, metrics, reference_label=reference_label)
        _pathway_rank_page(pdf, pyccc_result, reference_result, metrics, reference_label=reference_label)
        _pathway_heatmap_page(pdf, pyccc_result, reference_result, metrics, reference_label=reference_label)
        _lr_contribution_page(pdf, pyccc_result, reference_result, metrics, reference_label=reference_label)
        _bubble_page(pdf, pyccc_result, reference_result, metrics, reference_label=reference_label)
        _role_page(pdf, pyccc_result, reference_result, metrics, reference_label=reference_label)


def write_notes(
    path: Path,
    pyccc_result: pc.CCCResult,
    reference_result: pc.CCCResult,
    metrics: pd.DataFrame,
    *,
    adata: AnnData,
    lr_db: pc.CellChatDB,
    reference_label: str,
) -> None:
    max_diff = float(metrics["max_abs_diff"].max())
    min_pearson = float(metrics["pearson"].min())
    conclusion = "No material discrepancy remains in the checked analyses." if max_diff < 1e-10 and min_pearson > 0.999999 else "Some checked analyses differ; inspect the metric table and PDF pages."
    path.write_text(
        "\n".join(
            [
                f"# pyccc vs {reference_label}",
                "",
                f"- Dataset: official CellChat human skin AnnData reconstructed from `{HUMAN_SKIN_RDA_URL}`.",
                f"- Cells x genes: {adata.n_obs} x {adata.n_vars}.",
                f"- Condition compared: `{pyccc_result.condition}`.",
                f"- LR table: same `{lr_db.name}` Secreted Signaling table, {len(lr_db.interactions)} rows before gene filtering.",
                "- Parameters: triMean, global max-scaled signaling expression, Kh=0.5, n=1, cofactor adjustment on, population.size off; p-values are not used in this comparison.",
                f"- Reference: {reference_label}.",
                "",
                "## Conclusion",
                "",
                conclusion,
                "",
                "## Implementation Issues Checked",
                "",
                "- CellChat scales signaling expression by `max(data)` before group averaging.",
                "- CellChat applies co-activation/co-inhibition receptor factors before the Hill transform.",
                "- CellChat applies agonist/antagonist factors as source-target outer products.",
                "- CellChat population-size weighting uses direct source-target cell-frequency products.",
                "- Official RData factor-level cell-type labels are aligned with the CellChat R object.",
                "",
                "These semantics are now represented in pyccc's `score_method=\"cellchat\"` path.",
                "",
                "## Metrics",
                "",
                "```text",
                metrics.to_string(index=False),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _summary_page(pdf: PdfPages, pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, metrics: pd.DataFrame, *, adata: AnnData, lr_db: pc.CellChatDB, reference_label: str) -> None:
    fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
    ax.axis("off")
    ax.set_title(f"pyccc vs {reference_label}: same scRNA-seq data and same LR table", loc="left", fontsize=15, weight="bold")
    summary = [
        ["Dataset", "Official CellChat human skin"],
        ["Condition", str(pyccc_result.condition)],
        ["Cells x genes", f"{adata.n_obs} x {adata.n_vars}"],
        ["Cell groups", str(len(pyccc_result.groups))],
        ["LR table", f"{lr_db.name}; Secreted Signaling rows={len(lr_db.interactions)}"],
        ["pyccc nonzero rows", f"{len(pyccc_result.interactions):,}"],
        ["reference nonzero rows", f"{len(reference_result.interactions):,}"],
        ["Main conclusion", "all checked outputs agree numerically" if metrics["max_abs_diff"].max() < 1e-10 else "differences detected"],
    ]
    _draw_table(ax, pd.DataFrame(summary, columns=["Item", "Value"]), bbox=[0.02, 0.50, 0.96, 0.38], font_size=9)
    metric_table = metrics.copy()
    metric_table["analysis"] = metric_table["analysis"].map(_short_analysis_label)
    for col in ("pearson", "spearman", "max_abs_diff", "median_abs_diff", "top20_overlap"):
        metric_table[col] = metric_table[col].map(lambda value: f"{float(value):.4g}")
    _draw_table(ax, metric_table, bbox=[0.02, 0.08, 0.96, 0.34], font_size=8)
    ax.text(
        0.02,
        0.46,
        f"Each following A4 page places the pyccc output and the {reference_label} output for the same analysis side by side.",
        transform=ax.transAxes,
        fontsize=10,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _probability_page(pdf: PdfPages, pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, metrics: pd.DataFrame, *, reference_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE, constrained_layout=True)
    key_cols = _probability_key_cols(pyccc_result.interactions, reference_result.interactions)
    aligned = _aligned_probabilities(pyccc_result.interactions, reference_result.interactions, key_cols)
    axes[0].scatter(aligned["reference"], aligned["pyccc"], s=5, alpha=0.18, color="#5e4fa2")
    _identity_line(axes[0], aligned["reference"], aligned["pyccc"])
    axes[0].set_title("All nonzero LR-source-target probabilities")
    axes[0].set_xlabel(reference_label)
    axes[0].set_ylabel("pyccc")
    diffs = aligned.assign(abs_diff=lambda df: (df["pyccc"] - df["reference"]).abs()).sort_values("abs_diff", ascending=False).head(14)
    display_cols = [col for col in KEY_COLS + ["interaction_name"] if col in diffs.columns]
    _draw_table(axes[1], diffs[display_cols + ["pyccc", "reference", "abs_diff"]], font_size=6.2)
    axes[1].set_title(_metric_title(metrics, "lr_source_target_prob"))
    fig.suptitle("Analysis 1: LR-source-target communication probabilities", x=0.02, ha="left", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _network_page(pdf: PdfPages, pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, metrics: pd.DataFrame, *, reference_label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=A4_LANDSCAPE, constrained_layout=True)
    py_mat = pyccc_result.network()
    ref_mat = reference_result.network()
    vmax = max(float(py_mat.to_numpy().max()), float(ref_mat.to_numpy().max()))
    _heatmap(axes[0], py_mat, title="pyccc global weight", vmax=vmax)
    _heatmap(axes[1], ref_mat, title=f"{reference_label} global weight", vmax=vmax)
    _heatmap(axes[2], py_mat - ref_mat, title="pyccc - reference", cmap="coolwarm")
    fig.suptitle(f"Analysis 2: Global source-target network | {_metric_title(metrics, 'global_network_weight')}", x=0.02, ha="left", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _pathway_rank_page(pdf: PdfPages, pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, metrics: pd.DataFrame, *, reference_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE, constrained_layout=True)
    _pathway_rank(axes[0], pyccc_result.pathway_summary().head(15), "pyccc information flow")
    _pathway_rank(axes[1], reference_result.pathway_summary().head(15), f"{reference_label} information flow")
    fig.suptitle(f"Analysis 3: Pathway information-flow rank | {_metric_title(metrics, 'pathway_information_flow')}", x=0.02, ha="left", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _pathway_heatmap_page(pdf: PdfPages, pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, metrics: pd.DataFrame, *, reference_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE, constrained_layout=True)
    cp.pathway_heatmap(pyccc_result, top_pairs=18, compact_pairs=True, ax=axes[0], title="pyccc pathway heatmap")
    cp.pathway_heatmap(reference_result, top_pairs=18, compact_pairs=True, ax=axes[1], title=f"{reference_label} pathway heatmap")
    fig.suptitle("Analysis 4: Pathway by source-target heatmap", x=0.02, ha="left", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _lr_contribution_page(pdf: PdfPages, pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, metrics: pd.DataFrame, *, reference_label: str) -> None:
    pathway = str(pyccc_result.pathway_summary().iloc[0]["pathway"])
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE, constrained_layout=True)
    cp.lr_contribution(pyccc_result, pathway, top_n=10, ax=axes[0], title=f"pyccc {pathway} LR contribution")
    cp.lr_contribution(reference_result, pathway, top_n=10, ax=axes[1], title=f"{reference_label} {pathway} LR contribution")
    fig.suptitle(f"Analysis 5: LR contribution in top pathway | {_metric_title(metrics, f'{pathway}_lr_contribution')}", x=0.02, ha="left", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _bubble_page(pdf: PdfPages, pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, metrics: pd.DataFrame, *, reference_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE, constrained_layout=True)
    cp.bubble(pyccc_result, top_n=18, top_pairs=10, compact_pairs=True, ax=axes[0], title="pyccc LR bubble")
    cp.bubble(reference_result, top_n=18, top_pairs=10, compact_pairs=True, ax=axes[1], title=f"{reference_label} LR bubble")
    fig.suptitle("Analysis 6: Top ligand-receptor bubble", x=0.02, ha="left", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _role_page(pdf: PdfPages, pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, metrics: pd.DataFrame, *, reference_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE, constrained_layout=True)
    cp.signaling_role_scatter(pyccc_result, ax=axes[0], title="pyccc outgoing vs incoming roles")
    cp.signaling_role_scatter(reference_result, ax=axes[1], title=f"{reference_label} outgoing vs incoming roles")
    fig.suptitle("Analysis 7: Signaling role scatter", x=0.02, ha="left", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _filter_lr_to_available_genes(lr: pd.DataFrame, gene_lookup: dict[str, str]) -> pd.DataFrame:
    keep = []
    for row in lr.itertuples(index=False):
        genes = _complex_genes(row.ligand) + _complex_genes(row.receptor)
        keep.append(bool(genes) and all(gene.upper() in gene_lookup for gene in genes))
    return lr.loc[keep].reset_index(drop=True)


def _required_expression_genes(lr: pd.DataFrame, gene_lookup: dict[str, str], *, include_cofactors: bool) -> list[str]:
    columns = ["ligand", "receptor"]
    if include_cofactors:
        columns.extend(["co_A_receptor_genes", "co_I_receptor_genes", "agonist_genes", "antagonist_genes"])
    genes = []
    for col in columns:
        if col not in lr.columns:
            continue
        for value in lr[col].fillna("").astype(str):
            for gene in _complex_genes(value):
                resolved = gene_lookup.get(gene.upper())
                if resolved is not None:
                    genes.append(resolved)
    return list(dict.fromkeys(genes))


def _cellchat_r_lr_use(lr: pd.DataFrame) -> pd.DataFrame:
    required = {"interaction_name", "cellchat_ligand", "cellchat_receptor", "pathway", "annotation"}
    missing = sorted(required - set(map(str, lr.columns)))
    if missing:
        raise ValueError(f"CellChat R reference requires CellChat metadata columns: {missing}")
    out = pd.DataFrame(
        {
            "interaction_name": lr["interaction_name"].astype(str),
            "pathway_name": lr["pathway"].astype(str),
            "ligand": lr["cellchat_ligand"].astype(str),
            "receptor": lr["cellchat_receptor"].astype(str),
            "annotation": lr["annotation"].astype(str),
        }
    )
    for col in ["agonist", "antagonist", "co_A_receptor", "co_I_receptor", "interaction_name_2", "evidence"]:
        out[col] = lr[col].fillna("").astype(str) if col in lr.columns else ""
    return out.fillna("")


def _scaled_expression_frame(adata: AnnData, genes: list[str]) -> pd.DataFrame:
    x = adata[:, genes].X
    arr = x.toarray() if sparse.issparse(x) else np.asarray(x, dtype=float)
    max_value = float(np.nanmax(arr)) if arr.size else 0.0
    if np.isfinite(max_value) and max_value > 0:
        arr = arr / max_value
    return pd.DataFrame(arr, index=adata.obs_names, columns=genes)


def _group_expression(expr: pd.DataFrame, labels: np.ndarray, groups: list[str], *, aggregate: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg = pd.DataFrame(index=groups, columns=expr.columns, dtype=float)
    pct = pd.DataFrame(index=groups, columns=expr.columns, dtype=float)
    for group in groups:
        sub = expr.loc[labels == group]
        pct.loc[group] = (sub > 0).mean(axis=0).to_numpy(dtype=float)
        if aggregate == "tri_mean":
            q = np.percentile(sub.to_numpy(dtype=float), [25, 50, 75], axis=0)
            avg.loc[group] = ((q[0] + 2 * q[1] + q[2]) / 4).ravel()
        elif aggregate == "mean":
            avg.loc[group] = sub.mean(axis=0).to_numpy(dtype=float)
        else:
            raise ValueError("reference runner supports `tri_mean` and `mean`.")
    return avg, pct


def _complex_matrix(values: pd.DataFrame, complexes: pd.Series, gene_lookup: dict[str, str], *, mode: str = "geometric_mean") -> np.ndarray:
    out = []
    for complex_name in complexes.astype(str):
        genes = [gene_lookup[gene.upper()] for gene in _complex_genes(complex_name)]
        arr = values.loc[:, genes].to_numpy(dtype=float)
        if arr.shape[1] == 1:
            out.append(arr[:, 0])
        elif mode == "min":
            out.append(arr.min(axis=1))
        else:
            positive = arr > 0
            geo = np.zeros(arr.shape[0], dtype=float)
            keep = positive.all(axis=1)
            if keep.any():
                geo[keep] = np.exp(np.log(arr[keep]).mean(axis=1))
            out.append(geo)
    return np.vstack(out)


def _coreceptor_factor_matrix(lr: pd.DataFrame, expr_avg: pd.DataFrame, groups: list[str], gene_lookup: dict[str, str], column: str) -> np.ndarray:
    factors = np.ones((len(lr), len(groups)), dtype=float)
    if column not in lr.columns:
        return factors
    for i, complex_name in enumerate(lr[column].fillna("").astype(str)):
        for j, group in enumerate(groups):
            values = _optional_values(expr_avg.loc[group], complex_name, gene_lookup)
            if values.size:
                factors[i, j] = float(np.prod(1.0 + values))
    return factors


def _hill_factor_matrix(lr: pd.DataFrame, expr_avg: pd.DataFrame, groups: list[str], gene_lookup: dict[str, str], column: str, *, kh: float, hill: float, mode: str) -> np.ndarray:
    factors = np.ones((len(lr), len(groups)), dtype=float)
    if column not in lr.columns:
        return factors
    kh_pow = kh**hill
    for i, complex_name in enumerate(lr[column].fillna("").astype(str)):
        for j, group in enumerate(groups):
            values = _optional_values(expr_avg.loc[group], complex_name, gene_lookup)
            if not values.size:
                continue
            value_pow = np.power(values, hill)
            if mode == "agonist":
                factors[i, j] = float(np.prod(1.0 + value_pow / (kh_pow + value_pow)))
            else:
                factors[i, j] = float(np.prod(kh_pow / (kh_pow + value_pow)))
    return factors


def _optional_values(values: pd.Series, complex_name: str, gene_lookup: dict[str, str]) -> np.ndarray:
    genes = [gene_lookup[gene.upper()] for gene in _complex_genes(complex_name) if gene.upper() in gene_lookup]
    if not genes:
        return np.array([], dtype=float)
    return values.loc[genes].to_numpy(dtype=float)


def _complex_genes(name: str) -> list[str]:
    normalized = str(name).replace("+", "_").replace("&", "_").replace(":", "_")
    return [part.strip() for part in normalized.split("_") if part.strip()]


def _lr_metadata_columns(lr: pd.DataFrame) -> list[str]:
    core = {"ligand", "receptor", "pathway", "annotation"}
    return [col for col in lr.columns if col not in core and pd.api.types.is_string_dtype(lr[col])]


def _aligned_probabilities(a: pd.DataFrame, b: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    left = a[key_cols + ["prob"]].rename(columns={"prob": "pyccc"})
    right = b[key_cols + ["prob"]].rename(columns={"prob": "reference"})
    return left.merge(right, on=key_cols, how="outer").fillna({"pyccc": 0.0, "reference": 0.0})


def _probability_key_cols(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    if "interaction_name" in a.columns and "interaction_name" in b.columns:
        return ["source", "target", "interaction_name"]
    return KEY_COLS


def _aligned_matrices(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    b = b.reindex(index=a.index.union(b.index), columns=a.columns.union(b.columns), fill_value=0.0)
    a = a.reindex(index=b.index, columns=b.columns, fill_value=0.0)
    return pd.DataFrame({"pyccc": a.to_numpy().ravel(), "reference": b.to_numpy().ravel()})


def _aligned_summary(a: pd.DataFrame, b: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    return a[[key, value]].rename(columns={value: "pyccc"}).merge(b[[key, value]].rename(columns={value: "reference"}), on=key, how="outer").fillna(0.0)


def _aligned_lr_summary(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    keys = ["ligand", "receptor", "pathway"]
    return a[keys + ["prob"]].rename(columns={"prob": "pyccc"}).merge(b[keys + ["prob"]].rename(columns={"prob": "reference"}), on=keys, how="outer").fillna(0.0)


def _aligned_contribution(pyccc_result: pc.CCCResult, reference_result: pc.CCCResult, pathway: str) -> pd.DataFrame:
    def frame(result: pc.CCCResult, value: str) -> pd.DataFrame:
        df = result.interactions[result.interactions["pathway"].astype(str) == str(pathway)].copy()
        if df.empty:
            return pd.DataFrame({"interaction": [], value: []})
        df["interaction"] = df["ligand"].astype(str) + " - " + df["receptor"].astype(str)
        out = df.groupby("interaction", observed=True)["prob"].sum().reset_index()
        total = float(out["prob"].sum()) or 1.0
        out[value] = out["prob"] / total
        return out[["interaction", value]]

    return frame(pyccc_result, "pyccc").merge(frame(reference_result, "reference"), on="interaction", how="outer").fillna(0.0)


def _vector_metrics(name: str, aligned: pd.DataFrame) -> dict[str, float | str | int]:
    x = pd.to_numeric(aligned["pyccc"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(aligned["reference"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        pearson = 1.0 if np.allclose(x, y) else np.nan
        spearman = pearson
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman = float(stats.spearmanr(x, y).correlation)
    top_n = min(20, len(x))
    top_overlap = 1.0
    if top_n:
        top_a = set(np.argsort(x)[-top_n:])
        top_b = set(np.argsort(y)[-top_n:])
        top_overlap = len(top_a & top_b) / top_n
    return {
        "analysis": name,
        "n_values": int(len(x)),
        "pearson": pearson,
        "spearman": spearman,
        "max_abs_diff": float(np.max(np.abs(x - y))) if len(x) else 0.0,
        "median_abs_diff": float(np.median(np.abs(x - y))) if len(x) else 0.0,
        "top20_overlap": float(top_overlap),
    }


def _metric_title(metrics: pd.DataFrame, analysis: str) -> str:
    row = metrics[metrics["analysis"].astype(str) == str(analysis)]
    if row.empty:
        return ""
    r = row.iloc[0]
    return f"r={float(r['pearson']):.4f}, max_abs_diff={float(r['max_abs_diff']):.2e}, top20={float(r['top20_overlap']):.2f}"


def _short_analysis_label(label: str) -> str:
    mapping = {
        "lr_source_target_prob": "LR prob",
        "global_network_weight": "network",
        "pathway_information_flow": "pathway flow",
        "lr_information_flow": "LR flow",
    }
    text = str(label)
    return mapping.get(text, text.replace("_lr_contribution", " contrib"))


def _identity_line(ax, x, y) -> None:
    values = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    finite = values[np.isfinite(values)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if lo == hi:
        hi = lo + 1.0
    ax.plot([lo, hi], [lo, hi], color="#b8323b", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def _heatmap(ax, matrix: pd.DataFrame, *, title: str, cmap: str = "viridis", vmax: float | None = None) -> None:
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=None if cmap == "coolwarm" else 0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=6)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _pathway_rank(ax, df: pd.DataFrame, title: str) -> None:
    frame = df.sort_values("prob", ascending=True).tail(15)
    ax.barh(frame["pathway"].astype(str), frame["prob"].astype(float), color="#6f63b6")
    ax.set_title(title)
    ax.set_xlabel("Information flow")
    ax.tick_params(axis="y", labelsize=7)


def _draw_table(ax, df: pd.DataFrame, *, bbox=None, font_size: float = 7.0) -> None:
    ax.axis("off")
    frame = df.copy()
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].map(lambda value: f"{float(value):.4g}")
    table = ax.table(cellText=frame.astype(str).to_numpy(), colLabels=list(frame.columns), loc="center", cellLoc="left", colLoc="left", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.14)


if __name__ == "__main__":
    main()

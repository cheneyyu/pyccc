from __future__ import annotations

import importlib.util
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

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

try:
    from cellchat_reference_comparison import _cellchat_r_lr_use, _filter_lr_to_available_genes
except ModuleNotFoundError:  # pragma: no cover - used when imported from tests
    _comparison_path = Path(__file__).with_name("cellchat_reference_comparison.py")
    _spec = importlib.util.spec_from_file_location("cellchat_reference_comparison", _comparison_path)
    if _spec is None or _spec.loader is None:
        raise
    _comparison = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_comparison)
    _cellchat_r_lr_use = _comparison._cellchat_r_lr_use
    _filter_lr_to_available_genes = _comparison._filter_lr_to_available_genes


A4_LANDSCAPE = (11.69, 8.27)
TARGETS = ["cDC1", "cDC2", "LC", "Inflam. DC", "TC", "Inflam. TC", "CD40LG+ TC"]
RECEIVERS = TARGETS + ["NKT"]
PLOT_SPECS = [
    ("network_circle", "Overall source-target circle network"),
    ("bubble_cc_cxcl", "CCL/CXCL ligand-receptor bubble"),
    ("source_target_heatmap", "Source-target signaling heatmap"),
    ("pathway_rank", "Pathway information-flow rank"),
    ("lr_contribution_mif", "MIF ligand-receptor contribution"),
    ("role_scatter", "Outgoing/incoming signaling role scatter"),
    ("cxcl_hierarchy", "CXCL hierarchy network"),
    ("cxcl_circle", "CXCL circle network"),
    ("cxcl_chord", "CXCL chord network"),
    ("cxcl_chord_gene", "CXCL LR-mediated chord"),
    ("rank_compare", "LS vs NL rankNet comparison"),
    ("rank_compare_stacked", "LS vs NL stacked rankNet comparison"),
]

CELLCHAT_R_PLOT_RUNNER = r"""
suppressPackageStartupMessages(library(CellChat))
suppressPackageStartupMessages(library(ggplot2))

args <- commandArgs(trailingOnly = TRUE)
data_path <- args[[1]]
lr_path <- args[[2]]
out_dir <- args[[3]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

load(data_path)
pairLR.use <- read.delim(lr_path, check.names = FALSE, stringsAsFactors = FALSE)
rownames(pairLR.use) <- pairLR.use$interaction_name
targets <- c("cDC1", "cDC2", "LC", "Inflam. DC", "TC", "Inflam. TC", "CD40LG+ TC")

make_obj <- function(condition) {
  meta <- data_humanSkin$meta
  keep <- meta$condition == condition
  data.input <- data_humanSkin$data[, rownames(meta)[keep], drop = FALSE]
  meta.use <- droplevels(meta[keep, , drop = FALSE])
  meta.use$cell_type <- droplevels(meta.use$labels)
  obj <- createCellChat(object = data.input, meta = meta.use, group.by = "cell_type", datatype = "RNA")
  db.use <- CellChatDB.human
  db.use$interaction <- pairLR.use
  obj@DB <- db.use
  obj@LR$LRsig <- pairLR.use
  obj <- subsetData(obj)
  obj <- computeCommunProb(
    obj,
    type = "triMean",
    LR.use = pairLR.use,
    raw.use = TRUE,
    population.size = FALSE,
    nboot = 1,
    Kh = 0.5,
    n = 1
  )
  obj@LR$LRsig <- pairLR.use
  obj <- computeCommunProbPathway(obj, pairLR.use = pairLR.use, thresh = 1)
  obj <- aggregateNet(obj, thresh = 1, remove.isolate = FALSE)
  obj <- netAnalysis_computeCentrality(obj, slot.name = "netP", thresh = 1)
  obj
}

draw_returned <- function(x) {
  if (inherits(x, "ggplot")) {
    print(x)
  } else if (inherits(x, "Heatmap") || inherits(x, "HeatmapList")) {
    ComplexHeatmap::draw(x)
  } else if (!is.null(x)) {
    print(x)
  }
}

save_plot <- function(id, code, width = 1500, height = 1100, res = 170) {
  path <- file.path(out_dir, paste0(id, ".png"))
  status <- "ok"
  message <- ""
  tryCatch({
    png(path, width = width, height = height, res = res, type = "cairo-png")
    tryCatch({
      x <- force(code)
      draw_returned(x)
    }, finally = dev.off())
    if (requireNamespace("circlize", quietly = TRUE)) {
      try(circlize::circos.clear(), silent = TRUE)
    }
  }, error = function(e) {
    status <<- "error"
    message <<- conditionMessage(e)
    try(dev.off(), silent = TRUE)
    if (file.exists(path)) {
      file.remove(path)
    }
  })
  data.frame(id = id, path = path, status = status, message = message, stringsAsFactors = FALSE)
}

lsobj <- make_obj("LS")
nlobj <- make_obj("NL")
merged <- mergeCellChat(list(LS = lsobj, NL = nlobj), add.names = c("LS", "NL"), cell.prefix = TRUE)

manifest <- rbind(
  save_plot("network_circle", netVisual_circle(lsobj@net$weight, title.name = "CellChat R: overall network", vertex.weight = as.numeric(table(lsobj@idents)), weight.scale = TRUE, label.edge = FALSE), 1300, 1200),
  save_plot("bubble_cc_cxcl", netVisual_bubble(lsobj, sources.use = "Inflam. FIB", targets.use = targets, signaling = c("CCL", "CXCL"), remove.isolate = FALSE, thresh = 1, title.name = "CellChat R: CCL/CXCL bubble"), 1700, 1100),
  save_plot("source_target_heatmap", netVisual_heatmap(lsobj, measure = "weight", slot.name = "netP", color.heatmap = "Reds", title.name = "CellChat R: source-target heatmap", cluster.rows = FALSE, cluster.cols = FALSE), 1300, 1100),
  save_plot("pathway_rank", rankNet(lsobj, mode = "single", measure = "weight", stacked = FALSE, thresh = 1, title = "CellChat R: pathway information flow"), 1400, 1050),
  save_plot("lr_contribution_mif", netAnalysis_contribution(lsobj, signaling = "MIF", thresh = 1, title = "CellChat R: MIF LR contribution"), 1400, 900),
  save_plot("role_scatter", netAnalysis_signalingRole_scatter(lsobj, slot.name = "netP", title = "CellChat R: signaling roles"), 1200, 1000),
  save_plot("cxcl_hierarchy", netVisual_aggregate(lsobj, signaling = "CXCL", layout = "hierarchy", vertex.receiver = seq(5, 12), thresh = 1, remove.isolate = FALSE), 1500, 1100),
  save_plot("cxcl_circle", netVisual_aggregate(lsobj, signaling = "CXCL", layout = "circle", thresh = 1, remove.isolate = FALSE), 1300, 1200),
  save_plot("cxcl_chord", netVisual_aggregate(lsobj, signaling = "CXCL", layout = "chord", thresh = 1, remove.isolate = FALSE), 1300, 1200),
  save_plot("cxcl_chord_gene", netVisual_chord_gene(lsobj, signaling = "CXCL", sources.use = "Inflam. FIB", thresh = 1), 1400, 1200),
  save_plot("rank_compare", rankNet(merged, mode = "comparison", stacked = FALSE, comparison = c(1, 2), do.stat = FALSE, thresh = 1, title = "CellChat R: LS vs NL rankNet"), 1500, 1100),
  save_plot("rank_compare_stacked", rankNet(merged, mode = "comparison", stacked = TRUE, comparison = c(1, 2), do.stat = FALSE, thresh = 1, title = "CellChat R: stacked LS vs NL rankNet"), 1500, 1100)
)
write.table(manifest, file.path(out_dir, "r_plot_manifest.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
"""


@dataclass(frozen=True)
class PlotArtifact:
    plot_id: str
    title: str
    py_path: Path
    r_path: Path
    r_status: str
    r_message: str


def main() -> None:
    cache_dir = Path(os.environ.get("PYCCC_CACHE_DIR", Path.home() / ".cache" / "pyccc")) / "cellchat_official"
    out_dir = Path(os.environ.get("PYCCC_OUTPUT_DIR", "cellchat_plot_comparison"))
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
    gene_lookup = {str(gene).upper(): str(gene) for gene in adata.var_names}
    lr_for_r = _cellchat_r_lr_use(_filter_lr_to_available_genes(lr_db.interactions.copy(), gene_lookup))

    r_dir = out_dir / "cellchat_r_figures"
    py_dir = out_dir / "pyccc_python_figures"
    r_dir.mkdir(parents=True, exist_ok=True)
    py_dir.mkdir(parents=True, exist_ok=True)
    lr_path = out_dir / "cellchat_lr_use.tsv"
    lr_for_r.to_csv(lr_path, sep="\t", index=False)

    ls, nl = _compute_pyccc_results(adata, lr_db)
    diff = pc.compare_communication(ls, nl, label_a="LS", label_b="NL")
    py_paths = _save_pyccc_plots(py_dir, ls, diff)
    r_manifest = _run_cellchat_r_plots(data_path, lr_path, r_dir, out_dir / "run_cellchat_plot_comparison.R")

    artifacts = []
    for plot_id, title in PLOT_SPECS:
        row = r_manifest[r_manifest["id"].astype(str) == plot_id]
        if row.empty:
            r_status = "missing"
            r_message = "No manifest row was written by the R runner."
        else:
            r_status = str(row.iloc[0]["status"])
            r_message = "" if pd.isna(row.iloc[0]["message"]) else str(row.iloc[0]["message"])
        artifacts.append(
            PlotArtifact(
                plot_id=plot_id,
                title=title,
                py_path=py_paths[plot_id],
                r_path=r_dir / f"{plot_id}.png",
                r_status=r_status,
                r_message=r_message,
            )
        )

    pdf_path = out_dir / "pyccc_python_vs_cellchat_r_plots_a4.pdf"
    _save_a4_report(pdf_path, artifacts, adata=adata, lr_rows=len(lr_db.interactions))
    notes_path = out_dir / "pyccc_python_vs_cellchat_r_plots_notes.md"
    _write_notes(notes_path, artifacts, adata=adata, lr_rows=len(lr_db.interactions), pdf_path=pdf_path)

    print(f"Dataset: official CellChat human skin data ({adata.n_obs} cells x {adata.n_vars} genes)")
    print(f"LR table: {len(lr_db.interactions)} Secreted Signaling interactions")
    print(f"Saved Python figures: {py_dir.resolve()}")
    print(f"Saved CellChat R figures: {r_dir.resolve()}")
    print(f"Saved A4 plot comparison PDF: {pdf_path.resolve()}")
    print(f"Saved notes: {notes_path.resolve()}")
    print(r_manifest.to_string(index=False))


def _compute_pyccc_results(adata, lr_db: pc.CellChatDB):
    kwargs = dict(
        condition_key="condition",
        min_pct=0.0,
        min_expr=0.0,
        aggregate="tri_mean",
        score_method="cellchat",
        cofactor_adjust=True,
        cofactor_kh=0.5,
        cofactor_hill=1.0,
        population_size=False,
    )
    ls = pc.compute_communication(adata, "cell_type", lr_db, condition="LS", **kwargs)
    nl = pc.compute_communication(adata, "cell_type", lr_db, condition="NL", **kwargs)
    return ls, nl


def _save_pyccc_plots(out_dir: Path, ls: pc.CCCResult, diff) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def save(plot_id: str, fig) -> None:
        path = out_dir / f"{plot_id}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths[plot_id] = path

    fig, ax = plt.subplots(figsize=(7.6, 7.0), constrained_layout=True)
    cp.net_circle(ls, ax=ax, title="pyccc Python: overall network")
    save("network_circle", fig)

    fig, ax = plt.subplots(figsize=(9.4, 6.1), constrained_layout=True)
    cp.bubble(ls, sources=["Inflam. FIB"], targets=TARGETS, pathways=["CCL", "CXCL"], compact_pairs=True, ax=ax, title="pyccc Python: CCL/CXCL bubble")
    save("bubble_cc_cxcl", fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.4), constrained_layout=True)
    cp.net_heatmap(ls, pathway=None, ax=ax, title="pyccc Python: source-target heatmap")
    save("source_target_heatmap", fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.6), constrained_layout=True)
    _plot_pyccc_ranknet_single(ls, top_n=16, ax=ax, title="pyccc Python: pathway information flow")
    save("pathway_rank", fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    cp.lr_contribution(ls, "MIF", top_n=12, ax=ax, title="pyccc Python: MIF LR contribution")
    save("lr_contribution_mif", fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    cp.signaling_role_scatter(ls, ax=ax, title="pyccc Python: signaling roles")
    save("role_scatter", fig)

    sources = [group for group in ls.groups if group not in set(RECEIVERS)]
    fig, ax = plt.subplots(figsize=(8.6, 6.3), constrained_layout=True)
    cp.net_hierarchy(ls, pathway="CXCL", sources=sources, targets=RECEIVERS, ax=ax, title="pyccc Python: CXCL hierarchy")
    save("cxcl_hierarchy", fig)

    fig, ax = plt.subplots(figsize=(7.4, 7.0), constrained_layout=True)
    cp.net_circle(ls, pathway="CXCL", ax=ax, title="pyccc Python: CXCL circle")
    save("cxcl_circle", fig)

    fig, ax = plt.subplots(figsize=(7.4, 7.0), constrained_layout=True)
    cp.net_chord(ls, pathway="CXCL", top_n=60, ax=ax, title="pyccc Python: CXCL chord")
    save("cxcl_chord", fig)

    fig, ax = plt.subplots(figsize=(7.8, 7.0), constrained_layout=True)
    cp.net_chord_gene(ls, sources=["Inflam. FIB"], pathways=["CXCL"], top_n=30, ax=ax, title="pyccc Python: CXCL LR-mediated chord")
    save("cxcl_chord_gene", fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    _plot_pyccc_ranknet_compare(diff, top_n=20, stacked=False, ax=ax, title="pyccc Python: LS vs NL rankNet")
    save("rank_compare", fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    _plot_pyccc_ranknet_compare(diff, top_n=20, stacked=True, ax=ax, title="pyccc Python: stacked LS vs NL rankNet")
    save("rank_compare_stacked", fig)

    return paths


def _plot_pyccc_ranknet_single(result: pc.CCCResult, *, top_n: int, ax, title: str) -> None:
    frame = result.pathway_summary()[["pathway", "prob"]].copy()
    frame["information_flow"] = _cellchat_ranknet_scaled(frame["prob"])
    frame = frame.sort_values("information_flow", ascending=False).head(top_n).sort_values("information_flow", ascending=True)
    ax.barh(frame["pathway"].astype(str), frame["information_flow"].astype(float), color="#d95f5f")
    ax.set_xlabel("Information flow")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)


def _plot_pyccc_ranknet_compare(diff, *, top_n: int, stacked: bool, ax, title: str) -> None:
    left = diff.a.pathway_summary()[["pathway", "prob"]].rename(columns={"prob": diff.label_a})
    right = diff.b.pathway_summary()[["pathway", "prob"]].rename(columns={"prob": diff.label_b})
    frame = left.merge(right, on="pathway", how="outer").fillna(0.0)
    frame[f"{diff.label_a}_scaled"] = _cellchat_ranknet_scaled(frame[diff.label_a])
    frame[f"{diff.label_b}_scaled"] = _cellchat_ranknet_scaled(frame[diff.label_b])
    frame["total"] = frame[f"{diff.label_a}_scaled"] + frame[f"{diff.label_b}_scaled"]
    frame = frame.sort_values("total", ascending=False).head(top_n)
    if stacked:
        display = frame.sort_values("total", ascending=True)
        total = display[[diff.label_a, diff.label_b]].sum(axis=1).replace(0.0, np.nan)
        labels = display["pathway"].astype(str).to_numpy()
        left_values = (display[diff.label_a] / total).fillna(0.0).to_numpy(dtype=float)
        right_values = (display[diff.label_b] / total).fillna(0.0).to_numpy(dtype=float)
        ax.barh(labels, left_values, color="#d95f5f", label=diff.label_a)
        ax.barh(labels, right_values, left=left_values, color="#00bfc4", label=diff.label_b)
        ax.axvline(0.5, linestyle="--", color="#7a7a7a", linewidth=0.8)
        ax.set_xlabel("Relative information flow")
    else:
        long = frame.melt(
            id_vars="pathway",
            value_vars=[f"{diff.label_a}_scaled", f"{diff.label_b}_scaled"],
            var_name="condition",
            value_name="information_flow",
        )
        long["condition"] = long["condition"].str.replace("_scaled", "", regex=False)
        order = frame.sort_values("total", ascending=True)["pathway"].astype(str)
        sns.barplot(long, x="information_flow", y="pathway", hue="condition", order=order, ax=ax, palette=["#d95f5f", "#00bfc4"])
        ax.set_xlabel("Information flow")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(title="", frameon=False, loc="best")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6)


def _cellchat_ranknet_scaled(values: pd.Series) -> np.ndarray:
    raw = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scaled = np.zeros_like(raw, dtype=float)
    positive = raw > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled[positive] = -1.0 / np.log(raw[positive])
    invalid = positive & (~np.isfinite(scaled) | (scaled < 0))
    scaled[~positive] = 0.0
    scaled[invalid] = 0.0
    if invalid.any():
        max_valid = float(np.nanmax(scaled[~invalid])) if (~invalid).any() and np.nanmax(scaled[~invalid]) > 0 else 1.0
        invalid_idx = np.where(invalid)[0]
        order = np.argsort(raw[invalid_idx])
        assignments = np.linspace(max_valid * 1.1, max_valid * 1.5, num=len(invalid_idx))
        scaled[invalid_idx[order]] = assignments
    return scaled


def _run_cellchat_r_plots(data_path: Path, lr_path: Path, r_dir: Path, script_path: Path) -> pd.DataFrame:
    script_path.write_text(textwrap.dedent(CELLCHAT_R_PLOT_RUNNER).strip() + "\n", encoding="utf-8")
    env = os.environ.copy()
    local_r_lib = Path(".r-lib").resolve()
    if "R_LIBS_USER" not in env and local_r_lib.exists():
        env["R_LIBS_USER"] = str(local_r_lib)
    completed = subprocess.run(
        ["Rscript", str(script_path), str(data_path), str(lr_path), str(r_dir)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    log_path = r_dir / "cellchat_r_plot_run.log"
    log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    manifest_path = r_dir / "r_plot_manifest.tsv"
    if completed.returncode != 0:
        raise RuntimeError(f"CellChat R plot runner failed with exit code {completed.returncode}. See {log_path}.")
    if not manifest_path.exists():
        raise RuntimeError(f"CellChat R plot runner did not write {manifest_path}. See {log_path}.")
    return pd.read_csv(manifest_path, sep="\t")


def _save_a4_report(path: Path, artifacts: list[PlotArtifact], *, adata, lr_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        _summary_page(pdf, artifacts, adata=adata, lr_rows=lr_rows)
        for artifact in artifacts:
            _comparison_page(pdf, artifact)


def _summary_page(pdf: PdfPages, artifacts: list[PlotArtifact], *, adata, lr_rows: int) -> None:
    fig, ax = plt.subplots(figsize=A4_LANDSCAPE)
    ax.axis("off")
    ok = sum(artifact.r_status == "ok" and artifact.py_path.exists() for artifact in artifacts)
    summary = pd.DataFrame(
        [
            ["Dataset", "Official CellChat human skin"],
            ["Cells x genes", f"{adata.n_obs} x {adata.n_vars}"],
            ["LR table", f"CellChatDB human Secreted Signaling rows={lr_rows}"],
            ["Conditions", "LS for single-condition plots; LS vs NL for rankNet comparisons"],
            ["Plot comparisons", f"{ok}/{len(artifacts)} R/Python pairs rendered"],
            ["R source", "CellChat R plotting functions"],
            ["Python source", "pyccc Matplotlib plotting functions"],
        ],
        columns=["Item", "Value"],
    )
    ax.set_title("pyccc Python plots vs CellChat R plots", loc="left", fontsize=16, weight="bold")
    _draw_table(ax, summary, bbox=[0.02, 0.53, 0.96, 0.36], font_size=9)
    status = pd.DataFrame(
        {
            "analysis": [artifact.title for artifact in artifacts],
            "r_status": [artifact.r_status for artifact in artifacts],
            "r_message": [artifact.r_message[:70] for artifact in artifacts],
        }
    )
    _draw_table(ax, status, bbox=[0.02, 0.05, 0.96, 0.42], font_size=6.5)
    pdf.savefig(fig)
    plt.close(fig)


def _comparison_page(pdf: PdfPages, artifact: PlotArtifact) -> None:
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE, constrained_layout=True)
    _draw_image_or_message(axes[0], artifact.py_path, f"pyccc Python\n{artifact.py_path.name}")
    r_title = f"CellChat R\n{artifact.r_path.name}"
    if artifact.r_status != "ok":
        r_title += f"\n{artifact.r_status}: {artifact.r_message[:90]}"
    _draw_image_or_message(axes[1], artifact.r_path, r_title)
    fig.suptitle(artifact.title, x=0.02, ha="left", fontsize=14, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def _draw_image_or_message(ax, path: Path, title: str) -> None:
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)
    if path.exists():
        img = mpimg.imread(path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"Missing image:\n{path}", ha="center", va="center", fontsize=10)


def _draw_table(ax, df: pd.DataFrame, *, bbox, font_size: float) -> None:
    table = ax.table(cellText=df.astype(str).to_numpy(), colLabels=list(df.columns), loc="center", cellLoc="left", colLoc="left", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.14)


def _write_notes(path: Path, artifacts: list[PlotArtifact], *, adata, lr_rows: int, pdf_path: Path) -> None:
    lines = [
        "# pyccc Python plots vs CellChat R plots",
        "",
        f"- Dataset: official CellChat human skin data, {adata.n_obs} cells x {adata.n_vars} genes.",
        f"- LR table: same CellChatDB human Secreted Signaling set, {lr_rows} rows before gene filtering.",
        "- CellChat R figures are drawn by CellChat R plotting functions.",
        "- pyccc figures are drawn by pyccc Matplotlib plotting functions.",
        f"- Merged A4 report: `{pdf_path}`.",
        "",
        "## Plot Manifest",
        "",
        "| analysis | pyccc PNG | CellChat R PNG | R status |",
        "| --- | --- | --- | --- |",
    ]
    for artifact in artifacts:
        lines.append(f"| {artifact.title} | `{artifact.py_path}` | `{artifact.r_path}` | {artifact.r_status} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are visual/layout comparisons, not byte-for-byte plot clones.",
            "- The numerical inputs are the same LS/NL CellChatDB Secreted Signaling runs used in the parity comparison.",
            "- R plotting uses `thresh=1` so the one-bootstrap p-value gate does not remove nonzero communication probabilities.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import resource
import subprocess
import threading
import textwrap
import time
import traceback
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

import pyccc as pc
import pyccc.analysis as analysis
import pyccc.plotting as cp


try:
    from cellchat_official_plotnine_demo import HUMAN_SKIN_RDA_URL, _download_if_missing, read_official_human_skin
except ModuleNotFoundError:  # pragma: no cover
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
    from cellchat_reference_comparison import _cellchat_r_lr_use
except ModuleNotFoundError:  # pragma: no cover
    _comparison_path = Path(__file__).with_name("cellchat_reference_comparison.py")
    _spec = importlib.util.spec_from_file_location("cellchat_reference_comparison", _comparison_path)
    if _spec is None or _spec.loader is None:
        raise
    _comparison = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_comparison)
    _cellchat_r_lr_use = _comparison._cellchat_r_lr_use


DIRECT_CELLCHAT_R = r"""
suppressPackageStartupMessages(library(CellChat))
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
matrix_dir <- args[[1]]
lr_path <- args[[2]]
out_dir <- args[[3]]
condition_key <- args[[4]]
condition_a <- args[[5]]
condition_b <- args[[6]]
groupby <- args[[7]]
aggregate <- args[[8]]
cofactor_adjust <- as.logical(args[[9]])

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
timestamp <- function() as.numeric(proc.time()[["elapsed"]])
total_start <- timestamp()

mat <- readMM(file.path(matrix_dir, "matrix.mtx"))
genes <- readLines(file.path(matrix_dir, "genes.tsv"), warn = FALSE)
cells <- readLines(file.path(matrix_dir, "cells.tsv"), warn = FALSE)
meta <- read.delim(file.path(matrix_dir, "meta.tsv"), check.names = FALSE, stringsAsFactors = FALSE)
rownames(meta) <- meta$cell_id
rownames(mat) <- genes
colnames(mat) <- cells
pairLR.use <- read.delim(lr_path, check.names = FALSE, stringsAsFactors = FALSE)
rownames(pairLR.use) <- pairLR.use$interaction_name

plot_manifest <- data.frame(id = character(), path = character(), status = character(), message = character(), stringsAsFactors = FALSE)
draw_returned <- function(x) {
  if (inherits(x, "ggplot")) {
    print(x)
  } else if (inherits(x, "Heatmap") || inherits(x, "HeatmapList")) {
    ComplexHeatmap::draw(x)
  } else if (!is.null(x)) {
    print(x)
  }
}
save_plot <- function(id, code, width = 1300, height = 1000, res = 170) {
  path <- file.path(out_dir, paste0(id, ".png"))
  status <- "ok"
  message <- ""
  tryCatch({
    png(path, width = width, height = height, res = res, type = "cairo-png")
    tryCatch(draw_returned(force(code)), finally = dev.off())
    if (requireNamespace("circlize", quietly = TRUE)) try(circlize::circos.clear(), silent = TRUE)
  }, error = function(e) {
    status <<- "error"
    message <<- conditionMessage(e)
    try(dev.off(), silent = TRUE)
    if (file.exists(path)) file.remove(path)
  })
  data.frame(id = id, path = path, status = status, message = message, stringsAsFactors = FALSE)
}
skip_plot <- function(id, message) {
  data.frame(id = id, path = "", status = "skipped", message = message, stringsAsFactors = FALSE)
}
first_pathway <- function(obj) {
  if (!is.null(obj@netP$pathways) && length(obj@netP$pathways) > 0) obj@netP$pathways[[1]] else NA_character_
}

make_obj <- function(condition) {
  keep <- as.character(meta[[condition_key]]) == condition
  meta.use <- droplevels(meta[keep, , drop = FALSE])
  rownames(meta.use) <- meta.use$cell_id
  meta.use[[groupby]] <- droplevels(factor(meta.use[[groupby]]))
  obj <- createCellChat(object = mat[, rownames(meta.use), drop = FALSE], meta = meta.use, group.by = groupby, datatype = "RNA")
  db.use <- CellChatDB.human
  db.use$interaction <- pairLR.use
  obj@DB <- db.use
  obj@LR$LRsig <- pairLR.use
  obj <- subsetData(obj)
  obj <- computeCommunProb(
    obj,
    type = aggregate,
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
  obj <- tryCatch(netAnalysis_computeCentrality(obj, slot.name = "netP", thresh = 1), error = function(e) obj)
  obj
}

analysis_start <- timestamp()
obj_a <- make_obj(condition_a)
obj_b <- make_obj(condition_b)
merged <- mergeCellChat(stats::setNames(list(obj_a, obj_b), c(condition_a, condition_b)), add.names = c(condition_a, condition_b), cell.prefix = TRUE)
analysis_seconds <- timestamp() - analysis_start

plot_start <- timestamp()
pathway <- first_pathway(obj_a)
plot_manifest <- rbind(
  plot_manifest,
  save_plot("network_circle", netVisual_circle(obj_a@net$weight, vertex.weight = as.numeric(table(obj_a@idents)), weight.scale = TRUE, label.edge = FALSE, title.name = "CellChat network"), 1300, 1200),
  save_plot("pathway_heatmap", netVisual_heatmap(obj_a, measure = "weight", slot.name = "netP", color.heatmap = "Reds", cluster.rows = FALSE, cluster.cols = FALSE, title.name = "CellChat pathway heatmap"), 1300, 1000),
  save_plot("pathway_rank", rankNet(obj_a, mode = "single", measure = "weight", stacked = FALSE, thresh = 1, title = "CellChat pathway rank"), 1300, 950),
  save_plot("compare_interactions_weight", compareInteractions(merged, group = c(1, 2), measure = "weight", title.name = "Interaction weights"), 1100, 900),
  save_plot("rank_pathway_comparison", rankNet(merged, mode = "comparison", comparison = c(1, 2), stacked = FALSE, do.stat = FALSE, thresh = 1, title = "CellChat rank comparison"), 1400, 1000)
)
if (!is.na(pathway)) {
  plot_manifest <- rbind(plot_manifest, save_plot("bubble", netVisual_bubble(obj_a, signaling = pathway, remove.isolate = FALSE, thresh = 1, title.name = paste0(pathway, " bubble")), 1500, 1000))
} else {
  plot_manifest <- rbind(plot_manifest, skip_plot("bubble", "No pathway available."))
}
plot_seconds <- timestamp() - plot_start

write.table(plot_manifest, file.path(out_dir, "direct_cellchat_plot_manifest.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
runtime <- data.frame(
  total_seconds = timestamp() - total_start,
  analysis_seconds = analysis_seconds,
  plot_seconds = plot_seconds,
  n_cells = ncol(mat),
  n_genes = nrow(mat),
  n_lr = nrow(pairLR.use),
  stringsAsFactors = FALSE
)
write.table(runtime, file.path(out_dir, "direct_cellchat_runtime.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
saveRDS(merged, file.path(out_dir, "direct_cellchat_merged.rds"))
"""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / "matplotlib-cache"))
    if args.mode == "official":
        run_official(args, out_dir)
    elif args.mode == "prepared":
        run_prepared(args, out_dir)
    else:
        run_cellxgene_adaptive(args, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pyccc, pyccc-to-CellChat, and direct CellChat runtime.")
    parser.add_argument("--mode", choices=["official", "cellxgene", "prepared"], default="official")
    parser.add_argument("--out-dir", default="data/runtime_benchmark/official")
    parser.add_argument("--cache-dir", default="data/runtime_benchmark/cache")
    parser.add_argument("--h5ad", default="data/cellxgene/global_celltypist_immune_329k.h5ad")
    parser.add_argument("--lr-table", default=None, help="Prepared pyccc ligand-receptor interaction table for --mode prepared.")
    parser.add_argument("--r-input-dir", default=None, help="Prepared CellChat MatrixMarket input directory for --mode prepared.")
    parser.add_argument("--species", default="human", choices=["human", "mouse"])
    parser.add_argument("--annotation", default="Secreted Signaling")
    parser.add_argument("--condition-key", default=None)
    parser.add_argument("--condition-a", default=None)
    parser.add_argument("--condition-b", default=None)
    parser.add_argument("--groupby", default=None)
    parser.add_argument("--gene-symbols-key", default="gene_symbols")
    parser.add_argument("--aggregate", default="tri_mean")
    parser.add_argument("--cellchat-aggregate", default="triMean")
    parser.add_argument("--cofactor-adjust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-raw", action=argparse.BooleanOptionalAction, default=False, help="Use adata.raw.X when loading CELLxGENE h5ad input.")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--target-cells", type=int, default=1000000)
    parser.add_argument("--min-cells", type=int, default=25000)
    parser.add_argument("--n-groups", type=int, default=12, help="Number of shared groups to keep; use 0 to keep all shared groups.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Python worker count for pyccc permutation/downsampling paths.")
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--allow-repeated-cells", action="store_true", help="Allow upsampling by repeating cells when target cells exceed the real filtered dataset size.")
    parser.add_argument("--skip-direct-cellchat", action="store_true")
    return parser.parse_args()


def run_official(args: argparse.Namespace, out_dir: Path) -> None:
    cache_dir = Path(args.cache_dir) / "cellchat_official"
    data_path = cache_dir / "data_humanSkin_CellChat.rda"
    proxy = os.environ.get("PYCCC_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")
    _download_if_missing(HUMAN_SKIN_RDA_URL, data_path, proxy=proxy)
    adata = read_official_human_skin(data_path)
    defaults = {"condition_key": "condition", "condition_a": "LS", "condition_b": "NL", "groupby": "cell_type"}
    args = apply_defaults(args, defaults)
    if args.gene_symbols_key and args.gene_symbols_key not in adata.var:
        args.gene_symbols_key = None
    lr_db, lr_r = load_lr(args, adata)
    run_three_way_for_adata(args, out_dir, adata, lr_db, lr_r, scale_label="official_full")


def run_prepared(args: argparse.Namespace, out_dir: Path) -> None:
    args = apply_defaults(args, {"condition_key": "disease", "condition_a": "cytomegalovirus infection", "condition_b": "normal", "groupby": "cell_type"})
    if not args.lr_table:
        raise ValueError("--mode prepared requires --lr-table.")
    if not args.r_input_dir:
        raise ValueError("--mode prepared requires --r-input-dir.")
    start = time.perf_counter()
    adata = ad.read_h5ad(args.h5ad)
    sample_read_seconds = time.perf_counter() - start
    if args.gene_symbols_key and args.gene_symbols_key not in adata.var:
        args.gene_symbols_key = None
    lr = pd.read_csv(args.lr_table, sep="\t")
    lr_db = pc.CellChatDB(lr, name=Path(args.lr_table).stem)
    lr_r = _cellchat_r_lr_use(lr)
    input_dir = Path(args.r_input_dir)
    if not has_r_input(input_dir):
        raise FileNotFoundError(f"Prepared R input directory is incomplete: {input_dir}")
    run_three_way_for_adata(
        args,
        out_dir,
        adata,
        lr_db,
        lr_r,
        scale_label=f"cells_{adata.n_obs}",
        input_dir=input_dir,
        sample_read_seconds=sample_read_seconds,
    )


def run_cellxgene_adaptive(args: argparse.Namespace, out_dir: Path) -> None:
    defaults = {"condition_key": "tissue", "condition_a": "spleen", "condition_b": "lung", "groupby": "Predicted_labels_CellTypist"}
    args = apply_defaults(args, defaults)
    base, lr_db, lr_r = load_cellxgene_base(args)
    attempt_cells = int(args.target_cells)
    all_rows: list[pd.DataFrame] = []
    while attempt_cells >= int(args.min_cells):
        print(f"\n=== CELLxGENE adaptive attempt: {attempt_cells:,} cells ===", flush=True)
        attempt_dir = out_dir / f"cells_{attempt_cells}"
        adata = scale_adata(
            base,
            attempt_cells,
            condition_key=args.condition_key,
            random_state=args.random_state,
            allow_repeated_cells=args.allow_repeated_cells,
        )
        rows = run_three_way_for_adata(args, attempt_dir, adata, lr_db, lr_r, scale_label=f"cells_{adata.n_obs}")
        all_rows.append(rows)
        statuses = set(rows["status"].astype(str))
        if statuses <= {"ok", "skipped"} and "ok" in statuses:
            break
        attempt_cells //= 2
        del adata
        gc.collect()
    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        summary.to_csv(out_dir / "adaptive_runtime.tsv", sep="\t", index=False)
        write_speedup_summary(summary, out_dir / "adaptive_summary.md")
        write_runtime_plot(summary, out_dir / "runtime_speedup.png")


def apply_defaults(args: argparse.Namespace, defaults: dict[str, str]) -> argparse.Namespace:
    data = vars(args).copy()
    for key, value in defaults.items():
        if data.get(key) is None:
            data[key] = value
    return argparse.Namespace(**data)


def load_lr(args: argparse.Namespace, adata) -> tuple[pc.CellChatDB, pd.DataFrame]:
    db = pc.load_cellchatdb(args.species, cache_dir=args.cache_dir)
    lr = db.interactions.copy()
    if args.annotation:
        lr = lr[lr["annotation"].astype(str).eq(args.annotation)].reset_index(drop=True)
    matrix_names = analysis._matrix_var_names(adata.var, adata.var_names, gene_symbols_key=args.gene_symbols_key)
    gene_lookup = {str(gene).upper(): str(gene) for gene in matrix_names.astype(str)}
    lr = analysis._filter_lr_to_genes(lr, gene_lookup)
    if lr.empty:
        raise ValueError("No LR rows remain after filtering to available genes.")
    return pc.CellChatDB(lr, name=f"{db.name}_{args.annotation or 'all'}", metadata=db.metadata), _cellchat_r_lr_use(lr)


def load_cellxgene_base(args: argparse.Namespace):
    backed = ad.read_h5ad(args.h5ad, backed="r")
    try:
        source_var = backed.raw.var if args.use_raw and backed.raw is not None else backed.var
        source_var_names = pd.Index(backed.raw.var_names) if args.use_raw and backed.raw is not None else backed.var_names
        source_x = backed.raw.X if args.use_raw and backed.raw is not None else None
        if args.use_raw and backed.raw is None:
            raise ValueError("--use-raw was requested, but the h5ad does not contain adata.raw.")
        db = pc.load_cellchatdb(args.species, cache_dir=args.cache_dir)
        lr = db.interactions.copy()
        if args.annotation:
            lr = lr[lr["annotation"].astype(str).eq(args.annotation)].reset_index(drop=True)
        matrix_names = _matrix_var_names_allow_duplicates(source_var, source_var_names, gene_symbols_key=args.gene_symbols_key)
        gene_lookup = {str(gene).upper(): str(gene) for gene in matrix_names.astype(str)}
        lr = analysis._filter_lr_to_genes(lr, gene_lookup)
        selected = analysis._lr_expression_gene_candidates(lr, gene_lookup, include_cofactors=args.cofactor_adjust)
        gene_mask = np.asarray(matrix_names.isin(selected))
        symbol_values = matrix_names[gene_mask].astype(str).to_numpy()
        keep_unique = ~pd.Index(symbol_values).duplicated()
        gene_idx = np.flatnonzero(gene_mask)[keep_unique]
        symbol_values = symbol_values[keep_unique]

        obs = backed.obs
        cond = obs[args.condition_key].astype(str)
        group = obs[args.groupby].astype(str)
        tab = pd.crosstab(cond, group)
        common = tab.columns[(tab.loc[args.condition_a] > 0) & (tab.loc[args.condition_b] > 0)]
        totals = (tab.loc[args.condition_a, common] + tab.loc[args.condition_b, common]).sort_values(ascending=False)
        groups = totals.index.astype(str).tolist() if args.n_groups <= 0 else totals.head(args.n_groups).index.astype(str).tolist()
        mask = cond.isin([args.condition_a, args.condition_b]).to_numpy() & group.isin(groups).to_numpy()
        cell_idx = np.flatnonzero(mask)
        if cell_idx.size == 0:
            raise ValueError("No cells remain after condition and group filtering.")
        if source_x is None:
            base = backed[cell_idx, gene_idx].to_memory()
        else:
            base = ad.AnnData(source_x[cell_idx, :][:, gene_idx], obs=backed.obs.iloc[cell_idx].copy(), var=source_var.iloc[gene_idx].copy())
        base.var_names = pd.Index(symbol_values)
        base.var[args.gene_symbols_key] = symbol_values
        base.obs[args.condition_key] = base.obs[args.condition_key].astype(str)
        base.obs[args.groupby] = base.obs[args.groupby].astype(str)
    finally:
        backed.file.close()
    lr_db = pc.CellChatDB(lr, name=f"{db.name}_{args.annotation or 'all'}", metadata=db.metadata)
    return base, lr_db, _cellchat_r_lr_use(lr)


def run_three_way_for_adata(
    args: argparse.Namespace,
    out_dir: Path,
    adata,
    lr_db: pc.CellChatDB,
    lr_r: pd.DataFrame,
    *,
    scale_label: str,
    input_dir: Path | None = None,
    sample_read_seconds: float = 0.0,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    lr_path = out_dir / "cellchat_lr.tsv"
    lr_r.to_csv(lr_path, sep="\t", index=False)
    if input_dir is None:
        input_dir = out_dir / "r_input"
        write_r_input(adata, input_dir, groupby=args.groupby, condition_key=args.condition_key, gene_symbols_key=args.gene_symbols_key)
    elif not has_r_input(input_dir):
        raise FileNotFoundError(f"Prepared R input directory is incomplete: {input_dir}")
    write_input_runtime(out_dir / "input_runtime.tsv", adata, sample_read_seconds, input_dir)

    rows = []
    for strategy, runner in [
        ("pyccc_python", run_pyccc_python),
        ("pyccc_cellchat_bridge", run_pyccc_cellchat_bridge),
        ("direct_cellchat", run_direct_cellchat),
    ]:
        if strategy == "direct_cellchat" and args.skip_direct_cellchat:
            row = record_skip(strategy, scale_label, adata, "Skipped by --skip-direct-cellchat.")
            row.update(empty_cpu_peak_metadata())
            row.update(strategy_input_read_metadata(strategy, row, sample_read_seconds))
            row.update(runtime_metadata(args))
            rows.append(row)
            continue
        print(f"Running {strategy} on {adata.n_obs:,} cells", flush=True)
        monitor = ProcessTreeCpuMonitor()
        monitor.start()
        try:
            row = runner(args, out_dir / strategy, adata, lr_db, lr_path, input_dir, scale_label)
        except Exception as exc:
            row = record_failure(strategy, scale_label, adata, exc)
            (out_dir / f"{strategy}_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        row.update(monitor.stop())
        row.update(strategy_input_read_metadata(strategy, row, sample_read_seconds))
        row.update(runtime_metadata(args))
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "runtime_partial.tsv", sep="\t", index=False)
        gc.collect()
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "runtime.tsv", sep="\t", index=False)
    write_speedup_summary(frame, out_dir / "summary.md")
    write_runtime_plot(frame, out_dir / "runtime_speedup.png")
    return frame


def _matrix_var_names_allow_duplicates(var: pd.DataFrame, fallback: pd.Index, *, gene_symbols_key: str | None) -> pd.Index:
    if gene_symbols_key is None:
        return pd.Index(fallback)
    if gene_symbols_key not in var:
        raise KeyError(f"`gene_symbols_key={gene_symbols_key!r}` is not present in adata.var.")
    raw_names = var[gene_symbols_key].astype(object)
    names = pd.Index(raw_names.where(pd.notna(raw_names), "").astype(str))
    if (names == "").any():
        raise ValueError(f"`adata.var[{gene_symbols_key!r}]` contains empty gene names.")
    return names


def run_pyccc_python(args, out_dir: Path, adata, lr_db, lr_path, input_dir, scale_label: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    compute_start = time.perf_counter()
    res_a, res_b, diff = compute_pyccc_pair(args, adata, lr_db)
    compute_seconds = time.perf_counter() - compute_start
    plot_start = time.perf_counter()
    save_pyccc_plots(out_dir / "plots", res_a, diff)
    plot_seconds = time.perf_counter() - plot_start
    row = record_ok("pyccc_python", scale_label, adata, time.perf_counter() - start, compute_seconds, plot_seconds, export_seconds=0.0, r_seconds=0.0, n_interactions=len(diff.interactions))
    row.update(getattr(args, "_last_pyccc_pair_timing", {}))
    return row


def run_pyccc_cellchat_bridge(args, out_dir: Path, adata, lr_db, lr_path, input_dir, scale_label: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    compute_start = time.perf_counter()
    res_a, res_b, diff = compute_pyccc_pair(args, adata, lr_db)
    compute_seconds = time.perf_counter() - compute_start

    export_start = time.perf_counter()
    sample_a = adata[adata.obs[args.condition_key].astype(str).to_numpy() == args.condition_a]
    sample_b = adata[adata.obs[args.condition_key].astype(str).to_numpy() == args.condition_b]
    single_dir = pc.export_cellchat(res_a, out_dir / "sample_a_export", lr_table=lr_db, group_sizes=sample_a.obs[args.groupby].astype(str).value_counts())
    merged_dir = pc.export_cellchat_merged(
        diff,
        out_dir / "merged_export",
        lr_table=lr_db,
        group_sizes_a=sample_a.obs[args.groupby].astype(str).value_counts(),
        group_sizes_b=sample_b.obs[args.groupby].astype(str).value_counts(),
    )
    export_seconds = time.perf_counter() - export_start

    r_start = time.perf_counter()
    run_rscript([single_dir / "pyccc_to_cellchat.R", single_dir, out_dir / "sample_a_cellchat.rds", out_dir / "plots_single"], timeout=args.timeout_seconds)
    run_rscript([merged_dir / "pyccc_to_merged_cellchat.R", merged_dir, out_dir / "merged_cellchat.rds", out_dir / "plots_merged"], timeout=args.timeout_seconds)
    r_seconds = time.perf_counter() - r_start
    row = record_ok("pyccc_cellchat_bridge", scale_label, adata, time.perf_counter() - start, compute_seconds, plot_seconds=0.0, export_seconds=export_seconds, r_seconds=r_seconds, n_interactions=len(diff.interactions))
    row.update(getattr(args, "_last_pyccc_pair_timing", {}))
    return row


def run_direct_cellchat(args, out_dir: Path, adata, lr_db, lr_path, input_dir, scale_label: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "run_direct_cellchat.R"
    script_path.write_text(textwrap.dedent(DIRECT_CELLCHAT_R).strip() + "\n", encoding="utf-8")
    start = time.perf_counter()
    completed = run_rscript(
        [
            script_path,
            input_dir,
            lr_path,
            out_dir,
            args.condition_key,
            args.condition_a,
            args.condition_b,
            args.groupby,
            args.cellchat_aggregate,
            str(args.cofactor_adjust).upper(),
        ],
        timeout=args.timeout_seconds,
        check=False,
    )
    total_seconds = time.perf_counter() - start
    (out_dir / "direct_cellchat_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (out_dir / "direct_cellchat_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"direct CellChat failed with exit code {completed.returncode}: {(completed.stderr or '')[-1000:]}")
    runtime_path = out_dir / "direct_cellchat_runtime.tsv"
    if runtime_path.exists():
        runtime = pd.read_csv(runtime_path, sep="\t").iloc[0].to_dict()
        compute_seconds = float(runtime.get("analysis_seconds", np.nan))
        plot_seconds = float(runtime.get("plot_seconds", np.nan))
    else:
        compute_seconds = np.nan
        plot_seconds = np.nan
    return record_ok("direct_cellchat", scale_label, adata, total_seconds, compute_seconds, plot_seconds, export_seconds=0.0, r_seconds=total_seconds, n_interactions=np.nan)


def compute_pyccc_pair(args, adata, lr_db: pc.CellChatDB):
    kwargs = dict(
        condition_key=args.condition_key,
        groupby=args.groupby,
        lr_table=lr_db,
        min_pct=0.0,
        min_expr=0.0,
        aggregate=args.aggregate,
        score_method="cellchat",
        cofactor_adjust=args.cofactor_adjust,
        cofactor_kh=0.5,
        cofactor_hill=1.0,
        population_size=False,
        gene_symbols_key=args.gene_symbols_key,
        n_jobs=args.n_jobs,
    )
    start = time.perf_counter()
    res_a = pc.compute_communication(adata, condition=args.condition_a, **kwargs)
    condition_a_seconds = time.perf_counter() - start
    start = time.perf_counter()
    res_b = pc.compute_communication(adata, condition=args.condition_b, **kwargs)
    condition_b_seconds = time.perf_counter() - start
    start = time.perf_counter()
    diff = pc.compare_communication(res_a, res_b, label_a=args.condition_a, label_b=args.condition_b)
    compare_seconds = time.perf_counter() - start
    args._last_pyccc_pair_timing = {
        "condition_a_seconds": condition_a_seconds,
        "condition_b_seconds": condition_b_seconds,
        "compare_seconds": compare_seconds,
    }
    return res_a, res_b, diff


def save_pyccc_plots(out_dir: Path, result: pc.CCCResult, diff: pc.DifferentialCCC) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("network_circle", lambda ax: cp.net_circle(result, ax=ax, title="pyccc network"), (7.4, 7.0)),
        ("bubble", lambda ax: cp.bubble(result, top_n=30, compact_pairs=True, ax=ax, title="pyccc bubble"), (9.0, 6.2)),
        ("pathway_heatmap", lambda ax: cp.pathway_heatmap(result, top_pairs=25, compact_pairs=True, ax=ax, title="pyccc pathway heatmap"), (8.2, 6.2)),
        ("compare_interactions_weight", lambda ax: cp.compare_interactions(diff, ax=ax, title="pyccc interaction weights"), (6.5, 5.2)),
        ("rank_pathway_comparison", lambda ax: cp.rank_signaling_compare(diff, top_n=20, ax=ax, title="pyccc pathway rank comparison"), (8.0, 5.6)),
    ]
    for name, draw, size in plot_specs:
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
        draw(ax)
        fig.savefig(out_dir / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def write_r_input(adata, out_dir: Path, *, groupby: str, condition_key: str, gene_symbols_key: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    genes = adata.var[gene_symbols_key].astype(str).to_numpy() if gene_symbols_key and gene_symbols_key in adata.var else adata.var_names.astype(str).to_numpy()
    x = adata.X
    x_t = x.T if sparse.issparse(x) else sparse.csr_matrix(np.asarray(x).T)
    mmwrite(out_dir / "matrix.mtx", x_t)
    pd.Series(genes).to_csv(out_dir / "genes.tsv", index=False, header=False)
    pd.Series(adata.obs_names.astype(str)).to_csv(out_dir / "cells.tsv", index=False, header=False)
    meta = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            condition_key: adata.obs[condition_key].astype(str).to_numpy(),
            groupby: adata.obs[groupby].astype(str).to_numpy(),
        }
    )
    meta.to_csv(out_dir / "meta.tsv", sep="\t", index=False)


def has_r_input(path: Path) -> bool:
    return all((path / name).exists() for name in ["matrix.mtx", "genes.tsv", "cells.tsv", "meta.tsv"])


def write_input_runtime(path: Path, adata, sample_read_seconds: float, input_dir: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "stage": "read_prepared_h5ad",
                "seconds": float(sample_read_seconds),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "path": str(path.parent),
                "r_input_dir": str(input_dir),
            }
        ]
    )
    frame.to_csv(path, sep="\t", index=False)


def strategy_input_read_metadata(strategy: str, row: dict[str, object], sample_read_seconds: float) -> dict[str, object]:
    input_read_seconds = float(sample_read_seconds) if strategy in {"pyccc_python", "pyccc_cellchat_bridge"} else 0.0
    total_seconds = float(row["total_seconds"]) if row.get("status") == "ok" and pd.notna(row.get("total_seconds")) else np.nan
    total_with_input = total_seconds + input_read_seconds if np.isfinite(total_seconds) else np.nan
    return {
        "input_read_seconds": input_read_seconds,
        "total_with_input_read_seconds": total_with_input,
    }


def scale_adata(base, target_cells: int, *, condition_key: str, random_state: int, allow_repeated_cells: bool):
    rng = np.random.default_rng(random_state)
    if target_cells <= base.n_obs:
        idx = rng.choice(np.arange(base.n_obs), size=target_cells, replace=False)
        return base[np.sort(idx), :].copy()
    if not allow_repeated_cells:
        raise ValueError(
            f"Requested {target_cells:,} cells but only {base.n_obs:,} real filtered cells are available. "
            "Increase --n-groups, lower --target-cells, or pass --allow-repeated-cells."
        )
    full_repeats = target_cells // base.n_obs
    remainder = target_cells % base.n_obs
    x_parts = [base.X] * full_repeats
    obs_parts = []
    for i in range(full_repeats):
        obs = base.obs.copy()
        obs.index = [f"{idx}__rep{i}" for idx in base.obs_names]
        obs_parts.append(obs)
    if remainder:
        idx = rng.choice(np.arange(base.n_obs), size=remainder, replace=False)
        x_parts.append(base.X[idx])
        obs = base.obs.iloc[idx].copy()
        obs.index = [f"{idx}__rep{full_repeats}" for idx in obs.index]
        obs_parts.append(obs)
    x = sparse.vstack(x_parts, format="csr") if sparse.issparse(base.X) else np.vstack(x_parts)
    obs = pd.concat(obs_parts, axis=0)
    return ad.AnnData(x, obs=obs, var=base.var.copy())


def run_rscript(args: list[object], *, timeout: int, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    local_r_lib = Path(".r-lib").resolve()
    if "R_LIBS_USER" not in env and local_r_lib.exists():
        env["R_LIBS_USER"] = str(local_r_lib)
    completed = subprocess.run(
        ["Rscript", *[str(arg) for arg in args]],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"Rscript failed with exit code {completed.returncode}: {(completed.stderr or '')[-1000:]}")
    return completed


def record_ok(strategy, scale_label, adata, total_seconds, compute_seconds, plot_seconds, export_seconds, r_seconds, n_interactions) -> dict[str, object]:
    return {
        "strategy": strategy,
        "scale_label": scale_label,
        "status": "ok",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "total_seconds": float(total_seconds),
        "compute_seconds": float(compute_seconds) if pd.notna(compute_seconds) else np.nan,
        "plot_seconds": float(plot_seconds) if pd.notna(plot_seconds) else np.nan,
        "export_seconds": float(export_seconds) if pd.notna(export_seconds) else np.nan,
        "r_seconds": float(r_seconds) if pd.notna(r_seconds) else np.nan,
        "n_interactions": n_interactions,
        "maxrss_mb": maxrss_mb(),
        "error": "",
    }


def record_failure(strategy, scale_label, adata, exc: Exception) -> dict[str, object]:
    return {
        "strategy": strategy,
        "scale_label": scale_label,
        "status": "timeout" if isinstance(exc, subprocess.TimeoutExpired) else "failed",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "total_seconds": np.nan,
        "compute_seconds": np.nan,
        "plot_seconds": np.nan,
        "export_seconds": np.nan,
        "r_seconds": np.nan,
        "n_interactions": np.nan,
        "maxrss_mb": maxrss_mb(),
        "error": f"{type(exc).__name__}: {exc}",
    }


def record_skip(strategy, scale_label, adata, reason: str) -> dict[str, object]:
    row = record_failure(strategy, scale_label, adata, RuntimeError(reason))
    row["status"] = "skipped"
    row["error"] = reason
    return row


class ProcessTreeCpuMonitor:
    """Sample peak CPU usage for the current process and subprocess tree."""

    def __init__(self, pid: int | None = None, *, interval: float = 0.2) -> None:
        self.pid = int(pid or os.getpid())
        self.interval = float(interval)
        self.min_sample_seconds = max(0.05, self.interval * 0.5)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_time = 0.0
        self._last_cpu_seconds = 0.0
        self.peak_cpu_percent = 0.0
        self.sample_count = 0

    def start(self) -> None:
        self._last_cpu_seconds = process_tree_cpu_seconds(self.pid)
        self._last_time = time.perf_counter()
        self.peak_cpu_percent = 0.0
        self.sample_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2.0))
        self._sample()
        return {
            "peak_cpu_percent": float(self.peak_cpu_percent),
            "peak_cpu_cores": float(self.peak_cpu_percent / 100.0),
            "cpu_sample_count": int(self.sample_count),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        with self._lock:
            cpu_seconds = process_tree_cpu_seconds(self.pid)
            now = time.perf_counter()
            elapsed = now - self._last_time
            if elapsed < self.min_sample_seconds:
                return
            cpu_delta = cpu_seconds - self._last_cpu_seconds
            if cpu_delta >= 0:
                self.peak_cpu_percent = max(self.peak_cpu_percent, 100.0 * cpu_delta / elapsed)
                self.sample_count += 1
            self._last_time = now
            self._last_cpu_seconds = cpu_seconds


def empty_cpu_peak_metadata() -> dict[str, object]:
    return {"peak_cpu_percent": np.nan, "peak_cpu_cores": np.nan, "cpu_sample_count": 0}


def process_tree_cpu_seconds(root_pid: int) -> float:
    proc = Path("/proc")
    if not proc.exists():
        return time.process_time()

    children: dict[int, list[int]] = {}
    cpu_seconds: dict[int, float] = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        tgid = linux_proc_tgid(entry / "status")
        if tgid is not None and tgid != pid:
            continue
        parsed = parse_linux_proc_stat(entry / "stat")
        if parsed is None:
            continue
        ppid, seconds = parsed
        children.setdefault(ppid, []).append(pid)
        cpu_seconds[pid] = seconds

    total = 0.0
    stack = [int(root_pid)]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += cpu_seconds.get(pid, 0.0)
        stack.extend(children.get(pid, []))
    return total


def linux_proc_tgid(path: Path) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Tgid:"):
                return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


def parse_linux_proc_stat(path: Path) -> tuple[int, float] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    if len(fields) < 13:
        return None
    try:
        ppid = int(fields[1])
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        ticks_per_second = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        return None
    return ppid, float(user_ticks + system_ticks) / float(ticks_per_second)


def runtime_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "cpu_count": os.cpu_count() or "",
        "cpu_affinity_count": cpu_affinity_count(),
        "cpu_affinity": cpu_affinity_summary(),
        "n_jobs": getattr(args, "n_jobs", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        "blas_threads": _blas_thread_summary(),
    }


def cpu_affinity_count() -> int | str:
    if not hasattr(os, "sched_getaffinity"):
        return ""
    try:
        return len(os.sched_getaffinity(0))
    except OSError:
        return ""


def cpu_affinity_summary() -> str:
    if not hasattr(os, "sched_getaffinity"):
        return ""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except OSError:
        return ""
    if not cpus:
        return ""
    ranges: list[str] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(format_cpu_range(start, prev))
        start = prev = cpu
    ranges.append(format_cpu_range(start, prev))
    return ",".join(ranges)


def format_cpu_range(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _blas_thread_summary() -> str:
    try:
        from threadpoolctl import threadpool_info
    except Exception:
        return ""
    parts = []
    for item in threadpool_info():
        api = item.get("internal_api") or item.get("user_api") or "unknown"
        threads = item.get("num_threads", "")
        prefix = item.get("prefix", "")
        parts.append(f"{api}:{threads}:{prefix}")
    return ";".join(parts)


def maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def write_speedup_summary(frame: pd.DataFrame, path: Path) -> None:
    ok = frame[frame["status"].eq("ok")].copy()
    seconds_col = runtime_seconds_column(frame)
    lines = ["# Three-way CCC runtime benchmark", ""]
    if not ok.empty:
        lines.extend(["| Strategy | Cells | Total seconds | Speedup vs direct CellChat | Status |", "| --- | ---: | ---: | ---: | --- |"])
        direct = ok.loc[ok["strategy"].eq("direct_cellchat"), seconds_col]
        direct_seconds = float(direct.iloc[0]) if not direct.empty else np.nan
        for row in frame.itertuples(index=False):
            seconds = float(getattr(row, seconds_col))
            speedup = direct_seconds / seconds if row.status == "ok" and np.isfinite(direct_seconds) and seconds > 0 else np.nan
            speedup_text = f"{speedup:.2f}x" if np.isfinite(speedup) else "NA"
            seconds_text = f"{seconds:.3f}" if row.status == "ok" else "NA"
            lines.append(f"| {row.strategy} | {int(row.n_cells):,} | {seconds_text} | {speedup_text} | {row.status} |")
    lines.extend(["", "## Raw Rows", ""])
    lines.extend(["```text", frame.to_string(index=False), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runtime_plot(frame: pd.DataFrame, path: Path) -> None:
    ok = frame[frame["status"].eq("ok")].copy()
    if ok.empty:
        return
    seconds_col = runtime_seconds_column(ok)
    order = ["pyccc_python", "pyccc_cellchat_bridge", "direct_cellchat"]
    if "scale_label" in ok:
        labels = list(dict.fromkeys(ok["scale_label"].astype(str)))
        for label in labels:
            candidate = ok[ok["scale_label"].astype(str).eq(label)].copy()
            if set(order).issubset(set(candidate["strategy"].astype(str))):
                ok = candidate
                break
        else:
            ok = ok[ok["scale_label"].astype(str).eq(labels[0])].copy()
    ok["strategy"] = pd.Categorical(ok["strategy"], categories=order, ordered=True)
    ok = ok.sort_values("strategy")
    if ok.empty:
        return
    display = {
        "pyccc_python": "pyccc native",
        "pyccc_cellchat_bridge": "pyccc + CellChat R plots",
        "direct_cellchat": "direct CellChat R",
    }
    colors = {
        "pyccc_python": "#2C7FB8",
        "pyccc_cellchat_bridge": "#41AB5D",
        "direct_cellchat": "#D95F0E",
    }
    direct = ok.loc[ok["strategy"].astype(str).eq("direct_cellchat"), seconds_col]
    direct_seconds = float(direct.iloc[0]) if not direct.empty else np.nan

    fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
    y = np.arange(len(ok))
    ax.barh(y, ok[seconds_col].astype(float), color=[colors[str(s)] for s in ok["strategy"]], height=0.62)
    ax.set_yticks(y, [display.get(str(s), str(s)) for s in ok["strategy"]])
    ax.invert_yaxis()
    ax.set_xlabel("Total time from prepared input read through matched visualizations (seconds)")
    ax.set_title("Real 1M-cell CCC benchmark")
    ax.grid(axis="x", color="#D0D7DE", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    max_seconds = float(ok[seconds_col].max())
    for i, row in enumerate(ok.itertuples(index=False)):
        seconds = float(getattr(row, seconds_col))
        if np.isfinite(direct_seconds) and seconds > 0:
            speedup = direct_seconds / seconds
            label = f"{seconds:.1f}s, {speedup:.2f}x"
        else:
            label = f"{seconds:.1f}s"
        ax.text(seconds + max_seconds * 0.025, i, label, va="center", ha="left", fontsize=9)
    cells = int(ok["n_cells"].iloc[0])
    genes = int(ok["n_genes"].iloc[0])
    caption = (
        f"Human Immune Health Atlas; {cells:,} real cells, {genes} LR/cofactor genes; "
        "CMV infection vs normal, top 5 shared cell types"
    )
    ax.text(
        0.0,
        -0.26,
        textwrap.fill(caption, width=88),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max_seconds * 1.34)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def runtime_seconds_column(frame: pd.DataFrame) -> str:
    if "total_with_input_read_seconds" in frame.columns and frame["total_with_input_read_seconds"].notna().any():
        return "total_with_input_read_seconds"
    return "total_seconds"


if __name__ == "__main__":
    main()

suppressPackageStartupMessages(library(CellChat))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript pyccc_to_cellchat.R <pyccc_export_dir> <out.rds> [plot_dir]", call. = FALSE)
}

export_dir <- normalizePath(args[[1]], mustWork = TRUE)
out_rds <- args[[2]]
plot_dir <- if (length(args) >= 3 && nzchar(args[[3]]) && args[[3]] != "NA") args[[3]] else NA_character_

read_bridge <- function(name) {
  read.delim(file.path(export_dir, name), check.names = FALSE, stringsAsFactors = FALSE)
}

groups <- read_bridge("pyccc_cellchat_groups.tsv")
lr <- read_bridge("pyccc_cellchat_lr.tsv")
interactions <- read_bridge("pyccc_cellchat_interactions.tsv")

if (!all(c("group", "n_cells") %in% colnames(groups))) {
  stop("pyccc_cellchat_groups.tsv must contain group and n_cells columns")
}
if (!all(c("interaction_name", "interaction_name_2", "pathway_name", "ligand", "receptor") %in% colnames(lr))) {
  stop("pyccc_cellchat_lr.tsv is missing CellChat LR columns")
}
if (!all(c("source", "target", "interaction_name", "prob", "pval") %in% colnames(interactions))) {
  stop("pyccc_cellchat_interactions.tsv is missing interaction columns")
}

group_names <- as.character(groups$group)
lr$interaction_name <- as.character(lr$interaction_name)
rownames(lr) <- lr$interaction_name
lr <- lr[!duplicated(rownames(lr)), , drop = FALSE]
lr_names <- rownames(lr)

prob <- array(0, dim = c(length(group_names), length(group_names), length(lr_names)), dimnames = list(group_names, group_names, lr_names))
pval <- array(1, dim = dim(prob), dimnames = dimnames(prob))

interactions$source <- as.character(interactions$source)
interactions$target <- as.character(interactions$target)
interactions$interaction_name <- as.character(interactions$interaction_name)
interactions$prob <- as.numeric(interactions$prob)
interactions$pval <- as.numeric(interactions$pval)
interactions$pval[is.na(interactions$pval)] <- 0
interactions$prob[is.na(interactions$prob)] <- 0

interactions <- interactions[interactions$source %in% group_names & interactions$target %in% group_names & interactions$interaction_name %in% lr_names, , drop = FALSE]
if (nrow(interactions) > 0) {
  collapsed_prob <- aggregate(prob ~ source + target + interaction_name, interactions, sum)
  collapsed_pval <- aggregate(pval ~ source + target + interaction_name, interactions, min)
  for (i in seq_len(nrow(collapsed_prob))) {
    prob[collapsed_prob$source[i], collapsed_prob$target[i], collapsed_prob$interaction_name[i]] <- collapsed_prob$prob[i]
  }
  for (i in seq_len(nrow(collapsed_pval))) {
    pval[collapsed_pval$source[i], collapsed_pval$target[i], collapsed_pval$interaction_name[i]] <- collapsed_pval$pval[i]
  }
}
pval[prob == 0] <- 1

make_pathway_net <- function(prob, pval, lr) {
  pathways <- unique(as.character(lr$pathway_name))
  pathway_prob <- array(0, dim = c(dim(prob)[1], dim(prob)[2], length(pathways)), dimnames = list(dimnames(prob)[[1]], dimnames(prob)[[2]], pathways))
  pathway_pval <- array(1, dim = dim(pathway_prob), dimnames = dimnames(pathway_prob))
  for (pathway in pathways) {
    lr_use <- rownames(lr)[as.character(lr$pathway_name) == pathway]
    if (!length(lr_use)) next
    pathway_prob[, , pathway] <- apply(prob[, , lr_use, drop = FALSE], c(1, 2), sum)
    pathway_pval[, , pathway] <- apply(pval[, , lr_use, drop = FALSE], c(1, 2), min)
  }
  keep <- apply(pathway_prob, 3, sum) > 0
  if (!any(keep)) {
    keep[] <- TRUE
  }
  pathway_prob <- pathway_prob[, , keep, drop = FALSE]
  pathway_pval <- pathway_pval[, , keep, drop = FALSE]
  ord <- order(apply(pathway_prob, 3, sum), decreasing = TRUE)
  list(pathways = dimnames(pathway_prob)[[3]][ord], prob = pathway_prob[, , ord, drop = FALSE], pval = pathway_pval[, , ord, drop = FALSE])
}

cellchat <- methods::new("CellChat")
n_cells <- pmax(as.integer(groups$n_cells), 1)
idents <- factor(rep(group_names, n_cells), levels = group_names)
meta <- data.frame(labels = idents, group = idents, row.names = paste0("pyccc_cell_", seq_along(idents)))
data.empty <- Matrix::Matrix(0, nrow = 0, ncol = length(idents), sparse = TRUE)
colnames(data.empty) <- rownames(meta)

cellchat@data.raw <- data.empty
cellchat@data <- data.empty
cellchat@data.signaling <- data.empty
cellchat@data.scale <- matrix(0, nrow = 0, ncol = length(idents))
colnames(cellchat@data.scale) <- rownames(meta)
cellchat@data.smooth <- data.empty
cellchat@images <- list()
cellchat@meta <- meta
cellchat@idents <- idents
cellchat@DB <- list(interaction = lr)
cellchat@LR <- list(LRsig = lr)
cellchat@var.features <- list()
cellchat@dr <- list()
cellchat@options <- list(mode = "single", datatype = "RNA")
cellchat@net <- list(prob = prob, pval = pval)
cellchat@net$count <- apply(prob > 0, c(1, 2), sum)
cellchat@net$weight <- apply(prob, c(1, 2), sum)
cellchat@net$LRs <- lr_names[apply(prob, 3, sum) > 0]
cellchat@netP <- make_pathway_net(prob, pval, lr)

cellchat <- tryCatch(netAnalysis_computeCentrality(cellchat, slot.name = "netP", thresh = 1), error = function(e) cellchat)
saveRDS(cellchat, out_rds)

draw_returned <- function(x) {
  if (inherits(x, "ggplot")) {
    print(x)
  } else if (inherits(x, "Heatmap") || inherits(x, "HeatmapList")) {
    ComplexHeatmap::draw(x)
  } else if (!is.null(x)) {
    print(x)
  }
}

save_plot <- function(id, code, width = 1400, height = 1000, res = 170) {
  if (is.na(plot_dir)) return(invisible(NULL))
  dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(plot_dir, paste0(id, ".png"))
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

if (!is.na(plot_dir)) {
  first_pathway <- cellchat@netP$pathways[[1]]
  manifest <- rbind(
    save_plot("network_circle", netVisual_circle(cellchat@net$weight, vertex.weight = as.numeric(table(cellchat@idents)), weight.scale = TRUE, label.edge = FALSE, title.name = "pyccc exported network"), 1300, 1200),
    save_plot("bubble", netVisual_bubble(cellchat, signaling = first_pathway, remove.isolate = FALSE, thresh = 1, title.name = paste0("pyccc exported ", first_pathway, " bubble")), 1600, 1100),
    save_plot("pathway_heatmap", netVisual_heatmap(cellchat, measure = "weight", slot.name = "netP", color.heatmap = "Reds", cluster.rows = FALSE, cluster.cols = FALSE, title.name = "pyccc exported pathway heatmap"), 1300, 1100),
    save_plot("pathway_rank", rankNet(cellchat, mode = "single", measure = "weight", stacked = FALSE, thresh = 1, title = "pyccc exported pathway rank"), 1400, 1000),
    save_plot("pathway_circle", netVisual_aggregate(cellchat, signaling = first_pathway, layout = "circle", thresh = 1, remove.isolate = FALSE), 1300, 1200)
  )
  write.table(manifest, file.path(plot_dir, "pyccc_cellchat_plot_manifest.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

message("Saved CellChat RDS: ", normalizePath(out_rds, mustWork = FALSE))

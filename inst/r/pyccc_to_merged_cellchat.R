suppressPackageStartupMessages(library(CellChat))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript pyccc_to_merged_cellchat.R <pyccc_merged_export_dir> <out.rds> [plot_dir]", call. = FALSE)
}

export_dir <- normalizePath(args[[1]], mustWork = TRUE)
out_rds <- args[[2]]
plot_dir <- if (length(args) >= 3 && nzchar(args[[3]]) && args[[3]] != "NA") args[[3]] else NA_character_

samples_path <- file.path(export_dir, "pyccc_cellchat_samples.tsv")
if (!file.exists(samples_path)) {
  stop("Missing pyccc_cellchat_samples.tsv in merged export directory", call. = FALSE)
}

samples <- read.delim(samples_path, check.names = FALSE, stringsAsFactors = FALSE)
if (!all(c("role", "label", "directory") %in% colnames(samples))) {
  stop("pyccc_cellchat_samples.tsv must contain role, label, and directory columns", call. = FALSE)
}
samples <- samples[match(c("a", "b"), samples$role), , drop = FALSE]
samples <- samples[complete.cases(samples[, c("role", "label", "directory")]), , drop = FALSE]
if (nrow(samples) < 2) {
  stop("Merged export must contain two sample rows with roles a and b", call. = FALSE)
}

read_bridge <- function(sample_dir, name) {
  read.delim(file.path(sample_dir, name), check.names = FALSE, stringsAsFactors = FALSE)
}

make_pathway_net <- function(prob, pval, lr) {
  pathways <- unique(as.character(lr$pathway_name))
  if (!length(pathways)) {
    pathways <- "pyccc_empty"
  }
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

create_cellchat_from_export <- function(sample_dir, dataset_label) {
  groups <- read_bridge(sample_dir, "pyccc_cellchat_groups.tsv")
  lr <- read_bridge(sample_dir, "pyccc_cellchat_lr.tsv")
  interactions <- read_bridge(sample_dir, "pyccc_cellchat_interactions.tsv")

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

  cellchat <- methods::new("CellChat")
  n_cells <- pmax(as.integer(groups$n_cells), 1)
  idents <- factor(rep(group_names, n_cells), levels = group_names)
  label_prefix <- gsub("[^A-Za-z0-9_]+", "_", dataset_label)
  label_prefix <- gsub("^_+|_+$", "", label_prefix)
  if (!nzchar(label_prefix)) label_prefix <- "sample"
  cell_ids <- paste0("pyccc_", label_prefix, "_cell_", seq_along(idents))
  meta <- data.frame(labels = idents, group = idents, row.names = cell_ids)
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

  tryCatch(netAnalysis_computeCentrality(cellchat, slot.name = "netP", thresh = 1), error = function(e) cellchat)
}

labels <- make.unique(ifelse(nzchar(samples$label), samples$label, samples$role), sep = "_")
sample_dirs <- file.path(export_dir, samples$directory)
objects <- lapply(seq_len(nrow(samples)), function(i) create_cellchat_from_export(normalizePath(sample_dirs[[i]], mustWork = TRUE), labels[[i]]))
names(objects) <- labels

merged <- mergeCellChat(objects, add.names = labels, cell.prefix = FALSE)
saveRDS(merged, out_rds)

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

skip_plot <- function(id, message) {
  data.frame(id = id, path = "", status = "skipped", message = message, stringsAsFactors = FALSE)
}

has_network_delta <- function(object, comparison, measure) {
  net_a <- object@net[[comparison[[1]]]][[measure]]
  net_b <- object@net[[comparison[[2]]]][[measure]]
  any(abs(net_a - net_b) > 0)
}

save_diff_circle <- function(id, measure, comparison, width = 1300, height = 1200) {
  if (!has_network_delta(merged, comparison, measure)) {
    return(skip_plot(id, paste0("No nonzero ", measure, " network delta to draw.")))
  }
  save_plot(
    id,
    netVisual_diffInteraction(
      merged,
      comparison = comparison,
      measure = measure,
      weight.scale = TRUE,
      label.edge = FALSE,
      title.name = paste(labels[[1]], "vs", labels[[2]], measure)
    ),
    width,
    height
  )
}

if (!is.na(plot_dir)) {
  comparison <- c(1, 2)
  manifest <- do.call(
    rbind,
    list(
      save_plot("compare_interactions_count", compareInteractions(merged, group = comparison, measure = "count", title.name = "Number of interactions"), 1200, 950),
      save_plot("compare_interactions_weight", compareInteractions(merged, group = comparison, measure = "weight", title.name = "Interaction weights"), 1200, 950),
      save_diff_circle("diff_interaction_weight_circle", "weight", comparison, 1300, 1200),
      save_diff_circle("diff_interaction_count_circle", "count", comparison, 1300, 1200),
      save_plot("rank_pathway_comparison", rankNet(merged, mode = "comparison", comparison = comparison, stacked = FALSE, do.stat = FALSE, thresh = 1, title = paste(labels[[1]], "vs", labels[[2]], "pathway rank")), 1500, 1050),
      save_plot("rank_pathway_stacked", rankNet(merged, mode = "comparison", comparison = comparison, stacked = TRUE, do.stat = FALSE, thresh = 1, title = paste(labels[[1]], "vs", labels[[2]], "stacked pathway rank")), 1500, 1050)
    )
  )
  write.table(manifest, file.path(plot_dir, "pyccc_merged_cellchat_plot_manifest.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
}

message("Saved merged CellChat RDS: ", normalizePath(out_rds, mustWork = FALSE))

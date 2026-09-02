#!/usr/bin/env Rscript

# ============================================================
# Ovarian cancer metabolomics - pairwise differential analysis
# Module: 03_global_separation_and_pairwise_diff / 03B_pairwise_diff
#
# MERGED-only simplified figure version
#   - Use merged_deDuplicated_matrix.csv only
#   - Save differential result tables for all six pairwise comparisons
#   - Output ONLY 3 figures:
#       1) one grouped multi-comparison volcano plot
#       2) one selected endogenous-like feature heatmap with clinical annotations
#       3) one N_vs_M significant-feature heatmap with clinical annotations
#   - No metabolite names displayed on volcano plot
#   - Grouped volcano style follows the cluster/group-volcano logic:
#       x = comparison group, y = log2FC, point color = adjusted P significance,
#       with grey background bars and colored bottom tiles
# ============================================================

options(stringsAsFactors = FALSE)

# -----------------------------
# User settings
# -----------------------------
install_missing_packages <- FALSE
alpha_cutoff <- 0.05
logfc_cutoff <- log2(1.2)
multivolcano_p_cutoff <- 0.05
multivolcano_top_n <- 12
multivolcano_jitter_width <- 0.22
multivolcano_bar_quantile <- 0.975
multivolcano_tile_height <- 0.28
multivolcano_seed <- 123

allowed_groups <- c("N", "B", "BD", "M")
comparison_list <- list(
  c("N", "B"),
  c("N", "BD"),
  c("N", "M"),
  c("B", "BD"),
  c("B", "M"),
  c("BD", "M")
)
group_levels <- c("N", "B", "BD", "M")
style_candidates <- c(
  Sys.getenv("OV_PUBLICATION_STYLE", unset = NA_character_),
  file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]), winslash = "/", mustWork = FALSE)), "..", "style", "ov_publication_style.R"),
  file.path(getwd(), "scripts", "style", "ov_publication_style.R"),
  file.path(dirname(getwd()), "scripts", "style", "ov_publication_style.R")
)
style_candidates <- style_candidates[!is.na(style_candidates) & nzchar(style_candidates)]
style_file <- style_candidates[file.exists(style_candidates)][1]
if (!is.na(style_file) && file.exists(style_file)) {
  source(style_file)
} else {
  stop("Missing ov_publication_style.R. Use the bundled scripts/style file or set OV_PUBLICATION_STYLE.")
}
group_palette <- OV_GROUP_COLORS[group_levels]
comparison_strip_palette <- c(
  "N_vs_B"  = "#8DD3C7",
  "N_vs_BD" = "#FDB462",
  "N_vs_M"  = "#FB8072",
  "B_vs_BD" = "#BEBADA",
  "B_vs_M"  = "#FCCDE5",
  "BD_vs_M" = "#B3B3B3"
)
heatmap_detection_rate <- 0.70
heatmap_clip <- 3
full_heatmap_max_features <- Inf
nm_sig_heatmap_max_features <- 200
selected_clinical_heatmap_n <- 30

# -----------------------------
# Package setup
# -----------------------------
cran_pkgs <- c(
  "ggplot2", "dplyr", "tidyr", "readr", "stringr", "purrr",
  "tibble", "openxlsx", "RColorBrewer"
)
bioc_pkgs <- c("limma", "ComplexHeatmap", "circlize")

install_if_needed <- function(pkgs, bioc = FALSE) {
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      if (!install_missing_packages) {
        stop(sprintf("Package '%s' is required but not installed.", pkg))
      }
      if (bioc) {
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
          install.packages("BiocManager", repos = "https://cloud.r-project.org")
        }
        BiocManager::install(pkg, ask = FALSE, update = FALSE)
      } else {
        install.packages(pkg, repos = "https://cloud.r-project.org")
      }
    }
  }
}

install_if_needed(cran_pkgs, bioc = FALSE)
install_if_needed(bioc_pkgs, bioc = TRUE)

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(purrr)
  library(tibble)
  library(openxlsx)
  library(RColorBrewer)
  library(limma)
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
})

# -----------------------------
# Path helpers
# -----------------------------
get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  }
  normalizePath(getwd())
}

find_module_root <- function() {
  candidates <- unique(c(
    normalizePath(get_script_dir(), mustWork = FALSE),
    normalizePath(file.path(get_script_dir(), ".."), mustWork = FALSE),
    normalizePath(getwd(), mustWork = FALSE),
    normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
  ))
  for (cand in candidates) {
    if (dir.exists(file.path(cand, "data")) && dir.exists(file.path(cand, "figure_final"))) {
      return(cand)
    }
  }
  stop("Cannot locate module root containing both 'data' and 'figure_final'.")
}

module_root <- find_module_root()
data_dir <- file.path(module_root, "data")
fig_dir <- file.path(module_root, "figure_final")
res_dir <- file.path(data_dir, "results_tables")
coord_dir <- file.path(data_dir, "source_data")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(coord_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# General helpers
# -----------------------------
save_plot_dual <- function(plot_obj, file_stub, width = 7, height = 6, dpi = 320) {
  ggsave(paste0(file_stub, ".pdf"), plot_obj, width = width, height = height, units = "in")
  ggsave(paste0(file_stub, ".png"), plot_obj, width = width, height = height, units = "in", dpi = dpi)
}

safe_sheet_name <- function(x) {
  x <- gsub("[^A-Za-z0-9_]+", "_", x)
  substr(x, 1, 31)
}

find_label_col <- function(df) {
  candidates <- c("label", "Label", "group", "Group", "class", "Class")
  hit <- candidates[candidates %in% colnames(df)]
  if (length(hit) == 0) stop("Cannot find label/group column in matrix file.")
  hit[1]
}

find_id_col <- function(df) {
  candidates <- c("Alignment ID", "Alignment_ID", "SampleID", "sample_id", "ID", "id")
  hit <- candidates[candidates %in% colnames(df)]
  if (length(hit) == 0) return(NULL)
  hit[1]
}

pretty_comp_name <- function(x) {
  gsub("_vs_", " vs ", x)
}

read_matrix_file <- function(path) {
  message("Reading: ", basename(path))
  df <- suppressMessages(readr::read_csv(path, show_col_types = FALSE))
  label_col <- find_label_col(df)
  id_col <- find_id_col(df)

  feature_cols <- setdiff(colnames(df), c(label_col, id_col))
  out <- df %>% dplyr::rename(label = all_of(label_col))
  if (!is.null(id_col)) {
    out <- out %>% dplyr::rename(sample_id = all_of(id_col))
  } else {
    out$sample_id <- seq_len(nrow(out))
  }

  out <- out %>%
    dplyr::mutate(label = as.character(label), sample_id = as.character(sample_id))

  for (fc in feature_cols) {
    out[[fc]] <- suppressWarnings(as.numeric(out[[fc]]))
  }
  out
}

preprocess_matrix <- function(df, keep_groups = allowed_groups, detection_rate = heatmap_detection_rate) {
  feat_cols <- setdiff(colnames(df), c("sample_id", "label"))

  out <- df %>%
    filter(!is.na(label), label %in% keep_groups) %>%
    mutate(label = factor(label, levels = group_levels))

  mat <- as.matrix(out[, feat_cols, drop = FALSE])
  mode(mat) <- "numeric"

  keep_non_na <- colSums(!is.na(mat)) > 0
  mat <- mat[, keep_non_na, drop = FALSE]

  det_rate <- apply(mat, 2, function(x) mean(x > 0, na.rm = TRUE))
  keep_det <- det_rate >= detection_rate
  if (sum(keep_det) == 0) stop("No features passed the detection-rate filter.")
  mat <- mat[, keep_det, drop = FALSE]

  mat <- log2(mat + 1)

  sample_medians <- apply(mat, 1, median, na.rm = TRUE)
  overall_median <- median(sample_medians, na.rm = TRUE)
  mat <- sweep(mat, 1, sample_medians, FUN = "-") + overall_median

  for (j in seq_len(ncol(mat))) {
    if (anyNA(mat[, j])) {
      med_val <- median(mat[, j], na.rm = TRUE)
      if (!is.finite(med_val)) med_val <- 0
      mat[is.na(mat[, j]), j] <- med_val
    }
  }

  keep_var <- apply(mat, 2, stats::sd, na.rm = TRUE) > 0
  mat <- mat[, keep_var, drop = FALSE]

  bind_cols(
    out %>% select(sample_id, label),
    as.data.frame(mat, check.names = FALSE)
  )
}

run_pairwise_limma <- function(df_proc, g1, g2, alpha = alpha_cutoff, lfc = logfc_cutoff) {
  comp_name <- paste0(g1, "_vs_", g2)
  sub_df <- df_proc %>% filter(label %in% c(g1, g2))
  sub_df$label <- factor(sub_df$label, levels = c(g1, g2))

  feat_cols <- setdiff(colnames(sub_df), c("sample_id", "label"))
  expr <- t(as.matrix(sub_df[, feat_cols, drop = FALSE]))
  design <- model.matrix(~ 0 + label, data = sub_df)
  colnames(design) <- levels(sub_df$label)

  fit <- lmFit(expr, design)
  contrast_mat <- makeContrasts(contrasts = paste0(g2, "-", g1), levels = design)
  fit2 <- contrasts.fit(fit, contrast_mat)
  fit2 <- eBayes(fit2)
  tt <- topTable(fit2, number = Inf, adjust.method = "BH", sort.by = "P") %>%
    rownames_to_column("feature_id") %>%
    as_tibble()

  mean_g1 <- colMeans(sub_df[sub_df$label == g1, feat_cols, drop = FALSE], na.rm = TRUE)
  mean_g2 <- colMeans(sub_df[sub_df$label == g2, feat_cols, drop = FALSE], na.rm = TRUE)

  stat_df <- tibble(
    feature_id = feat_cols,
    mean_group1 = unname(mean_g1),
    mean_group2 = unname(mean_g2)
  )

  tt %>%
    left_join(stat_df, by = "feature_id") %>%
    mutate(
      comparison = comp_name,
      comparison_pretty = pretty_comp_name(comp_name),
      group1 = g1,
      group2 = g2,
      significant = adj.P.Val < alpha & abs(logFC) >= lfc,
      neglog10_adjP = -log10(adj.P.Val + 1e-300)
    ) %>%
    arrange(adj.P.Val, desc(abs(logFC)))
}

# -----------------------------
# Grouped multi-volcano helper
# -----------------------------
make_grouped_multivolcano_plot <- function(all_results) {
  volcano_df <- bind_rows(all_results) %>%
    mutate(
      comparison = factor(comparison, levels = names(all_results)),
      comparison_pretty = factor(
        pretty_comp_name(as.character(comparison)),
        levels = pretty_comp_name(names(all_results))
      ),
      label_sig = ifelse(
        adj.P.Val < multivolcano_p_cutoff,
        paste0("adjust P-val<", multivolcano_p_cutoff),
        paste0("adjust P-val>=", multivolcano_p_cutoff)
      ),
      sig_class = dplyr::case_when(
        adj.P.Val >= multivolcano_p_cutoff ~ "FDR >= 0.05",
        logFC >= 0 ~ "FDR < 0.05, logFC > 0",
        TRUE ~ "FDR < 0.05, logFC < 0"
      )
    )
  
  top_df <- volcano_df %>%
    group_by(comparison, comparison_pretty) %>%
    arrange(desc(abs(logFC)), .by_group = TRUE) %>%
    slice_head(n = multivolcano_top_n) %>%
    ungroup()
  
  base_df <- volcano_df %>%
    mutate(is_top = feature_id %in% top_df$feature_id)
  
  dt <- volcano_df %>% filter(adj.P.Val >= multivolcano_p_cutoff)
  sig_pos <- volcano_df %>% filter(adj.P.Val < multivolcano_p_cutoff, logFC >= 0)
  sig_neg <- volcano_df %>% filter(adj.P.Val < multivolcano_p_cutoff, logFC < 0)
  
  # 上下灰色背景范围：用更稳的分位数，不要太夸张
  y_upper_df <- volcano_df %>%
    group_by(comparison, comparison_pretty) %>%
    summarise(
      y = quantile(logFC, probs = multivolcano_bar_quantile, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(y = pmax(y, 0))
  
  y_lower_df <- volcano_df %>%
    group_by(comparison, comparison_pretty) %>%
    summarise(
      y = quantile(logFC, probs = 1 - multivolcano_bar_quantile, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(y = pmin(y, 0))
  
  comp_num <- tibble(
    comparison = factor(names(all_results), levels = names(all_results)),
    comparison_pretty = factor(
      pretty_comp_name(names(all_results)),
      levels = pretty_comp_name(names(all_results))
    ),
    x = seq_along(names(all_results)),
    tile_fill = unname(comparison_strip_palette[names(all_results)]),
    tile_label = c("N vs B", "N vs BD", "N vs M", "B vs BD", "B vs M", "BD vs M")
  )

  sig_counts <- volcano_df %>%
    filter(adj.P.Val < multivolcano_p_cutoff) %>%
    mutate(direction = ifelse(logFC >= 0, "up", "down")) %>%
    count(comparison, direction) %>%
    tidyr::pivot_wider(names_from = direction, values_from = n, values_fill = 0)

  comp_num <- comp_num %>%
    left_join(sig_counts, by = "comparison") %>%
    mutate(
      up = ifelse(is.na(up), 0L, up),
      down = ifelse(is.na(down), 0L, down),
      count_label = paste0(up, "/", down)
    )
  
  y_upper_df <- left_join(y_upper_df, comp_num, by = c("comparison", "comparison_pretty"))
  y_lower_df <- left_join(y_lower_df, comp_num, by = c("comparison", "comparison_pretty"))
  
  set.seed(multivolcano_seed)
  dt <- dt %>%
    mutate(
      x = match(as.character(comparison), names(all_results)),
      x_jit = x + runif(n(), -multivolcano_jitter_width, multivolcano_jitter_width)
    )

  sig_pos <- sig_pos %>%
    mutate(
      x = match(as.character(comparison), names(all_results)),
      x_jit = x + runif(n(), -multivolcano_jitter_width, multivolcano_jitter_width)
    )

  sig_neg <- sig_neg %>%
    mutate(
      x = match(as.character(comparison), names(all_results)),
      x_jit = x + runif(n(), -multivolcano_jitter_width, multivolcano_jitter_width)
    )
  
  top_df <- top_df %>%
    mutate(
      x = match(as.character(comparison), names(all_results)),
      x_jit = x + runif(n(), -multivolcano_jitter_width, multivolcano_jitter_width)
    )
  
  y_max_plot <- max(volcano_df$logFC, na.rm = TRUE)
  y_min_plot <- min(volcano_df$logFC, na.rm = TRUE)
  y_span <- y_max_plot - y_min_plot
  if (!is.finite(y_span) || y_span <= 0) y_span <- 1
  
  tile_y <- y_min_plot - y_span * 0.10
  tile_h <- multivolcano_tile_height
  
  color_map <- stats::setNames(
    c(OV_BACKGROUND_GREY, OV_SIGNAL_RED),
    c(
      paste0("adjust P-val>=", multivolcano_p_cutoff),
      paste0("adjust P-val<", multivolcano_p_cutoff)
    )
  )
  
  p <- ggplot() +
    geom_rect(
      data = y_upper_df,
      aes(
        xmin = x - 0.38, xmax = x + 0.38,
        ymin = 0, ymax = y
      ),
      fill = "#D9D9D9",
      alpha = 0.35,
      inherit.aes = FALSE
    ) +
    geom_rect(
      data = y_lower_df,
      aes(
        xmin = x - 0.38, xmax = x + 0.38,
        ymin = y, ymax = 0
      ),
      fill = "#D9D9D9",
      alpha = 0.35,
      inherit.aes = FALSE
    ) +
    geom_hline(yintercept = 0, color = "grey70", linewidth = 0.4) +
    geom_point(
      data = dt,
      aes(x = x_jit, y = logFC, color = sig_class, shape = sig_class),
      size = 0.95,
      alpha = 0.26
    ) +
    geom_point(
      data = sig_pos,
      aes(x = x_jit, y = logFC, color = sig_class, shape = sig_class),
      size = 1.55,
      alpha = 0.90
    ) +
    geom_point(
      data = sig_neg,
      aes(x = x_jit, y = logFC, color = sig_class, shape = sig_class),
      size = 1.65,
      alpha = 0.90
    ) +
    geom_tile(
      data = comp_num,
      aes(x = x, y = tile_y),
      height = tile_h,
      width = 0.76,
      color = "black",
      fill = comp_num$tile_fill,
      alpha = 0.95,
      show.legend = FALSE
    ) +
    geom_text(
      data = comp_num,
      aes(x = x, y = tile_y, label = tile_label),
      size = 6.0,
      color = "white",
      fontface = "bold"
    ) +
    geom_text(
      data = comp_num,
      aes(x = x, y = tile_y - tile_h * 1.15, label = count_label),
      size = 3.1,
      color = "grey15"
    ) +
    scale_color_manual(
      name = NULL,
      values = c(
        "FDR >= 0.05" = OV_BACKGROUND_GREY,
        "FDR < 0.05, logFC > 0" = OV_SIGNAL_RED,
        "FDR < 0.05, logFC < 0" = OV_SIGNAL_BLUE
      )
    ) +
    scale_shape_manual(
      name = NULL,
      values = c(
        "FDR >= 0.05" = 16,
        "FDR < 0.05, logFC > 0" = 16,
        "FDR < 0.05, logFC < 0" = 17
      )
    ) +
    scale_x_continuous(
      breaks = comp_num$x,
      labels = comp_num$comparison_pretty,
      limits = c(0.5, nrow(comp_num) + 0.5)
    ) +
    labs(
      x = "Comparison",
      y = "Average log2 fold change",
      title = "Grouped multi-comparison differential features"
    ) +
    coord_cartesian(
      ylim = c(tile_y - tile_h * 0.9, y_max_plot * 1.04),
      clip = "off"
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "plain", hjust = 0, size = 16),
      axis.title = element_text(size = 15, color = "black", face = "bold"),
      axis.line.y = element_line(color = "black", linewidth = 0.9),
      axis.line.x = element_blank(),
      axis.text.x = element_blank(),
      axis.text.y = element_text(color = "black", size = 11),
      panel.grid = element_blank(),
      legend.position = c(0.96, 0.90),
      legend.justification = c(1, 1),
      legend.direction = "vertical",
      legend.text = element_text(size = 11),
      plot.margin = margin(12, 18, 30, 12)
    )
  
  list(
    plot = p,
    plot_df = volcano_df,
    top_df = top_df,
    bar_up = y_upper_df,
    bar_down = y_lower_df,
    tile_df = comp_num
  )
}

# -----------------------------
# Heatmap helpers
# -----------------------------
build_heatmap_matrix <- function(df_proc, feature_ids = NULL, sample_groups = group_levels,
                                 max_features = Inf, clip_value = heatmap_clip) {
  sub_df <- df_proc %>%
    filter(label %in% sample_groups) %>%
    mutate(label = factor(label, levels = group_levels)) %>%
    arrange(label, sample_id)

  all_feat_cols <- setdiff(colnames(sub_df), c("sample_id", "label"))
  if (is.null(feature_ids)) {
    feature_ids <- all_feat_cols
  } else {
    feature_ids <- intersect(feature_ids, all_feat_cols)
  }

  if (length(feature_ids) == 0) stop("No feature IDs available for heatmap.")

  if (is.finite(max_features) && length(feature_ids) > max_features) {
    feature_ids <- feature_ids[seq_len(max_features)]
  }

  mat <- sub_df %>% select(all_of(feature_ids)) %>% as.matrix()
  rownames(mat) <- make.unique(paste0(as.character(sub_df$label), "_", sub_df$sample_id))

  feature_means <- apply(mat, 2, mean, na.rm = TRUE)
  feature_sds <- apply(mat, 2, sd, na.rm = TRUE)
  feature_sds[feature_sds == 0 | is.na(feature_sds)] <- 1e-10
  mat_z <- sweep(mat, 2, feature_means, FUN = "-")
  mat_z <- sweep(mat_z, 2, feature_sds, FUN = "/")
  mat_z[mat_z > clip_value] <- clip_value
  mat_z[mat_z < -clip_value] <- -clip_value

  if (anyNA(mat_z)) {
    col_medians <- apply(mat_z, 2, median, na.rm = TRUE)
    for (i in seq_len(ncol(mat_z))) {
      na_idx <- is.na(mat_z[, i])
      if (any(na_idx)) mat_z[na_idx, i] <- col_medians[i]
    }
  }

  list(
    matrix = t(mat_z),
    sample_info = sub_df %>% select(sample_id, label) %>% mutate(sample_name = rownames(mat))
  )
}

draw_complex_heatmap <- function(hm_obj, file_stub, width = 16, height = 10) {
  pdf(paste0(file_stub, ".pdf"), width = width, height = height)
  draw(hm_obj, heatmap_legend_side = "right", annotation_legend_side = "right")
  dev.off()

  png(paste0(file_stub, ".png"), width = width, height = height, units = "in", res = 320)
  draw(hm_obj, heatmap_legend_side = "right", annotation_legend_side = "right")
  dev.off()
}

make_grouped_heatmap <- function(df_proc, feature_ids = NULL, sample_groups = group_levels,
                                 title_text, file_stub, max_features = Inf,
                                 split_mode = c("four_groups", "N_vs_M")) {
  split_mode <- match.arg(split_mode)
  hm_dat <- build_heatmap_matrix(
    df_proc = df_proc,
    feature_ids = feature_ids,
    sample_groups = sample_groups,
    max_features = max_features
  )
  heatmap_mat <- hm_dat$matrix
  sample_info <- hm_dat$sample_info

  if (nrow(heatmap_mat) < 1 || ncol(heatmap_mat) < 2) {
    message("Heatmap skipped due to too-small matrix: ", title_text)
    return(invisible(NULL))
  }

  annotation_df <- data.frame(Group = sample_info$label)
  rownames(annotation_df) <- sample_info$sample_name

  ha <- HeatmapAnnotation(
    Group = annotation_df$Group,
    col = list(Group = group_palette),
    annotation_name_side = "left",
    annotation_legend_param = list(
      title = "Group",
      title_position = "topcenter",
      ncol = 1,
      legend_direction = "vertical"
    )
  )

  col_fun <- circlize::colorRamp2(
    c(min(heatmap_mat, na.rm = TRUE), 0, max(heatmap_mat, na.rm = TRUE)),
    c(OV_HEATMAP_BLUE, OV_HEATMAP_WHITE, OV_HEATMAP_RED)
  )

  split_vector <- if (split_mode == "four_groups") {
    factor(sample_info$label, levels = group_levels)
  } else {
    factor(ifelse(sample_info$label == "N", "N", "M"), levels = c("N", "M"))
  }

  ht <- Heatmap(
    heatmap_mat,
    name = "Expression",
    col = col_fun,
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    clustering_method_rows = "ward.D2",
    show_row_names = FALSE,
    show_column_names = FALSE,
    top_annotation = ha,
    column_split = split_vector,
    column_gap = unit(4, "mm"),
    row_title = paste0("Metabolites (n=", nrow(heatmap_mat), ")"),
    column_title = title_text,
    heatmap_legend_param = list(
      title = "Z-score",
      title_position = "topcenter",
      legend_width = unit(5, "cm")
    ),
    use_raster = TRUE,
    raster_quality = 2
  )

  draw_complex_heatmap(
    ht,
    file_stub = file_stub,
    width = ifelse(split_mode == "four_groups", 18, 14),
    height = 10
  )

  invisible(list(matrix = heatmap_mat, sample_info = sample_info))
}

truncate_label <- function(x, width = 32) {
  x <- gsub(" \\(not validated\\)", "", x, ignore.case = TRUE)
  x <- gsub("Not validated", "", x, ignore.case = TRUE)
  ifelse(nchar(x) > width, paste0(substr(x, 1, width - 3), "..."), x)
}

flag_endogenous_like_features <- function(feature_ids) {
  feature_ids <- as.character(feature_ids)
  include_pattern <- paste(
    c(
      "acid", "FA ", "LPC", "LPE", "carnitine", "inosine",
      "dehydroisoandrosterone", "pregnenolone", "HHTrE", "PGI2",
      "nicotinamide", "Ala-Ile", "Leucylalanine", "Cyclo\\(",
      "ketoleucine", "indoxyl", "Niacinamide", "glycocholic",
      "sulfochenodeoxycholic", "pyroglutamic", "glycerol",
      "creatine", "creatinine", "phenylalanine", "tryptophan",
      "arginine", "ornithine", "isoleucine", "threonine",
      "asparagine", "uridine", "butanoic", "decanoic",
      "octanoic", "dodecanedioic", "octadecanedioic",
      "oxindole", "hydroxyphenylacetic", "sulfate", "amide",
      "betaine"
    ),
    collapse = "|"
  )
  exclude_pattern <- paste(
    c(
      "phthalate", "NCGC", "CCMSLIB", "not validated", "isomer",
      "atrazine", "terbuthylazine", "benzothiazole", "benzoxazolinone",
      "acetoxyphenol", "methoxycinnamic", "trimethylbenzoic",
      "camphanic", "perillic", "paraxanthine", "caffeine",
      "salicylic", "catechol", "naphthalene", "benzenesulfonate",
      "piperidine", "quinazolin", "isoquinoline", "xanthene",
      "aniline", "didanosine", "exemestane", "ethoxyquin",
      "tanshinone", "viridiflorine", "heliocurassavicine",
      "psoralen", "coumaraldehyde", "diphenyl", "lysergol",
      "acetanilide", "aminophenol", "aminobenzoic", "methylindole",
      "hydroxypyrene", "acetoin", "theobromine", "anisic",
      "nordeprenyl", "flavone", "methylpyrazole", "chaulmoogric",
      "acridone", "securinine", "nalidixic", "tuberostemonine",
      "lauryl", "diethanolamide", "hydroxymethyl", "oxononanoic",
      "dihydroxybenzoic", "aminotetrahydro", "methanesulfinyl",
      "naphthyridin", "lichesterinic", "fructosyl", "methylpyrimidin"
    ),
    collapse = "|"
  )
  grepl(include_pattern, feature_ids, ignore.case = TRUE) &
    !grepl(exclude_pattern, feature_ids, ignore.case = TRUE)
}

select_endogenous_heatmap_features <- function(all_results, n = selected_clinical_heatmap_n) {
  ranked <- bind_rows(all_results) %>%
    filter(significant) %>%
    mutate(
      abs_logFC = abs(logFC),
      endogenous_like = flag_endogenous_like_features(feature_id)
    ) %>%
    arrange(adj.P.Val, desc(abs_logFC)) %>%
    distinct(feature_id, .keep_all = TRUE)

  write.csv(
    ranked %>% filter(!endogenous_like),
    file.path(coord_dir, "MERGED__Heatmap_selected_features_excluded_nonendogenous_like.csv"),
    row.names = FALSE
  )

  selected <- ranked %>%
    filter(endogenous_like) %>%
    slice_head(n = n)

  if (nrow(selected) == 0) {
    stop("No endogenous-like significant features were available for selected clinical heatmap.")
  }
  selected
}

make_clinical_annotation_heatmap <- function(df_proc, feature_ids, file_stub, title_text,
                                             sample_groups = group_levels,
                                             feature_table = NULL,
                                             max_features = Inf,
                                             show_feature_names = TRUE,
                                             width = 11.5,
                                             height = 7.6) {
  hm_dat <- build_heatmap_matrix(
    df_proc = df_proc,
    feature_ids = feature_ids,
    sample_groups = sample_groups,
    max_features = max_features
  )
  heatmap_mat <- hm_dat$matrix
  sample_info <- hm_dat$sample_info

  clinical_candidates <- c(
    file.path(module_root, "..", "01_cohort_and_design", "data_clean", "clinical_merged_analysis_ready.csv"),
    file.path(module_root, "..", "..", "01_cohort_and_design", "data_clean", "clinical_merged_analysis_ready.csv")
  )
  clinical_path <- normalizePath(clinical_candidates[file.exists(clinical_candidates)][1], mustWork = FALSE)
  if (!file.exists(clinical_path)) {
    stop("Cannot find clinical_merged_analysis_ready.csv for selected clinical heatmap.")
  }
  clinical_df <- suppressMessages(readr::read_csv(clinical_path, show_col_types = FALSE)) %>%
    mutate(sample_id = as.character(sample_id)) %>%
    select(sample_id, age_group, CA125, stage_binary, pathology_class)

  sample_ann <- sample_info %>%
    mutate(sample_id = as.character(sample_id)) %>%
    left_join(clinical_df, by = "sample_id") %>%
    mutate(
      Group = factor(label, levels = group_levels),
      Age = factor(ifelse(is.na(age_group), "Missing", age_group), levels = c("<40", "40-49", ">=50", "Missing")),
      CA125_cat = dplyr::case_when(
        is.na(CA125) ~ "Missing",
        CA125 <= 35 ~ "<=35",
        TRUE ~ ">35"
      ),
      CA125_cat = factor(CA125_cat, levels = c("<=35", ">35", "Missing")),
      Stage = factor(ifelse(is.na(stage_binary), "Not malignant/NA", stage_binary), levels = c("Early (I-II)", "Advanced (III-IV)", "Not malignant/NA")),
      Pathology = factor(ifelse(is.na(pathology_class), "Not malignant/NA", pathology_class), levels = c("Epithelial", "Sex cord-stromal", "Germ cell", "Carcinosarcoma", "Not malignant/NA"))
    )

  ha <- HeatmapAnnotation(
    Group = sample_ann$Group,
    Age = sample_ann$Age,
    CA125 = sample_ann$CA125_cat,
    Stage = sample_ann$Stage,
    Pathology = sample_ann$Pathology,
    col = list(
      Group = group_palette,
      Age = c("<40" = "#D9D9D9", "40-49" = "#BEBADA", ">=50" = "#80B1D3", "Missing" = "#F0F0F0"),
      CA125 = c("<=35" = "#8DD3C7", ">35" = "#FB8072", "Missing" = "#F0F0F0"),
      Stage = c("Early (I-II)" = "#8DD3C7", "Advanced (III-IV)" = "#FB8072", "Not malignant/NA" = "#F0F0F0"),
      Pathology = c("Epithelial" = "#FB8072", "Sex cord-stromal" = "#FDB462", "Germ cell" = "#80B1D3", "Carcinosarcoma" = "#BEBADA", "Not malignant/NA" = "#F0F0F0")
    ),
    annotation_name_side = "left"
  )

  col_fun <- circlize::colorRamp2(c(-2.5, 0, 2.5), c(OV_HEATMAP_BLUE, OV_HEATMAP_WHITE, OV_HEATMAP_RED))
  ht <- Heatmap(
    heatmap_mat,
    name = "Z-score",
    col = col_fun,
    cluster_rows = TRUE,
    cluster_columns = FALSE,
    show_row_names = show_feature_names,
    row_labels = truncate_label(rownames(heatmap_mat), width = 34),
    row_names_gp = grid::gpar(fontsize = 5),
    show_column_names = FALSE,
    top_annotation = ha,
    column_split = factor(sample_info$label, levels = sample_groups),
    column_gap = unit(2.5, "mm"),
    column_title = paste0(title_text, " (n=", nrow(heatmap_mat), ")"),
    heatmap_legend_param = list(title = "Z-score", title_position = "topcenter"),
    use_raster = TRUE,
    raster_quality = 2
  )

  draw_complex_heatmap(ht, file_stub = file_stub, width = width, height = height)
  source_stub <- file.path(coord_dir, basename(file_stub))
  write.csv(as.data.frame(heatmap_mat), paste0(source_stub, "_matrix.csv"))
  write.csv(sample_ann, paste0(source_stub, "_sample_info.csv"), row.names = FALSE)
  if (!is.null(feature_table)) {
    write.csv(feature_table, paste0(source_stub, "_feature_list.csv"), row.names = FALSE)
  } else {
    write.csv(data.frame(feature_id = rownames(heatmap_mat)), paste0(source_stub, "_feature_list.csv"), row.names = FALSE)
  }
  invisible(list(matrix = heatmap_mat, sample_info = sample_ann, feature_list = feature_table))
}

make_selected_clinical_heatmap <- function(df_proc, all_results, file_stub) {
  selected <- select_endogenous_heatmap_features(all_results)
  make_clinical_annotation_heatmap(
    df_proc = df_proc,
    feature_ids = selected$feature_id,
    file_stub = file_stub,
    title_text = "Selected endogenous-like differential features",
    sample_groups = group_levels,
    feature_table = selected,
    max_features = selected_clinical_heatmap_n,
    show_feature_names = TRUE,
    width = 11.5,
    height = 7.6
  )
}

# -----------------------------
# Load data
# -----------------------------
merged_path <- file.path(data_dir, "merged_deDuplicated_matrix.csv")
if (!file.exists(merged_path)) {
  stop("merged_deDuplicated_matrix.csv was not found in ./data.")
}

raw_df <- read_matrix_file(merged_path)
proc_df <- preprocess_matrix(raw_df)
write.csv(proc_df, file.path(coord_dir, "MERGED_processed_matrix.csv"), row.names = FALSE)

# -----------------------------
# Pairwise differential analysis
# -----------------------------
workbook <- createWorkbook()
all_results <- list()
summary_list <- list()

for (cmp in comparison_list) {
  g1 <- cmp[1]
  g2 <- cmp[2]
  comp_name <- paste0(g1, "_vs_", g2)

  res_df <- run_pairwise_limma(proc_df, g1, g2)
  all_results[[comp_name]] <- res_df

  out_csv <- file.path(res_dir, paste0("MERGED__", comp_name, "__diff_results.csv"))
  write.csv(res_df, out_csv, row.names = FALSE)

  addWorksheet(workbook, safe_sheet_name(comp_name))
  writeData(workbook, safe_sheet_name(comp_name), res_df)

  summary_list[[comp_name]] <- res_df %>%
    summarise(
      dataset = "MERGED",
      comparison = comp_name,
      total_features = n(),
      significant_features = sum(significant),
      median_abs_logFC_sig = ifelse(sum(significant) > 0, median(abs(logFC[significant])), NA_real_)
    )
}

saveWorkbook(workbook, file.path(res_dir, "MERGED_pairwise_diff_results.xlsx"), overwrite = TRUE)
summary_df <- bind_rows(summary_list)
write.csv(summary_df, file.path(res_dir, "MERGED__pairwise_summary.csv"), row.names = FALSE)

# -----------------------------
# Figure 1: grouped multi-volcano
# -----------------------------
volcano_out <- make_grouped_multivolcano_plot(all_results)
save_plot_dual(
  volcano_out$plot,
  file.path(fig_dir, "MERGED__Fig3_grouped_multivolcano"),
  width = 10.8,
  height = 7.2
)
write.csv(volcano_out$plot_df, file.path(coord_dir, "MERGED__Fig3_grouped_multivolcano_source.csv"), row.names = FALSE)
write.csv(volcano_out$top_df, file.path(coord_dir, "MERGED__Fig3_grouped_multivolcano_top_points.csv"), row.names = FALSE)
write.csv(volcano_out$bar_up, file.path(coord_dir, "MERGED__Fig3_grouped_multivolcano_bar_up.csv"), row.names = FALSE)
write.csv(volcano_out$bar_down, file.path(coord_dir, "MERGED__Fig3_grouped_multivolcano_bar_down.csv"), row.names = FALSE)
write.csv(volcano_out$tile_df, file.path(coord_dir, "MERGED__Fig3_grouped_multivolcano_tiles.csv"), row.names = FALSE)

# -----------------------------
# Figure 2: selected endogenous-like feature heatmap with clinical annotations
# -----------------------------
selected_clinical_hm <- make_selected_clinical_heatmap(
  df_proc = proc_df,
  all_results = all_results,
  file_stub = file.path(fig_dir, "MERGED__Heatmap_selected_features_clinical_annotations")
)
if (!is.null(selected_clinical_hm)) {
  write.csv(as.data.frame(selected_clinical_hm$matrix), file.path(coord_dir, "MERGED__Heatmap_selected_features_clinical_annotations_matrix.csv"))
  write.csv(selected_clinical_hm$sample_info, file.path(coord_dir, "MERGED__Heatmap_selected_features_clinical_annotations_sample_info.csv"), row.names = FALSE)
  write.csv(selected_clinical_hm$feature_list, file.path(coord_dir, "MERGED__Heatmap_selected_features_clinical_annotations_feature_list.csv"), row.names = FALSE)
}

# -----------------------------
# Figure 3: N_vs_M significant-feature heatmap
# -----------------------------
nm_res <- all_results[["N_vs_M"]]
if (!is.null(nm_res)) {
  nm_sig_features <- nm_res %>%
    filter(significant) %>%
    arrange(adj.P.Val, desc(abs(logFC))) %>%
    pull(feature_id) %>%
    unique()

  if (length(nm_sig_features) > 0) {
    nm_feature_table <- nm_res %>%
      filter(feature_id %in% nm_sig_features) %>%
      arrange(match(feature_id, nm_sig_features)) %>%
      distinct(feature_id, .keep_all = TRUE)

    nm_hm <- make_clinical_annotation_heatmap(
      df_proc = proc_df,
      feature_ids = nm_sig_features,
      file_stub = file.path(fig_dir, "MERGED__Heatmap_N_vs_M_significant_features"),
      title_text = "N vs M significant features with clinical annotations",
      sample_groups = c("N", "M"),
      feature_table = nm_feature_table,
      max_features = nm_sig_heatmap_max_features,
      show_feature_names = length(nm_sig_features) <= 80,
      width = 10.5,
      height = 8.2
    )
    if (!is.null(nm_hm)) {
      write.csv(as.data.frame(nm_hm$matrix), file.path(coord_dir, "MERGED__Heatmap_N_vs_M_significant_features_matrix.csv"))
      write.csv(nm_hm$sample_info, file.path(coord_dir, "MERGED__Heatmap_N_vs_M_significant_features_sample_info.csv"), row.names = FALSE)
      write.csv(nm_hm$feature_list, file.path(coord_dir, "MERGED__Heatmap_N_vs_M_significant_feature_list.csv"), row.names = FALSE)
    }
  } else {
    message("No significant features for N_vs_M under current thresholds; N_vs_M heatmap skipped.")
  }
}

message("All analyses completed successfully. Grouped volcano, selected clinical heatmap and annotated N-vs-M heatmap were generated.")


#!/usr/bin/env Rscript

# ============================================================
# Script: 02_metabolomics_qc_and_overview.R
# Purpose:
#   1. Read four metabolomics matrices in data_clean/
#   2. Compare before-vs-after batch correction for pos/neg mode
#   3. Generate PCA plots and several QC/overview plots
#   4. Export summary tables to data_clean/ and figures to figure_final/
#
# Expected input files (in ../data_clean or ./data_clean):
#   - OV_4.9_neg_raw_clean.csv
#   - OV_4.9_neg_raw_clean_aftercor.csv
#   - OV_4.9_pos_raw_clean.csv
#   - OV_4.9_pos_raw_clean_aftercor.csv
#
# Data structure assumption:
#   - Rows = samples
#   - One column named "Alignment ID" for sample ID
#   - One column named "label" for sample group (e.g. QC / N / B / BD / M)
#   - All remaining columns are numeric metabolite features
#
# Outputs:
#   - ../data_clean/metabolomics_overview_summary.csv
#   - ../data_clean/metabolomics_pca_variance_summary.csv
#   - ../data_clean/metabolomics_qc_cv_summary.csv
#   - ../figure_final/*.pdf and *.png
# ============================================================

suppressPackageStartupMessages({
  required_pkgs <- c("ggplot2", "dplyr", "tidyr", "readr", "stringr",
                     "forcats", "patchwork", "purrr", "scales")
  missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    stop(
      "Please install required packages first:\n",
      paste(missing_pkgs, collapse = ", "),
      call. = FALSE
    )
  }

  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(forcats)
  library(patchwork)
  library(purrr)
  library(scales)
})
source("D:/xwdata/OVdatarewrite/00_project_style/ov_publication_style.R")

# -----------------------------
# Helper: resolve directories
# -----------------------------
get_module_root <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(dirname(normalizePath(sub("^--file=", "", file_arg), winslash = "/"))))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(dirname(normalizePath(sys.frames()[[1]]$ofile, winslash = "/"))))
  }
  cwd <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

  candidates <- c(
    cwd,
    file.path(cwd, ".."),
    file.path(cwd, "../.."),
    file.path(cwd, "02_metabolomics_qc_and_overview")
  )

  candidates <- unique(normalizePath(candidates, winslash = "/", mustWork = FALSE))

  for (p in candidates) {
    if (dir.exists(file.path(p, "data_clean")) &&
        dir.exists(file.path(p, "figure_final")) &&
        dir.exists(file.path(p, "scripts"))) {
      return(normalizePath(p, winslash = "/", mustWork = TRUE))
    }
  }

  stop("Cannot locate module root containing data_clean/, figure_final/, and scripts/.", call. = FALSE)
}

module_root <- get_module_root()
data_dir    <- file.path(module_root, "data_clean")
figure_dir  <- file.path(module_root, "figure_final")

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

group_colors <- c(OV_GROUP_COLORS, QC = OV_BACKGROUND_GREY, UNKNOWN = OV_BACKGROUND_GREY)
status_colors <- c(Before = OV_BACKGROUND_GREY, After = OV_SIGNAL_BLUE)
supp_qc_base_size <- 10.5

message("Module root: ", module_root)
message("Data dir    : ", data_dir)
message("Figure dir  : ", figure_dir)

# -----------------------------
# Input files
# -----------------------------
file_map <- tibble::tribble(
  ~mode, ~status,    ~filename,
  "NEG", "Before",   "OV_4.9_neg_raw_clean.csv",
  "NEG", "After",    "OV_4.9_neg_raw_clean_aftercor.csv",
  "POS", "Before",   "OV_4.9_pos_raw_clean.csv",
  "POS", "After",    "OV_4.9_pos_raw_clean_aftercor.csv"
) %>%
  mutate(path = file.path(data_dir, filename))

missing_files <- file_map %>% filter(!file.exists(path))
if (nrow(missing_files) > 0) {
  stop(
    "Missing input files:\n",
    paste(missing_files$path, collapse = "\n"),
    call. = FALSE
  )
}

# -----------------------------
# Helper functions
# -----------------------------
detect_id_col <- function(df) {
  id_candidates <- c("Alignment ID", "alignment_id", "SampleID", "sample_id", "ID", "id")
  id_col <- id_candidates[id_candidates %in% colnames(df)][1]
  if (is.na(id_col)) stop("Cannot find sample ID column.", call. = FALSE)
  id_col
}

detect_label_col <- function(df) {
  label_candidates <- c("label", "Label", "group", "Group", "Class", "class")
  label_col <- label_candidates[label_candidates %in% colnames(df)][1]
  if (is.na(label_col)) stop("Cannot find label column.", call. = FALSE)
  label_col
}

prepare_matrix <- function(df, mode, status) {
  id_col <- detect_id_col(df)
  label_col <- detect_label_col(df)

  feature_cols <- setdiff(colnames(df), c(id_col, label_col))

  # force numeric on features
  df[feature_cols] <- lapply(df[feature_cols], function(x) suppressWarnings(as.numeric(x)))

  # sample order factor
  df <- df %>%
    mutate(
      sample_id = as.character(.data[[id_col]]),
      label = as.character(.data[[label_col]]),
      mode = mode,
      status = status
    ) %>%
    select(sample_id, label, mode, status, all_of(feature_cols))

  # standardize labels
  df <- df %>%
    mutate(
      label = ifelse(is.na(label) | label == "", "Unknown", label),
      label = toupper(label),
      sample_class = ifelse(label == "QC", "QC", "Biological")
    )

  df
}

safe_log2 <- function(mat) {
  log2(mat + 1)
}

make_pca_df <- function(df, center = TRUE, scale. = TRUE, remove_qc = FALSE) {
  feature_cols <- setdiff(colnames(df), c("sample_id", "label", "mode", "status", "sample_class"))
  sub_df <- df

  if (remove_qc) {
    sub_df <- sub_df %>% filter(sample_class != "QC")
  }

  mat <- as.matrix(sub_df[, feature_cols, drop = FALSE])
  mat <- safe_log2(mat)

  # remove zero-variance / all-missing features
  keep <- apply(mat, 2, function(x) {
    x <- x[is.finite(x)]
    length(x) >= 2 && sd(x) > 0
  })
  mat <- mat[, keep, drop = FALSE]

  # impute missing values by feature median
  for (j in seq_len(ncol(mat))) {
    idx <- !is.finite(mat[, j])
    if (any(idx)) {
      med <- stats::median(mat[, j], na.rm = TRUE)
      if (!is.finite(med)) med <- 0
      mat[idx, j] <- med
    }
  }

  pca <- prcomp(mat, center = center, scale. = scale.)

  var_exp <- (pca$sdev^2) / sum(pca$sdev^2)
  pca_df <- as.data.frame(pca$x[, 1:5, drop = FALSE]) %>%
    mutate(
      sample_id = sub_df$sample_id,
      label = sub_df$label,
      mode = sub_df$mode,
      status = sub_df$status,
      sample_class = sub_df$sample_class
    )

  list(
    pca = pca,
    pca_df = pca_df,
    variance = tibble::tibble(
      mode = unique(sub_df$mode),
      status = unique(sub_df$status),
      remove_qc = remove_qc,
      PC = paste0("PC", seq_along(var_exp)),
      variance_explained = var_exp
    )
  )
}

calc_sample_metrics <- function(df) {
  feature_cols <- setdiff(colnames(df), c("sample_id", "label", "mode", "status", "sample_class"))
  mat <- as.matrix(df[, feature_cols, drop = FALSE])

  tibble::tibble(
    sample_id = df$sample_id,
    label = df$label,
    mode = df$mode,
    status = df$status,
    sample_class = df$sample_class,
    n_features = ncol(mat),
    missing_fraction = rowMeans(!is.finite(mat) | is.na(mat)),
    total_intensity = rowSums(mat, na.rm = TRUE),
    median_intensity = apply(mat, 1, median, na.rm = TRUE)
  )
}

calc_feature_metrics <- function(df) {
  feature_cols <- setdiff(colnames(df), c("sample_id", "label", "mode", "status", "sample_class"))
  mat <- as.matrix(df[, feature_cols, drop = FALSE])

  tibble::tibble(
    feature = colnames(mat),
    mode = unique(df$mode),
    status = unique(df$status),
    missing_fraction = colMeans(!is.finite(mat) | is.na(mat)),
    mean_intensity = colMeans(mat, na.rm = TRUE),
    median_intensity = apply(mat, 2, median, na.rm = TRUE),
    sd_intensity = apply(mat, 2, sd, na.rm = TRUE)
  )
}

calc_qc_cv <- function(df) {
  feature_cols <- setdiff(colnames(df), c("sample_id", "label", "mode", "status", "sample_class"))
  qc_df <- df %>% filter(sample_class == "QC")

  if (nrow(qc_df) < 2) {
    return(tibble::tibble(
      feature = character(),
      mode = character(),
      status = character(),
      qc_mean = numeric(),
      qc_sd = numeric(),
      qc_cv = numeric()
    ))
  }

  mat <- as.matrix(qc_df[, feature_cols, drop = FALSE])

  qc_mean <- colMeans(mat, na.rm = TRUE)
  qc_sd   <- apply(mat, 2, sd, na.rm = TRUE)
  qc_cv   <- ifelse(qc_mean == 0, NA_real_, qc_sd / qc_mean)

  tibble::tibble(
    feature = names(qc_mean),
    mode = unique(df$mode),
    status = unique(df$status),
    qc_mean = qc_mean,
    qc_sd = qc_sd,
    qc_cv = qc_cv
  )
}

plot_pca <- function(pca_df, var_df, title_text, remove_qc = FALSE) {
  vx <- percent(var_df$variance_explained[var_df$PC == "PC1"][1], accuracy = 0.1)
  vy <- percent(var_df$variance_explained[var_df$PC == "PC2"][1], accuracy = 0.1)

  label_levels <- c("N", "B", "BD", "M", "QC", "UNKNOWN")
  pca_df <- pca_df %>%
    mutate(label = factor(label, levels = unique(c(label_levels, sort(unique(as.character(label)))))))

  ggplot(pca_df, aes(x = PC1, y = PC2, color = label, shape = sample_class)) +
    geom_point(size = 2.35, alpha = 0.82) +
    labs(
      title = title_text,
      x = paste0("PC1 (", vx, ")"),
      y = paste0("PC2 (", vy, ")"),
      color = "Group",
      shape = "Sample type"
    ) +
    scale_color_manual(values = group_colors, na.value = OV_BACKGROUND_GREY) +
    ov_theme(base_size = supp_qc_base_size, base_family = "Helvetica") +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = supp_qc_base_size + 1.4),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.box.spacing = margin(0, 0, 0, 4)
    ) +
    guides(
      color = guide_legend(override.aes = list(size = 3.2, alpha = 1)),
      shape = guide_legend(override.aes = list(size = 3.2, alpha = 1))
    )
}

plot_variance_bar <- function(var_df, title_text) {
  sub <- var_df %>% slice_head(n = 10) %>%
    mutate(PC = factor(PC, levels = PC))

  ggplot(sub, aes(x = PC, y = variance_explained)) +
    geom_col(width = 0.75) +
    scale_y_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title = title_text,
      x = NULL,
      y = "Explained variance"
    ) +
    ov_theme(base_size = 7, base_family = "Helvetica") +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid.major.x = element_blank()
    )
}

plot_sample_metric <- function(metric_df, value_col, ylab, title_text, log10_y = FALSE) {
  p <- ggplot(metric_df, aes(x = status, y = .data[[value_col]], fill = status)) +
    geom_boxplot(width = 0.65, outlier.shape = NA, alpha = 0.8) +
    geom_jitter(aes(color = label), width = 0.14, size = 1.1, alpha = 0.72) +
    facet_wrap(~ mode, scales = "free_y") +
    labs(
      title = title_text,
      x = NULL,
      y = ylab,
      fill = NULL,
      color = "Group"
    ) +
    scale_fill_manual(values = status_colors) +
    scale_color_manual(values = group_colors, na.value = OV_BACKGROUND_GREY) +
    ov_theme(base_size = 7, base_family = "Helvetica") +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid.major.x = element_blank(),
      legend.position = "right"
    )
  if (log10_y) {
    p <- p + scale_y_continuous(trans = "log10", labels = label_number())
  }
  p
}

plot_qc_cv <- function(qc_cv_df, title_text) {
  ggplot(qc_cv_df %>% filter(is.finite(qc_cv)), aes(x = qc_cv, fill = status)) +
    geom_histogram(bins = 40, alpha = 0.58, position = "identity", colour = "white", linewidth = 0.15) +
    labs(
      title = title_text,
      x = "QC coefficient of variation (CV)",
      y = "Feature count",
      fill = NULL
    ) +
    scale_fill_manual(values = status_colors) +
    ov_theme(base_size = supp_qc_base_size, base_family = "Helvetica") +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = supp_qc_base_size + 1.4),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.box.spacing = margin(0, 0, 0, 4)
    ) +
    guides(fill = guide_legend(override.aes = list(alpha = 0.85)))
}

plot_qc_cv_threshold <- function(qc_cv_df, title_text) {
  summary_df <- qc_cv_df %>%
    filter(is.finite(qc_cv)) %>%
    group_by(mode, status) %>%
    summarise(
      cv_20 = mean(qc_cv <= 0.20),
      cv_30 = mean(qc_cv <= 0.30),
      cv_50 = mean(qc_cv <= 0.50),
      .groups = "drop"
    ) %>%
    pivot_longer(cols = starts_with("cv_"), names_to = "threshold", values_to = "proportion") %>%
    mutate(
      threshold = recode(threshold, cv_20 = "CV ≤ 20%", cv_30 = "CV ≤ 30%", cv_50 = "CV ≤ 50%"),
      threshold = factor(threshold, levels = c("CV ≤ 20%", "CV ≤ 30%", "CV ≤ 50%"))
    )

  ggplot(summary_df, aes(x = threshold, y = proportion, fill = status)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    facet_wrap(~ mode) +
    scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
    labs(
      title = title_text,
      x = NULL,
      y = "Proportion of features",
      fill = NULL
    ) +
    scale_fill_manual(values = status_colors) +
    ov_theme(base_size = 7, base_family = "Helvetica") +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid.major.x = element_blank()
    )
}

save_plot_pdf_png <- function(plot_obj, basename, width = 8, height = 6, dpi = 320) {
  pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else "pdf"
  ggsave(filename = file.path(figure_dir, paste0(basename, ".pdf")),
         plot = plot_obj, width = width, height = height, dpi = dpi,
         device = pdf_device, fallback_resolution = dpi)
  ggsave(filename = file.path(figure_dir, paste0(basename, ".png")),
         plot = plot_obj, width = width, height = height, dpi = dpi)
  if (requireNamespace("svglite", quietly = TRUE)) {
    tryCatch({
      svglite::svglite(file.path(figure_dir, paste0(basename, ".svg")), width = width, height = height)
      print(plot_obj)
      dev.off()
    }, error = function(e) {
      if (grDevices::dev.cur() != 1) grDevices::dev.off()
      message("Skipping SVG export for ", basename, ": ", conditionMessage(e))
    })
  }
}

# -----------------------------
# Read and prepare all matrices
# -----------------------------
data_list <- vector("list", nrow(file_map))
names(data_list) <- paste(file_map$mode, file_map$status, sep = "_")

for (i in seq_len(nrow(file_map))) {
  raw_df <- readr::read_csv(file_map$path[i], show_col_types = FALSE)
  data_list[[i]] <- prepare_matrix(raw_df, mode = file_map$mode[i], status = file_map$status[i])
}

# -----------------------------
# Compute summaries
# -----------------------------
sample_metrics <- purrr::map_dfr(data_list, calc_sample_metrics)
feature_metrics <- purrr::map_dfr(data_list, calc_feature_metrics)
qc_cv_metrics <- purrr::map_dfr(data_list, calc_qc_cv)

overview_summary <- sample_metrics %>%
  group_by(mode, status) %>%
  summarise(
    n_samples = n(),
    n_qc = sum(sample_class == "QC"),
    n_biological = sum(sample_class != "QC"),
    mean_missing_fraction = mean(missing_fraction, na.rm = TRUE),
    median_total_intensity = median(total_intensity, na.rm = TRUE),
    median_median_intensity = median(median_intensity, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(
    feature_metrics %>%
      group_by(mode, status) %>%
      summarise(
        n_features = n(),
        mean_feature_missing_fraction = mean(missing_fraction, na.rm = TRUE),
        .groups = "drop"
      ),
    by = c("mode", "status")
  ) %>%
  left_join(
    qc_cv_metrics %>%
      filter(is.finite(qc_cv)) %>%
      group_by(mode, status) %>%
      summarise(
        median_qc_cv = median(qc_cv, na.rm = TRUE),
        prop_qc_cv_20 = mean(qc_cv <= 0.20, na.rm = TRUE),
        prop_qc_cv_30 = mean(qc_cv <= 0.30, na.rm = TRUE),
        prop_qc_cv_50 = mean(qc_cv <= 0.50, na.rm = TRUE),
        .groups = "drop"
      ),
    by = c("mode", "status")
  )

readr::write_csv(overview_summary, file.path(data_dir, "metabolomics_overview_summary.csv"))
readr::write_csv(sample_metrics, file.path(data_dir, "metabolomics_sample_metrics.csv"))
readr::write_csv(feature_metrics, file.path(data_dir, "metabolomics_feature_metrics.csv"))
readr::write_csv(qc_cv_metrics, file.path(data_dir, "metabolomics_qc_cv_metrics.csv"))

# -----------------------------
# PCA
# -----------------------------
pca_results_all <- purrr::map(data_list, ~ make_pca_df(.x, remove_qc = FALSE))
pca_results_bio <- purrr::map(data_list, ~ make_pca_df(.x, remove_qc = TRUE))

pca_var_all <- purrr::map_dfr(pca_results_all, "variance")
pca_var_bio <- purrr::map_dfr(pca_results_bio, "variance") %>% mutate(remove_qc = TRUE)

readr::write_csv(
  bind_rows(
    pca_var_all %>% mutate(dataset = "All samples"),
    pca_var_bio %>% mutate(dataset = "Biological only")
  ),
  file.path(data_dir, "metabolomics_pca_variance_summary.csv")
)

# -----------------------------
# Build retained plots only
# -----------------------------
# Current manuscript architecture keeps only three QC panels from this module:
#   1) NEG after-correction PCA
#   2) POS after-correction PCA
#   3) QC CV distribution before vs after correction
# Other overview, threshold-summary and before/after comparison plots are
# intentionally not generated to avoid old figures returning after reruns.
p_neg_after <- plot_pca(
  pca_results_all[["NEG_After"]]$pca_df,
  pca_results_all[["NEG_After"]]$variance,
  title_text = "Negative ion mode - after correction"
)

p_pos_after <- plot_pca(
  pca_results_all[["POS_After"]]$pca_df,
  pca_results_all[["POS_After"]]$variance,
  title_text = "Positive ion mode - after correction"
)

save_plot_pdf_png(p_neg_after,  "Fig2_metabolomics_overview_combined_panel1_NEG_after_PCA", width = 7, height = 6)
save_plot_pdf_png(p_pos_after,  "Fig2_metabolomics_overview_combined_panel2_POS_after_PCA", width = 7, height = 6)

if (nrow(qc_cv_metrics) > 0) {
  p_qc_cv_hist <- plot_qc_cv(qc_cv_metrics, "QC feature CV distribution before vs after correction")
  save_plot_pdf_png(p_qc_cv_hist, "Fig2_QC_CV_distribution_before_after", width = 7, height = 6)
}

message("Analysis completed successfully.")
message("Summary tables written to: ", data_dir)
message("Figures written to      : ", figure_dir)

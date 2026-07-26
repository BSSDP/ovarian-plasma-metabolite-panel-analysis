# ============================================================
# 03A_global_plsda_umap_permanova.R
# 模块: 03_global_separation_and_pairwise_diff / 03A_global
# 目的:
#   1) 读取正负离子(以及可选 merged)批次矫正后的代谢矩阵
#   2) 去除 QC 样本，仅保留 N/B/BD/M 四组
#   3) 进行全局四组 PLS-DA、UMAP、PCA 可视化
#   4) 进行全局与两两 PERMANOVA
#   5) 输出主图 Fig2E 与补图 Sup_Fig
#
# 数据要求:
#   - 行: 样本
#   - 列: 特征
#   - 至少包含两列: "Alignment ID" 和 "label"
#   - label 至少包含: QC / N / B / BD / M
#
# 建议运行位置:
#   - 03A_global/scripts/ 目录下 source() 或 Rscript 运行
# ============================================================

options(stringsAsFactors = FALSE)

# -----------------------------
# 0. 依赖包
# -----------------------------
required_pkgs <- c(
  "ggplot2", "dplyr", "readr", "tibble", "patchwork", "vegan",
  "mixOmics", "uwot", "ggrepel", "openxlsx", "purrr", "tidyr", "stringr", "grid"
)

install_if_missing <- function(pkgs) {
  missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs) > 0) {
    message("Installing missing packages: ", paste(missing_pkgs, collapse = ", "))
    install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
  }
  invisible(lapply(pkgs, library, character.only = TRUE))
}
install_if_missing(required_pkgs)

# -----------------------------
# 1. 路径设置
# -----------------------------
get_root_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(dirname(normalizePath(sys.frames()[[1]]$ofile))))
  }
  wd <- getwd()
  if (basename(wd) == "scripts") return(dirname(wd))
  return(normalizePath(file.path(wd, "03A_global"), mustWork = FALSE))
}

root_dir   <- get_root_dir()
data_dir   <- file.path(root_dir, "data")
fig_dir    <- file.path(root_dir, "figure_final")
result_dir <- data_dir

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(result_dir, showWarnings = FALSE, recursive = TRUE)

message("Root dir   : ", root_dir)
message("Data dir   : ", data_dir)
message("Figure dir : ", fig_dir)

# -----------------------------
# 2. 颜色与工具函数
# -----------------------------
group_levels <- c("N", "B", "BD", "M")
style_file <- file.path(dirname(root_dir), "00_project_style", "ov_publication_style.R")
if (file.exists(style_file)) {
  source(style_file)
} else {
  stop("Missing project style file: ", style_file)
}
group_colors <- OV_GROUP_COLORS[group_levels]

safe_numeric <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

median_impute_matrix <- function(mat) {
  for (j in seq_len(ncol(mat))) {
    med_j <- median(mat[, j], na.rm = TRUE)
    if (!is.finite(med_j)) med_j <- 0
    mat[is.na(mat[, j]), j] <- med_j
  }
  mat
}

save_plot_both <- function(plot_obj, out_prefix, width = 7, height = 6, dpi = 320) {
  allowed_stems <- c(
    "NEG_PLSDA_global",
    "POS_PLSDA_global",
    "MERGED_PLSDA_global",
    "SupFig_MERGED_pairwise_PLSDA"
  )
  stem <- basename(out_prefix)
  if (!(stem %in% allowed_stems)) {
    message("Skipping deprecated 03A figure output: ", stem)
    return(invisible(NULL))
  }
  ggsave(filename = paste0(out_prefix, ".pdf"), plot = plot_obj,
         width = width, height = height, units = "in", device = cairo_pdf)
  ggsave(filename = paste0(out_prefix, ".png"), plot = plot_obj,
         width = width, height = height, units = "in", dpi = dpi)
}

# -----------------------------
# 3. 读取与预处理函数
# -----------------------------
read_metabolomics_matrix <- function(file_path, dataset_name) {
  message("\nReading: ", basename(file_path))
  df <- readr::read_csv(file_path, show_col_types = FALSE)

  required_cols <- c("Alignment ID", "label")
  if (!all(required_cols %in% colnames(df))) {
    stop(dataset_name, " 缺少必要列: ", paste(setdiff(required_cols, colnames(df)), collapse = ", "))
  }

  df <- df %>%
    dplyr::rename(sample_id = `Alignment ID`, group = label) %>%
    dplyr::mutate(
      sample_id = as.character(sample_id),
      group = stringr::str_trim(as.character(group)),
      group = toupper(group)
    )

  # 只保留四组样本，自动去掉 QC 和其他标签
  keep_idx <- df$group %in% group_levels
  removed_n <- sum(!keep_idx)
  if (removed_n > 0) {
    message("  Removed non-biological/QC samples: ", removed_n)
  }
  df_bio <- df[keep_idx, , drop = FALSE]

  # 特征矩阵
  feature_df <- df_bio %>% dplyr::select(-sample_id, -group)
  feature_df[] <- lapply(feature_df, safe_numeric)

  # 保留至少有一个非缺失值的特征
  keep_feature <- colSums(!is.na(feature_df)) > 0
  feature_df <- feature_df[, keep_feature, drop = FALSE]

  # 负值裁零（防止 log 出问题）
  feature_mat_raw <- as.matrix(feature_df)
  feature_mat_raw[is.na(feature_mat_raw)] <- NA
  feature_mat_raw[feature_mat_raw < 0] <- 0

  # 缺失统计（样本层面）
  sample_missing_rate <- rowMeans(is.na(feature_mat_raw))
  feature_missing_rate <- colMeans(is.na(feature_mat_raw))

  # 中位数填补 + log2(x + 1)
  feature_mat_imp <- median_impute_matrix(feature_mat_raw)
  feature_mat_log <- log2(feature_mat_imp + 1)

  # 去掉零方差特征
  sd_vec <- apply(feature_mat_log, 2, sd, na.rm = TRUE)
  keep_sd <- is.finite(sd_vec) & sd_vec > 0
  feature_mat_log <- feature_mat_log[, keep_sd, drop = FALSE]

  # 标准化，用于距离/PLS-DA/UMAP
  feature_mat_scaled <- scale(feature_mat_log, center = TRUE, scale = TRUE)
  rownames(feature_mat_scaled) <- df_bio$sample_id

  meta <- tibble::tibble(
    sample_id = df_bio$sample_id,
    group = factor(df_bio$group, levels = group_levels),
    dataset = dataset_name,
    sample_missing_rate = sample_missing_rate
  )

  list(
    dataset_name = dataset_name,
    meta = meta,
    X_log = feature_mat_log,
    X_scaled = feature_mat_scaled,
    n_samples = nrow(feature_mat_scaled),
    n_features = ncol(feature_mat_scaled),
    sample_missing_rate = sample_missing_rate,
    feature_missing_rate = feature_missing_rate
  )
}

# -----------------------------
# 4. 降维与统计
# -----------------------------
run_pca <- function(X, meta) {
  fit <- prcomp(X, center = FALSE, scale. = FALSE)
  coords <- as.data.frame(fit$x[, 1:2, drop = FALSE])
  colnames(coords) <- c("Dim1", "Dim2")
  coords <- dplyr::bind_cols(meta, coords)

  var_expl <- (fit$sdev^2) / sum(fit$sdev^2)
  attr(coords, "axis_labels") <- c(
    paste0("PC1 (", sprintf("%.1f", 100 * var_expl[1]), "%)"),
    paste0("PC2 (", sprintf("%.1f", 100 * var_expl[2]), "%)")
  )
  coords
}

run_plsda <- function(X, meta) {
  fit <- mixOmics::plsda(X = X, Y = meta$group, ncomp = 2)
  coords <- as.data.frame(fit$variates$X[, 1:2, drop = FALSE])
  colnames(coords) <- c("Dim1", "Dim2")
  coords <- dplyr::bind_cols(meta, coords)
  attr(coords, "axis_labels") <- c("PLS-DA component 1", "PLS-DA component 2")
  coords
}

run_umap_embed <- function(X, meta, seed = 1234) {
  set.seed(seed)
  nn_use <- max(5, min(15, nrow(X) - 1))
  emb <- uwot::umap(
    X,
    n_neighbors = nn_use,
    min_dist = 0.25,
    metric = "euclidean",
    verbose = FALSE,
    ret_model = FALSE,
    scale = FALSE
  )
  coords <- as.data.frame(emb)
  colnames(coords) <- c("Dim1", "Dim2")
  coords <- dplyr::bind_cols(meta, coords)
  attr(coords, "axis_labels") <- c("UMAP1", "UMAP2")
  coords
}

run_permanova_global <- function(X, meta, dataset_name, ordination_method) {
  dist_mat <- vegan::vegdist(X, method = "euclidean")
  fit <- vegan::adonis2(dist_mat ~ group, data = meta, permutations = 999)
  tibble::tibble(
    dataset = dataset_name,
    ordination_base = ordination_method,
    Df = fit$Df[1],
    SumOfSqs = fit$SumOfSqs[1],
    R2 = fit$R2[1],
    F = fit$F[1],
    p_value = fit$`Pr(>F)`[1]
  )
}

run_permanova_pairwise <- function(X, meta, dataset_name) {
  pair_list <- combn(group_levels, 2, simplify = FALSE)
  out <- purrr::map_dfr(pair_list, function(grp_pair) {
    idx <- meta$group %in% grp_pair
    meta_sub <- droplevels(meta[idx, , drop = FALSE])
    X_sub <- X[idx, , drop = FALSE]
    dist_sub <- vegan::vegdist(X_sub, method = "euclidean")
    fit <- vegan::adonis2(dist_sub ~ group, data = meta_sub, permutations = 999)
    tibble::tibble(
      dataset = dataset_name,
      group1 = grp_pair[1],
      group2 = grp_pair[2],
      Df = fit$Df[1],
      SumOfSqs = fit$SumOfSqs[1],
      R2 = fit$R2[1],
      F = fit$F[1],
      p_value = fit$`Pr(>F)`[1]
    )
  })

  out %>% dplyr::mutate(p_adj_bh = p.adjust(p_value, method = "BH"))
}

# -----------------------------
# 5. 绘图函数
# -----------------------------
make_ordination_plot <- function(coord_df, title_text, subtitle_text = NULL,
                                 point_size = 2.0, ellipse_level = 0.80,
                                 legend_position = "right") {
  axis_labels <- attr(coord_df, "axis_labels")
  if (is.null(axis_labels)) axis_labels <- c("Dim1", "Dim2")

  ggplot(coord_df, aes(x = Dim1, y = Dim2, color = group)) +
    stat_ellipse(
      aes(fill = group),
      geom = "polygon", type = "norm", level = ellipse_level,
      alpha = 0.10, color = NA, show.legend = FALSE
    ) +
    stat_ellipse(
      geom = "path", type = "norm", level = ellipse_level,
      linewidth = 0.6, alpha = 0.9, show.legend = FALSE
    ) +
    geom_point(size = point_size, alpha = 0.88) +
    scale_color_manual(values = group_colors, drop = FALSE) +
    scale_fill_manual(values = group_colors, drop = FALSE, guide = "none") +
    labs(
      title = title_text,
      subtitle = subtitle_text,
      x = axis_labels[1],
      y = axis_labels[2],
      color = NULL
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(size = 10),
      panel.grid = element_blank(),
      aspect.ratio = 1,
      legend.position = legend_position,
      legend.key = element_rect(fill = "white", color = NA)
    )
}

make_missingness_plot <- function(meta_df, dataset_name) {
  ggplot(meta_df, aes(x = group, y = sample_missing_rate, fill = group)) +
    geom_violin(trim = FALSE, alpha = 0.28, color = NA) +
    geom_boxplot(width = 0.16, outlier.shape = NA, alpha = 0.85) +
    geom_jitter(width = 0.12, height = 0, size = 1.2, alpha = 0.5) +
    scale_fill_manual(values = group_colors, drop = FALSE) +
    labs(
      title = paste0(dataset_name, ": sample missingness"),
      x = NULL, y = "Missing rate per sample"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid = element_blank(),
      legend.position = "none"
    )
}

make_group_legend_plot <- function() {
  legend_df <- tibble::tibble(
    group = factor(group_levels, levels = group_levels),
    x = 0,
    y = 0
  )

  legend_source <- ggplot(legend_df, aes(x = x, y = y, color = group)) +
    geom_point(size = 3.0) +
    scale_color_manual(values = group_colors, drop = FALSE) +
    labs(color = "group") +
    theme_void(base_size = 12) +
    theme(
      legend.position = "right",
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 11),
      legend.key.height = grid::unit(4.0, "mm"),
      legend.key.width = grid::unit(4.0, "mm"),
      legend.spacing.y = grid::unit(0.2, "mm"),
      legend.margin = margin(0, 0, 0, 0),
      plot.margin = margin(0, 0, 0, 0)
  )

  legend_grob <- ggplotGrob(legend_source)
  guide_index <- which(grepl("guide-box", legend_grob$layout$name))[1]
  if (is.na(guide_index)) {
    stop("Unable to extract compact group legend from ggplot object.")
  }
  patchwork::wrap_elements(full = legend_grob$grobs[[guide_index]])
}

# -----------------------------
# 6. 文件定义
# -----------------------------
file_map <- c(
  "NEG" = file.path(data_dir, "OV_4.9_neg_raw_clean_aftercor.csv"),
  "POS" = file.path(data_dir, "OV_4.9_pos_raw_clean_aftercor.csv"),
  "MERGED" = file.path(data_dir, "merged_deDuplicated_matrix.csv")
)

available_files <- file_map[file.exists(file_map)]
if (length(available_files) < 1) {
  stop("data/ 目录下未找到可用输入文件。")
}
message("\nAvailable input files: ")
print(available_files)

# -----------------------------
# 7. 主流程
# -----------------------------
analysis_list <- list()
global_perm_tbl <- list()
pairwise_perm_tbl <- list()
summary_tbl <- list()

for (dataset_name in names(available_files)) {
  file_path <- available_files[[dataset_name]]

  dat <- read_metabolomics_matrix(file_path, dataset_name = dataset_name)
  analysis_list[[dataset_name]] <- dat

  # 概览统计表
  group_count_tbl <- dat$meta %>%
    dplyr::count(group, name = "n") %>%
    dplyr::mutate(dataset = dataset_name)

  summary_tbl[[dataset_name]] <- tibble::tibble(
    dataset = dataset_name,
    n_samples = dat$n_samples,
    n_features = dat$n_features,
    mean_sample_missing_rate = mean(dat$sample_missing_rate, na.rm = TRUE),
    median_sample_missing_rate = median(dat$sample_missing_rate, na.rm = TRUE)
  )

  write.csv(group_count_tbl,
            file = file.path(result_dir, paste0(dataset_name, "_group_counts.csv")),
            row.names = FALSE)

  # PCA / PLS-DA / UMAP 坐标
  pca_coords   <- run_pca(dat$X_scaled, dat$meta)
  plsda_coords <- run_plsda(dat$X_scaled, dat$meta)
  umap_coords  <- run_umap_embed(dat$X_scaled, dat$meta)

  write.csv(pca_coords,
            file = file.path(result_dir, paste0(dataset_name, "_PCA_coordinates.csv")),
            row.names = FALSE)
  write.csv(plsda_coords,
            file = file.path(result_dir, paste0(dataset_name, "_PLSDA_coordinates.csv")),
            row.names = FALSE)
  write.csv(umap_coords,
            file = file.path(result_dir, paste0(dataset_name, "_UMAP_coordinates.csv")),
            row.names = FALSE)

  # PERMANOVA
  global_perm_tbl[[dataset_name]] <- run_permanova_global(
    X = dat$X_scaled,
    meta = dat$meta,
    dataset_name = dataset_name,
    ordination_method = "scaled log2 matrix"
  )
  pairwise_perm_tbl[[dataset_name]] <- run_permanova_pairwise(
    X = dat$X_scaled,
    meta = dat$meta,
    dataset_name = dataset_name
  )

  # 单数据集全局图
  p_pca <- make_ordination_plot(
    pca_coords,
    title_text = paste0(dataset_name, " PCA"),
    subtitle_text = paste0("n = ", dat$n_samples, "; features = ", dat$n_features)
  )
  p_pls <- make_ordination_plot(
    plsda_coords,
    title_text = paste0(dataset_name, " PLS-DA"),
    subtitle_text = NULL
  )
  p_umap <- make_ordination_plot(
    umap_coords,
    title_text = paste0(dataset_name, " UMAP"),
    subtitle_text = paste0("n_neighbors = ", max(5, min(15, dat$n_samples - 1)), "; min_dist = 0.25")
  )
  p_miss <- make_missingness_plot(dat$meta, dataset_name)

  save_plot_both(p_pca,  file.path(fig_dir, paste0(dataset_name, "_PCA_global")),  width = 6.4, height = 5.4)
  save_plot_both(p_pls,  file.path(fig_dir, paste0(dataset_name, "_PLSDA_global")), width = 6.4, height = 5.4)
  save_plot_both(p_umap, file.path(fig_dir, paste0(dataset_name, "_UMAP_global")),  width = 6.4, height = 5.4)
  save_plot_both(p_miss, file.path(fig_dir, paste0(dataset_name, "_sample_missingness")), width = 5.2, height = 4.8)

  # pairwise PLS-DA / UMAP 补图
  pair_list <- combn(group_levels, 2, simplify = FALSE)
  pair_pls_plots <- list()
  pair_umap_plots <- list()

  for (i in seq_along(pair_list)) {
    grp_pair <- pair_list[[i]]
    idx <- dat$meta$group %in% grp_pair
    meta_sub <- droplevels(dat$meta[idx, , drop = FALSE])
    X_sub <- dat$X_scaled[idx, , drop = FALSE]

    coords_pls_sub <- run_plsda(X_sub, meta_sub)
    coords_umap_sub <- run_umap_embed(X_sub, meta_sub, seed = 1234 + i)

    pair_title <- paste0(grp_pair[1], " vs ", grp_pair[2])

    pair_pls_plots[[i]] <- make_ordination_plot(
      coords_pls_sub,
      title_text = pair_title,
      subtitle_text = NULL,
      legend_position = "none"
    )
    pair_umap_plots[[i]] <- make_ordination_plot(
      coords_umap_sub,
      title_text = pair_title,
      subtitle_text = NULL,
      legend_position = "none"
    )

    write.csv(coords_pls_sub,
              file = file.path(result_dir, paste0(dataset_name, "_", grp_pair[1], "_vs_", grp_pair[2], "_PLSDA_coordinates.csv")),
              row.names = FALSE)
    write.csv(coords_umap_sub,
              file = file.path(result_dir, paste0(dataset_name, "_", grp_pair[1], "_vs_", grp_pair[2], "_UMAP_coordinates.csv")),
              row.names = FALSE)
  }

  legend_plot <- make_group_legend_plot()
  combined_pair_pls <- patchwork::wrap_plots(pair_pls_plots, ncol = 3) | legend_plot
  combined_pair_pls <- combined_pair_pls + patchwork::plot_layout(widths = c(1, 0.065))
  combined_pair_umap <- patchwork::wrap_plots(pair_umap_plots, ncol = 3) | legend_plot
  combined_pair_umap <- combined_pair_umap + patchwork::plot_layout(widths = c(1, 0.065))

  save_plot_both(combined_pair_pls,
                 file.path(fig_dir, paste0("SupFig_", dataset_name, "_pairwise_PLSDA")),
                 width = 15.2, height = 9.4)
  save_plot_both(combined_pair_umap,
                 file.path(fig_dir, paste0("SupFig_", dataset_name, "_pairwise_UMAP")),
                 width = 15.2, height = 9.4)
}

# -----------------------------
# 8. 汇总表输出
# -----------------------------
summary_df <- dplyr::bind_rows(summary_tbl)
global_perm_df <- dplyr::bind_rows(global_perm_tbl)
pairwise_perm_df <- dplyr::bind_rows(pairwise_perm_tbl)

write.csv(summary_df,
          file = file.path(result_dir, "global_ordination_dataset_summary.csv"),
          row.names = FALSE)
write.csv(global_perm_df,
          file = file.path(result_dir, "PERMANOVA_global_summary.csv"),
          row.names = FALSE)
write.csv(pairwise_perm_df,
          file = file.path(result_dir, "PERMANOVA_pairwise_summary.csv"),
          row.names = FALSE)

wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "dataset_summary")
openxlsx::writeData(wb, "dataset_summary", summary_df)
openxlsx::addWorksheet(wb, "PERMANOVA_global")
openxlsx::writeData(wb, "PERMANOVA_global", global_perm_df)
openxlsx::addWorksheet(wb, "PERMANOVA_pairwise")
openxlsx::writeData(wb, "PERMANOVA_pairwise", pairwise_perm_df)
openxlsx::saveWorkbook(wb,
                       file = file.path(result_dir, "ordination_and_PERMANOVA_summary.xlsx"),
                       overwrite = TRUE)

# -----------------------------
# 9. 组合主图 Fig2E
# -----------------------------
make_main_figure_grid <- function(analysis_list, method = c("PLSDA", "UMAP")) {
  method <- match.arg(method)
  plots <- list()

  for (nm in names(analysis_list)) {
    dat <- analysis_list[[nm]]
    coords <- switch(
      method,
      "PLSDA" = run_plsda(dat$X_scaled, dat$meta),
      "UMAP"  = run_umap_embed(dat$X_scaled, dat$meta, seed = 4321)
    )
    plots[[nm]] <- make_ordination_plot(
      coords,
      title_text = paste0(nm, " ", method),
      subtitle_text = paste0("n = ", dat$n_samples, "; features = ", dat$n_features)
    )
  }

  ncol_use <- ifelse(length(plots) >= 3, 3, length(plots))
  patchwork::wrap_plots(plots, ncol = ncol_use)
}

main_fig_pls <- make_main_figure_grid(analysis_list, method = "PLSDA")
main_fig_umap <- make_main_figure_grid(analysis_list, method = "UMAP")

fig2e <- main_fig_pls / main_fig_umap +
  patchwork::plot_annotation(
    title = "Fig2E. Global four-group separation across metabolomics datasets",
    subtitle = "Top row: PLS-DA; bottom row: UMAP; QC samples removed"
  )

save_plot_both(fig2e,
               file.path(fig_dir, "Fig2E_global_four_group_separation"),
               width = 16, height = 10)

# -----------------------------
# 10. 文本摘要
# -----------------------------
summary_txt <- file.path(result_dir, "ordination_run_summary.txt")
cat(
  "03A_global analysis finished.\n",
  "====================================\n",
  "Input files processed:\n",
  paste0("- ", names(available_files), ": ", basename(available_files), collapse = "\n"), "\n\n",
  "Key retained figure outputs:\n",
  "- NEG_PLSDA_global.[pdf/png]\n",
  "- POS_PLSDA_global.[pdf/png]\n",
  "- MERGED_PLSDA_global.[pdf/png]\n",
  "- SupFig_MERGED_pairwise_PLSDA.[pdf/png]\n",
  "- PERMANOVA_global_summary.csv\n",
  "- PERMANOVA_pairwise_summary.csv\n",
  "- ordination_and_PERMANOVA_summary.xlsx\n",
  file = summary_txt
)

message("\nAll done.")
message("Retained 03A figure outputs have been saved.")

OV_GROUP_COLORS <- c(
  Normal = "#80B1D3", Benign = "#8DD3C7", Borderline = "#FDB462", Malignant = "#FB8072",
  N = "#80B1D3", B = "#8DD3C7", BD = "#FDB462", M = "#FB8072"
)

OV_COHORT_COLORS <- c(
  "Discovery cohort" = "#8DD3C7",
  "Temporal same-centre validation cohort" = "#BEBADA"
)

OV_MODEL_COLORS <- c(
  "Baseline" = "#B3B3B3", "CA125-only" = "#B3B3B3",
  "Augmented" = "#FB8072", "Targeted model" = "#FB8072"
)

OV_DIRECTION_COLORS <- c(
  up = "#FB8072", down = "#80B1D3", positive = "#FB8072", negative = "#80B1D3",
  background = "#D9D9D9", missing = "#B3B3B3"
)

OV_PATHWAY_COLORS <- c(
  "Arginine/proline" = "#8DD3C7", "Phenylalanine" = "#FFFFB3",
  "Tryptophan" = "#BEBADA", "Steroid hormone" = "#FB8072"
)

OV_HEATMAP_BLUE <- "#2166AC"
OV_HEATMAP_WHITE <- "#F7F7F7"
OV_HEATMAP_RED <- "#B2182B"
OV_SIGNAL_RED <- "#FB8072"
OV_SIGNAL_BLUE <- "#80B1D3"
OV_BACKGROUND_GREY <- "#D9D9D9"
OV_TEXT_DARK <- "#000000"
OV_AXIS_GREY <- "#000000"

ov_theme <- function(base_size = 7, base_family = "Arial") {
  ggplot2::theme_classic(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      axis.line = ggplot2::element_line(linewidth = 0.35, colour = OV_AXIS_GREY),
      axis.ticks = ggplot2::element_line(linewidth = 0.35, colour = OV_AXIS_GREY),
      axis.text = ggplot2::element_text(colour = OV_TEXT_DARK, size = base_size - 0.5),
      axis.title = ggplot2::element_text(colour = OV_TEXT_DARK, size = base_size),
      legend.title = ggplot2::element_text(size = base_size - 0.5),
      legend.text = ggplot2::element_text(size = base_size - 0.7),
      plot.title = ggplot2::element_text(size = base_size + 0.5, face = "bold"),
      strip.text = ggplot2::element_text(size = base_size, face = "bold"),
      legend.background = ggplot2::element_blank(),
      panel.grid = ggplot2::element_blank()
    )
}

ov_save_pdf_png <- function(plot, file_stem, width = 5, height = 3.5, dpi = 600) {
  grDevices::pdf(paste0(file_stem, ".pdf"), width = width, height = height, family = "Arial", useDingbats = FALSE)
  print(plot)
  grDevices::dev.off()
  ggplot2::ggsave(paste0(file_stem, ".png"), plot, width = width, height = height, units = "in", dpi = dpi)
  if (requireNamespace("svglite", quietly = TRUE)) {
    svglite::svglite(paste0(file_stem, ".svg"), width = width, height = height)
    print(plot)
    grDevices::dev.off()
  }
}

#!/usr/bin/env Rscript

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

run_child <- function(script_path) {
  message("\n>>> Running ", basename(script_path))
  status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(script_path))
  if (!identical(status, 0L)) {
    stop("Child script failed: ", script_path, call. = FALSE)
  }
}

script_dir <- get_script_dir()

publication_scripts <- c(
  file.path(script_dir, "TCGA_GATM_survival_audit.R")
)

missing_scripts <- publication_scripts[!file.exists(publication_scripts)]
if (length(missing_scripts) > 0) {
  stop("Missing required publication script(s):\n", paste(missing_scripts, collapse = "\n"))
}

for (script_path in publication_scripts) {
  run_child(script_path)
}

message("\nTCGA publication-support workflow complete.")
message("- Publication figures: ", file.path(script_dir, "figure"))
message("- GATM exploratory survival source tables: ", file.path(script_dir, "tables"))

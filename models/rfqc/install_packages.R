script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_argument)) {
  sub("^--file=", "", script_argument[[1L]])
} else {
  "models/rfqc/install_packages.R"
}
project_root <- normalizePath(
  file.path(dirname(script_path), "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
library_path <- file.path(project_root, ".r-library")
dir.create(library_path, recursive = TRUE, showWarnings = FALSE)

packages <- c(
  "randomForestSRC",
  "arrow",
  "data.table",
  "jsonlite",
  "digest",
  "tidyselect"
)

install.packages(
  packages,
  lib = library_path,
  repos = "https://cloud.r-project.org",
  dependencies = c("Depends", "Imports", "LinkingTo"),
  Ncpus = max(1L, parallel::detectCores() - 1L)
)

.libPaths(c(library_path, .libPaths()))
for (package in packages) {
  cat(
    package,
    requireNamespace(package, quietly = TRUE),
    as.character(utils::packageVersion(package)),
    "\n"
  )
}

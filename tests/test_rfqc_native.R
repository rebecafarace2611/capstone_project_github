script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- sub("^--file=", "", script_argument[[1L]])
project_root <- normalizePath(
  file.path(dirname(script_path), ".."),
  winslash = "/",
  mustWork = TRUE
)
.libPaths(c(file.path(project_root, ".r-library"), .libPaths()))
source(file.path(project_root, "models", "rfqc", "workflow.R"))

target <- factor(
  c(0, 0, 0, 1, 1, 1),
  levels = c(0, 1),
  labels = c(NEGATIVE_CLASS, POSITIVE_CLASS)
)
probability <- c(0.05, 0.20, 0.40, 0.35, 0.70, 0.90)
optimized <- optimize_gmean_threshold(target, probability)
stopifnot(identical(optimized$best$threshold[[1L]], 0.35))
stopifnot(abs(optimized$best$sensitivity[[1L]] - 1) < 1e-12)
stopifnot(abs(optimized$best$specificity[[1L]] - 2 / 3) < 1e-12)

metrics <- classification_metrics(
  factor(
    c(0, 0, 1, 1),
    levels = c(0, 1),
    labels = c(NEGATIVE_CLASS, POSITIVE_CLASS)
  ),
  c(0.1, 0.8, 0.4, 0.9),
  0.5
)
stopifnot(metrics$tn == 1L)
stopifnot(metrics$fp == 1L)
stopifnot(metrics$fn == 1L)
stopifnot(metrics$tp == 1L)
stopifnot(abs(metrics$gmean - 0.5) < 1e-12)
stopifnot(abs(metrics$p4 - 0.0625) < 1e-12)
stopifnot(abs(metrics$fpr - 0.5) < 1e-12)
stopifnot(abs(metrics$fnr - 0.5) < 1e-12)

training_features <- prepare_feature_frame(
  data.frame(category = c("b", "a"), value = c(1, 2)),
  c("category", "value")
)
training_levels <- attr(training_features, "categorical_levels")
prediction_features <- prepare_feature_frame(
  data.frame(category = c("a", "new"), value = c(3, 4)),
  c("category", "value"),
  categorical_levels = training_levels
)
stopifnot(identical(
  levels(prediction_features$category),
  c("a", "b", "new")
))

curve <- threshold_curve(target, probability)
stopifnot(all(curve$quantile >= 0 & curve$quantile < 1))
stopifnot(all(diff(curve$quantile) <= 0))

grid <- make_parameter_grid("full")
stopifnot(nrow(grid) == 64L)
stopifnot(setequal(unique(grid$splitrule), c("gini", "auc")))

auc_grid <- make_parameter_grid("auc", tuning_trees = 1000L)
stopifnot(nrow(auc_grid) == 32L)
stopifnot(identical(unique(auc_grid$splitrule), "auc"))
stopifnot(identical(unique(auc_grid$ntree), 1000L))

local_grid <- make_parameter_grid("local")
stopifnot(nrow(local_grid) == 4L)
stopifnot(setequal(unique(local_grid$mtry), c(24L, 48L)))
stopifnot(setequal(unique(local_grid$nodesize), c(10L, 20L)))
stopifnot(identical(unique(local_grid$nsplit), 10L))
stopifnot(identical(unique(local_grid$splitrule), "gini"))
stopifnot(identical(unique(local_grid$ntree), 500L))

cat("Native RFQ workflow tests passed.\n")

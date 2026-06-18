script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_argument)) {
  sub("^--file=", "", script_argument[[1L]])
} else {
  "models/rfqc/run_shap.R"
}
PROJECT_ROOT <- normalizePath(
  file.path(dirname(script_path), "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
.libPaths(c(file.path(PROJECT_ROOT, ".r-library"), .libPaths()))
source(file.path(PROJECT_ROOT, "models", "rfqc", "workflow.R"))

normalize_arg_name <- function(name) {
  gsub("-", "_", sub("^--", "", name), fixed = TRUE)
}

parse_args <- function(defaults) {
  raw <- commandArgs(trailingOnly = TRUE)
  args <- defaults
  index <- 1L
  while (index <= length(raw)) {
    token <- raw[[index]]
    if (!startsWith(token, "--")) {
      stop("Unexpected positional argument: ", token)
    }
    if (grepl("=", token, fixed = TRUE)) {
      pieces <- strsplit(sub("^--", "", token), "=", fixed = TRUE)[[1L]]
      args[[gsub("-", "_", pieces[[1L]], fixed = TRUE)]] <- pieces[[2L]]
    } else {
      key <- normalize_arg_name(token)
      next_index <- index + 1L
      if (next_index > length(raw) || startsWith(raw[[next_index]], "--")) {
        args[[key]] <- TRUE
      } else {
        args[[key]] <- raw[[next_index]]
        index <- next_index
      }
    }
    index <- index + 1L
  }
  args
}

as_integer_arg <- function(args, key) {
  value <- suppressWarnings(as.integer(args[[key]]))
  if (is.na(value) || value <= 0L) {
    stop("--", gsub("_", "-", key, fixed = TRUE), " must be a positive integer.")
  }
  value
}

require_package <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(
      "Missing R package '", package, "'. Run: bash models/rfqc/run_autodl.sh install"
    )
  }
}

write_parquet_atomic <- function(frame, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  arrow::write_parquet(frame, temporary, compression = "snappy")
  if (file.exists(path)) {
    file.remove(path)
  }
  if (!file.rename(temporary, path)) {
    stop("Could not atomically write Parquet file: ", path)
  }
}

sample_values <- function(values, size) {
  if (size <= 0L || !length(values)) {
    return(values[0L])
  }
  values[sample.int(length(values), size)]
}

select_explain_rows <- function(
  selection,
  explain_size,
  features,
  target = NULL,
  prediction_file = NULL
) {
  row_count <- nrow(features)
  if (identical(selection, "random")) {
    return(sort(sample.int(row_count, min(explain_size, row_count))))
  }

  if (identical(selection, "actual_positive")) {
    if (is.null(target)) {
      stop("selection=actual_positive requires a dataset with the target column.")
    }
    candidates <- which(target == POSITIVE_CLASS)
    return(sort(sample_values(candidates, min(explain_size, length(candidates)))))
  }

  if (identical(selection, "stratified")) {
    if (is.null(target)) {
      stop("selection=stratified requires a dataset with the target column.")
    }
    positives <- which(target == POSITIVE_CLASS)
    negatives <- which(target == NEGATIVE_CLASS)
    positive_n <- min(length(positives), ceiling(explain_size / 2))
    negative_n <- min(length(negatives), explain_size - positive_n)
    selected <- c(
      sample_values(positives, positive_n),
      sample_values(negatives, negative_n)
    )
    if (length(selected) < min(explain_size, row_count)) {
      remainder <- setdiff(seq_len(row_count), selected)
      selected <- c(
        selected,
        sample_values(
          remainder,
          min(length(remainder), explain_size - length(selected))
        )
      )
    }
    return(sort(selected))
  }

  if (!selection %in% c("top_score", "predicted_positive")) {
    stop(
      "--selection must be one of random, top_score, predicted_positive, ",
      "actual_positive, stratified."
    )
  }
  if (is.null(prediction_file) || !file.exists(prediction_file)) {
    stop("selection=", selection, " requires --prediction-file.")
  }
  predictions <- arrow::read_parquet(prediction_file, as_data_frame = TRUE)
  required <- c("row_index", "fraud_probability", "primary_prediction")
  missing <- setdiff(required, names(predictions))
  if (length(missing)) {
    stop("Prediction file is missing columns: ", paste(missing, collapse = ", "))
  }
  if (nrow(predictions) != row_count) {
    stop("Prediction file row count does not match the explanation dataset.")
  }
  if (identical(selection, "top_score")) {
    ordered <- predictions[order(-predictions$fraud_probability), , drop = FALSE]
  } else {
    ordered <- predictions[predictions$primary_prediction == 1L, , drop = FALSE]
    ordered <- ordered[order(-ordered$fraud_probability), , drop = FALSE]
  }
  selected <- ordered$row_index[seq_len(min(explain_size, nrow(ordered)))] + 1L
  sort(as.integer(selected))
}

coerce_prediction_features <- function(frame, approved_features, categorical_levels) {
  frame <- as.data.frame(frame[, approved_features, drop = FALSE])
  categorical <- intersect(names(categorical_levels), approved_features)
  for (feature in categorical) {
    frame[[feature]] <- factor(
      as.character(frame[[feature]]),
      levels = categorical_levels[[feature]]
    )
  }
  for (feature in setdiff(approved_features, categorical)) {
    frame[[feature]] <- as.numeric(frame[[feature]])
  }
  if (anyNA(frame)) {
    stop("SHAP prediction frame contains missing or invalid values after coercion.")
  }
  frame
}

build_feature_value_long <- function(explain_frame, row_metadata, approved_features) {
  values <- data.table::as.data.table(explain_frame[, approved_features, drop = FALSE])
  values[, shap_row_id := seq_len(.N)]
  long <- data.table::melt(
    values,
    id.vars = "shap_row_id",
    variable.name = "feature",
    value.name = "feature_value",
    variable.factor = FALSE
  )
  long[, feature_value := as.character(feature_value)]
  long[row_metadata, source_row_index := i.source_row_index, on = "shap_row_id"]
  data.table::setcolorder(
    long,
    c("shap_row_id", "source_row_index", "feature", "feature_value")
  )
  long[]
}

defaults <- list(
  model = file.path(PROJECT_ROOT, "runs", "final_local_qstar_3000", "rfqc_native_model.rds"),
  train = file.path(PROJECT_ROOT, "data", "train_model_dataset.parquet"),
  test = file.path(PROJECT_ROOT, "data", "test_model_dataset.parquet"),
  approved_features = file.path(PROJECT_ROOT, "outputs", "leakage_analysis", "approved_features.json"),
  prediction_file = file.path(PROJECT_ROOT, "runs", "final_local_qstar_3000", "final_test_predictions.parquet"),
  output_dir = file.path(PROJECT_ROOT, "runs", "final_local_qstar_3000", "shap_top_score"),
  dataset = "test",
  selection = "top_score",
  background_size = "1000",
  explain_size = "500",
  nsim = "20",
  chunk_size = "100",
  seed = "42"
)
args <- parse_args(defaults)

require_package("arrow")
require_package("data.table")
require_package("fastshap")
require_package("jsonlite")
require_package("randomForestSRC")
require_package("tidyselect")

background_size <- as_integer_arg(args, "background_size")
explain_size <- as_integer_arg(args, "explain_size")
nsim <- as_integer_arg(args, "nsim")
chunk_size <- as_integer_arg(args, "chunk_size")
seed <- as_integer_arg(args, "seed")
set.seed(seed)

for (path in c(args$model, args$train, args$approved_features)) {
  if (!file.exists(path)) {
    stop("Missing required file: ", path)
  }
}
if (identical(args$dataset, "test") && !file.exists(args$test)) {
  stop("Missing test dataset: ", args$test)
}
if (!args$dataset %in% c("test", "train")) {
  stop("--dataset must be either test or train.")
}

dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
approved_features <- load_approved_features(args$approved_features)

cat("Loading training frame for background rows.\n")
training <- load_training_frame(args$train, approved_features)
training_feature_frame <- training[, approved_features, drop = FALSE]
training_levels <- attr(training_feature_frame, "categorical_levels")
background_indices <- sort(sample.int(
  nrow(training_feature_frame),
  min(background_size, nrow(training_feature_frame))
))
background <- training_feature_frame[background_indices, , drop = FALSE]

cat("Loading explanation dataset.\n")
if (identical(args$dataset, "test")) {
  evaluation <- load_evaluation_frame(
    args$test,
    approved_features,
    training_levels
  )
  explain_pool <- evaluation$features
  explain_target <- evaluation$target
} else {
  explain_pool <- training_feature_frame
  explain_target <- training[[TARGET_COLUMN]]
}

selected_indices <- select_explain_rows(
  args$selection,
  explain_size,
  explain_pool,
  explain_target,
  args$prediction_file
)
explain_frame <- explain_pool[selected_indices, approved_features, drop = FALSE]
row_metadata <- data.table::data.table(
  shap_row_id = seq_along(selected_indices),
  source_row_index = selected_indices - 1L,
  dataset = args$dataset,
  target = as.character(explain_target[selected_indices])
)

if (file.exists(args$prediction_file) && identical(args$dataset, "test")) {
  prediction_rows <- arrow::read_parquet(args$prediction_file, as_data_frame = TRUE)
  prediction_rows <- data.table::as.data.table(prediction_rows)
  prediction_rows[, source_row_index := as.integer(row_index)]
  row_metadata <- prediction_rows[
    row_metadata,
    on = "source_row_index"
  ][
    ,
    .(
      shap_row_id,
      source_row_index,
      dataset,
      target,
      fraud_probability,
      primary_prediction
    )
  ]
}

combined_levels <- lapply(names(training_levels), function(feature) {
  unique(c(
    training_levels[[feature]],
    as.character(background[[feature]]),
    as.character(explain_frame[[feature]])
  ))
})
names(combined_levels) <- names(training_levels)
background <- coerce_prediction_features(background, approved_features, combined_levels)
explain_frame <- coerce_prediction_features(explain_frame, approved_features, combined_levels)

rm(training, training_feature_frame, explain_pool)
if (exists("evaluation")) {
  rm(evaluation)
}
gc()

cat("Loading final forest model. This can take several minutes for the 37 GB RDS.\n")
model <- readRDS(args$model)
if (!inherits(model, "forest")) {
  stop("The saved model is not a randomForestSRC forest object.")
}

predict_positive <- function(object, newdata) {
  newdata <- coerce_prediction_features(newdata, approved_features, combined_levels)
  prediction <- predict(
    object,
    newdata,
    perf.type = "none",
    block.size = NULL
  )
  positive_probabilities(prediction$predicted)
}

baseline <- mean(predict_positive(model, background))
cat("Background baseline fraud probability:", baseline, "\n")

shap_chunks <- list()
chunk_starts <- seq(1L, nrow(explain_frame), by = chunk_size)
for (chunk_id in seq_along(chunk_starts)) {
  start <- chunk_starts[[chunk_id]]
  end <- min(start + chunk_size - 1L, nrow(explain_frame))
  cat("Computing SHAP chunk", chunk_id, "rows", start, "to", end, "\n")
  chunk <- explain_frame[start:end, approved_features, drop = FALSE]
  shap_values <- fastshap::explain(
    object = model,
    X = background,
    newdata = chunk,
    pred_wrapper = predict_positive,
    nsim = nsim,
    adjust = TRUE,
    baseline = baseline,
    shap_only = TRUE
  )
  shap_values <- as.data.frame(shap_values, check.names = FALSE)
  shap_values$shap_row_id <- seq(from = start, to = end)
  shap_chunks[[chunk_id]] <- shap_values
  gc()
}

shap_wide <- data.table::rbindlist(shap_chunks, fill = TRUE)
data.table::setcolorder(shap_wide, c("shap_row_id", approved_features))
shap_wide[row_metadata, source_row_index := i.source_row_index, on = "shap_row_id"]
data.table::setcolorder(shap_wide, c("shap_row_id", "source_row_index", approved_features))

shap_long <- data.table::melt(
  shap_wide,
  id.vars = c("shap_row_id", "source_row_index"),
  measure.vars = approved_features,
  variable.name = "feature",
  value.name = "shap_value",
  variable.factor = FALSE
)

summary <- shap_long[
  ,
  .(
    mean_abs_shap = mean(abs(shap_value)),
    mean_shap = mean(shap_value),
    sd_shap = stats::sd(shap_value),
    min_shap = min(shap_value),
    q25_shap = as.numeric(stats::quantile(shap_value, 0.25)),
    median_shap = stats::median(shap_value),
    q75_shap = as.numeric(stats::quantile(shap_value, 0.75)),
    max_shap = max(shap_value),
    positive_share = mean(shap_value > 0)
  ),
  by = feature
]
summary <- summary[order(-mean_abs_shap)]
summary[, rank := seq_len(.N)]
data.table::setcolorder(summary, c("rank", "feature", setdiff(names(summary), c("rank", "feature"))))

feature_values_long <- build_feature_value_long(
  explain_frame,
  row_metadata,
  approved_features
)

write_parquet_atomic(shap_wide, file.path(args$output_dir, "shap_values_wide.parquet"))
write_parquet_atomic(shap_long, file.path(args$output_dir, "shap_values_long.parquet"))
write_parquet_atomic(feature_values_long, file.path(args$output_dir, "shap_feature_values_long.parquet"))
write_csv_atomic(file.path(args$output_dir, "shap_summary.csv"), summary)
write_csv_atomic(file.path(args$output_dir, "shap_explain_rows.csv"), row_metadata)
write_json_atomic(
  file.path(args$output_dir, "shap_metadata.json"),
  list(
    generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    implementation = "fastshap_model_agnostic_randomForestSRC_RFQ",
    model = normalizePath(args$model, winslash = "/", mustWork = TRUE),
    train = normalizePath(args$train, winslash = "/", mustWork = TRUE),
    test = if (identical(args$dataset, "test")) {
      normalizePath(args$test, winslash = "/", mustWork = TRUE)
    } else {
      NULL
    },
    approved_features = normalizePath(args$approved_features, winslash = "/", mustWork = TRUE),
    prediction_file = if (file.exists(args$prediction_file)) {
      normalizePath(args$prediction_file, winslash = "/", mustWork = TRUE)
    } else {
      NULL
    },
    dataset = args$dataset,
    selection = args$selection,
    background_size = nrow(background),
    explain_size = nrow(explain_frame),
    nsim = nsim,
    chunk_size = chunk_size,
    seed = seed,
    baseline_fraud_probability = baseline,
    feature_count = length(approved_features),
    output_files = list(
      shap_values_wide = "shap_values_wide.parquet",
      shap_values_long = "shap_values_long.parquet",
      shap_feature_values_long = "shap_feature_values_long.parquet",
      shap_summary = "shap_summary.csv",
      shap_explain_rows = "shap_explain_rows.csv"
    )
  )
)

cat("SHAP outputs written to", normalizePath(args$output_dir, winslash = "/"), "\n")

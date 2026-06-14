TARGET_COLUMN <- "respuesta_dicot_c"
NEGATIVE_CLASS <- "nonfraud"
POSITIVE_CLASS <- "fraud"

NUMERIC_CATEGORICAL_FEATURES <- c(
  "comarcaid",
  "tomadorcodigopostal",
  "tomadormunicipioid",
  "tomadorprovinciaid",
  "tomadorcomarcaid",
  "tomadornacionalidadid"
)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x)) y else x
}

write_json_atomic <- function(path, payload) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  jsonlite::write_json(
    payload,
    temporary,
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null",
    na = "null",
    digits = 16
  )
  if (file.exists(path)) {
    file.remove(path)
  }
  if (!file.rename(temporary, path)) {
    stop("Could not atomically write JSON file: ", path)
  }
}

write_csv_atomic <- function(path, frame) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp")
  data.table::fwrite(frame, temporary)
  if (file.exists(path)) {
    file.remove(path)
  }
  if (!file.rename(temporary, path)) {
    stop("Could not atomically write CSV file: ", path)
  }
}

load_approved_features <- function(path) {
  payload <- jsonlite::read_json(path, simplifyVector = TRUE)
  features <- payload$approved_features
  if (is.null(features) || !length(features)) {
    stop("No approved feature list found in ", path)
  }
  if (TARGET_COLUMN %in% features) {
    stop("The target must not appear in approved_features.")
  }
  if (anyDuplicated(features)) {
    stop("The approved feature list contains duplicates.")
  }
  as.character(features)
}

normalize_target <- function(values) {
  numeric_values <- suppressWarnings(as.integer(as.character(values)))
  if (anyNA(numeric_values) || !all(numeric_values %in% c(0L, 1L))) {
    stop("Target must contain only 0 and 1.")
  }
  factor(
    numeric_values,
    levels = c(0L, 1L),
    labels = c(NEGATIVE_CLASS, POSITIVE_CLASS)
  )
}

prepare_feature_frame <- function(
  frame,
  approved_features,
  categorical_levels = NULL
) {
  missing <- setdiff(approved_features, names(frame))
  if (length(missing)) {
    stop("Approved features missing from model frame: ", paste(missing, collapse = ", "))
  }

  frame <- as.data.frame(frame[, approved_features, drop = FALSE])
  categorical <- if (is.null(categorical_levels)) {
    unique(c(
      names(frame)[vapply(
        frame,
        function(x) is.character(x) || is.factor(x),
        logical(1)
      )],
      intersect(NUMERIC_CATEGORICAL_FEATURES, approved_features)
    ))
  } else {
    intersect(names(categorical_levels), approved_features)
  }
  for (feature in categorical) {
    if (is.null(categorical_levels)) {
      frame[[feature]] <- factor(frame[[feature]])
    } else {
      values <- as.character(frame[[feature]])
      unseen <- setdiff(unique(values), categorical_levels[[feature]])
      frame[[feature]] <- factor(
        values,
        levels = unique(c(categorical_levels[[feature]], unseen))
      )
    }
  }
  for (feature in setdiff(approved_features, categorical)) {
    frame[[feature]] <- as.numeric(frame[[feature]])
  }
  if (anyNA(frame)) {
    stop("Prepared model features contain missing or invalid values.")
  }
  attr(frame, "categorical_features") <- categorical
  attr(frame, "categorical_levels") <- lapply(frame[categorical], levels)
  frame
}

load_training_frame <- function(path, approved_features) {
  columns <- c(approved_features, TARGET_COLUMN)
  frame <- arrow::read_parquet(
    path,
    col_select = tidyselect::all_of(columns),
    as_data_frame = TRUE
  )
  if (anyNA(frame)) {
    stop("Training data contain missing values.")
  }
  target <- normalize_target(frame[[TARGET_COLUMN]])
  features <- prepare_feature_frame(frame, approved_features)
  features[[TARGET_COLUMN]] <- target
  features
}

load_feature_frame <- function(path, approved_features) {
  frame <- arrow::read_parquet(
    path,
    col_select = tidyselect::all_of(approved_features),
    as_data_frame = TRUE
  )
  if (anyNA(frame)) {
    stop("Prediction features contain missing values.")
  }
  prepare_feature_frame(frame, approved_features)
}

load_target_only <- function(path) {
  frame <- arrow::read_parquet(
    path,
    col_select = tidyselect::all_of(TARGET_COLUMN),
    as_data_frame = TRUE
  )
  normalize_target(frame[[TARGET_COLUMN]])
}

load_evaluation_frame <- function(
  path,
  approved_features,
  categorical_levels
) {
  columns <- c(approved_features, TARGET_COLUMN)
  frame <- arrow::read_parquet(
    path,
    col_select = tidyselect::all_of(columns),
    as_data_frame = TRUE
  )
  if (anyNA(frame)) {
    stop("Test data contain missing values.")
  }
  target <- normalize_target(frame[[TARGET_COLUMN]])
  features <- prepare_feature_frame(
    frame,
    approved_features,
    categorical_levels = categorical_levels
  )
  list(features = features, target = target)
}

positive_probabilities <- function(prediction_matrix) {
  if (is.null(dim(prediction_matrix))) {
    stop("RFQ predictions are not a probability matrix.")
  }
  positive_position <- match(POSITIVE_CLASS, colnames(prediction_matrix))
  if (is.na(positive_position)) {
    stop("RFQ probability matrix does not contain class ", POSITIVE_CLASS)
  }
  as.numeric(prediction_matrix[, positive_position])
}

binary_target <- function(target) {
  values <- as.integer(target == POSITIVE_CLASS)
  if (!all(values %in% c(0L, 1L)) || length(unique(values)) != 2L) {
    stop("Both target classes are required.")
  }
  values
}

validate_probabilities <- function(target, probabilities) {
  y <- binary_target(target)
  scores <- as.numeric(probabilities)
  if (length(y) != length(scores)) {
    stop("Target and probabilities must have equal length.")
  }
  valid <- is.finite(scores) & scores >= 0 & scores <= 1
  if (!all(valid)) {
    stop("Probabilities contain invalid values.")
  }
  list(target = y, scores = scores)
}

threshold_curve <- function(target, probabilities) {
  validated <- validate_probabilities(target, probabilities)
  y <- validated$target
  scores <- validated$scores
  order_index <- order(scores, decreasing = TRUE, method = "radix")
  sorted_scores <- scores[order_index]
  sorted_target <- y[order_index]

  positives <- sum(sorted_target)
  negatives <- length(sorted_target) - positives
  cumulative_tp <- cumsum(sorted_target)
  cumulative_fp <- cumsum(1L - sorted_target)
  last_for_score <- c(sorted_scores[-1L] != sorted_scores[-length(sorted_scores)], TRUE)
  last_positions <- which(last_for_score)

  threshold <- sorted_scores[last_for_score]
  tp <- cumulative_tp[last_for_score]
  fp <- cumulative_fp[last_for_score]
  fn <- positives - tp
  tn <- negatives - fp
  sensitivity <- tp / positives
  specificity <- tn / negatives

  data.frame(
    threshold = threshold,
    quantile = (length(scores) - last_positions) / length(scores),
    tn = tn,
    fp = fp,
    fn = fn,
    tp = tp,
    sensitivity = sensitivity,
    specificity = specificity,
    fpr = 1 - specificity,
    fnr = 1 - sensitivity,
    precision = ifelse((tp + fp) > 0, tp / (tp + fp), 0),
    gmean = sqrt(sensitivity * specificity),
    informedness = sensitivity + specificity - 1,
    balance_gap = abs(sensitivity - specificity)
  )
}

optimize_gmean_threshold <- function(target, probabilities) {
  curve <- threshold_curve(target, probabilities)
  ranked <- curve[
    order(
      -curve$gmean,
      -curve$sensitivity,
      -curve$specificity,
      curve$threshold,
      method = "radix"
    ),
  ]
  list(best = ranked[1L, , drop = FALSE], curve = curve[order(curve$threshold), ])
}

roc_auc_score <- function(target, probabilities) {
  y <- binary_target(target)
  positive <- y == 1L
  negative <- !positive
  ranks <- rank(probabilities, ties.method = "average")
  (sum(ranks[positive]) - sum(seq_len(sum(positive)))) /
    (sum(positive) * sum(negative))
}

pr_auc_score <- function(target, probabilities) {
  y <- binary_target(target)
  order_index <- order(probabilities, decreasing = TRUE, method = "radix")
  sorted_target <- y[order_index]
  cumulative_tp <- cumsum(sorted_target)
  precision <- cumulative_tp / seq_along(sorted_target)
  sum(precision[sorted_target == 1L]) / sum(sorted_target)
}

classification_metrics <- function(target, probabilities, threshold) {
  validated <- validate_probabilities(target, probabilities)
  y <- validated$target
  scores <- validated$scores
  predictions <- as.integer(scores >= threshold)

  tn <- sum(y == 0L & predictions == 0L)
  fp <- sum(y == 0L & predictions == 1L)
  fn <- sum(y == 1L & predictions == 0L)
  tp <- sum(y == 1L & predictions == 1L)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
  npv <- if ((tn + fn) > 0) tn / (tn + fn) else 0
  f1 <- if ((2 * tp + fp + fn) > 0) 2 * tp / (2 * tp + fp + fn) else 0

  list(
    threshold = as.numeric(threshold),
    tn = as.integer(tn),
    fp = as.integer(fp),
    fn = as.integer(fn),
    tp = as.integer(tp),
    sensitivity = sensitivity,
    specificity = specificity,
    fpr = 1 - specificity,
    fnr = 1 - sensitivity,
    gmean = sqrt(sensitivity * specificity),
    precision = precision,
    npv = npv,
    f1 = f1,
    p4 = sensitivity * specificity * precision * npv,
    informedness = sensitivity + specificity - 1,
    markedness = precision + npv - 1,
    roc_auc = roc_auc_score(target, scores),
    pr_auc = pr_auc_score(target, scores),
    predicted_positive = as.integer(sum(predictions)),
    predicted_positive_rate = mean(predictions),
    balance_gap = abs(sensitivity - specificity)
  )
}

threshold_for_rule <- function(rule, target, oob_probabilities) {
  if (identical(rule, "gmean_optimized")) {
    optimized <- optimize_gmean_threshold(target, oob_probabilities)
    return(list(
      threshold = optimized$best$threshold[[1L]],
      quantile = optimized$best$quantile[[1L]],
      training_gmean = optimized$best$gmean[[1L]],
      curve = optimized$curve
    ))
  }
  if (identical(rule, "q_star_prevalence")) {
    threshold <- mean(target == POSITIVE_CLASS)
    metrics <- classification_metrics(target, oob_probabilities, threshold)
    return(list(
      threshold = threshold,
      quantile = mean(oob_probabilities < threshold),
      training_gmean = metrics$gmean,
      curve = NULL
    ))
  }
  stop("Unknown threshold rule: ", rule)
}

rfq_formula <- function(approved_features) {
  reformulate(approved_features, response = TARGET_COLUMN)
}

fit_rfq <- function(data, approved_features, parameters, seed) {
  randomForestSRC::imbalanced(
    rfq_formula(approved_features),
    data = data,
    ntree = as.integer(parameters$ntree),
    method = "rfq",
    splitrule = as.character(parameters$splitrule),
    perf.type = "gmean",
    block.size = NULL,
    mtry = as.integer(parameters$mtry),
    nodesize = as.integer(parameters$nodesize),
    nsplit = as.integer(parameters$nsplit),
    importance = FALSE,
    forest = TRUE,
    seed = as.integer(seed)
  )
}

make_parameter_grid <- function(profile, tuning_trees = 0L) {
  if (identical(profile, "quick")) {
    grid <- expand.grid(
      mtry = c(12L, 24L),
      nodesize = c(1L, 10L),
      nsplit = 10L,
      splitrule = c("gini", "auc"),
      stringsAsFactors = FALSE
    )
    grid$ntree <- 500L
  } else if (identical(profile, "local")) {
    grid <- expand.grid(
      mtry = c(24L, 48L),
      nodesize = c(10L, 20L),
      nsplit = 10L,
      splitrule = "gini",
      stringsAsFactors = FALSE
    )
    grid$ntree <- 500L
  } else if (identical(profile, "paper")) {
    grid <- expand.grid(
      mtry = c(12L, 24L, 48L, 72L),
      nodesize = c(1L, 5L, 10L, 20L),
      nsplit = c(10L, 25L),
      splitrule = "gini",
      stringsAsFactors = FALSE
    )
    grid$ntree <- 3000L
  } else if (identical(profile, "auc")) {
    grid <- expand.grid(
      mtry = c(12L, 24L, 48L, 72L),
      nodesize = c(1L, 5L, 10L, 20L),
      nsplit = c(10L, 25L),
      splitrule = "auc",
      stringsAsFactors = FALSE
    )
    grid$ntree <- 3000L
  } else if (identical(profile, "full")) {
    grid <- expand.grid(
      mtry = c(12L, 24L, 48L, 72L),
      nodesize = c(1L, 5L, 10L, 20L),
      nsplit = c(10L, 25L),
      splitrule = c("gini", "auc"),
      stringsAsFactors = FALSE
    )
    grid$ntree <- 3000L
  } else {
    stop("Unknown tuning profile: ", profile)
  }
  tuning_trees <- as.integer(tuning_trees)
  if (tuning_trees > 0L) {
    grid$ntree <- tuning_trees
  }
  grid$candidate <- seq_len(nrow(grid))
  grid[, c("candidate", "ntree", "mtry", "nodesize", "nsplit", "splitrule")]
}

rank_cv_results <- function(results) {
  grouping <- c(
    "candidate", "ntree", "mtry", "nodesize", "nsplit",
    "splitrule", "threshold_rule"
  )
  summary <- results[, .(
    completed_folds = data.table::uniqueN(fold),
    mean_validation_gmean = mean(gmean),
    std_validation_gmean = stats::sd(gmean),
    mean_sensitivity = mean(sensitivity),
    mean_specificity = mean(specificity),
    mean_balance_gap = mean(balance_gap),
    mean_precision = mean(precision),
    mean_f1 = mean(f1),
    mean_p4 = mean(p4),
    mean_pr_auc = mean(pr_auc),
    mean_roc_auc = mean(roc_auc),
    mean_threshold = mean(selected_threshold),
    std_threshold = stats::sd(selected_threshold),
    mean_fit_seconds = mean(fit_seconds)
  ), by = grouping]
  data.table::setorder(
    summary,
    -mean_validation_gmean,
    std_validation_gmean,
    mean_balance_gap,
    -mean_pr_auc
  )
  summary[, rank := seq_len(.N)]
  data.table::setcolorder(summary, c("rank", setdiff(names(summary), "rank")))
  summary[]
}

flatten_metrics_record <- function(
  candidate,
  fold,
  parameters,
  threshold_rule,
  threshold_result,
  metrics,
  fit_seconds
) {
  as.list(c(
    list(
      candidate = as.integer(candidate),
      fold = as.integer(fold),
      ntree = as.integer(parameters$ntree),
      mtry = as.integer(parameters$mtry),
      nodesize = as.integer(parameters$nodesize),
      nsplit = as.integer(parameters$nsplit),
      splitrule = as.character(parameters$splitrule),
      threshold_rule = threshold_rule,
      selected_threshold = threshold_result$threshold,
      selected_quantile = threshold_result$quantile,
      training_oob_gmean = threshold_result$training_gmean,
      fit_seconds = fit_seconds
    ),
    metrics
  ))
}

sample_stratified_rows <- function(target, size, seed) {
  if (length(target) <= size) {
    return(seq_along(target))
  }
  set.seed(seed)
  positive <- which(target == POSITIVE_CLASS)
  negative <- which(target == NEGATIVE_CLASS)
  positive_size <- max(2L, round(size * length(positive) / length(target)))
  positive_size <- min(positive_size, length(positive))
  negative_size <- min(size - positive_size, length(negative))
  sort(c(
    sample(positive, positive_size),
    sample(negative, negative_size)
  ))
}

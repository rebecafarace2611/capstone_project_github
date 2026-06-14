script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- if (length(script_argument)) {
  sub("^--file=", "", script_argument[[1L]])
} else {
  "models/rfqc/run.R"
}
SCRIPT_DIR <- normalizePath(dirname(script_path), winslash = "/", mustWork = TRUE)
PROJECT_ROOT <- normalizePath(
  file.path(SCRIPT_DIR, "..", ".."),
  winslash = "/",
  mustWork = TRUE
)
local_library <- file.path(PROJECT_ROOT, ".r-library")
if (dir.exists(local_library)) {
  .libPaths(c(local_library, .libPaths()))
}

raw_arguments <- commandArgs(trailingOnly = TRUE)
thread_option <- match("--threads", raw_arguments)
if (!is.na(thread_option) && thread_option < length(raw_arguments)) {
  requested_threads <- as.integer(raw_arguments[[thread_option + 1L]])
  Sys.setenv(
    RF_CORES = requested_threads,
    OMP_NUM_THREADS = requested_threads,
    MC_CORES = 1L
  )
}

required_packages <- c(
  "randomForestSRC", "arrow", "data.table", "jsonlite", "digest", "tidyselect"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Missing R packages: ", paste(missing_packages, collapse = ", "),
    ". Install them before running native RFQ."
  )
}

workflow_candidates <- c(
  file.path(SCRIPT_DIR, "workflow.R"),
  file.path(PROJECT_ROOT, "models", "rfqc", "workflow.R")
)
workflow_path <- workflow_candidates[file.exists(workflow_candidates)][1L]
if (is.na(workflow_path)) {
  stop(
    "workflow.R was not found beside models/rfqc/run.R."
  )
}
source(workflow_path)

parse_cli <- function(arguments) {
  defaults <- list(
    stage = "smoke",
    train = file.path(PROJECT_ROOT, "data", "train_model_dataset.parquet"),
    test = file.path(PROJECT_ROOT, "data", "test_model_dataset.parquet"),
    approved_features = file.path(
      PROJECT_ROOT, "outputs", "leakage_analysis", "approved_features.json"
    ),
    folds = file.path(
      PROJECT_ROOT, "outputs", "rfqc", "folds", "fold_assignments.parquet"
    ),
    output_dir = file.path(PROJECT_ROOT, "outputs", "rfqc", "runs"),
    profile = "full",
    random_state = 42L,
    smoke_rows = 30000L,
    threads = 0L,
    baseline_trees = 3000L,
    tuning_trees = 0L,
    final_trees = 10000L,
    restart = FALSE,
    no_save_model = FALSE
  )
  if (length(arguments) && !startsWith(arguments[[1L]], "--")) {
    defaults$stage <- arguments[[1L]]
    arguments <- arguments[-1L]
  }
  flag_names <- c("restart", "no-save-model")
  index <- 1L
  while (index <= length(arguments)) {
    option <- arguments[[index]]
    if (!startsWith(option, "--")) {
      stop("Unexpected argument: ", option)
    }
    name <- sub("^--", "", option)
    if (name %in% flag_names) {
      defaults[[gsub("-", "_", name)]] <- TRUE
      index <- index + 1L
      next
    }
    if (index == length(arguments)) {
      stop("Missing value for option ", option)
    }
    value <- arguments[[index + 1L]]
    key <- gsub("-", "_", name)
    if (!key %in% names(defaults)) {
      stop("Unknown option: ", option)
    }
    defaults[[key]] <- value
    index <- index + 2L
  }
  integer_keys <- c(
    "random_state", "smoke_rows", "threads", "baseline_trees",
    "tuning_trees", "final_trees"
  )
  for (key in integer_keys) {
    defaults[[key]] <- as.integer(defaults[[key]])
  }
  valid_stages <- c("smoke", "baseline", "tune", "final", "recover", "resume")
  if (!defaults$stage %in% valid_stages) {
    stop("Stage must be one of: ", paste(valid_stages, collapse = ", "), ".")
  }
  if (!defaults$profile %in% c("quick", "local", "paper", "auc", "full")) {
    stop("Profile must be one of: quick, local, paper, auc, full.")
  }
  if (defaults$tuning_trees < 0L) {
    stop("--tuning-trees must be zero or a positive integer.")
  }
  defaults
}

optional_normalized_path <- function(path) {
  if (file.exists(path)) {
    normalizePath(path, winslash = "/", mustWork = TRUE)
  } else {
    NULL
  }
}

write_run_context <- function(
  args,
  approved_features,
  schema,
  locked_configuration = NULL
) {
  package_versions <- vapply(
    required_packages,
    function(package) as.character(utils::packageVersion(package)),
    character(1)
  )
  actual_train_sha256 <- digest::digest(file = args$train, algo = "sha256")
  actual_features_sha256 <- digest::digest(
    file = args$approved_features,
    algo = "sha256"
  )
  if (!is.null(locked_configuration)) {
    expected <- locked_configuration$expected_input_sha256
    if (
      !is.null(expected$train) &&
        !identical(as.character(expected$train), actual_train_sha256)
    ) {
      stop("Training data hash does not match the locked configuration.")
    }
    if (
      !is.null(expected$approved_features) &&
        !identical(
          as.character(expected$approved_features),
          actual_features_sha256
        )
    ) {
      stop("Approved-feature hash does not match the locked configuration.")
    }
  }
  payload <- list(
    implementation = "native_randomForestSRC_RFQ",
    r_version = R.version.string,
    platform = R.version$platform,
    packages = as.list(package_versions),
    stage = args$stage,
    random_state = args$random_state,
    rf_cores = getOption("rf.cores"),
    omp_num_threads = Sys.getenv("OMP_NUM_THREADS"),
    mc_cores = getOption("mc.cores"),
    inputs = list(
      train = normalizePath(args$train, winslash = "/", mustWork = TRUE),
      test = optional_normalized_path(args$test),
      approved_features = normalizePath(
        args$approved_features, winslash = "/", mustWork = TRUE
      )
    ),
    input_sha256 = list(
      train = actual_train_sha256,
      test = if (is.null(locked_configuration)) {
        NULL
      } else {
        locked_configuration$expected_input_sha256$test %||% NULL
      },
      approved_features = actual_features_sha256
    ),
    test_sha256_status = if (is.null(locked_configuration)) {
      "not_recorded"
    } else {
      "expected_hash_from_locked_configuration_not_recomputed"
    },
    locked_configuration_sha256 = if (is.null(locked_configuration)) {
      NULL
    } else {
      digest::digest(
        file = file.path(args$output_dir, "best_configuration.json"),
        algo = "sha256"
      )
    },
    approved_feature_count = length(approved_features),
    categorical_features = schema,
    missing_data_policy = "Model-ready Parquet inputs must contain no missing values.",
    resampling_policy = "No undersampling, oversampling, SMOTE, or class weighting.",
    final_test_policy = if (args$stage %in% c("final", "recover", "resume")) {
      paste(
        "The locked model and training-derived threshold are persisted before",
        "one Parquet read loads test features and labels. A test-access marker",
        "prevents a second final evaluation."
      )
    } else {
      "Test rows are not loaded outside the final stage."
    }
  )
  write_json_atomic(file.path(args$output_dir, "run_context.json"), payload)
}

run_baseline <- function(args, training, approved_features) {
  cat("Native RFQ baseline started. Test data will not be read.\n")
  parameters <- list(
    ntree = args$baseline_trees,
    mtry = 12L,
    nodesize = 1L,
    nsplit = 10L,
    splitrule = "gini"
  )
  started <- proc.time()[["elapsed"]]
  model <- fit_rfq(training, approved_features, parameters, args$random_state)
  fit_seconds <- proc.time()[["elapsed"]] - started
  probabilities <- positive_probabilities(model$predicted.oob)
  valid <- is.finite(probabilities)
  target <- training[[TARGET_COLUMN]][valid]
  probabilities <- probabilities[valid]

  optimized <- threshold_for_rule(
    "gmean_optimized",
    target,
    probabilities
  )
  q_star <- threshold_for_rule(
    "q_star_prevalence",
    target,
    probabilities
  )
  optimized_metrics <- classification_metrics(
    target,
    probabilities,
    optimized$threshold
  )
  q_star_metrics <- classification_metrics(
    target,
    probabilities,
    q_star$threshold
  )

  write_csv_atomic(
    file.path(args$output_dir, "baseline_threshold_curve.csv"),
    data.table::as.data.table(optimized$curve)
  )
  arrow::write_parquet(
    data.frame(
      fraud_probability = probabilities,
      gmean_optimized_prediction = as.integer(
        probabilities >= optimized$threshold
      ),
      q_star_prediction = as.integer(probabilities >= q_star$threshold)
    ),
    file.path(args$output_dir, "baseline_oob_predictions.parquet"),
    compression = "snappy"
  )
  result <- list(
    implementation = "native_randomForestSRC_RFQ",
    model_role = "untuned_training_oob_baseline",
    parameters = parameters,
    fit_seconds = fit_seconds,
    gmean_optimized = list(
      threshold = optimized$threshold,
      metrics = optimized_metrics
    ),
    q_star_prevalence = list(
      threshold = q_star$threshold,
      metrics = q_star_metrics
    )
  )
  metrics_path <- file.path(args$output_dir, "baseline_oob_metrics.json")
  write_json_atomic(metrics_path, result)
  if (!args$no_save_model) {
    saveRDS(
      model,
      file.path(args$output_dir, "rfqc_native_baseline_model.rds"),
      compress = FALSE
    )
  }
  cat("Native RFQ baseline completed successfully.\n")
  cat(
    sprintf(
      "OOB G-mean: %.4f; sensitivity: %.4f; specificity: %.4f\n",
      optimized_metrics$gmean,
      optimized_metrics$sensitivity,
      optimized_metrics$specificity
    )
  )
  cat("Metrics:", metrics_path, "\n")
}

run_smoke <- function(args, training, approved_features) {
  cat("Native RFQ smoke check started. Test data will not be read.\n")
  rows <- sample_stratified_rows(
    training[[TARGET_COLUMN]],
    args$smoke_rows,
    args$random_state
  )
  sample_data <- training[rows, , drop = FALSE]
  parameters <- list(
    ntree = 100L,
    mtry = 12L,
    nodesize = 5L,
    nsplit = 10L,
    splitrule = "gini"
  )
  model <- fit_rfq(sample_data, approved_features, parameters, args$random_state)
  probabilities <- positive_probabilities(model$predicted.oob)
  valid <- is.finite(probabilities)
  optimized <- threshold_for_rule(
    "gmean_optimized",
    sample_data[[TARGET_COLUMN]][valid],
    probabilities[valid]
  )
  q_star <- threshold_for_rule(
    "q_star_prevalence",
    sample_data[[TARGET_COLUMN]][valid],
    probabilities[valid]
  )
  result <- list(
    rows = nrow(sample_data),
    fraud_cases = sum(sample_data[[TARGET_COLUMN]] == POSITIVE_CLASS),
    parameters = parameters,
    gmean_optimized = list(
      threshold = optimized$threshold,
      metrics = classification_metrics(
        sample_data[[TARGET_COLUMN]][valid],
        probabilities[valid],
        optimized$threshold
      )
    ),
    q_star_prevalence = list(
      threshold = q_star$threshold,
      metrics = classification_metrics(
        sample_data[[TARGET_COLUMN]][valid],
        probabilities[valid],
        q_star$threshold
      )
    )
  )
  path <- file.path(args$output_dir, "smoke_result.json")
  write_json_atomic(path, result)
  cat("Native RFQ smoke check completed successfully.\n")
  cat("Result:", path, "\n")
}

run_tune <- function(args, training, approved_features) {
  if (!file.exists(args$folds)) {
    stop(
      paste(
        "Fold assignments not found. Run:",
        "python models/rfqc/prepare_folds.py"
      )
    )
  }
  assignments <- arrow::read_parquet(args$folds, as_data_frame = TRUE)
  if (
    nrow(assignments) != nrow(training) ||
      !identical(as.integer(assignments$row_index), seq_len(nrow(training)) - 1L)
  ) {
    stop("Fold assignments do not match the current training data order.")
  }
  fold <- as.integer(assignments$fold)
  fold_values <- sort(unique(fold))
  if (!identical(fold_values, seq.int(0L, max(fold_values)))) {
    stop("Fold labels must be consecutive integers beginning at zero.")
  }

  grid <- make_parameter_grid(args$profile, args$tuning_trees)
  threshold_rules <- c("gmean_optimized", "q_star_prevalence")
  manifest <- list(
    implementation = "native_randomForestSRC_RFQ",
    profile = args$profile,
    tuning_trees_override = args$tuning_trees,
    random_state = args$random_state,
    folds_sha256 = digest::digest(file = args$folds, algo = "sha256"),
    training_sha256 = digest::digest(file = args$train, algo = "sha256"),
    parameters = grid,
    threshold_rules = threshold_rules
  )
  signature <- digest::digest(manifest, algo = "sha256")
  manifest$run_signature <- signature

  checkpoint_path <- file.path(args$output_dir, "cv_fold_results.csv")
  manifest_path <- file.path(args$output_dir, "tuning_manifest.json")
  ranking_path <- file.path(args$output_dir, "cv_ranking.csv")
  best_path <- file.path(args$output_dir, "best_configuration.json")
  if (args$restart) {
    unlink(c(checkpoint_path, manifest_path, ranking_path, best_path))
  }
  if (file.exists(manifest_path)) {
    existing_manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)
    if (!identical(existing_manifest$run_signature, signature)) {
      stop("Existing tuning checkpoint is incompatible. Use --restart.")
    }
  } else {
    write_json_atomic(manifest_path, manifest)
  }

  results <- if (file.exists(checkpoint_path)) {
    data.table::fread(checkpoint_path)
  } else {
    data.table::data.table()
  }
  total_fits <- nrow(grid) * length(fold_values)
  completed <- if (nrow(results)) {
    unique(results[, .(candidate, fold)])
  } else {
    data.table::data.table(candidate = integer(), fold = integer())
  }
  completed_fits <- nrow(completed)
  cat(
    "Native RFQ tuning:", nrow(grid), "candidates x",
    length(fold_values), "folds =", total_fits, "forest fits.\n"
  )
  if (completed_fits) {
    cat("Resuming from", completed_fits, "completed forest fits.\n")
  }

  for (candidate_index in seq_len(nrow(grid))) {
    parameters <- as.list(grid[candidate_index, ])
    for (fold_value in fold_values) {
      already_done <- nrow(completed[
        candidate == candidate_index & fold == fold_value
      ]) > 0L
      if (already_done) {
        next
      }
      cat(
        sprintf(
          "Starting fit %d/%d: candidate %d/%d, fold %d/%d, %s\n",
          completed_fits + 1L,
          total_fits,
          candidate_index,
          nrow(grid),
          fold_value + 1L,
          length(fold_values),
          paste(
            sprintf("%s=%s", names(parameters)[-1L], parameters[-1L]),
            collapse = ", "
          )
        )
      )
      fit_index <- which(fold != fold_value)
      validation_index <- which(fold == fold_value)
      started <- proc.time()[["elapsed"]]
      model <- fit_rfq(
        training[fit_index, , drop = FALSE],
        approved_features,
        parameters,
        args$random_state + fold_value
      )
      fit_seconds <- proc.time()[["elapsed"]] - started
      oob_probabilities <- positive_probabilities(model$predicted.oob)
      valid_oob <- is.finite(oob_probabilities)
      validation_prediction <- predict(
        model,
        training[validation_index, approved_features, drop = FALSE],
        perf.type = "none",
        block.size = NULL
      )
      validation_probabilities <- positive_probabilities(
        validation_prediction$predicted
      )

      records <- lapply(threshold_rules, function(rule) {
        threshold_result <- threshold_for_rule(
          rule,
          training[[TARGET_COLUMN]][fit_index][valid_oob],
          oob_probabilities[valid_oob]
        )
        metrics <- classification_metrics(
          training[[TARGET_COLUMN]][validation_index],
          validation_probabilities,
          threshold_result$threshold
        )
        flatten_metrics_record(
          candidate_index,
          fold_value,
          parameters,
          rule,
          threshold_result,
          metrics,
          fit_seconds
        )
      })
      results <- data.table::rbindlist(
        list(results, data.table::rbindlist(records, fill = TRUE)),
        fill = TRUE
      )
      write_csv_atomic(checkpoint_path, results)
      completed_fits <- completed_fits + 1L
      completed <- unique(results[, .(candidate, fold)])
      current <- results[
        candidate == candidate_index & fold == fold_value
      ]
      cat(
        sprintf(
          "Completed in %.1fs. Validation G-mean: optimized=%.4f, q*=%.4f\n",
          fit_seconds,
          current[threshold_rule == "gmean_optimized", gmean],
          current[threshold_rule == "q_star_prevalence", gmean]
        )
      )
      rm(model, validation_prediction)
      invisible(gc())
    }
  }

  expected_rows <- nrow(grid) * length(fold_values) * length(threshold_rules)
  if (nrow(results) != expected_rows) {
    stop(
      "Tuning checkpoint is incomplete: expected ", expected_rows,
      " rows but found ", nrow(results), "."
    )
  }
  ranking <- rank_cv_results(results)
  write_csv_atomic(ranking_path, ranking)
  best <- as.list(ranking[1L])
  best$selection_profile <- args$profile
  best$selection_folds <- length(fold_values)
  best$recommended_final_trees <- 10000L
  write_json_atomic(best_path, best)
  cat("Native RFQ tuning completed successfully.\n")
  cat(
    sprintf(
      "Best validation G-mean: %.4f; splitrule=%s; threshold rule=%s\n",
      best$mean_validation_gmean,
      best$splitrule,
      best$threshold_rule
    )
  )
  cat("Best configuration:", best_path, "\n")
}

validate_locked_final_configuration <- function(args, training) {
  best_path <- file.path(args$output_dir, "best_configuration.json")
  if (!file.exists(best_path)) {
    stop("A locked best_configuration.json is required for final evaluation.")
  }
  best <- jsonlite::read_json(best_path, simplifyVector = TRUE)
  if (!identical(as.character(best$status), "locked_for_final_test")) {
    stop("best_configuration.json is not locked for final testing.")
  }
  if (as.integer(best$ntree) != args$final_trees) {
    stop(
      "Locked ntree=", best$ntree,
      " does not match --final-trees=", args$final_trees, "."
    )
  }
  if (as.integer(best$random_state) != args$random_state) {
    stop("Locked random_state does not match the final-run random state.")
  }
  selected_rule <- as.character(best$threshold_rule)
  if (!identical(selected_rule, "q_star_prevalence")) {
    stop("This locked final run requires threshold_rule=q_star_prevalence.")
  }
  locked_threshold <- as.numeric(best$locked_threshold)
  training_prevalence <- mean(training[[TARGET_COLUMN]] == POSITIVE_CLASS)
  if (
    !is.finite(locked_threshold) ||
      abs(locked_threshold - training_prevalence) > 1e-15
  ) {
    stop("The locked q* threshold does not equal full-training prevalence.")
  }
  list(
    best = best,
    parameters = list(
      ntree = args$final_trees,
      mtry = as.integer(best$mtry),
      nodesize = as.integer(best$nodesize),
      nsplit = as.integer(best$nsplit),
      splitrule = as.character(best$splitrule)
    ),
    selected_rule = selected_rule,
    locked_threshold = locked_threshold
  )
}

evaluate_locked_final_test <- function(
  args,
  prediction_model,
  approved_features,
  categorical_levels,
  locked,
  fit_seconds = NULL,
  recovery_mode = FALSE,
  resume_mode = FALSE
) {
  test_access_path <- file.path(
    args$output_dir,
    ".test_access_started.json"
  )
  if (resume_mode) {
    if (!file.exists(test_access_path)) {
      stop("Resume mode requires the original test-access marker.")
    }
    original_access <- jsonlite::read_json(
      test_access_path,
      simplifyVector = TRUE
    )
    write_json_atomic(
      file.path(args$output_dir, ".test_access_failed_preprocessing.json"),
      list(
        recorded_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
        original_access = original_access,
        failure_stage = "feature_preparation_before_model_prediction",
        failure_reason = paste(
          "The first test read was rejected by an overly strict unseen-factor",
          "level check before predictions or test metrics were produced."
        ),
        predictions_produced = FALSE,
        metrics_produced = FALSE,
        locked_configuration_changed = FALSE
      )
    )
    write_json_atomic(
      file.path(args$output_dir, ".test_access_retry_started.json"),
      list(
        started_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
        test_path = args$test,
        attempt = 2L,
        reason = "Technical recovery after pre-prediction preprocessing failure."
      )
    )
  } else {
    write_json_atomic(
      test_access_path,
      list(
        started_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
        test_path = args$test,
        recovery_mode = recovery_mode,
        access_policy = paste(
          "Single arrow::read_parquet call for approved features and target;",
          "rerun prohibited after this marker is created."
        )
      )
    )
  }
  cat(
    "Model and q* threshold are fixed. Reading test data once for final evaluation.\n"
  )
  evaluation <- load_evaluation_frame(
    args$test,
    approved_features,
    categorical_levels
  )
  if (
    !is.null(locked$best$expected_test_rows) &&
      nrow(evaluation$features) !=
        as.integer(locked$best$expected_test_rows)
  ) {
    stop("Test row count does not match the locked configuration.")
  }
  if (
    !is.null(locked$best$expected_test_fraud) &&
      sum(evaluation$target == POSITIVE_CLASS) !=
        as.integer(locked$best$expected_test_fraud)
  ) {
    stop("Test fraud count does not match the locked configuration.")
  }
  test_prediction <- predict(
    prediction_model,
    evaluation$features,
    perf.type = "none",
    block.size = NULL
  )
  test_probabilities <- positive_probabilities(test_prediction$predicted)
  primary_metrics <- classification_metrics(
    evaluation$target,
    test_probabilities,
    locked$locked_threshold
  )
  result <- list(
    implementation = "native_randomForestSRC_RFQ",
    evaluation_role = "single_untouched_final_test",
    parameters = locked$parameters,
    locked_candidate = as.integer(locked$best$candidate),
    threshold_rule = locked$selected_rule,
    final_fit_seconds = fit_seconds,
    recovered_from_saved_model = recovery_mode,
    resumed_after_preprocessing_failure = resume_mode,
    test_read_attempts = if (resume_mode) 2L else 1L,
    first_read_produced_predictions = if (resume_mode) FALSE else NULL,
    protocol_deviation = if (resume_mode) {
      paste(
        "The test Parquet was read a second time only because the first read",
        "failed during factor preprocessing before model prediction or metric",
        "calculation. Model parameters and threshold remained locked."
      )
    } else {
      NULL
    },
    selected_threshold = locked$locked_threshold,
    primary_test_metrics = primary_metrics
  )
  metrics_path <- file.path(args$output_dir, "final_test_metrics.json")
  predictions <- data.frame(
    row_index = seq_along(test_probabilities) - 1L,
    fraud_probability = test_probabilities,
    primary_prediction = as.integer(
      test_probabilities >= locked$locked_threshold
    )
  )
  arrow::write_parquet(
    predictions,
    file.path(args$output_dir, "final_test_predictions.parquet"),
    compression = "snappy"
  )
  write_json_atomic(metrics_path, result)
  write_json_atomic(
    file.path(args$output_dir, ".final_evaluation_complete.json"),
    list(
      completed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
      recovery_mode = recovery_mode,
      resume_mode = resume_mode,
      test_read_attempts = if (resume_mode) 2L else 1L,
      metrics = basename(metrics_path),
      predictions = "final_test_predictions.parquet"
    )
  )
  cat("Final native RFQ evaluation completed successfully.\n")
  cat(
    sprintf(
      "Primary test G-mean: %.4f; sensitivity: %.4f; specificity: %.4f\n",
      primary_metrics$gmean,
      primary_metrics$sensitivity,
      primary_metrics$specificity
    )
  )
  cat("Metrics:", metrics_path, "\n")
}

run_final <- function(
  args,
  training,
  approved_features,
  categorical_levels
) {
  locked <- validate_locked_final_configuration(args, training)
  if (args$no_save_model) {
    stop("The locked final run must save rfqc_native_model.rds.")
  }
  parameters <- locked$parameters
  selected_rule <- locked$selected_rule
  locked_threshold <- locked$locked_threshold

  protected_paths <- file.path(
    args$output_dir,
    c(
      "final_test_metrics.json",
      "final_test_predictions.parquet",
      "final_threshold_curve.csv",
      "final_tree_convergence.csv",
      "rfqc_native_model.rds",
      "rfqc_native_model.rds.tmp",
      ".test_access_started.json",
      ".final_evaluation_complete.json"
    )
  )
  existing_outputs <- protected_paths[file.exists(protected_paths)]
  if (length(existing_outputs)) {
    stop(
      "Final-run outputs or access markers already exist; refusing to rerun: ",
      paste(basename(existing_outputs), collapse = ", ")
    )
  }

  cat("Final native RFQ training started with", args$final_trees, "trees.\n")
  started <- proc.time()[["elapsed"]]
  model <- fit_rfq(training, approved_features, parameters, args$random_state)
  fit_seconds <- proc.time()[["elapsed"]] - started

  convergence_points <- c(500L, 1000L, 2000L, 3000L, 5000L, args$final_trees)
  convergence_points <- unique(sort(
    convergence_points[convergence_points <= args$final_trees]
  ))
  convergence <- lapply(convergence_points, function(tree_count) {
    probabilities <- if (tree_count == args$final_trees) {
      positive_probabilities(model$predicted.oob)
    } else {
      restored <- predict(
        model,
        get.tree = seq_len(tree_count),
        outcome = "train",
        perf.type = "none",
        block.size = NULL
      )
      restored_probabilities <- positive_probabilities(restored$predicted.oob)
      rm(restored)
      restored_probabilities
    }
    valid <- is.finite(probabilities)
    metrics <- classification_metrics(
      training[[TARGET_COLUMN]][valid],
      probabilities[valid],
      locked_threshold
    )
    data.frame(
      trees = tree_count,
      threshold_rule = selected_rule,
      threshold = locked_threshold,
      sensitivity = metrics$sensitivity,
      specificity = metrics$specificity,
      fpr = metrics$fpr,
      gmean = metrics$gmean,
      precision = metrics$precision,
      roc_auc = metrics$roc_auc,
      pr_auc = metrics$pr_auc,
      balance_gap = metrics$balance_gap
    )
  })
  convergence <- data.table::rbindlist(convergence)
  write_csv_atomic(
    file.path(args$output_dir, "final_tree_convergence.csv"),
    convergence
  )

  oob_probabilities <- positive_probabilities(model$predicted.oob)
  valid_oob <- is.finite(oob_probabilities)
  selected_threshold <- threshold_for_rule(
    selected_rule,
    training[[TARGET_COLUMN]][valid_oob],
    oob_probabilities[valid_oob]
  )
  if (abs(selected_threshold$threshold - locked_threshold) > 1e-15) {
    stop("The fitted-model q* threshold differs from the locked threshold.")
  }

  threshold_curve_frame <- threshold_curve(
    training[[TARGET_COLUMN]][valid_oob],
    oob_probabilities[valid_oob]
  )
  threshold_curve_frame$is_selected_threshold <- FALSE
  threshold_curve_frame$threshold_rule <- selected_rule
  selected_oob_metrics <- classification_metrics(
    training[[TARGET_COLUMN]][valid_oob],
    oob_probabilities[valid_oob],
    locked_threshold
  )
  selected_curve_row <- data.frame(
    threshold = locked_threshold,
    quantile = mean(oob_probabilities[valid_oob] < locked_threshold),
    tn = selected_oob_metrics$tn,
    fp = selected_oob_metrics$fp,
    fn = selected_oob_metrics$fn,
    tp = selected_oob_metrics$tp,
    sensitivity = selected_oob_metrics$sensitivity,
    specificity = selected_oob_metrics$specificity,
    fpr = selected_oob_metrics$fpr,
    fnr = selected_oob_metrics$fnr,
    precision = selected_oob_metrics$precision,
    gmean = selected_oob_metrics$gmean,
    informedness = selected_oob_metrics$informedness,
    balance_gap = selected_oob_metrics$balance_gap,
    is_selected_threshold = TRUE,
    threshold_rule = selected_rule
  )
  threshold_curve_frame <- data.table::rbindlist(
    list(threshold_curve_frame, selected_curve_row),
    fill = TRUE
  )
  data.table::setorder(threshold_curve_frame, threshold)
  write_csv_atomic(
    file.path(args$output_dir, "final_threshold_curve.csv"),
    threshold_curve_frame
  )

  deployable_forest <- model$forest
  deployable_forest$xvar <- NULL
  validation_rows <- seq_len(min(1000L, nrow(training)))
  original_validation_probabilities <- positive_probabilities(
    predict(
      model,
      training[validation_rows, approved_features, drop = FALSE],
      perf.type = "none",
      block.size = NULL
    )$predicted
  )
  compact_validation_probabilities <- positive_probabilities(
    predict(
      deployable_forest,
      training[validation_rows, approved_features, drop = FALSE],
      perf.type = "none",
      block.size = NULL
    )$predicted
  )
  if (
    !isTRUE(all.equal(
      original_validation_probabilities,
      compact_validation_probabilities,
      tolerance = sqrt(.Machine$double.eps)
    ))
  ) {
    stop("Compact forest does not reproduce fitted-model predictions.")
  }
  model_path <- file.path(args$output_dir, "rfqc_native_model.rds")
  temporary_model_path <- paste0(model_path, ".tmp")
  saveRDS(
    deployable_forest,
    temporary_model_path,
    compress = "gzip"
  )
  rm(deployable_forest)
  if (!file.rename(temporary_model_path, model_path)) {
    unlink(temporary_model_path)
    stop("Could not atomically finalize rfqc_native_model.rds.")
  }

  evaluate_locked_final_test(
    args,
    model,
    approved_features,
    categorical_levels,
    locked,
    fit_seconds = fit_seconds,
    recovery_mode = FALSE
  )
}

run_recover_final <- function(
  args,
  training,
  approved_features,
  categorical_levels
) {
  locked <- validate_locked_final_configuration(args, training)
  required_training_artifacts <- file.path(
    args$output_dir,
    c("final_threshold_curve.csv", "final_tree_convergence.csv")
  )
  missing_artifacts <- required_training_artifacts[
    !file.exists(required_training_artifacts)
  ]
  if (length(missing_artifacts)) {
    stop(
      "Cannot recover without completed training artifacts: ",
      paste(basename(missing_artifacts), collapse = ", ")
    )
  }
  prohibited_paths <- file.path(
    args$output_dir,
    c(
      ".test_access_started.json",
      ".final_evaluation_complete.json",
      "final_test_metrics.json",
      "final_test_predictions.parquet"
    )
  )
  existing_prohibited <- prohibited_paths[file.exists(prohibited_paths)]
  if (length(existing_prohibited)) {
    stop(
      "Test access or final outputs already exist; refusing recovery: ",
      paste(basename(existing_prohibited), collapse = ", ")
    )
  }
  model_path <- file.path(args$output_dir, "rfqc_native_model.rds")
  temporary_model_path <- paste0(model_path, ".tmp")
  candidates <- c(model_path, temporary_model_path)
  candidates <- candidates[file.exists(candidates)]
  if (length(candidates) != 1L) {
    stop(
      "Recovery requires exactly one saved model candidate (.rds or .rds.tmp)."
    )
  }
  recovery_path <- candidates[[1L]]
  cat("Loading saved compact forest for pre-test recovery:", recovery_path, "\n")
  recovered_model <- readRDS(recovery_path)
  if (!inherits(recovered_model, "forest")) {
    stop("Saved recovery object is not a randomForestSRC forest.")
  }
  if (as.integer(recovered_model$ntree) != args$final_trees) {
    stop("Saved recovery forest tree count does not match the locked model.")
  }
  validation_rows <- seq_len(min(1000L, nrow(training)))
  validation_probabilities <- positive_probabilities(
    predict(
      recovered_model,
      training[validation_rows, approved_features, drop = FALSE],
      perf.type = "none",
      block.size = NULL
    )$predicted
  )
  if (
    length(validation_probabilities) != length(validation_rows) ||
      any(!is.finite(validation_probabilities))
  ) {
    stop("Saved recovery forest failed the pre-test prediction check.")
  }
  if (identical(recovery_path, temporary_model_path)) {
    if (!file.rename(temporary_model_path, model_path)) {
      stop("Could not finalize the recovered model file.")
    }
  }
  cat("Saved compact forest passed the pre-test recovery check.\n")
  evaluate_locked_final_test(
    args,
    recovered_model,
    approved_features,
    categorical_levels,
    locked,
    fit_seconds = NULL,
    recovery_mode = TRUE
  )
}

run_resume_final <- function(
  args,
  training,
  approved_features,
  categorical_levels
) {
  locked <- validate_locked_final_configuration(args, training)
  test_access_path <- file.path(
    args$output_dir,
    ".test_access_started.json"
  )
  if (!file.exists(test_access_path)) {
    stop("Resume requires the original .test_access_started.json marker.")
  }
  prohibited_paths <- file.path(
    args$output_dir,
    c(
      ".final_evaluation_complete.json",
      "final_test_metrics.json",
      "final_test_predictions.parquet",
      ".test_access_retry_started.json"
    )
  )
  existing_prohibited <- prohibited_paths[file.exists(prohibited_paths)]
  if (length(existing_prohibited)) {
    stop(
      "Final outputs or a retry marker already exist; refusing resume: ",
      paste(basename(existing_prohibited), collapse = ", ")
    )
  }
  required_paths <- file.path(
    args$output_dir,
    c(
      "rfqc_native_model.rds",
      "final_threshold_curve.csv",
      "final_tree_convergence.csv"
    )
  )
  missing_required <- required_paths[!file.exists(required_paths)]
  if (length(missing_required)) {
    stop(
      "Resume is missing required saved artifacts: ",
      paste(basename(missing_required), collapse = ", ")
    )
  }
  model_path <- file.path(args$output_dir, "rfqc_native_model.rds")
  cat("Loading saved compact forest for test-read recovery:", model_path, "\n")
  recovered_model <- readRDS(model_path)
  if (
    !inherits(recovered_model, "forest") ||
      as.integer(recovered_model$ntree) != args$final_trees
  ) {
    stop("Saved model does not match the locked final forest.")
  }
  validation_rows <- seq_len(min(1000L, nrow(training)))
  validation_probabilities <- positive_probabilities(
    predict(
      recovered_model,
      training[validation_rows, approved_features, drop = FALSE],
      perf.type = "none",
      block.size = NULL
    )$predicted
  )
  if (
    length(validation_probabilities) != length(validation_rows) ||
      any(!is.finite(validation_probabilities))
  ) {
    stop("Saved model failed the pre-test resume prediction check.")
  }
  write_json_atomic(
    file.path(args$output_dir, "test_retry_context.json"),
    list(
      recorded_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
      reason = paste(
        "Resume after the first test read failed during strict unseen-factor",
        "validation before prediction."
      ),
      model_parameters_changed = FALSE,
      threshold_changed = FALSE,
      first_attempt_predictions_produced = FALSE,
      first_attempt_metrics_produced = FALSE
    )
  )
  cat("Saved compact forest passed the resume prediction check.\n")
  evaluate_locked_final_test(
    args,
    recovered_model,
    approved_features,
    categorical_levels,
    locked,
    fit_seconds = NULL,
    recovery_mode = TRUE,
    resume_mode = TRUE
  )
}

main <- function() {
  args <- parse_cli(raw_arguments)
  args$train <- normalizePath(args$train, winslash = "/", mustWork = TRUE)
  if (args$stage %in% c("final", "recover", "resume")) {
    args$test <- normalizePath(args$test, winslash = "/", mustWork = TRUE)
  } else {
    args$test <- normalizePath(args$test, winslash = "/", mustWork = FALSE)
  }
  args$approved_features <- normalizePath(
    args$approved_features, winslash = "/", mustWork = TRUE
  )
  args$output_dir <- normalizePath(
    args$output_dir, winslash = "/", mustWork = FALSE
  )
  args$folds <- normalizePath(args$folds, winslash = "/", mustWork = FALSE)
  dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)
  if (
    args$stage %in% c("final", "recover") &&
      file.exists(file.path(args$output_dir, ".test_access_started.json"))
  ) {
    stop(
      "Test access has already started for this directory; refusing to rerun."
    )
  }

  if (args$threads > 0L) {
    Sys.setenv(
      RF_CORES = args$threads,
      OMP_NUM_THREADS = args$threads,
      MC_CORES = 1L
    )
    options(rf.cores = args$threads, mc.cores = 1L)
  }
  approved_features <- load_approved_features(args$approved_features)
  training <- load_training_frame(args$train, approved_features)
  schema <- attr(training[, approved_features, drop = FALSE], "categorical_features")
  if (is.null(schema)) {
    schema <- names(training)[vapply(training, is.factor, logical(1))]
    schema <- setdiff(schema, TARGET_COLUMN)
  }
  categorical_levels <- lapply(training[, schema, drop = FALSE], levels)
  locked_configuration <- if (args$stage %in% c("final", "recover", "resume")) {
    configuration_path <- file.path(
      args$output_dir,
      "best_configuration.json"
    )
    if (!file.exists(configuration_path)) {
      stop("Locked best_configuration.json was not found.")
    }
    jsonlite::read_json(configuration_path, simplifyVector = TRUE)
  } else {
    NULL
  }
  write_run_context(
    args,
    approved_features,
    schema,
    locked_configuration
  )

  if (identical(args$stage, "smoke")) {
    run_smoke(args, training, approved_features)
  } else if (identical(args$stage, "baseline")) {
    run_baseline(args, training, approved_features)
  } else if (identical(args$stage, "tune")) {
    run_tune(args, training, approved_features)
  } else if (identical(args$stage, "final")) {
    run_final(
      args,
      training,
      approved_features,
      categorical_levels
    )
  } else if (identical(args$stage, "recover")) {
    run_recover_final(
      args,
      training,
      approved_features,
      categorical_levels
    )
  } else {
    run_resume_final(
      args,
      training,
      approved_features,
      categorical_levels
    )
  }
}

main()

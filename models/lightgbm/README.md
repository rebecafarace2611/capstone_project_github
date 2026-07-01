# LightGBM Baseline

This directory contains the first-stage LightGBM workflow. It deliberately
uses the original training distribution with no resampling and no class
weights. The final test dataset is not accepted by the baseline command and is
therefore unavailable to model selection.

## Baseline protocol

- 144 leakage-approved predictors.
- Native LightGBM handling for 24 categorical predictors.
- Five-fold stratified grouped cross-validation.
- Binary GBDT with `learning_rate=0.05`, `num_leaves=31`, and an upper limit of
  5,000 trees.
- Early stopping after 100 rounds without validation average-precision
  improvement.
- Model selection evidence based on out-of-fold predictions.
- Primary operating point: the lowest OOF FPR attainable at recall of at least
  0.80.
- No sampling, SMOTE, `class_weight`, `is_unbalance`, or `scale_pos_weight`.

The workflow reuses `outputs/rfqc/folds/fold_assignments.parquet` when that
local file is available. Because it is confidential and excluded from Git, a
fresh AutoDL checkout may not contain it. In that case the script recreates the
same type of deterministic grouped folds inside the private run directory.

## AutoDL preparation

Only the training Parquet is required for this stage. Place it at:

```text
data/train_model_dataset.parquet
```

Do not upload the raw CSV, variable workbook, or final test Parquet for this
baseline run. Confirm that use of the selected cloud instance is permitted for
the confidential company data.

Use a persistent AutoDL filesystem and start a `tmux` session so that a browser
or SSH disconnect does not stop training:

```bash
cd /root/autodl-tmp/capstone_project_github
tmux new -s lgbm
bash models/lightgbm/run_autodl.sh install
bash models/lightgbm/run_autodl.sh check
LGBM_THREADS=16 bash models/lightgbm/run_autodl.sh baseline
```

Detach from `tmux` with `Ctrl-b`, then `d`. Reconnect with:

```bash
tmux attach -t lgbm
```

If the Python process itself is interrupted, run the same baseline command
again. Completed fold directories are treated as checkpoints and are reused.
Do not change parameters while resuming into the same run directory. Use a new
`LGBM_RUN_DIR` for a different configuration.

## Important outputs

All outputs are written below `runs/lightgbm/baseline/` by default:

- `status.json`: running, failed, or complete status.
- `run_spec.json`: immutable data hashes and training configuration.
- `fold_metrics.csv`: per-fold early-stopping and validation results.
- `oof_operating_points.csv`: FPR, recall, precision, and alert counts at the
  requested recall levels.
- `baseline_summary.json`: pooled OOF result and recommended final tree count.
- `feature_importance_summary.csv`: mean gain importance across folds.
- `fold_*/`: checkpointed model, validation predictions, metrics, and feature
  importance.
- `oof_predictions.parquet`: confidential row-level OOF scores.

The entire `runs/` directory is ignored by Git. Keep row-level predictions,
fold models, and generated fold assignments private. Only reviewed aggregate
metrics and figures should later be copied into `outputs/lightgbm/`.

## Direct Python invocation

```bash
python -m models.lightgbm.run_baseline \
  --threads 16 \
  --output-dir runs/lightgbm/baseline
```

Use CPU for the first reproducible baseline. The script supports `--device
gpu` or `--device cuda`, but those modes require a compatible LightGBM build
and are not deterministic across hardware in the same way as the CPU run.

## Imbalance strategy screen

After the baseline is complete, run the controlled imbalance screen before
structural hyperparameter tuning:

```bash
LGBM_THREADS=16 bash models/lightgbm/run_autodl.sh screen
```

The screen imports the completed baseline when its data hashes and modelling
configuration match. It then evaluates these strategies on the same grouped
folds:

- `scale_pos_weight`: 2, 5, 10, and 20.
- Random undersampling: one fraud row to 15, 30, or 60 legitimate rows.

Weighting and undersampling are never combined. Random undersampling is
performed separately inside each training fold; validation folds retain their
original prevalence. All strategies retain PR-AUC early stopping and the same
untuned tree structure as the baseline.

The primary ranking field is the mean of the five fold-specific FPR values at
recall of at least 0.80. Pooled OOF metrics are retained as secondary evidence
because independently early-stopped fold models can have different score
scales.

Important screen outputs below `runs/lightgbm/imbalance_screen/` are:

- `strategy_ranking.csv`: primary comparison and relative FPR improvement.
- `screen_summary.json`: recommended strategy and pooled secondary metrics.
- `run_spec.json` and `run_context.json`: immutable configuration and software
  context.
- `strategies/*/fold_metrics.csv`: per-fold evidence.
- `strategies/*/oof_operating_points.csv`: pooled threshold trade-offs.

As with the baseline, rerunning the same command resumes completed folds. Use a
new `LGBM_SCREEN_DIR` if any strategy or model setting is changed.

## Optuna tuning of random undersampling

After the imbalance screen, jointly tune the undersampling ratio and LightGBM
structure with 60 sequential TPE trials:

```bash
LGBM_THREADS=16 LGBM_N_TRIALS=60 \
  bash models/lightgbm/run_autodl.sh tune
```

The search includes:

- Legitimate-to-fraud ratios: 5, 10, 15, 20, 30, and 60.
- Learning rate from 0.01 to 0.10 on a log scale.
- Valid paired tree-depth and leaf-count configurations.
- Minimum leaf samples, feature fraction, L1/L2 regularization, split gain,
  and categorical smoothing/regularization.

Every trial uses all five locked grouped folds. Each training fold is
undersampled independently with a deterministic seed, while validation data
remain untouched. PR-AUC early stopping is retained. The single Optuna
objective is mean fold FPR at recall of at least 0.80; FPR standard deviation,
worst-fold FPR, PR-AUC, ROC-AUC, and precision are stored for secondary review.

The Optuna study is stored in `optuna_study.sqlite3`. Repeating the same command
continues until the requested total number of completed trials is reached; it
does not add 60 more trials each time. Use `LGBM_N_TRIALS=80` later to extend a
completed 60-trial study to 80.

Important outputs below `runs/lightgbm/optuna_rus/` are:

- `optuna_study.sqlite3`: private resumable study database.
- `trial_ranking.csv`: completed trials ranked by mean FPR, then stability.
- `best_configuration.json`: provisional best configuration requiring a
  separate confirmation run.
- `optuna_trials.csv`: full Optuna trial history.
- `trials/trial_*/fold_metrics.csv`: fold evidence for each completed trial.

Trial models and row-level predictions are deliberately not saved. The test
dataset remains unused. A successful tuning run selects candidates for a
fixed-configuration confirmation stage; it does not lock the final model.

## Boundary-aware local refinement

If the broad search places its best trials on the lowest undersampling ratio,
run the separate local profile rather than merely extending the same study:

```bash
LGBM_THREADS=16 LGBM_REFINE_TRIALS=40 \
  bash models/lightgbm/run_autodl.sh refine
```

This creates an independent study in `runs/lightgbm/optuna_refine/`. It expands
the legitimate-to-fraud ratios downward to 1, 2, 3, 5, 7, and 10, while
narrowing the other parameters around the region identified by the broad
search. The search includes only shallow tree configurations, minimum leaf
sizes of 50 or 100, feature fractions of 0.8 to 1.0, and focused categorical
and split regularization values.

The broad-search winners are enqueued as anchor trials. The refinement output
also reports relative FPR improvement against the completed broad-search best
configuration. It uses a separate SQLite database and is independently
resumable. Repeating the command continues to the requested total of 40
completed trials.

Local refinement is still model selection on the same cross-validation folds.
Its winner must subsequently be confirmed across several independent
undersampling seeds before final tree count and threshold selection.

## Fixed-candidate RUS seed confirmation

Do not extend either Optuna study after local refinement. Confirm the three
pre-specified candidates instead:

- A: local trial 34, RUS 1:7.
- B: local trial 8, RUS 1:10.
- C: broad trial 43, RUS 1:5.

Run the confirmation on AutoDL with only the training data available:

```bash
LGBM_THREADS=24 bash models/lightgbm/run_autodl.sh confirm
```

The command evaluates five new, pre-specified RUS seed offsets on all five
locked folds for every candidate, for 75 fits in total. The zero offset used
during tuning is deliberately excluded. Only the training-fold legitimate
rows change between seeds; features, folds, candidate hyperparameters,
LightGBM random state, early stopping, and validation distributions remain
fixed.

The primary analysis unit is each seed's five-fold mean FPR, giving five
replicates per candidate. The 75 individual fits are retained as descriptive
evidence but are not treated as 75 independent observations. An absolute mean
FPR gap of 0.002 is marked as practically close; candidate selection still
requires review of seed-to-seed standard deviation, worst-seed performance,
paired seed wins, PR-AUC, and precision.

A mean-FPR leader is marked as a clear winner only when it is ahead by at
least 0.002 absolute FPR and wins at least four of five paired seed comparisons
against every challenger. Otherwise the provisional order uses seed-level
standard deviation, worst-seed FPR, mean FPR, and PR-AUC, in that order, and
remains subject to review before anything is locked.

Important outputs below `runs/lightgbm/seed_confirmation/` are:

- `all_fold_metrics.csv`: all 75 candidate/seed/fold aggregate results.
- `seed_summary.csv`: five-fold summaries for the 15 candidate/seed runs.
- `candidate_ranking.csv`: candidate means, seed variability, worst seed,
  worst fold, discrimination metrics, and best-iteration distribution.
- `paired_seed_differences.csv` and `paired_comparison_summary.csv`: paired
  seed-level candidate comparisons.
- `confirmation_summary.json`: leading candidate and the required next step.
- `candidates/*/seed_offset_*/fold_*.json`: resumable fit checkpoints.

Confirmation does not use the final test set and does not itself lock the
deployment threshold. After reviewing the confirmation, lock one candidate
and a unified tree-count rule, retrain fixed-tree fold models to create
comparable OOF scores, and choose one threshold on pooled original-prevalence
OOF predictions. Only then may the final test set be evaluated once.

## Fixed-tree ensemble OOF and threshold lock

The completed seed confirmation selects candidate C, the broad-search trial
43 configuration with RUS 1:5. Its confirmation mean FPR is 0.190844 and its
seed-level standard deviation is 0.000627. It beats A and B on four of five
paired seeds. The locked tree count is 248, the median best iteration across
candidate C's 25 confirmation fits.

Generate comparable fixed-tree OOF scores with:

```bash
LGBM_THREADS=24 bash models/lightgbm/run_autodl.sh lock-oof
```

This stage first verifies the complete 75-fit confirmation evidence and all
locked input hashes. It then trains five fixed 248-tree models per fold using
the same five seed offsets, for 25 fits in total. No early stopping is used.
The five probabilities for each validation row are averaged, producing one
five-seed ensemble OOF score for every training row.

The single business threshold is the highest pooled ensemble OOF score that
attains recall of at least 0.80. Fold-specific thresholds are diagnostic only
and are never averaged. The test dataset is not accepted by this command.

Important outputs below `runs/lightgbm/fixed_oof_lock/` are:

- `status.json`: progress through the 25 fixed fits and final lock status.
- `model_fold_metrics.csv`: diagnostics for each fold/seed member model.
- `ensemble_fold_metrics.csv`: ensemble discrimination and operating metrics
  by fold, including performance at the single pooled threshold.
- `oof_operating_points.csv`: pooled OOF operating point evidence.
- `locked_model_spec.json`: candidate C parameters, 248 trees, five-member
  ensemble rule, sampling seeds, and the locked threshold for final testing.
- `oof_ensemble_predictions.parquet`: confidential row-level averaged OOF
  scores.
- `folds/*/seed_offset_*/predictions.parquet`: confidential resumable member
  predictions.

After reviewing these outputs, the next and final modelling action is to train
the five locked models on all training data and evaluate the averaged score on
the sealed test set exactly once at the locked threshold.

## One-time final test

The final-test command requires a literal acknowledgement phrase and has no
force/reselection option:

```bash
bash models/lightgbm/run_autodl.sh final-test \
  --confirm-final-test I_UNDERSTAND_TEST_IS_ONE_TIME
```

Before opening the test Parquet, the command verifies the train, test, feature,
and fixed-OOF lock hashes. It creates `FINAL_TEST_OPENED.json` when the sealed
test is first accessed. It also writes `FINAL_TEST_CONSUMPTION.json` into the
fixed-OOF lock directory, preventing the same locked model from being tested
again through a different output directory. Interrupted member training can
resume only with the identical specification and original output directory;
after `final_test_summary.json` is written, rerunning is prohibited.

Five 248-tree models are trained on all training fraud rows and five separately
undersampled legitimate sets using the locked seeds `105042`, `205042`,
`305042`, `405042`, and `505042`. Their test fraud probabilities are averaged
and evaluated only at threshold `0.23669952663465765`. No test-label-derived
threshold or alternate model comparison is calculated. Categories not present
in training are handled as LightGBM missing values and counted in the private
run context.

The command saves all five private LightGBM model text files for subsequent
SHAP analysis, member predictions, averaged final predictions, fixed-threshold
metrics, and an immutable final summary below `runs/lightgbm/final_test/`.

## Post-test ensemble SHAP

Run SHAP only after the final-test status is complete:

```bash
bash models/lightgbm/run_autodl.sh shap
```

The default explains the 500 highest-scoring final-test rows. It uses
LightGBM's native exact TreeSHAP contributions for each saved member and
averages the five signed contributions. Member-level and ensemble additivity
are verified against raw model scores, and reloaded model probabilities are
verified against the saved final predictions.

TreeSHAP contributions are in raw-score/log-odds space. They exactly explain
the mean member raw score. Because the deployed ensemble averages member
probabilities rather than raw scores, the SHAP sum must not be interpreted as
the ensemble fraud probability itself. This limitation is explicitly recorded
in `shap_metadata.json`.

Important SHAP outputs under
`runs/lightgbm/final_test/shap_top_score/` are:

- `shap_summary.csv`: global importance and contribution distributions.
- `shap_values_wide.parquet` and `shap_values_long.parquet`: averaged exact
  TreeSHAP contributions for the explained rows.
- `shap_feature_values_long.parquet`: corresponding original feature values.
- `shap_explain_rows.csv`: source rows, test predictions, raw scores, and
  additivity errors.
- `shap_member_checks.csv`: member model hashes and exactness checks.
- `shap_metadata.json`: method, selection, hashes, interpretation, and QA.

SHAP is explanation-only: it may not change the model, seeds, threshold, or
reported final-test metrics.

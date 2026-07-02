# LightGBM final outputs

The LightGBM modelling workflow is complete. The locked model is candidate C
(global trial 43), using a 5-member RUS 1:5 ensemble and the fixed OOF threshold
`0.23669952663465765`.

## Final records

- `final_results_summary.json`: authoritative final metrics, locked configuration,
  RFQC comparison, SHAP summary, and source archive hashes.
- `final_model_comparison.csv`: compact LightGBM versus RFQC test comparison.
- `stage_progression.csv`: modelling-stage performance history.
- `experiment_archive/`: versioned experiment records and source archives.
- `private_backup_manifest.json`: checksum and handling record for the external
  private backup.

## Privacy and final-test policy

The repository archive excludes trained model files, row-level predictions, and
row-level SHAP/feature-value files. These remain only in the external private
backup recorded by `private_backup_manifest.json`.

The final test set has already been consumed. Its recorded metrics are final and
the test workflow must not be rerun for model selection or further tuning.

See `experiment_archive/README.md` for the archive layout and provenance details.

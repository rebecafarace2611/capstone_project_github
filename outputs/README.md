# Analysis Outputs

This private repository includes reviewed analysis reports, aggregate metrics,
configuration files, threshold curves, tuning records, and other
non-row-level results from this directory.

Current local structure:

```text
outputs/leakage_analysis/
outputs/descriptive_analysis/
outputs/rfqc/folds/
outputs/rfqc/experiment_archive/
```

The following remain excluded:

- Parquet files containing predictions or fold assignments.
- Any row-level CSV predictions.
- Fitted model files.
- Raw, processed, training, and test data stored under `data/`.

These outputs are intended only for authorised members of the private
repository and must not be copied into public issues or repositories.

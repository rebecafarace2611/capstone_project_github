# RFQC Experiment Archive

## Baselines

- `baselines/baseline_500`: untuned 500-tree OOB baseline.
- `baselines/baseline_1000`: untuned 1,000-tree OOB baseline.
- `baselines/baseline_2000`: untuned 2,000-tree OOB baseline.
- `baselines/baseline_3000`: untuned 3,000-tree OOB baseline.

Archived baseline filenames include the tree count so they remain identifiable
when downloaded separately.

## Tuning

- `tuning/01_quick_search_500`: 8-candidate quick search.
- `tuning/02_local_refine_gini_500`: 4-candidate local Gini refinement.

All baseline and tuning runs used the same training data, approved feature set,
fold assignments where applicable, and random seed.

## Final Evaluation

- `final/final_local_qstar_3000`: the single locked final evaluation.

This directory contains the locked configuration, run context, final metrics,
111,020 row-level predictions, threshold curve, tree-convergence results, and
the combined baseline/CV/test summary. The approximately 38 GB fitted model is
retained on the server and is not included in this local archive.

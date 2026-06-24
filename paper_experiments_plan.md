# Config-Driven Incremental Paper Experiments Plan

Build a clean experiment framework under `paper_experiments/` with a YAML config
file controlling datasets, metrics, seeds, k-grid, output paths, and resume
behavior. Experiments are incremental: adding a dataset or metric should only
compute missing artifacts, while existing completed results are skipped unless
`--force` is used.

Default protocol:

- Stratified 80/20 splits over seeds `0, 1, 2, 3, 4`.
- Preprocess targets with integer label encoding.
- Use numeric columns as interval features without scaling.
- One-hot encode categorical columns as boolean features.
- Stream raw LazyFCA explanations one test query at a time; do not cache full
  explanations.
- Evaluate vanilla/all-hypotheses LazyFCA, global top-k ranking, and random
  top-k sanity baselines.
- Plot `k` increasing from 1 upward with vanilla LazyFCA as a horizontal
  reference line.

Outputs are written under `paper_experiments/results/<run_name>/`:

- `chunks/`: per dataset/seed/method/metric result chunks
- `manifest.jsonl`: incremental execution log
- `topk_results.csv`: combined top-k rows
- `vanilla_lazyfca.csv`: all-hypotheses reference rows
- `summary_by_dataset_metric.csv`: mean/std over seeds
- `compactness_summary.csv`: best and compact-k summaries
- `dataset_diagnostics.csv`: feature/class/classifier/singleton diagnostics
- `plots/`: compactness-first top-k plots

Deferred:

- Randomized LazyFCA is not implemented in v1, but the method registry is kept
  explicit so it can be added later.
- IPS-KNN is not included in v1.

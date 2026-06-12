# Experiment Context

This file is a local handoff note for Codex chats working on the LazyFCA
top-k ranking experiments. Keep it updated incrementally as the analysis
progresses.

## Project Goal

The repo studies LazyFCA classifiers for tabular classification. For each test
object, LazyFCA generates many local positive and negative classifiers. The
experiments pool those classifiers, rank them by a metric, keep only top-k, and
predict from the surviving class balance.

The main research question is:

Can we rank local FCA classifiers so well that we keep very few classifiers,
possibly a single classifier, while preserving high F1 and gaining
interpretability?

## Current Metric Work

Recently added/tested metrics include:

- `precision`
- `precision_log_tp`
- `precision_sqrt_tp`
- `log_odds_ratio`
- `log_odds_ratio_log_tp`
- `query_weighted_precision`
- `query_weighted_precision_log_tp`
- `query_weighted_precision_sqrt_tp`

The new metrics are implemented in:

- `lazyfca/metrics.py`
- `lazyfca/calculators.py`

They have also been added to top-k experiment notebooks, including:

- `experiments/rice_topk_ranking_raw.ipynb`
- `experiments/sonar_topk_ranking_raw.ipynb`
- `experiments/spambase_topk_ranking_raw.ipynb`
- `experiments/parkinsons_topk_ranking_raw.ipynb`

## Important TP Clarification

For a generated classifier, `tp` / `supporters_covered` is computed over the
training supporters only. The test query object is not counted in `tp`.

However, the training object used to generate the classifier is counted. A
classifier is built from the query and one same-class training source object:

```text
numeric_minimum = min(query_numeric, source_numeric)
numeric_maximum = max(query_numeric, source_numeric)
```

Then coverage is measured over all same-class training supporters. Because the
source object lies inside the interval by construction, `tp >= 1`.

Therefore:

```text
tp = 1
```

means the classifier covered only its generating training source and no
additional same-class training objects. It does not mean the query was counted,
but it also is not independent support beyond the source. Treat such classifiers
as local source-query rules, close to a nearest-neighbor style explanation, not
as rules validated by multiple training supporters.

This is important: every generated same-class classifier has at least this
tautological source support. For support-aware metrics, the meaningful extra
support is closer to:

```text
extra_tp = tp - 1
```

If a dataset has mostly or entirely `tp = 1`, then raw `tp`, `precision`, and
smoothed log-odds can be misleading as evidence of rule support. In that case
the experiment is mostly ranking nearest-neighbor-like source-query intervals,
not genuinely multi-object FCA rules. Sonar is exactly this case: every
classifier has `tp=1`. Spambase is mixed: many classifiers have `tp=1`, but
some negative-class classifiers have very large `tp`, which is why direct
support multipliers become biased toward broad negative rules.

Updated methodological conclusion after running the `tp >= 2` ablation:
classifiers with `tp = 1` should not be dropped by default. They are not fake
classifiers: they cover one same-class training object and, when `fp = 0`, no
counter-class objects. They should be interpreted as singleton source-query
classifiers, close to a nearest-neighbor-style local explanation. The important
caveat is that `tp = 1` is tautological source support, not evidence that the
description generalizes to additional same-class training objects.

The stricter multi-support filter is:

```text
supporters_covered >= 2
```

for both positive and negative classifiers. This should be reported as an
ablation/diagnostic, not used as the default classifier definition. Experiments
should also report how many queries lose all classifiers under this filter,
because datasets such as Sonar may collapse completely. If that happens, it is
itself an important finding: the dataset/rule construction mostly or entirely
produces singleton source-query rules under the current setup.

This supported-rule filter has been added as a separate section to
the Rice, Sonar, and Spambase raw top-k notebooks. It creates
`supported_explanations` with `tp >= 2`, reports classifier-retention
diagnostics, and writes supported-run outputs to dataset-specific folders:

```text
experiments/generated_plots/rice_topk_ranking_raw_supported_tp2/
experiments/generated_plots/sonar_topk_ranking_raw_supported_tp2/
experiments/generated_plots/spambase_topk_ranking_raw_supported_tp2/
```

Observed `tp >= 2` ablation results:

- Sonar collapses completely: every generated classifier has `tp = 1`, so the
  supported run keeps zero classifiers for all 42 test objects and all F1 scores
  are 0. The raw singleton geometric rankings were very strong, with
  `query_weighted_precision` / `query_similarity` / `interval_tightness` /
  `simplicity_prior` around F1 0.9268 at `k = 2`.
- Rice becomes too sparse under the filter. In the currently generated
  supported output, many queries lose at least one side and some lose all
  classifiers; top F1 drops to about 0.7857, far below the raw run where the
  best metrics reached roughly 0.91-0.92. Re-run Rice from a clean full
  notebook before treating the exact supported numbers as final, because the
  saved diagnostics currently contain only 20 rows.
- Spambase survives the filter, but performance still drops. Raw
  `query_weighted_precision` reached about F1 0.908 at `k = 5`; the supported
  `tp >= 2` version reaches about F1 0.812 at `k = 4`. This shows that
  multi-support rules exist in Spambase, but singleton local classifiers were
  carrying useful signal.

Current recommendation: keep `tp = 1` classifiers in the main top-k ranking
experiments, but label them clearly as singleton/source-query classifiers. Use
`tp >= 2` as a stricter ablation to measure how much performance depends on
multi-object support. For support-aware ranking metrics, do not blindly reward
the tautological source support; if support is used, consider using
`extra_tp = tp - 1` or using support only as a tie-breaker among pure/local
classifiers.

## Important Rice Finding

The Rice notebook is the active analysis focus:

- `experiments/rice_topk_ranking_raw.ipynb`

Do not analyze Rice only from `topk_summary.csv`; inspect plots and low-k
behavior. The key observation is that some metrics work very well at `k=1`.

Plain `precision` and smoothed `log_odds_ratio` both measure purity, but their
low-k behavior differs sharply.

Plain precision:

```text
precision = tp / (tp + fp)
```

Every classifier with `fp = 0` receives score `1.0`, regardless of whether it
covers `tp = 1` or `tp = 200`. In Rice, every test object has perfect-precision
classifiers from both classes. Because the pooled list is built with positive
classifiers first and Python sorting is stable, `precision` at `k=1` chooses a
positive classifier for every test object.

Observed Rice top-1 behavior:

```text
precision, k=1:
TP=163, TN=0, FP=218, FN=0
F1=0.5993
```

So the low-k drop for precision is mostly a tie/order artifact, not evidence
that purity is useless.

Smoothed `log_odds_ratio`:

```text
log_odds_ratio = (2 * tp + 1) / (2 * fp + 1)
```

When `fp = 0`, this becomes:

```text
2 * tp + 1
```

So among pure classifiers it breaks ties by support (`tp`). This is why
`log_odds_ratio` does not collapse near `k=1`.

Observed Rice top-1 behavior:

```text
log_odds_ratio, k=1:
TP=147, TN=201, FP=17, FN=16
F1=0.8991

log_odds_ratio_log_tp, k=1:
TP=148, TN=202, FP=16, FN=15
F1=0.9052
```

Interpretation: the current "log-odds" metric is useful less because it is a
true statistical log-odds ratio and more because smoothing turns it into a
support-sensitive purity score. It prefers low `fp`, and among pure rules it
prefers larger `tp`.

## Why Precision Support Multipliers Got Worse

Metrics like:

```text
precision * log(1 + tp)
precision * sqrt(tp)
```

do not preserve "purity first." A large but impure classifier can outrank a
small pure classifier. On Rice this happened often:

```text
precision_log_tp top-1 impure classifiers: 343 / 381
precision_sqrt_tp top-1 impure classifiers: 372 / 381
```

So the support multiplier variants are not automatically better. The current
lesson is that support should help break ties among near-pure classifiers, but
should not overpower the false-positive penalty.

## Current Working Hypothesis

For Rice:

- Best single-classifier interpretability: `log_odds_ratio_log_tp`
- Strong runner-up at `k=1`: `log_odds_ratio`
- Best compact peak: `query_weighted_precision` around small k, but not as
  strong at exactly `k=1`
- Best raw F1 may still be `precision`, but it needs many classifiers and is
  less interpretable
- `precision_log_tp`, `precision_sqrt_tp`, and
  `query_weighted_precision_sqrt_tp` look weak on Rice

## Next Analysis Direction

The plan was to go dataset by dataset:

1. Finish Rice analysis, especially explaining metric behavior from plots and
   classifier-level statistics.
2. Move to Sonar next:
   `experiments/sonar_topk_ranking_raw.ipynb`
3. Compare whether the Rice mechanism generalizes or is dataset-specific.

When analyzing a notebook, check:

- `experiments/generated_plots/<notebook_stem>/all_metrics_topk_f1.png`
- individual metric plots
- `topk_summary.csv`
- the last retention/ranking cells in the notebook
- low-k behavior, especially `k=1`
- whether metric differences are caused by true scoring behavior or tie/order
  artifacts

## Recovered Chat Context

This note summarizes context recovered from local Codex session files. The chat
named `Check work status` was a bridge/recovery chat. It recovered an earlier
chat called `Explain log-odds metrics`, updated notebooks for the new metrics,
and then started the Rice result analysis.

## Sonar Finding

Active notebook:

- `experiments/sonar_topk_ranking_raw.ipynb`

Sonar has 166 train objects, 42 test objects, and 60 numeric features. The
baseline threshold-only classifier predicts every test object as positive:

```text
TP=22, TN=0, FP=20, FN=0
F1=0.6875
```

The key mechanism is different from Rice. In Sonar, every generated classifier
in the pooled explanations has the same contingency values:

```text
tp = 1
fp = 0
precision = 1.0
log_odds_ratio = 3.0
```

Therefore all contingency/purity metrics are constant and cannot rank the
classifiers:

- `precision`
- `precision_log_tp`
- `precision_sqrt_tp`
- `log_odds_ratio`
- `log_odds_ratio_log_tp`
- `balanced_precision_proxy`

Because the pooled list is built as positive classifiers first, then negative
classifiers, stable sorting keeps positives first under ties. Since there are
89 positive and 77 negative classifiers per query, these metrics predict
positive for every k and stay flat at F1 `0.6875`.

The geometric metrics do work:

```text
interval_tightness, k=1: F1=0.9130
interval_tightness, k=2: F1=0.9268

simplicity_prior, k=2: F1=0.9268
query_similarity, k=2: F1=0.9268
query_weighted_precision*, k=2: F1=0.9268
```

Definitions in the current implementation:

```text
normalized_widths = interval_widths / dataset_numeric_range
interval_tightness = 1 - mean(normalized_widths)
description_volume = product(normalized_widths)
query_similarity = interval_tightness  # for Sonar, because there are no binary features
simplicity_prior = 1 / (1 + (1 - interval_tightness))  # for Sonar, no binary complexity
```

For Sonar, `query_similarity` and `simplicity_prior` are effectively monotonic
variants of `interval_tightness`, so they produce the same ranking. The
`query_weighted_precision*` metrics also collapse to the same ranking because
`precision = 1` and `tp = 1` for every classifier, making their contingency
part constant.

`description_volume` is weaker because Sonar is high-dimensional. It multiplies
60 normalized widths, so many volumes become extremely tiny or exactly zero.
Any zero/near-zero width dominates the product, creating many ties and making
the metric much less stable than the mean-width based `interval_tightness`.

## Spambase Finding

Active notebook:

- `experiments/spambase_topk_ranking_raw.ipynb`

Spambase uses an 80% stratified subset, then a 90/10 train/test split:

```text
train = 3312 objects
test = 368 objects
features = 57 numeric, 0 binary
train class counts: pos=1305, neg=2007
test class counts: pos=145, neg=223
```

The baseline threshold-only classifier predicts every test object as negative:

```text
TP=0, TN=223, FP=0, FN=145
F1=0.0
```

The best compact metric is `query_weighted_precision`:

```text
query_weighted_precision, k=1:
TP=126, TN=208, FP=15, FN=19
F1=0.8811

query_weighted_precision, k=5:
TP=129, TN=213, FP=10, FN=16
F1=0.9085
```

Since Spambase is numeric-only:

```text
query_similarity = interval_tightness
query_weighted_precision = precision * query_similarity
```

So the successful metric is selecting very local, usually pure classifiers near
the query.

The support-weighted variants fail:

```text
query_weighted_precision_log_tp, k=1: F1=0.0
query_weighted_precision_sqrt_tp, k=1: F1=0.0
```

At top-1, both variants predict every test object as negative:

```text
query_weighted_precision_log_tp:
truth_pos predicted_pos = 0 / 145
truth_neg predicted_neg = 223 / 223

query_weighted_precision_sqrt_tp:
truth_pos predicted_pos = 0 / 145
truth_neg predicted_neg = 223 / 223
```

The reason is class/support asymmetry. Positive classifiers have small support,
while negative classifiers can have very large support:

```text
positive classifier tp: median=1, max=21
negative classifier tp: median=1, max=205
```

`query_weighted_precision` is bounded by the locality/precision scale, so a
very local pure rule can win. But multiplying by `log(1 + tp)` or `sqrt(tp)`
lets broad high-support negative rules dominate. Example from a positive test
case:

```text
local positive rule:
label=1, tp=1, fp=0, precision=1.0, similarity=1.0
query_weighted_precision = 1.0
query_weighted_precision_log_tp = 0.693
query_weighted_precision_sqrt_tp = 1.0

broad negative rule:
label=0, tp=170, fp=7, precision=0.960, similarity=0.968
query_weighted_precision = 0.930
query_weighted_precision_log_tp = 4.781
query_weighted_precision_sqrt_tp = 12.124
```

Thus support weighting overwhelms the local-query signal and shifts the top
ranked classifiers toward negative-class broad rules. This is the same general
warning as Rice, but stronger: support should not be multiplied in directly
unless it is constrained to act only as a tie-breaker or is normalized by class.

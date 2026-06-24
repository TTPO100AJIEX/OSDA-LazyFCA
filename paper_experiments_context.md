# Paper Experiments Context

This repo studies LazyFCA classifiers for tabular classification. For each test
object, LazyFCA generates local pattern classifiers by intersecting the query
with each training object of every class. Each generated classifier has a
pattern description and contingency-style coverage metrics (`tp`, `fp`, `tn`,
`fn`) measured against same-class supporters and opposite-class opposers.

The paper experiment goal is to test whether a global ranking over all generated
class hypotheses can discard harmful/noisy classifiers and keep a compact set
without losing predictive quality. Vanilla LazyFCA/all-hypotheses aggregation is
included as a reference, but it is not treated as the ideal baseline: it can be
bad because many generated classifiers are harmful when all are counted equally.

Important caveat: every classifier is generated from one same-class source
object, so `tp >= 1` by construction. `tp = 1` means singleton/source-query
support, not independent multi-object support. Diagnostics should report
singleton rates, and conclusions should distinguish compact local classifiers
from multi-object generalized rules.

The default config now enables all registered LazyFCA metric fields for
systematic comparison. Individual metrics can be disabled in
`paper_experiments/config.yaml` or selected from the CLI with `--metrics`.

Primary method for v1:

- Put all class-specific classifiers for a query into one pool.
- Rank globally by one metric.
- Keep top `k`.
- Predict by retained classifier counts per class.
- Break equal-count ties by summed ranking scores, then training class prior,
  then lower encoded class label.

Experiments live outside the old exploratory `experiments/` folder under
`paper_experiments/`. They are config-driven and incremental. Full explanations
are not cached because they can exceed RAM/disk limits on larger datasets.
Instead, the runner streams one query at a time, computes all missing metric
chunks requested for that dataset/seed, writes compact result CSVs, and discards
the explanation before moving to the next query. Existing result chunks are
skipped unless `--force` is used.

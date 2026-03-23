# Changes By Alan

This document explains the code changes made to the core `lazyfca` package, why they were made, and the mathematical meaning of the modified and newly added metrics.

It focuses on the following files:

- `lazyfca/classifier.py`
- `lazyfca/dataset.py`
- `lazyfca/lazyfca.py`

## 1. High-Level Goal

The original repository already implemented the main LazyFCA idea:

1. Take a query object.
2. Intersect it with training objects to form hypotheses.
3. Keep the hypotheses that pass some metric thresholds.
4. Aggregate surviving positive and negative hypotheses by counting them.

The changes introduced here had three goals:

1. Correct the metrics that were mathematically incorrect.
2. Add new metrics for:
   - interval specificity / simplicity,
   - query-aware ranking,
   - FCA-friendly stability proxies.
3. Add optional ranking and top-k pruning, so the system can do more than thresholding.

## 2. Changes In `lazyfca/classifier.py`

This file originally contained:

- the `Hypothesis` class,
- the `Classifier` class,
- the `Metrics` dataclass,
- the `get_metrics()` function.

It was already responsible for computing the confusion-style statistics of each hypothesis. Most of the work happened in this file.

### 2.1 What Was Added Structurally

The following attributes were added to each `Classifier`:

- `query`
- `source`
- `dataset`

These were not stored before.

### Why They Were Added

The original code could compute only class-coverage metrics such as:

- support,
- precision,
- lift,
- WRAcc,
- etc.

The new metrics require additional context:

- `query`: needed for query-aware similarity,
- `source`: useful because the hypothesis is generated from the query-source pair,
- `dataset`: needed for global numeric ranges and normalization.

Without storing these, interval- and query-based metrics could not be defined cleanly.

## 3. Corrections To Existing Metrics

Let:

- `tp` = number of supporters covered by the hypothesis
- `fp` = number of opposers covered by the hypothesis
- `fn` = number of supporters not covered
- `tn` = number of opposers not covered

and:

- `P = tp + fn`
- `N = fp + tn`
- `T = P + N`

The hypothesis acts like a binary test:

- covered
- uncovered

Therefore, the correct impurity-based metrics must evaluate the class distribution in:

- the covered group: `(tp, fp)`
- the uncovered group: `(fn, tn)`

and compare it to the parent distribution `(P, N)`.

### 3.1 Information Gain

#### Old Implementation

The old code effectively used:

\[
IG_{\text{old}} = -\left(\frac{tp}{P}\ln\frac{tp}{P} + \frac{tn}{N}\ln\frac{tn}{N}\right)
\]

#### Why It Was Incorrect

This is not the information gain of the classifier split.

Problems:

- it used only `tp/P` and `tn/N`,
- it ignored `fp` and `fn`,
- it did not compare the impurity before and after the split,
- it did not model the two induced groups (`covered`, `uncovered`).

So it was not measuring "how much the hypothesis improves class separation".

#### New Implementation

The corrected formula is:

\[
IG_{\text{new}} = H(P,N) - \frac{tp+fp}{T}H(tp,fp) - \frac{fn+tn}{T}H(fn,tn)
\]

where:

\[
H(a,b) = -\frac{a}{a+b}\log_2\frac{a}{a+b} - \frac{b}{a+b}\log_2\frac{b}{a+b}
\]

#### Why The New Version Is Correct

This is the standard entropy reduction produced by the rule-induced split:

- before using the rule: parent impurity,
- after using the rule: weighted child impurity,
- gain: reduction in impurity.

This still captures purity for classification, but in the correct split-based sense.

### 3.2 Gini Gain

#### Old Implementation

The old code effectively used:

\[
Gini_{\text{old}} = 1 - \left(\frac{tp}{P}\right)^2 - \left(\frac{tn}{N}\right)^2
\]

#### Why It Was Incorrect

This again used only partial information from the confusion table.

It was not a true Gini reduction for the split induced by the hypothesis.

#### New Implementation

\[
Gini_{\text{new}} = G(P,N) - \frac{tp+fp}{T}G(tp,fp) - \frac{fn+tn}{T}G(fn,tn)
\]

where:

\[
G(a,b) = 1 - \left(\frac{a}{a+b}\right)^2 - \left(\frac{b}{a+b}\right)^2
\]

#### Why The New Version Is Correct

This is the weighted decrease in Gini impurity caused by splitting the dataset into:

- covered objects,
- uncovered objects.

### 3.3 Chi-Squared

#### Old Implementation

The old code effectively used:

\[
\chi^2_{\text{old}} = \frac{(tp-P)^2}{P} + \frac{(tn-N)^2}{N}
\]

#### Why It Was Incorrect

This is not the standard chi-squared test for a `2 \times 2` contingency table.

Problems:

- it compared observed counts to class totals rather than expected counts,
- it used only two cells,
- it ignored the full independence structure of the contingency table.

#### New Implementation

First compute the expected counts:

\[
E_{tp} = \frac{(tp+fp)(tp+fn)}{T}
\]

\[
E_{fp} = \frac{(tp+fp)(fp+tn)}{T}
\]

\[
E_{fn} = \frac{(fn+tn)(tp+fn)}{T}
\]

\[
E_{tn} = \frac{(fn+tn)(fp+tn)}{T}
\]

Then:

\[
\chi^2_{\text{new}} =
\frac{(tp-E_{tp})^2}{E_{tp}} +
\frac{(fp-E_{fp})^2}{E_{fp}} +
\frac{(fn-E_{fn})^2}{E_{fn}} +
\frac{(tn-E_{tn})^2}{E_{tn}}
\]

#### Why The New Version Is Correct

This is the standard Pearson chi-squared statistic for a `2 \times 2` table.

### 3.4 G-Test

#### Old Implementation

The old code effectively used:

\[
G_{\text{old}} = 2\left(tp\ln\frac{tp}{P} + tn\ln\frac{tn}{N}\right)
\]

#### Why It Was Incorrect

Again, this was not the standard likelihood-ratio test.

Problems:

- wrong baseline,
- only two cells,
- ignored expected counts,
- could become negative.

#### New Implementation

Using the same expected counts as above:

\[
G_{\text{new}} =
2\left(
tp\ln\frac{tp}{E_{tp}} +
fp\ln\frac{fp}{E_{fp}} +
fn\ln\frac{fn}{E_{fn}} +
tn\ln\frac{tn}{E_{tn}}
\right)
\]

with the usual convention that terms with zero observed count contribute `0`.

#### Why The New Version Is Correct

This is the standard likelihood-ratio statistic for a `2 \times 2` contingency table.

## 4. Numerical Stability / Guarding Issues

The original file also had numerical fragility:

- division by zero could occur,
- `log(0)` could occur,
- `MCC` could fail when its denominator became zero.

### What Was Added

Three helper routines were added:

- safe division,
- safe `x log(x/E)` handling,
- expected contingency table computation.

### Why

Metric code should not crash on edge cases such as:

- a rule covering no objects,
- a perfect rule with zero false positives,
- a degenerate confusion table.

These changes did not alter the intended meaning of the metrics. They only made them robust.

## 5. New Metric Families Added To `Metrics`

The following new metrics were added:

- `interval_tightness`
- `description_volume`
- `simplicity_prior`
- `query_binary_similarity`
- `query_numeric_similarity`
- `query_similarity`
- `query_weighted_precision`
- `query_weighted_wracc`
- `stability`
- `robustness`
- `delta_stability`

They were also added to the display/export dictionary so that notebooks can show them.

## 6. Interval-Specific Metrics

These metrics are especially relevant when most datasets are numerical and hypotheses are interval descriptions.

### 6.1 Hypothesis Interval

For numeric feature `j`, the hypothesis interval is:

\[
[a_j, b_j] = [\min(x_j, y_j), \max(x_j, y_j)]
\]

where:

- `x_j` is the query value,
- `y_j` is the value of the source object used to generate the hypothesis.

So for point-valued numerical data, the width is:

\[
b_j - a_j = |x_j - y_j|
\]

### 6.2 Interval Tightness

Let the global dataset range for feature `j` be:

\[
[m_j, M_j]
\]

Then the normalized width of the hypothesis interval is:

\[
w_j = \frac{b_j - a_j}{M_j - m_j}
\]

when `M_j > m_j`.

Then:

\[
\text{IntervalTightness}(h) = 1 - \frac{1}{d}\sum_{j=1}^{d} w_j
\]

where `d` is the number of numeric features.

### Interpretation

- close to `1`: intervals are narrow, therefore specific,
- close to `0`: intervals are wide, therefore vague.

### Why It Was Added

In interval pattern structures, interpretability depends not only on class purity but also on how sharp the description is.

Two hypotheses may have similar precision, but one may describe:

- a very narrow interval,
- a very broad interval.

The narrow one is often more interpretable.

### 6.3 Description Volume

The description volume is:

\[
\text{DescriptionVolume}(h) = \prod_{j=1}^{d} w_j
\]

where `w_j` are the normalized widths.

### Interpretation

- low volume: compact description,
- high volume: broad description.

### Why It Was Added

This gives a multiplicative notion of interval complexity. It is stronger than average width because it penalizes hypotheses that are broad across many dimensions.

### 6.4 Simplicity Prior

Let:

- binary complexity = proportion of active binary features kept in the hypothesis,
- interval complexity = `1 - interval_tightness`.

Then:

\[
\text{DescriptionComplexity}(h) = \text{BinaryComplexity}(h) + \text{IntervalComplexity}(h)
\]

and:

\[
\text{SimplicityPrior}(h) = \frac{1}{1 + \text{DescriptionComplexity}(h)}
\]

### Interpretation

- higher simplicity prior means a simpler, more compact hypothesis,
- lower simplicity prior means a more complex description.

### Why It Was Added

This gives a simple and explicit interpretability-oriented regularizer.

## 7. Query-Aware Metrics

The original implementation treated all surviving hypotheses equally. But LazyFCA is query-local: the same hypothesis quality should depend partly on how close the hypothesis stays to the query object.

### 7.1 Binary Query Similarity

Let:

- the query have some active binary features,
- the hypothesis retain only those that survive intersection.

Then:

\[
\text{QueryBinarySimilarity}(h,x) =
\frac{\# \text{ active query binary features preserved in } h}
{\# \text{ active query binary features in } x}
\]

### Interpretation

- high value: much of the query’s binary identity is retained,
- low value: the hypothesis became too generic in the binary part.

### 7.2 Numeric Query Similarity

For numerical features, similarity is defined through interval width.

For one numeric feature:

\[
[a_j,b_j] = [\min(x_j,y_j), \max(x_j,y_j)]
\]

So:

\[
b_j-a_j = |x_j-y_j|
\]

Normalizing by the dataset range:

\[
\text{LocalNumericSimilarity}_j = 1 - \frac{|x_j-y_j|}{M_j-m_j}
\]

The implemented aggregate is exactly the interval tightness:

\[
\text{QueryNumericSimilarity}(h,x) = \text{IntervalTightness}(h)
\]

### Interpretation

- narrow interval means source is close to query,
- broad interval means source is far from query.

### 7.3 Overall Query Similarity

The total query similarity is the average of:

- binary similarity, when binary features exist,
- numeric similarity, when numeric features exist.

### 7.4 Query-Weighted Precision

\[
\text{QueryWeightedPrecision}(h,x) = \text{Precision}(h)\cdot \text{QuerySimilarity}(h,x)
\]

### 7.5 Query-Weighted WRAcc

\[
\text{QueryWeightedWRAcc}(h,x) = \text{WRAcc}(h)\cdot \text{QuerySimilarity}(h,x)
\]

### Why These Were Added

They support the intended paper direction:

- rank hypotheses,
- keep only those that are not only class-discriminative,
- but also close to the current query.

This is especially natural in LazyFCA because classification is performed one query at a time.

## 8. FCA-Friendly Stability In The Lazy Setting

### 8.1 Classical Concept Stability

For a formal concept `C = (Ext(C), Int(C))`, classical stability is:

\[
\text{Stab}(C) =
\frac{|\{S \subseteq Ext(C) \mid S' = Int(C)\}|}
{2^{|Ext(C)|}}
\]

This measures how often subsets of the extent regenerate exactly the same intent.

### 8.2 Why A New Definition Was Needed

LazyFCA does not build the full concept lattice.

So, in the lazy setting:

- we do not have the lattice order,
- we do not enumerate all subconcepts,
- we do not directly have concept extents in the FCA sense.

What we do have is:

- a local hypothesis `h`,
- the set of same-class training objects covered by `h`.

So the question becomes:

> If we keep only a random subset of the covered supporters, how likely is it that the same hypothesis is still regenerated?

This is the local lazy analogue of concept stability.

### 8.3 Covered Supporter Set

For a hypothesis `h`, let:

\[
E_h = \{g \in \text{supporters} \mid h \sqsubseteq \delta(g)\}
\]

In code, this is the set of covered supporters.

### 8.4 Witness Sets For Binary Features

Suppose the query has a binary feature active, but the hypothesis does not retain it.

That means the feature was dropped during intersection.

To preserve this dropped state, a subset of `E_h` must still contain at least one covered supporter contradicting that feature.

For each dropped binary feature `f`, define the witness set:

\[
W_f^{bin} = \{g \in E_h \mid g_f = 0\}
\]

If a subset loses all objects in `W_f^{bin}`, then that feature may reappear and the hypothesis changes.

### 8.5 Witness Sets For Interval Endpoints

For numeric feature `j`, the hypothesis interval is:

\[
[a_j,b_j]
\]

To preserve this same interval in a subset, the subset must still contain:

- at least one object achieving the lower bound `a_j`,
- at least one object achieving the upper bound `b_j`.

So define:

\[
W_{j,\min} = \{g \in E_h \mid g_j = a_j\}
\]

\[
W_{j,\max} = \{g \in E_h \mid g_j = b_j\}
\]

If `a_j = b_j`, only one witness family is necessary.

### 8.6 Witness Family Of A Hypothesis

Let:

\[
\mathcal{W}(h) =
\{W_f^{bin}\} \cup \{W_{j,\min}, W_{j,\max}\}
\]

for all relevant binary contradictions and numeric interval boundaries.

These sets define the evidence that keeps the hypothesis unchanged.

### 8.7 Lazy Delta Stability

The implemented local analogue of `\Delta` is:

\[
\Delta_{\text{lazy}}(h) = \min_{W \in \mathcal{W}(h)} |W|
\]

### Interpretation

- small value: at least one defining boundary is supported by very few objects,
- large value: all defining boundaries are well supported.

This mirrors the lattice idea that fragility is determined by the weakest path to a more specific descendant.

### 8.8 Lazy Stability

Assume each covered supporter is kept independently with probability `1/2`.

A witness set `W` survives if at least one of its objects is kept. The probability of that is:

\[
1 - 2^{-|W|}
\]

The implemented local approximation is:

\[
\text{Stability}_{\text{lazy}}(h) \approx \prod_{W \in \mathcal{W}(h)} \left(1 - 2^{-|W|}\right)
\]

### Interpretation

- high stability: many subsets preserve all defining interval boundaries,
- low stability: one or more boundaries depend on only a few objects.

### Important Note

This is an approximation, not exact FCA stability.

Reason:

- witness sets are not generally independent,
- the exact counting would be combinatorial.

Still, the measure is faithful to the intended intuition:

- a hypothesis is stable if many subsets preserve its defining boundaries.

### 8.9 Robustness

In the code, `robustness` is currently exposed with the same numerical value as the lazy stability approximation.

Why:

- in the FCA literature, robustness is closely related to stability,
- the implementation currently uses the equal-weight subset-preservation viewpoint.

This makes the metric naming easier for experimentation while preserving an FCA-friendly interpretation.

## 9. Changes To Threshold Comparison Logic

The `Metrics.is_better_than()` method originally compared only the old metrics.

It now also compares all newly added metrics so that thresholds can be placed on them too.

One special case had to be added:

- `description_volume` is a metric to minimize,
- so its comparison direction is reversed.

This is because:

- smaller volume means a more compact interval description,
- unlike support or precision where larger is better.

## 10. Changes In `lazyfca/dataset.py`

Originally this file only:

- identified boolean columns,
- identified numeric columns,
- split the dataset into positive and negative subsets,
- converted rows to internal samples.

### What Was Added

The dataset now also stores:

- number of binary features,
- number of numeric features,
- dataset-wide numeric minima,
- dataset-wide numeric maxima,
- dataset-wide numeric ranges.

### Why These Were Added

The new interval metrics require normalization.

Without normalization, interval width is not comparable across features:

- width `0.5` is large if the feature lives in `[0,1]`,
- width `0.5` is tiny if the feature lives in `[0,1000]`.

So dataset-level numeric ranges were needed to compute:

- interval tightness,
- description volume,
- numeric query similarity.

### What Was Deleted

No substantive functionality was removed from this file. It was only extended.

## 11. Changes In `lazyfca/lazyfca.py`

Originally `LazyFCA` supported only threshold-based selection.

The logic was:

1. generate classifiers,
2. keep those passing thresholds,
3. count positive and negative survivors.

There was no built-in ranking or top-k pruning.

### What Was Added

The constructor now supports:

- `pos_rank_by`
- `neg_rank_by`
- `pos_top_k`
- `neg_top_k`

and a helper method was added to:

- sort hypotheses by a chosen metric,
- keep only the top-k.

### Why This Was Added

The original code could only answer:

> Is this hypothesis above the threshold?

But the research goal is stronger:

> Which hypotheses are most important, and how many low-importance ones can be removed?

So ranking support was necessary.

### What Changed In Classification

Previously:

- classification used all surviving hypotheses equally.

Now:

- hypotheses can first be threshold-filtered,
- then ranked,
- then truncated to top-k,
- and only then counted.

This enables experiments such as:

- threshold only,
- top-k only,
- threshold plus top-k,
- ranking by query-aware metrics,
- ranking by stability-aware metrics.

### What Was Deleted

No original threshold logic was removed.

Instead, the old behavior was preserved and ranking was added on top of it.

## 12. Summary Of What Was Corrected, Added, And Removed

### Corrected

- information gain
- gini gain
- chi-squared
- g-test
- numerical safety around metric computation

### Added

- interval specificity metrics
- query-aware similarity metrics
- query-weighted importance metrics
- lazy stability / robustness / delta-stability proxies
- ranking and top-k pruning support
- dataset-wide normalization statistics

### Removed

No major algorithmic component was deleted.

The only real removals were:

- the old incorrect active formulas for the four corrected statistical metrics,
- unsafe direct arithmetic that could fail on edge cases.

## 13. Practical Interpretation For Numerical Datasets

When the data are mostly numerical, the new metrics can be read as follows:

- `precision`, `lift`, `WRAcc`, `balanced_precision_proxy`:
  how discriminative the hypothesis is for the class label.

- `interval_tightness`, `description_volume`, `simplicity_prior`:
  how specific and compact the interval description is.

- `query_similarity`, `query_weighted_precision`, `query_weighted_wracc`:
  how local the hypothesis remains to the current query object.

- `stability`, `delta_stability`:
  how fragile or robust the interval boundaries are with respect to the covered supporters.

This gives three complementary views of a hypothesis:

1. class quality,
2. interpretability / compactness,
3. robustness.

These are precisely the ingredients needed for ranking and pruning hypotheses in a paper about interpretability-oriented LazyFCA.

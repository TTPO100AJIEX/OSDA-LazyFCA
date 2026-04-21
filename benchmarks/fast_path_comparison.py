"""
Benchmark: compare the two Numba fast-path designs for one explain_sample() call.

  A) Current fast path  — Numba tp/fp, _score_from_contingency, argsort top-k,
                          create only top-k Classifier objects.

  B) Simplified path    — Numba tp/fp, create ALL n_train Classifier objects with
                          pre-injected metrics, use existing _rank/_get_top_k.

Run from the repo root:
    python3.13 benchmarks/fast_path_comparison.py
"""

import sys, gc, time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn.model_selection

from lazyfca.lazyfca import LazyFCA
from lazyfca.classifier import Classifier
from lazyfca.explanation import Explanation
from lazyfca.numba_kernels import compute_tp_fp

# ---------------------------------------------------------------------------
# Synthetic data matching taxi profile
# ---------------------------------------------------------------------------

N_TOTAL   = 10_000
N_NUMERIC = 10
N_BINARY  = 16
REPEATS   = 7
TOP_K     = 2
RANK_BY   = "youdens_j"

rng = np.random.default_rng(42)
X = pd.DataFrame({f"n{i}": rng.standard_normal(N_TOTAL) for i in range(N_NUMERIC)})
for i in range(4):
    for j in range(4):
        X[f"cat{i}_{j}"] = rng.choice([True, False], N_TOTAL).astype(bool)
y = pd.Series((rng.standard_normal(N_TOTAL) > 0).astype(int))

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.003, stratify=y, random_state=42
)
print(f"Train: {len(X_train)} ({y_train.sum()} pos / {(1-y_train).sum()} neg), "
      f"Test: {len(X_test)}")

clf = LazyFCA(rank_by=RANK_BY, top_k=TOP_K)
clf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# B) Simplified path: Numba + ALL Classifier objects + existing machinery
# ---------------------------------------------------------------------------

def explain_simplified(clf, sample_series):
    sample = clf.dataset.make_sample(sample_series)
    pos, neg = clf.dataset.positive, clf.dataset.negative
    n_pos, n_neg = len(pos), len(neg)

    q_bin     = np.ascontiguousarray(sample.binary)
    q_num     = np.ascontiguousarray(sample.numeric)
    pos_bin_c = np.ascontiguousarray(pos.binary)
    pos_num_c = np.ascontiguousarray(pos.numeric)
    neg_bin_c = np.ascontiguousarray(neg.binary)
    neg_num_c = np.ascontiguousarray(neg.numeric)

    pos_tp, pos_fp = compute_tp_fp(
        q_bin, q_num, pos_bin_c, pos_num_c, pos_bin_c, pos_num_c, neg_bin_c, neg_num_c
    )
    neg_tp, neg_fp = compute_tp_fp(
        q_bin, q_num, neg_bin_c, neg_num_c, neg_bin_c, neg_num_c, pos_bin_c, pos_num_c
    )

    positive_classifiers = [
        clf._make_precached_classifier(
            sample, pos, i, Classifier.Type.POSITIVE, int(pos_tp[i]), int(pos_fp[i])
        )
        for i in range(n_pos)
    ]
    negative_classifiers = [
        clf._make_precached_classifier(
            sample, neg, i, Classifier.Type.NEGATIVE, int(neg_tp[i]), int(neg_fp[i])
        )
        for i in range(n_neg)
    ]

    positive_classifiers, negative_classifiers = clf._get_top_k(
        positive_classifiers, negative_classifiers
    )
    explanation = Explanation(sample, positive_classifiers, negative_classifiers)
    gc.collect()
    return explanation

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

rows = list(X_test.iterrows())

def bench(fn, label, repeats=REPEATS):
    fn(rows[0][1])   # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _, row in rows:
            fn(row)
        times.append(time.perf_counter() - t0)
    per = [t / len(rows) * 1000 for t in times]
    print(f"  {label:<50}  mean {sum(per)/len(per):6.1f} ms/sample"
          f"   min {min(per):6.1f} ms/sample")

print("\nWarming up Numba...", end=" ", flush=True)
t0 = time.perf_counter()
clf.explain_sample(rows[0][1])
explain_simplified(clf, rows[0][1])
print(f"done ({1000*(time.perf_counter()-t0):.0f} ms)\n")

print(f"Benchmarking {len(rows)} test samples × {REPEATS} repeats\n")
bench(clf.explain_sample,                    "A) current  (Numba + top-k objects only)")
bench(lambda r: explain_simplified(clf, r),  "B) simplified (Numba + ALL objects + existing rank)")

# ---------------------------------------------------------------------------
# Correctness: both paths should give identical predictions
# ---------------------------------------------------------------------------

print("\nVerifying predictions match...")
preds_a, preds_b = [], []
for _, row in rows:
    ea = clf.explain_sample(row)
    eb = explain_simplified(clf, row)
    preds_a.append(clf.classify_explanation(ea, trust=True))
    preds_b.append(clf.classify_explanation(eb, trust=True))

match = all(abs(a[0]-b[0]) < 1e-9 for a, b in zip(preds_a, preds_b))
print(f"  Predictions identical: {match}")

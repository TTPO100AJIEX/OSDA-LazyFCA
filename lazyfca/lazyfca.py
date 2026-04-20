from __future__ import annotations

import typing
import math

import dataclasses
import gc
import tqdm
import numpy
import pandas
import joblib

from lazyfca.dataset import Dataset
from lazyfca.dataset import Sample
from lazyfca.dataset import Subset
from lazyfca.explanation import Explanation
from lazyfca.classifier import Classifier
from lazyfca.metrics import Metrics
from lazyfca.metrics import METADATA
from lazyfca.numba_kernels import NUMBA_AVAILABLE
from lazyfca.numba_kernels import compute_tp_fp


_CONTINGENCY_METRICS = frozenset([
    "tp", "fp", "tn", "fn",
    "supporters_covered", "opposers_covered", "supporter_opposer_ratio",
    "support", "error_rate", "precision", "lift", "wracc",
    "balanced_precision_proxy", "youdens_j", "log_odds_ratio",
    "matthews_correlation", "information_gain", "gini_gain",
    "chi_squared", "g_test",
])


def _score_from_contingency(
    metric: str,
    tp: numpy.ndarray,
    fp: numpy.ndarray,
    n_sup: int,
    n_opp: int,
) -> numpy.ndarray:
    """
    Vectorised ranking score (higher = better, minimised metrics already negated)
    from contingency counts, mirroring score_for_ranking() over all classifiers at once.
    n_sup / n_opp are len(supporters) / len(opposers) for the given classifier type.
    """
    tn = n_opp - fp
    fn = n_sup - tp
    tp_f = tp.astype(float)
    fp_f = fp.astype(float)
    tn_f = tn.astype(float)
    fn_f = fn.astype(float)

    def sdiv(a, b):
        b_arr = numpy.full(len(a), b, dtype=float) if numpy.isscalar(b) else numpy.asarray(b, dtype=float)
        return numpy.divide(a, b_arr, out=numpy.zeros_like(a, dtype=float), where=(b_arr != 0))

    if metric in ("tp", "supporters_covered"):
        return tp_f
    elif metric in ("fp", "opposers_covered"):
        return -fp_f
    elif metric == "tn":
        return tn_f
    elif metric == "fn":
        return -fn_f
    elif metric == "supporter_opposer_ratio":
        return sdiv(tp_f, fp_f)
    elif metric == "support":
        return sdiv(tp_f, n_sup)
    elif metric == "error_rate":
        return -sdiv(fp_f, n_opp)
    elif metric == "precision":
        return sdiv(tp_f, tp_f + fp_f)
    elif metric == "lift":
        total = n_sup + n_opp
        base  = n_sup / total if total else 0.0
        return sdiv(sdiv(tp_f, tp_f + fp_f), base) if base else numpy.zeros(len(tp))
    elif metric == "wracc":
        total = n_sup + n_opp
        base  = n_sup / total if total else 0.0
        return (tp_f + fp_f) / total * (sdiv(tp_f, tp_f + fp_f) - base)
    elif metric in ("balanced_precision_proxy", "youdens_j"):
        return sdiv(tp_f, n_sup) - sdiv(fp_f, n_opp)
    elif metric == "log_odds_ratio":
        return (tp_f + 0.5) / (fp_f + 0.5)
    elif metric == "matthews_correlation":
        num   = tp_f * tn_f - fp_f * fn_f
        denom = numpy.sqrt((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f))
        return numpy.where(denom != 0, num / denom, 0.0)
    elif metric == "information_gain":
        total = n_sup + n_opp
        def H(p, n):
            t = p + n
            if t == 0 or p == 0 or n == 0:
                return 0.0
            r = p / t
            return -(r * math.log2(r) + (1 - r) * math.log2(1 - r))
        prior = H(n_sup, n_opp)
        return numpy.array([
            prior - ((tp[i] + fp[i]) * H(tp[i], fp[i]) + (fn[i] + tn[i]) * H(fn[i], tn[i])) / total
            for i in range(len(tp))
        ])
    elif metric == "gini_gain":
        total = n_sup + n_opp
        def G(p, n):
            t = p + n
            if t == 0:
                return 0.0
            r = p / t
            return 1.0 - r ** 2 - (1 - r) ** 2
        prior = G(n_sup, n_opp)
        return numpy.array([
            prior - ((tp[i] + fp[i]) * G(tp[i], fp[i]) + (fn[i] + tn[i]) * G(fn[i], tn[i])) / total
            for i in range(len(tp))
        ])
    elif metric in ("chi_squared", "g_test"):
        scores = numpy.empty(len(tp))
        for i in range(len(tp)):
            t, f, fnn, tnn = int(tp[i]), int(fp[i]), int(fn[i]), int(tn[i])
            tot = t + f + fnn + tnn
            if tot == 0:
                scores[i] = 0.0
                continue
            etp = (t + f) * (t + fnn) / tot
            efp = (t + f) * (f + tnn) / tot
            efn = (fnn + tnn) * (t + fnn) / tot
            etn = (fnn + tnn) * (f + tnn) / tot
            if metric == "chi_squared":
                scores[i] = sum(
                    (o - e) ** 2 / e if e else 0
                    for o, e in [(t, etp), (f, efp), (fnn, efn), (tnn, etn)]
                )
            else:
                scores[i] = 2.0 * sum(
                    o * math.log(o / e) if o > 0 and e > 0 else 0
                    for o, e in [(t, etp), (f, efp), (fnn, efn), (tnn, etn)]
                )
        return scores
    else:
        raise ValueError(
            f"'{metric}' cannot be computed from the contingency table alone; "
            "use a contingency-based rank_by or set use_numba=False"
        )


class LazyFCA:
    Params = Metrics

    def __init__(
        self,
        pos_params: LazyFCA.Params = Metrics(),
        neg_params: LazyFCA.Params = Metrics(),
        pos_weight: float = 1.0,
        pos_rank_by: typing.Optional[str] = None,
        neg_rank_by: typing.Optional[str] = None,
        pos_top_k: typing.Optional[int] = None,
        neg_top_k: typing.Optional[int] = None,
        rank_by: typing.Optional[str] = None,
        top_k: typing.Optional[int] = None,
        use_numba: bool = True,
    ):
        self.pos_params = pos_params
        self.neg_params = neg_params
        self.pos_weight = pos_weight
        self.pos_rank_by = pos_rank_by
        self.neg_rank_by = neg_rank_by
        self.pos_top_k = pos_top_k
        self.neg_top_k = neg_top_k
        self.rank_by = rank_by
        self.top_k = top_k
        self.use_numba = use_numba

    # ------------------------------------------------------------------
    # Numba fast path
    # ------------------------------------------------------------------

    def _numba_applicable(self) -> bool:
        return (
            NUMBA_AVAILABLE
            and self.use_numba
            and self.rank_by in _CONTINGENCY_METRICS
            and self.top_k is not None
            and self.pos_rank_by is None
            and self.neg_rank_by is None
            and self.pos_top_k is None
            and self.neg_top_k is None
            and not any(getattr(self.pos_params, m.attr) is not None for m in METADATA)
            and not any(getattr(self.neg_params, m.attr) is not None for m in METADATA)
        )

    def _make_precached_classifier(
        self,
        query: Sample,
        subset: Subset,
        idx: int,
        type: Classifier.Type,
        tp: int,
        fp: int,
    ) -> Classifier:
        source = Sample(subset.binary[idx], subset.numeric[idx])
        clf = Classifier(query, source, self.dataset, type)
        n_sup = len(clf.supporters)
        n_opp = len(clf.opposers)
        clf.metrics.tp = tp
        clf.metrics.fp = fp
        clf.metrics.tn = n_opp - fp
        clf.metrics.fn = n_sup - tp
        clf.metrics.supporters_covered = tp
        clf.metrics.opposers_covered = fp
        clf.metrics.supporter_opposer_ratio = (tp / fp) if fp != 0 else float("inf")
        return clf

    def _explain_sample_fast(self, sample: Sample) -> Explanation:
        pos, neg = self.dataset.positive, self.dataset.negative
        n_pos, n_neg = len(pos), len(neg)

        q_bin     = numpy.ascontiguousarray(sample.binary)
        q_num     = numpy.ascontiguousarray(sample.numeric)
        pos_bin_c = numpy.ascontiguousarray(pos.binary)
        pos_num_c = numpy.ascontiguousarray(pos.numeric)
        neg_bin_c = numpy.ascontiguousarray(neg.binary)
        neg_num_c = numpy.ascontiguousarray(neg.numeric)

        # Positive classifiers: hypothesis = query ∩ pos_train[i]
        #   tp = covers(positive subset), fp = covers(negative subset)
        pos_tp, pos_fp = compute_tp_fp(
            q_bin, q_num, pos_bin_c, pos_num_c, pos_bin_c, pos_num_c, neg_bin_c, neg_num_c
        )
        # Negative classifiers: hypothesis = query ∩ neg_train[i]
        #   tp = covers(negative subset), fp = covers(positive subset)
        neg_tp, neg_fp = compute_tp_fp(
            q_bin, q_num, neg_bin_c, neg_num_c, neg_bin_c, neg_num_c, pos_bin_c, pos_num_c
        )

        pos_scores = _score_from_contingency(self.rank_by, pos_tp, pos_fp, n_pos, n_neg)
        neg_scores = _score_from_contingency(self.rank_by, neg_tp, neg_fp, n_neg, n_pos)

        # Global top-k via merge of two sorted arrays (same semantics as _get_top_k)
        pos_order = numpy.argsort(pos_scores)[::-1]
        neg_order = numpy.argsort(neg_scores)[::-1]
        top_pos = top_neg = 0
        k = min(self.top_k, n_pos + n_neg)
        while top_pos + top_neg < k:
            if top_neg >= n_neg:
                top_pos += 1
            elif top_pos >= n_pos:
                top_neg += 1
            elif pos_scores[pos_order[top_pos]] >= neg_scores[neg_order[top_neg]]:
                top_pos += 1
            else:
                top_neg += 1

        positive_classifiers = [
            self._make_precached_classifier(
                sample, pos, int(pos_order[i]),
                Classifier.Type.POSITIVE,
                int(pos_tp[pos_order[i]]), int(pos_fp[pos_order[i]]),
            )
            for i in range(top_pos)
        ]
        negative_classifiers = [
            self._make_precached_classifier(
                sample, neg, int(neg_order[i]),
                Classifier.Type.NEGATIVE,
                int(neg_tp[neg_order[i]]), int(neg_fp[neg_order[i]]),
            )
            for i in range(top_neg)
        ]

        explanation = Explanation(sample, positive_classifiers, negative_classifiers)
        gc.collect()
        return explanation

    # ------------------------------------------------------------------
    # Ranking helpers (used by legacy path)
    # ------------------------------------------------------------------

    def _rank(self, classifiers: typing.List[Classifier], rank_by: typing.Optional[str]) -> typing.List[Classifier]:
        return sorted(
            classifiers, key=lambda classifier: classifier.metrics.score_for_ranking(rank_by), reverse=True
        )

    def _rank_and_trim(
        self, classifiers: typing.List[Classifier], rank_by: typing.Optional[str], top_k: typing.Optional[int]
    ) -> typing.List[Classifier]:
        if rank_by is not None:
            classifiers = self._rank(classifiers, rank_by)
        if top_k is not None:
            classifiers = classifiers[:top_k]
        return classifiers

    def _get_top_k(
        self,
        positive_classifiers: typing.List[Classifier],
        negative_classifiers: typing.List[Classifier],
    ) -> typing.Tuple[typing.List[Classifier], typing.List[Classifier]]:
        if self.rank_by is not None:
            positive_classifiers = self._rank(positive_classifiers, self.rank_by)
            negative_classifiers = self._rank(negative_classifiers, self.rank_by)
        if self.top_k is not None:
            top_positive, top_negative = 0, 0
            while top_positive + top_negative < min(self.top_k, len(positive_classifiers) + len(negative_classifiers)):
                next_positive = positive_classifiers[top_positive].metrics.score_for_ranking(self.rank_by)
                next_negative = negative_classifiers[top_negative].metrics.score_for_ranking(self.rank_by)
                if next_positive > next_negative:
                    top_positive += 1
                else:
                    top_negative += 1
            positive_classifiers = positive_classifiers[:top_positive]
            negative_classifiers = negative_classifiers[:top_negative]

        return positive_classifiers, negative_classifiers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = ["LazyFCA"]
        lines.append("=" * 40)

        fitted = hasattr(self, "dataset")
        lines.append(f"  Status       : {'fitted' if fitted else 'not fitted'}")

        if fitted:
            ds = self.dataset
            total = len(ds.positive) + len(ds.negative)
            lines.append(f"  Dataset      : {total} samples "
                         f"(pos={len(ds.positive)}, neg={len(ds.negative)})")
            lines.append(f"  Features     : {ds.binary_feature_count} binary, "
                         f"{ds.numeric_feature_count} numeric")

        lines.append("")
        lines.append("  Filtering thresholds")
        lines.append("  " + "-" * 36)

        def fmt_params(label: str, params) -> None:
            active = {k: v for k, v in dataclasses.asdict(params).items() if v is not None}
            if active:
                kv = ", ".join(f"{k}={v}" for k, v in active.items())
                lines.append(f"  {label:<14}: {kv}")
            else:
                lines.append(f"  {label:<14}: (none)")

        fmt_params("pos_params", self.pos_params)
        fmt_params("neg_params", self.neg_params)
        lines.append(f"  {'pos_weight':<14}: {self.pos_weight}")

        lines.append("")
        lines.append("  Ranking & top-k")
        lines.append("  " + "-" * 36)

        def fmt_opt(label: str, value) -> str:
            return f"  {label:<14}: {value if value is not None else '—'}"

        lines.append(fmt_opt("pos_rank_by", self.pos_rank_by))
        lines.append(fmt_opt("neg_rank_by", self.neg_rank_by))
        lines.append(fmt_opt("rank_by", self.rank_by))
        lines.append(fmt_opt("pos_top_k", self.pos_top_k))
        lines.append(fmt_opt("neg_top_k", self.neg_top_k))
        lines.append(fmt_opt("top_k", self.top_k))
        lines.append(fmt_opt("use_numba", self.use_numba))

        lines.append("=" * 40)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def fit(self, X_train: pandas.DataFrame, y_train: pandas.Series):
        self.dataset = Dataset(X_train, y_train)
        return self

    def classify_explanation(
        self, explanation: Explanation, trust: bool = False, probs: bool = True
    ) -> typing.Tuple[float, float]:
        if trust:
            positive_classifiers = explanation.positive_classifiers
            negative_classifiers = explanation.negative_classifiers
        else:
            positive_classifiers = list(
                filter(
                    lambda classifier: classifier.metrics.is_better_than(self.pos_params),
                    explanation.positive_classifiers,
                )
            )
            negative_classifiers = list(
                filter(
                    lambda classifier: classifier.metrics.is_better_than(self.neg_params),
                    explanation.negative_classifiers,
                )
            )
            positive_classifiers = self._rank_and_trim(positive_classifiers, self.pos_rank_by, self.pos_top_k)
            negative_classifiers = self._rank_and_trim(negative_classifiers, self.neg_rank_by, self.neg_top_k)
            positive_classifiers, negative_classifiers = self._get_top_k(positive_classifiers, negative_classifiers)
        positive, negative = len(positive_classifiers), len(negative_classifiers)
        if not probs:
            return (negative, positive)
        positive *= self.pos_weight
        total = negative + positive
        return (0.5, 0.5) if total == 0 else ((negative / total), (positive / total))

    def classify_explanations(
        self, explanations: typing.List[Explanation], trust: bool = False, probs: bool = True
    ) -> numpy.ndarray:
        return numpy.array([self.classify_explanation(explanation, trust, probs) for explanation in explanations])

    def classify_sample(self, sample: pandas.Series) -> typing.Tuple[float, float]:
        return self.classify_explanation(self.explain_sample(sample), trust=True, probs=True)

    def predict(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> numpy.ndarray:
        return numpy.array(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.classify_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

    def explain_sample(self, sample: pandas.Series) -> Explanation:
        sample = self.dataset.make_sample(sample)
        if self._numba_applicable():
            return self._explain_sample_fast(sample)
        return self._explain_sample_legacy(sample)

    def _explain_sample_legacy(self, sample: Sample) -> Explanation:
        def make_classifiers(
            type: Classifier.Type,
            subset: Subset,
            params: LazyFCA.Params,
            rank_by: typing.Optional[str],
            top_k: typing.Optional[int],
        ):
            classifiers = [Classifier(sample, example, self.dataset, type) for example in subset]
            if any(getattr(params, m.attr) is not None for m in METADATA):
                classifiers = [c for c in classifiers if c.metrics.is_better_than(params)]
            return self._rank_and_trim(classifiers, rank_by, top_k)

        positive_classifiers = make_classifiers(
            Classifier.Type.POSITIVE, self.dataset.positive, self.pos_params, self.pos_rank_by, self.pos_top_k
        )
        negative_classifiers = make_classifiers(
            Classifier.Type.NEGATIVE, self.dataset.negative, self.neg_params, self.neg_rank_by, self.neg_top_k
        )
        positive_classifiers, negative_classifiers = self._get_top_k(positive_classifiers, negative_classifiers)
        explanation = Explanation(sample, positive_classifiers, negative_classifiers)
        gc.collect()
        return explanation

    def explain(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> typing.List[Explanation]:
        return list(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.explain_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

from __future__ import annotations

import typing
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
from lazyfca.numba_kernels import compute_tp_fp


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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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

    def _rank(self, classifiers: typing.List[Classifier], rank_by: typing.Optional[str]) -> typing.List[Classifier]:
        return sorted(classifiers, key=lambda classifier: classifier.metrics.score_for_ranking(rank_by), reverse=True)

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
                while top_positive + top_negative < min(
                    self.top_k, len(positive_classifiers) + len(negative_classifiers)
                ):
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
            lines.append(f"  Dataset      : {total} samples (pos={len(ds.positive)}, neg={len(ds.negative)})")
            lines.append(f"  Features     : {ds.binary_feature_count} binary, {ds.numeric_feature_count} numeric")

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
        pos, neg = self.dataset.positive, self.dataset.negative
        n_pos, n_neg = len(pos), len(neg)

        q_bin = numpy.ascontiguousarray(sample.binary)
        q_num = numpy.ascontiguousarray(sample.numeric)
        pos_bin_c = numpy.ascontiguousarray(pos.binary)
        pos_num_c = numpy.ascontiguousarray(pos.numeric)
        neg_bin_c = numpy.ascontiguousarray(neg.binary)
        neg_num_c = numpy.ascontiguousarray(neg.numeric)

        pos_tp, pos_fp = compute_tp_fp(q_bin, q_num, pos_bin_c, pos_num_c, pos_bin_c, pos_num_c, neg_bin_c, neg_num_c)
        neg_tp, neg_fp = compute_tp_fp(q_bin, q_num, neg_bin_c, neg_num_c, neg_bin_c, neg_num_c, pos_bin_c, pos_num_c)

        positive_classifiers = [
            self._make_precached_classifier(sample, pos, i, Classifier.Type.POSITIVE, int(pos_tp[i]), int(pos_fp[i]))
            for i in range(n_pos)
        ]
        negative_classifiers = [
            self._make_precached_classifier(sample, neg, i, Classifier.Type.NEGATIVE, int(neg_tp[i]), int(neg_fp[i]))
            for i in range(n_neg)
        ]

        if any(getattr(self.pos_params, m.attr) is not None for m in METADATA):
            positive_classifiers = [c for c in positive_classifiers if c.metrics.is_better_than(self.pos_params)]
        if any(getattr(self.neg_params, m.attr) is not None for m in METADATA):
            negative_classifiers = [c for c in negative_classifiers if c.metrics.is_better_than(self.neg_params)]

        positive_classifiers = self._rank_and_trim(positive_classifiers, self.pos_rank_by, self.pos_top_k)
        negative_classifiers = self._rank_and_trim(negative_classifiers, self.neg_rank_by, self.neg_top_k)

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

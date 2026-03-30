from __future__ import annotations
import math
import dataclasses

import numpy

from .dataset import Sample
from .dataset import Subset
from .dataset import Dataset


class Hypothesis:
    def __init__(self, lhs: Sample, rhs: Sample):
        numeric_stacked = numpy.vstack([lhs.numeric, rhs.numeric])

        self.binary = lhs.binary & rhs.binary
        self.numeric_minimum = numeric_stacked.min(axis=0)
        self.numeric_maximum = numeric_stacked.max(axis=0)

    def covers(self, subset: Subset) -> numpy.ndarray:
        # Returns true/false for every object in the subset
        covers_binary = numpy.all(subset.binary | ~self.binary, axis=1)
        covers_numeric_minimum = self.numeric_minimum <= subset.numeric
        covers_numeric_maximum = subset.numeric <= self.numeric_maximum
        covers_numeric = numpy.all(covers_numeric_minimum & covers_numeric_maximum, axis=1)
        return covers_binary & covers_numeric

    def to_string(self):
        parts = []
        for binary in self.binary:
            parts.append("1" if binary else "0")
        for minimum, maximum in zip(self.numeric_minimum, self.numeric_maximum):
            parts.append(f"[{minimum}, {maximum}]")
        return "; ".join(parts)


class Classifier:
    class Type:
        POSITIVE = "POSITIVE"
        NEGATIVE = "NEGATIVE"

    def __init__(self, lhs: Sample, rhs: Sample, dataset: Dataset, type: Type):
        self.query = lhs
        self.source = rhs
        self.dataset = dataset
        self.hypothesis = Hypothesis(lhs, rhs)
        self.type = type
        match type:
            case Classifier.Type.POSITIVE:
                self.supporters = dataset.positive
                self.opposers = dataset.negative
            case Classifier.Type.NEGATIVE:
                self.supporters = dataset.negative
                self.opposers = dataset.positive
        self.metrics = None

    @dataclasses.dataclass
    class Metrics:
        supporters_covered: int = 0
        opposers_covered: int = numpy.inf
        supporter_opposer_ratio: float = 0.0
        support: float = 0.0
        error_rate: float = 1.0
        precision: float = 0.0
        lift: float = 0.0
        wracc: float = -numpy.inf
        balanced_precision_proxy: float = -numpy.inf
        youdens_j: float = -numpy.inf
        matthews_correlation: float = -numpy.inf
        information_gain: float = -numpy.inf
        gini_gain: float = -numpy.inf
        log_odds_ratio: float = -numpy.inf
        chi_squared: float = -numpy.inf
        g_test: float = -numpy.inf
        interval_tightness: float = -numpy.inf
        description_volume: float = numpy.inf
        simplicity_prior: float = 0.0
        query_binary_similarity: float = 0.0
        query_numeric_similarity: float = 0.0
        query_similarity: float = 0.0
        query_weighted_precision: float = 0.0
        query_weighted_wracc: float = -numpy.inf
        stability: float = 0.0
        robustness: float = 0.0
        delta_stability: float = 0.0

        @dataclasses.dataclass
        class Metadata:
            name: str
            attr: str
            is_minimized: bool = False

        METADATA = [
            Metadata(name="Supporters covered", attr="supporters_covered"),
            Metadata(name="Opposers covered", attr="opposers_covered", is_minimized=True),
            Metadata(name="Supporters to opposers ratio", attr="supporter_opposer_ratio"),
            Metadata(name="Support", attr="support"),
            Metadata(name="Error rate", attr="error_rate", is_minimized=True),
            Metadata(name="Precision", attr="precision"),
            Metadata(name="Lift", attr="lift"),
            Metadata(name="WRAcc", attr="wracc"),
            Metadata(name="Balanced precision proxy", attr="balanced_precision_proxy"),
            Metadata(name="Youden's J", attr="youdens_j"),
            Metadata(name="Matthews correlation", attr="matthews_correlation"),
            Metadata(name="Information gain", attr="information_gain"),
            Metadata(name="Gini gain", attr="gini_gain"),
            Metadata(name="Log odds ratio", attr="log_odds_ratio"),
            Metadata(name="Chi squared", attr="chi_squared"),
            Metadata(name="G-test", attr="g_test"),
            Metadata(name="Interval tightness", attr="interval_tightness"),
            Metadata(name="Description volume", attr="description_volume", is_minimized=True),
            Metadata(name="Simplicity prior", attr="simplicity_prior"),
            Metadata(name="Query binary similarity", attr="query_binary_similarity"),
            Metadata(name="Query numeric similarity", attr="query_numeric_similarity"),
            Metadata(name="Query similarity", attr="query_similarity"),
            Metadata(name="Query weighted precision", attr="query_weighted_precision"),
            Metadata(name="Query weighted WRAcc", attr="query_weighted_wracc"),
            Metadata(name="Stability", attr="stability"),
            Metadata(name="Robustness", attr="robustness"),
            Metadata(name="Delta stability", attr="delta_stability"),
        ]

        def to_dict(self):
            return {metadata.name: getattr(self, metadata.attr) for metadata in Classifier.Metrics.METADATA}

        @staticmethod
        def minimized_fields() -> set[str]:
            return [metadata.attr for metadata in Classifier.Metrics.METADATA if metadata.is_minimized]

        def score_for_ranking(self, field: str) -> float:
            value = getattr(self, field)
            return -value if field in self.minimized_fields() else value

        @staticmethod
        def from_dict(dictionary: dict) -> Classifier.Metrics:
            result = Classifier.Metrics()
            for metadata in Classifier.Metrics.METADATA:
                if metadata.name in dictionary:
                    setattr(result, metadata.attr, dictionary[metadata.name])
            return result

        def is_better_than(self, other: Classifier.Metrics) -> bool:
            for metadata in Classifier.Metrics.METADATA:
                if metadata.is_minimized:
                    if getattr(self, metadata.attr) > getattr(other, metadata.attr):
                        return False
                else:
                    if getattr(self, metadata.attr) < getattr(other, metadata.attr):
                        return False
            return True

    @staticmethod
    def _binary_entropy(p: int, n: int) -> float:
        total = p + n
        if total == 0 or p == 0 or n == 0:
            return 0.0
        p_ratio = p / total
        n_ratio = n / total
        return -(p_ratio * math.log2(p_ratio) + n_ratio * math.log2(n_ratio))

    @staticmethod
    def _gini_impurity(p: int, n: int) -> float:
        total = p + n
        if total == 0:
            return 0.0
        p_ratio = p / total
        n_ratio = n / total
        return 1.0 - p_ratio**2 - n_ratio**2

    @staticmethod
    def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
        return numerator / denominator if denominator != 0 else default

    @staticmethod
    def _xlogy(observed: float, expected: float) -> float:
        if observed <= 0 or expected <= 0:
            return 0.0
        return observed * math.log(observed / expected)

    @staticmethod
    def _contingency_expected(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float, float]:
        total = tp + fp + fn + tn
        positive_row = tp + fp
        negative_row = fn + tn
        supporter_col = tp + fn
        opposer_col = fp + tn
        return (
            positive_row * supporter_col / total,
            positive_row * opposer_col / total,
            negative_row * supporter_col / total,
            negative_row * opposer_col / total,
        )

    def _query_binary_similarity(self) -> float:
        query_active = int(self.query.binary.sum())
        if query_active == 0:
            return 1.0
        matched = int(self.hypothesis.binary.sum())
        return matched / query_active

    def _interval_tightness(self) -> tuple[float, float]:
        if len(self.dataset.numeric_range) == 0:
            return 1.0, 0.0

        widths = self.hypothesis.numeric_maximum - self.hypothesis.numeric_minimum
        normalized_widths = numpy.divide(
            widths,
            self.dataset.numeric_range,
            out=numpy.zeros_like(widths, dtype=numpy.float64),
            where=self.dataset.numeric_range > 0,
        )
        normalized_widths = numpy.clip(normalized_widths, 0.0, 1.0)
        interval_tightness = 1.0 - float(normalized_widths.mean())
        description_volume = float(numpy.prod(normalized_widths))
        return interval_tightness, description_volume

    def _simplicity_prior(self, interval_tightness: float) -> float:
        binary_complexity = self._safe_div(float(self.hypothesis.binary.sum()), self.dataset.binary_feature_count)
        interval_complexity = 1.0 - interval_tightness
        description_complexity = binary_complexity + interval_complexity
        return 1.0 / (1.0 + description_complexity)

    def _stability_metrics(self, supporters_covered: numpy.ndarray) -> tuple[float, float]:
        covered_binary = self.supporters.binary[supporters_covered]
        covered_numeric = self.supporters.numeric[supporters_covered]
        witness_sizes = [len(covered_binary)]  # The regenerating subset must be non-empty.

        dropped_binary = self.query.binary & ~self.hypothesis.binary
        for index in numpy.flatnonzero(dropped_binary):
            witness_sizes.append(int((~covered_binary[:, index]).sum()))

        for index in range(len(self.hypothesis.numeric_minimum)):
            min_witnesses = int((covered_numeric[:, index] == self.hypothesis.numeric_minimum[index]).sum())
            max_witnesses = int((covered_numeric[:, index] == self.hypothesis.numeric_maximum[index]).sum())
            witness_sizes.append(min_witnesses)
            if self.hypothesis.numeric_minimum[index] != self.hypothesis.numeric_maximum[index]:
                witness_sizes.append(max_witnesses)

        witness_sizes = [size for size in witness_sizes if size > 0]
        if not witness_sizes:
            return 0.0, 0.0

        stability = float(numpy.prod([1.0 - 2.0 ** (-size) for size in witness_sizes]))
        delta_stability = float(min(witness_sizes))
        return stability, delta_stability

    def get_metrics(self, eps: float = 0.5):
        if self.metrics is not None:
            return self.metrics

        supporters_covered = self.hypothesis.covers(self.supporters)
        opposers_covered = self.hypothesis.covers(self.opposers)

        tp = int(supporters_covered.sum())
        fp = int(opposers_covered.sum())
        tn = int((~opposers_covered).sum())
        fn = int((~supporters_covered).sum())

        p = len(self.supporters)
        n = len(self.opposers)
        total = p + n
        covered_total = tp + fp
        uncovered_total = fn + tn
        baseline_positive_rate = self._safe_div(p, total)
        precision = self._safe_div(tp, covered_total)
        support = self._safe_div(tp, p)
        error_rate = self._safe_div(fp, n)
        lift = self._safe_div(precision, baseline_positive_rate)
        wracc = self._safe_div(covered_total, total) * (precision - baseline_positive_rate)
        balanced_precision_proxy = self._safe_div(tp, p) - self._safe_div(fp, n)

        mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        matthews_correlation = self._safe_div(tp * tn - fp * fn, mcc_denominator)

        information_gain = (
            self._binary_entropy(p, n)
            - (covered_total * self._binary_entropy(tp, fp) + uncovered_total * self._binary_entropy(fn, tn)) / total
        )
        gini_gain = (
            self._gini_impurity(p, n)
            - (covered_total * self._gini_impurity(tp, fp) + uncovered_total * self._gini_impurity(fn, tn)) / total
        )

        expected_tp, expected_fp, expected_fn, expected_tn = self._contingency_expected(tp, fp, fn, tn)
        chi_squared = sum(
            [
                self._safe_div((tp - expected_tp) ** 2, expected_tp),
                self._safe_div((fp - expected_fp) ** 2, expected_fp),
                self._safe_div((fn - expected_fn) ** 2, expected_fn),
                self._safe_div((tn - expected_tn) ** 2, expected_tn),
            ]
        )
        g_test = 2.0 * sum(
            [
                self._xlogy(tp, expected_tp),
                self._xlogy(fp, expected_fp),
                self._xlogy(fn, expected_fn),
                self._xlogy(tn, expected_tn),
            ]
        )

        query_binary_similarity = self._query_binary_similarity()
        interval_tightness, description_volume = self._interval_tightness()
        query_numeric_similarity = interval_tightness
        similarity_parts = [query_binary_similarity] if self.dataset.binary_feature_count > 0 else []
        if self.dataset.numeric_feature_count > 0:
            similarity_parts.append(query_numeric_similarity)
        query_similarity = float(numpy.mean(similarity_parts)) if similarity_parts else 1.0
        simplicity_prior = self._simplicity_prior(interval_tightness)
        stability, delta_stability = self._stability_metrics(supporters_covered)

        self.metrics = Classifier.Metrics(
            supporters_covered=tp,
            opposers_covered=fp,
            supporter_opposer_ratio=(tp / fp if fp != 0 else numpy.inf),
            support=support,
            error_rate=error_rate,
            precision=precision,
            lift=lift,
            wracc=wracc,
            balanced_precision_proxy=balanced_precision_proxy,
            youdens_j=tp / (tp + fn) - fp / (fp + tn),
            matthews_correlation=matthews_correlation,
            information_gain=information_gain,
            gini_gain=gini_gain,
            log_odds_ratio=(tp + eps) / (fp + eps),
            chi_squared=chi_squared,
            g_test=g_test,
            interval_tightness=interval_tightness,
            description_volume=description_volume,
            simplicity_prior=simplicity_prior,
            query_binary_similarity=query_binary_similarity,
            query_numeric_similarity=query_numeric_similarity,
            query_similarity=query_similarity,
            query_weighted_precision=precision * query_similarity,
            query_weighted_wracc=wracc * query_similarity,
            stability=stability,
            robustness=stability,
            delta_stability=delta_stability,
        )
        return self.metrics

    def to_dict(self, with_metrics: bool = True):
        return {
            "Hypothesis": self.hypothesis.to_string(),
            "Type": self.type,
            "Supporters": len(self.supporters),
            "Opposers": len(self.opposers),
            **(self.get_metrics().to_dict() if with_metrics else {}),
        }

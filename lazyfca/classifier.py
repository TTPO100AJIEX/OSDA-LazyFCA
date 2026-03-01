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
        self.hypothesis = Hypothesis(lhs, rhs)
        self.type = type
        match type:
            case Classifier.Type.POSITIVE:
                self.supporters = dataset.positive
                self.opposers = dataset.negative
            case Classifier.Type.NEGATIVE:
                self.supporters = dataset.negative
                self.opposers = dataset.positive

    @dataclasses.dataclass
    class Metrics:
        supporters_covered: int = 0
        opposers_covered: int = numpy.inf
        supporter_opposer_ratio: float = 1.0
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

        def to_dict(self):
            return {
                "Supporters covered": self.supporters_covered,
                "Opposers covered": self.opposers_covered,
                "Supporters to opposers ratio": self.supporter_opposer_ratio,
                "Support": self.support,
                "Error rate": self.error_rate,
                "Precision": self.precision,
                "Lift": self.lift,
                "WRAcc": self.wracc,
                "Balanced precision proxy": self.balanced_precision_proxy,
                "Youden's J": self.youdens_j,
                "Matthews correlation": self.matthews_correlation,
                "Information gain": self.information_gain,
                "Gini gain": self.gini_gain,
                "Log odds ratio": self.log_odds_ratio,
                "Chi squared": self.chi_squared,
                "G-test": self.g_test,
            }

        def is_better_than(self, other: Classifier.Metrics) -> bool:
            if self.supporters_covered < other.supporters_covered:
                return False
            if self.opposers_covered > other.opposers_covered:
                return False
            if self.supporter_opposer_ratio < other.supporter_opposer_ratio:
                return False
            if self.support < other.support:
                return False
            if self.error_rate > other.error_rate:
                return False
            if self.precision < other.precision:
                return False
            if self.lift < other.lift:
                return False
            if self.wracc < other.wracc:
                return False
            if self.balanced_precision_proxy < other.balanced_precision_proxy:
                return False
            if self.youdens_j < other.youdens_j:
                return False
            if self.matthews_correlation < other.matthews_correlation:
                return False
            if self.information_gain < other.information_gain:
                return False
            if self.gini_gain < other.gini_gain:
                return False
            if self.log_odds_ratio < other.log_odds_ratio:
                return False
            if self.chi_squared < other.chi_squared:
                return False
            if self.g_test < other.g_test:
                return False
            return True

    def _binary_entropy(p: int, n: int) -> float:
        total = p + n
        if total == 0 or p == 0 or n == 0:
            return 0.0
        p_ratio = p / total
        n_ratio = n / total
        return -(p_ratio * math.log2(p_ratio) + n_ratio * math.log2(n_ratio))

    def _gini_impurity(p: int, n: int) -> float:
        total = p + n
        if total == 0:
            return 0.0
        p_ratio = p / total
        n_ratio = n / total
        return 1.0 - p_ratio ** 2 - n_ratio ** 2

    def get_metrics(self, eps: float = 0.5):
        supporters_covered = self.hypothesis.covers(self.supporters)
        opposers_covered = self.hypothesis.covers(self.opposers)

        tp = int(supporters_covered.sum())
        fp = int(opposers_covered.sum())
        tn = int((~opposers_covered).sum())
        fn = int((~supporters_covered).sum())

        p = len(self.supporters)
        n = len(self.opposers)

        return Classifier.Metrics(
            supporters_covered=tp,
            opposers_covered=fp,
            supporter_opposer_ratio=(tp / fp if fp != 0 else numpy.inf),
            support=tp / p,
            error_rate=fp / n,
            precision=tp / (tp + fp),
            lift=(tp / (tp + fp)) / (p / (p + n)),
            wracc=((tp + fp) / (p + n)) * (tp / (tp + fp) - p / (p + n)),
            balanced_precision_proxy=tp / p - fp / n,
            youdens_j=tp / (tp + fn) - fp / (fp + tn),
            matthews_correlation=(tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
            # information_gain=(
            #     self._binary_entropy(p, n)
            #     - ((tp + fp) * self._binary_entropy(tp, fp) + (fn + tn) * self._binary_entropy(fn, tn)) / (p + n)
            # ),
            information_gain = - ((tp / p) * math.log(tp / p) + (tn / n) * math.log(tn / n)),
            # gini_gain=(
            #     self._gini_impurity(p, n)
            #     - ((tp + fp) * self._gini_impurity(tp, fp) + (fn + tn) * self._gini_impurity(fn, tn)) / (p + n)
            # ),
            gini_gain = 1 - ((tp / p) ** 2 + (tn / n) ** 2),
            log_odds_ratio=(tp + eps) / (fp + eps),
            chi_squared=((tp - p) ** 2 / p) + ((tn - n) ** 2 / n),
            g_test=2 * ((tp * math.log(tp / p)) + tn * math.log(tn / n)),
        )

    def to_dict(self, with_metrics: bool = True):
        return {
            "Hypothesis": self.hypothesis.to_string(),
            "Type": self.type,
            "Supporters": len(self.supporters),
            "Opposers": len(self.opposers),
            **(self.get_metrics().to_dict() if with_metrics else {}),
        }

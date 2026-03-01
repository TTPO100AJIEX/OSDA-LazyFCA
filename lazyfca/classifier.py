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
            return True

    def get_metrics(self):
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
        )

    def to_dict(self, with_metrics: bool = True):
        return {
            "Hypothesis": self.hypothesis.to_string(),
            "Type": self.type,
            "Supporters": len(self.supporters),
            "Opposers": len(self.opposers),
            **(self.get_metrics().to_dict() if with_metrics else {}),
        }

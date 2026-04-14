from __future__ import annotations

import numpy

from lazyfca.dataset import Sample
from lazyfca.dataset import Subset
from lazyfca.dataset import Dataset


from lazyfca.metrics import Metrics
from lazyfca.metrics import LazyMetrics


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
        self.metrics = LazyMetrics(self)

    def get_metrics(self) -> Metrics:
        return self.metrics

    def to_dict(self, with_metrics: bool = True) -> dict:
        return {
            "Hypothesis": self.hypothesis.to_string(),
            "Type": self.type,
            "Supporters": len(self.supporters),
            "Opposers": len(self.opposers),
            **(self.metrics.to_dict() if with_metrics else {}),
        }

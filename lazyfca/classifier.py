import math

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

    def __init__(self, hypothesis: Hypothesis, dataset: Dataset, type: Type):
        self.hypothesis = hypothesis
        self.type = type
        match type:
            case Classifier.Type.POSITIVE:
                self.supporters = dataset.positive
                self.opposers = dataset.negative
            case Classifier.Type.NEGATIVE:
                self.supporters = dataset.negative
                self.opposers = dataset.positive

    def get_metrics(self):
        supporters_covered = self.hypothesis.covers(self.supporters)
        opposers_covered = self.hypothesis.covers(self.opposers)

        tp = int(supporters_covered.sum())
        fp = int(opposers_covered.sum())
        tn = int((~opposers_covered).sum())
        fn = int((~supporters_covered).sum())

        p = len(self.supporters)
        n = len(self.opposers)

        return {
            "Hypothesis": self.hypothesis.to_string(),
            "Type": self.type,
            "Supporters": p,
            "Opposers": n,
            "Supporters covered": p,
            "Opposers covered": n,
            "Support": tp / p,
            "Error rate": fp / n,
            "Precision": tp / (tp + fp),
            "Lift": (tp / (tp + fp)) / (p / (p + n)),
            "WRAcc": ((tp + fp) / (p + n)) * (tp / (tp + fp) - p / (p + n)),
            "Balanced precision proxy": tp / p - fp / n,
            "Youden's J": tp / (tp + fn) - fp / (fp + tn),
            "Matthews correlation": (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
        }

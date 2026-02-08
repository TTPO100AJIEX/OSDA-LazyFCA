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

        return {
            "Type": self.type,
            "Supporters": len(self.supporters),
            "Opposers": len(self.opposers),
            "Supporters covered": supporters_covered.sum(),
            "Opposers covered": opposers_covered.sum(),
        }

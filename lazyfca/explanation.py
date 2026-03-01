import typing

import pandas

from .classifier import Classifier
from .dataset import Sample


class Explanation:
    def __init__(
        self,
        sample: Sample,
        positive_classifiers: typing.List[Classifier],
        negative_classifiers: typing.List[Classifier],
    ):
        self.sample = sample
        self.positive_classifiers = positive_classifiers
        self.negative_classifiers = negative_classifiers

    def display(self):
        return pandas.DataFrame(
            [
                *[classifier.to_dict(with_metrics=True) for classifier in self.positive_classifiers],
                *[classifier.to_dict(with_metrics=True) for classifier in self.negative_classifiers],
            ]
        )

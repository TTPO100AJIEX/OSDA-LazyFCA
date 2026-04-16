import typing

import pandas

from lazyfca.classifier import Classifier
from lazyfca.dataset import Sample


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

    def __repr__(self) -> str:
        pos_n = len(self.positive_classifiers)
        neg_n = len(self.negative_classifiers)
        lines = ["Explanation"]
        lines.append("=" * 46)
        lines.append(f"  {'Positive classifiers':<22}: {pos_n}")
        lines.append(f"  {'Negative classifiers':<22}: {neg_n}")
        lines.append(f"  {'Total':<22}: {pos_n + neg_n}")

        def _show_top(clf, label: str) -> None:
            lines.append("")
            lines.append(f"  {label}")
            lines.append("  " + "-" * 42)
            lines.append(f"    {'Hypothesis':<14}: {clf.hypothesis.to_string()}")
            lines.append(f"    {'Supporters':<14}: {len(clf.supporters)}")
            lines.append(f"    {'Opposers':<14}: {len(clf.opposers)}")

        if self.positive_classifiers:
            _show_top(self.positive_classifiers[0], "Top positive classifier")
        if self.negative_classifiers:
            _show_top(self.negative_classifiers[0], "Top negative classifier")

        lines.append("=" * 46)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def display(self):
        return pandas.DataFrame(
            [
                *[classifier.to_dict(with_metrics=True) for classifier in self.positive_classifiers],
                *[classifier.to_dict(with_metrics=True) for classifier in self.negative_classifiers],
            ]
        )

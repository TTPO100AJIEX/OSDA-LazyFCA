from __future__ import annotations
import typing

import math
import numpy
import pandas
import matplotlib.pyplot as plt

from lazyfca.classifier import Classifier
from lazyfca.dataset import Dataset
from lazyfca.dataset import Sample
from lazyfca.metrics import Metrics


def graph_layout(num_items: int):
    for i in range(3, 8):
        if num_items % i == 0:
            return num_items // i, i
    return num_items // 5 + 1, 5


def minimize(classifiers: typing.List[Classifier]) -> typing.List[Classifier]:
    result: typing.List[Classifier] = []
    for i in range(len(classifiers)):
        for j in range(i + 1, len(classifiers)):
            if classifiers[j].is_more_general_than(classifiers[i]):
                break
        else:
            result.append(classifiers[i])
    return result


def rank(classifiers: typing.List[Classifier], rank_by: str) -> typing.List[Classifier]:
    return sorted(classifiers, key=lambda classifier: classifier.metrics.score_for_ranking(rank_by), reverse=True)


def filt(classifiers: typing.List[Classifier], thresholds: Metrics) -> typing.List[Classifier]:
    return [c for c in classifiers if c.metrics.is_better_than(thresholds)]


class Explanation:
    __slots__ = (
        "dataset",
        "sample",
        "positive_classifiers",
        "negative_classifiers",
        "positive_ranked_by",
        "negative_ranked_by",
    )

    def __init__(
        self,
        dataset: Dataset,
        sample: Sample,
        positive_classifiers: typing.List[Classifier],
        negative_classifiers: typing.List[Classifier],
        positive_ranked_by: typing.Optional[str] = None,
        negative_ranked_by: typing.Optional[str] = None,
    ):
        self.dataset = dataset
        self.sample = sample
        self.positive_classifiers = positive_classifiers
        self.negative_classifiers = negative_classifiers
        self.positive_ranked_by = positive_ranked_by
        self.negative_ranked_by = negative_ranked_by

    def __repr__(self) -> str:
        pos_n = len(self.positive_classifiers)
        neg_n = len(self.negative_classifiers)
        lines = ["Explanation"]
        lines.append("=" * 46)
        lines.append(f"  {'Positive classifiers':<22}: {pos_n}")
        lines.append(f"  {'Negative classifiers':<22}: {neg_n}")
        lines.append(f"  {'Total':<22}: {pos_n + neg_n}")

        def _show_top(clf: Classifier, label: str) -> None:
            lines.append("")
            lines.append(f"  {label}")
            lines.append("  " + "-" * 42)
            lines.append(f"    {'Hypothesis':<14}: {clf.to_string()}")
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

    def _modify(
        self,
        positive_classifiers: typing.List[Classifier],
        negative_classifiers: typing.List[Classifier],
        inplace: bool,
    ):
        if inplace:
            self.positive_classifiers = positive_classifiers
            self.negative_classifiers = negative_classifiers
            return self
        else:
            return Explanation(
                dataset=self.dataset,
                sample=self.sample,
                positive_classifiers=positive_classifiers,
                negative_classifiers=negative_classifiers,
                positive_ranked_by=self.positive_ranked_by,
                negative_ranked_by=self.negative_ranked_by,
            )

    def filter(self, positive_thresholds: Metrics, negative_thresholds: Metrics, inplace: bool = True) -> Explanation:
        return self._modify(
            filt(self.positive_classifiers, positive_thresholds),
            filt(self.negative_classifiers, negative_thresholds),
            inplace,
        )

    def rank(self, positive: typing.Optional[str], negative: typing.Optional[str], inplace: bool = True) -> Explanation:
        result = self._modify(
            rank(self.positive_classifiers, positive) if positive is not None else self.positive_classifiers,
            rank(self.negative_classifiers, negative) if negative is not None else self.negative_classifiers,
            inplace,
        )
        result.positive_ranked_by = result.positive_ranked_by or positive
        result.negative_ranked_by = result.negative_ranked_by or negative
        return result

    def trim(self, positive: typing.Optional[int], negative: typing.Optional[int], inplace: bool = True) -> Explanation:
        if positive is not None:
            assert self.positive_ranked_by is not None, f"The explanation (positive) must be ranked before trimming"
        if negative is not None:
            assert self.negative_ranked_by is not None, f"The explanation (negative) must be ranked before trimming"
        return self._modify(self.positive_classifiers[:positive], self.negative_classifiers[:negative], inplace)

    def keep_top_k(self, top_k: int, inplace: bool = True) -> Explanation:
        assert self.positive_ranked_by is not None, f"The explanation (positive) must be ranked before trimming"
        assert self.negative_ranked_by is not None, f"The explanation (negative) must be ranked before trimming"
        assert self.positive_ranked_by == self.negative_ranked_by, (
            f"Positive and negative explanations must be ranked by the same metric before trimming"
        )

        top_positive, top_negative = 0, 0
        at_most_k = min(top_k, len(self.positive_classifiers) + len(self.negative_classifiers))
        while top_positive + top_negative < at_most_k:
            next_positive = self.positive_classifiers[top_positive].metrics.score_for_ranking(self.positive_ranked_by)
            next_negative = self.negative_classifiers[top_negative].metrics.score_for_ranking(self.negative_ranked_by)
            if next_positive > next_negative:
                top_positive += 1
            else:
                top_negative += 1
        return self._modify(self.positive_classifiers[:top_positive], self.negative_classifiers[:top_negative], inplace)

    def minimize(self, inplace: bool = True) -> Explanation:
        return self._modify(minimize(self.positive_classifiers), minimize(self.negative_classifiers), inplace)

    def display(self):
        return pandas.DataFrame(
            [
                *[classifier.to_dict(with_metrics=True) for classifier in self.positive_classifiers],
                *[classifier.to_dict(with_metrics=True) for classifier in self.negative_classifiers],
            ]
        )

    def display_binary(self):
        def collect_stats(classifiers: typing.List[Classifier]):
            stats = {}
            for clf in classifiers:
                for feature, value in zip(self.dataset.bool_columns, clf.binary):
                    if value:
                        stats[feature] = stats.get(feature, 0) + 1
            return stats

        positive = pandas.DataFrame(collect_stats(self.positive_classifiers).items())
        negative = pandas.DataFrame(collect_stats(self.negative_classifiers).items())
        return positive.merge(negative, on=0).rename(columns={0: "feature", "1_x": "positive", "1_y": "negative"})

    def display_numeric(self):
        def collect_stats(classifiers: typing.List[Classifier]):
            stats = {}
            for clf in classifiers:
                columns = self.dataset.numeric_columns
                for feature, min_value, max_value in zip(columns, clf.numeric_minimum, clf.numeric_maximum):
                    stats[feature] = numpy.hstack(
                        [
                            stats.get(feature, numpy.array([])),
                            numpy.arange(math.floor(min_value), math.ceil(max_value) + 1, step=1),
                        ]
                    )
            return stats

        positive = collect_stats(self.positive_classifiers)
        negative = collect_stats(self.negative_classifiers)
        n_rows, n_cols = graph_layout(len(self.dataset.numeric_columns))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        for ax, feature in zip(axes.flat, self.dataset.numeric_columns):
            pos, neg = positive[feature], negative[feature]
            bins = numpy.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 50)
            ax.hist(pos, bins=bins, alpha=0.5, label="positive")
            ax.hist(neg, bins=bins, alpha=0.5, label="negative")
            ax.set_title(feature)
            ax.get_yaxis().set_ticks([])
            ax.legend()
        fig.tight_layout()
        return fig

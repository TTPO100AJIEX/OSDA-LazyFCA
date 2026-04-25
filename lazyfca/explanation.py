import typing

import math
import numpy
import pandas
import matplotlib.pyplot as plt

from lazyfca.classifier import Classifier
from lazyfca.dataset import Dataset
from lazyfca.dataset import Sample


def graph_layout(num_items: int):
    for i in range(3, 8):
        if num_items % i == 0:
            return num_items // i, i
    return num_items // 5 + 1, 5


def minimize(classifiers: typing.List[Classifier]) -> typing.List[Classifier]:
    result: typing.List[Classifier] = []
    for key in classifiers:
        for key2 in result:
            if numpy.all(key.binary == key2.binary):
                break
        else:
            for key2 in classifiers:
                if key2.is_more_general_than(key, only_binary=True):
                    break
            else:
                result.append(key.clone())

    for classifier in result:
        for key in classifiers:
            if classifier.is_more_general_than(key, only_binary=True):
                classifier.numeric_minimum = numpy.minimum(classifier.numeric_minimum, key.numeric_minimum)
                classifier.numeric_maximum = numpy.maximum(classifier.numeric_maximum, key.numeric_maximum)
    return result


class Explanation:
    def __init__(
        self,
        dataset: Dataset,
        sample: Sample,
        positive_classifiers: typing.List[Classifier],
        negative_classifiers: typing.List[Classifier],
    ):
        self.dataset = dataset
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

    def minimize(self):
        self.positive_classifiers = minimize(self.positive_classifiers)
        self.negative_classifiers = minimize(self.negative_classifiers)
        return self

    def display(self):
        return pandas.DataFrame(
            [
                *[classifier.to_dict(with_metrics=True) for classifier in self.positive_classifiers],
                *[classifier.to_dict(with_metrics=True) for classifier in self.negative_classifiers],
            ]
        )

    def display_binary(self):
        features = {}
        for hypothesis in self.positive_classifiers:
            for feature, value in zip(hypothesis.dataset.bool_columns, hypothesis.binary):
                if value:
                    features[feature] = features.get(feature, 0) + 1
        df = pandas.DataFrame(features.items()).transpose()
        if len(df) != 0:
            df = df.sort_values(axis=1, by=1, ascending=False)
        return df

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

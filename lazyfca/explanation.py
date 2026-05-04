from __future__ import annotations
import typing

import math
import numba
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


def rank(classifiers: typing.List[Classifier], rank_by: str) -> typing.List[Classifier]:
    return sorted(classifiers, key=lambda classifier: classifier.metrics.score_for_ranking(rank_by), reverse=True)


def filt(classifiers: typing.List[Classifier], thresholds: Metrics) -> typing.List[Classifier]:
    return [c for c in classifiers if c.metrics.is_better_than(thresholds)]


class Explanation:
    """A multi-class explanation: one list of classifiers per class.

    Internally classifiers are stored in :attr:`class_classifiers`, a list of
    length ``n_classes``. The legacy attributes :attr:`positive_classifiers`
    and :attr:`negative_classifiers` remain available when there are exactly
    two classes (mapping to class ``1`` and class ``0`` respectively).
    """

    @typing.overload
    def __init__(
        self,
        dataset: Dataset,
        sample: Sample,
        positive_classifiers: typing.List[Classifier],
        negative_classifiers: typing.List[Classifier],
        positive_ranked_by: typing.Optional[str] = None,
        negative_ranked_by: typing.Optional[str] = None,
    ): ...

    @typing.overload
    def __init__(
        self,
        dataset: Dataset,
        sample: Sample,
        *,
        class_classifiers: typing.List[typing.List[Classifier]],
        class_ranked_by: typing.Optional[typing.List[typing.Optional[str]]] = None,
    ): ...

    def __init__(
        self,
        dataset: Dataset,
        sample: Sample,
        positive_classifiers: typing.Optional[typing.List[Classifier]] = None,
        negative_classifiers: typing.Optional[typing.List[Classifier]] = None,
        positive_ranked_by: typing.Optional[str] = None,
        negative_ranked_by: typing.Optional[str] = None,
        *,
        class_classifiers: typing.Optional[typing.List[typing.List[Classifier]]] = None,
        class_ranked_by: typing.Optional[typing.List[typing.Optional[str]]] = None,
    ):
        self.dataset = dataset
        self.sample = sample
        self.n_classes = dataset.n_classes

        if class_classifiers is not None:
            assert len(class_classifiers) == self.n_classes, (
                f"class_classifiers has length {len(class_classifiers)}, expected {self.n_classes}"
            )
            self.class_classifiers: typing.List[typing.List[Classifier]] = [list(cs) for cs in class_classifiers]
        else:
            assert self.n_classes == 2, (
                "Binary positive_classifiers/negative_classifiers constructor is only valid when n_classes == 2"
            )
            self.class_classifiers = [
                list(negative_classifiers or []),
                list(positive_classifiers or []),
            ]

        if class_ranked_by is not None:
            assert len(class_ranked_by) == self.n_classes
            self.class_ranked_by: typing.List[typing.Optional[str]] = list(class_ranked_by)
        else:
            self.class_ranked_by = [None] * self.n_classes
            if self.n_classes == 2:
                self.class_ranked_by[0] = negative_ranked_by
                self.class_ranked_by[1] = positive_ranked_by

    # ------------------------------------------------------------------ #
    # Backwards-compatible binary-style accessors                         #
    # ------------------------------------------------------------------ #

    @property
    def positive_classifiers(self) -> typing.List[Classifier]:
        assert self.n_classes == 2, ".positive_classifiers is only defined for binary explanations (n_classes == 2)"
        return self.class_classifiers[1]

    @positive_classifiers.setter
    def positive_classifiers(self, value: typing.List[Classifier]) -> None:
        assert self.n_classes == 2
        self.class_classifiers[1] = list(value)

    @property
    def negative_classifiers(self) -> typing.List[Classifier]:
        assert self.n_classes == 2, ".negative_classifiers is only defined for binary explanations (n_classes == 2)"
        return self.class_classifiers[0]

    @negative_classifiers.setter
    def negative_classifiers(self, value: typing.List[Classifier]) -> None:
        assert self.n_classes == 2
        self.class_classifiers[0] = list(value)

    @property
    def positive_ranked_by(self) -> typing.Optional[str]:
        assert self.n_classes == 2
        return self.class_ranked_by[1]

    @positive_ranked_by.setter
    def positive_ranked_by(self, value: typing.Optional[str]) -> None:
        assert self.n_classes == 2
        self.class_ranked_by[1] = value

    @property
    def negative_ranked_by(self) -> typing.Optional[str]:
        assert self.n_classes == 2
        return self.class_ranked_by[0]

    @negative_ranked_by.setter
    def negative_ranked_by(self, value: typing.Optional[str]) -> None:
        assert self.n_classes == 2
        self.class_ranked_by[0] = value

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        counts = [len(cs) for cs in self.class_classifiers]
        lines = ["Explanation"]
        lines.append("=" * 46)
        for c, count in enumerate(counts):
            label = (
                "Positive classifiers"
                if (self.n_classes == 2 and c == 1)
                else "Negative classifiers"
                if (self.n_classes == 2 and c == 0)
                else f"Class {c} classifiers"
            )
            lines.append(f"  {label:<22}: {count}")
        lines.append(f"  {'Total':<22}: {sum(counts)}")

        def _show_top(clf: Classifier, label: str) -> None:
            lines.append("")
            lines.append(f"  {label}")
            lines.append("  " + "-" * 42)
            lines.append(f"    {'Hypothesis':<14}: {clf.to_string()}")
            lines.append(f"    {'Supporters':<14}: {len(clf.supporters)}")
            lines.append(f"    {'Opposers':<14}: {len(clf.opposers)}")

        for c, classifiers in enumerate(self.class_classifiers):
            if classifiers:
                if self.n_classes == 2:
                    label = "Top positive classifier" if c == 1 else "Top negative classifier"
                else:
                    label = f"Top class-{c} classifier"
                _show_top(classifiers[0], label)

        lines.append("=" * 46)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def _modify(
        self,
        class_classifiers: typing.List[typing.List[Classifier]],
        inplace: bool,
    ):
        if inplace:
            self.class_classifiers = [list(cs) for cs in class_classifiers]
            return self
        else:
            return Explanation(
                self.dataset,
                self.sample,
                class_classifiers=class_classifiers,
                class_ranked_by=list(self.class_ranked_by),
            )

    def filter(
        self,
        positive_thresholds: typing.Optional[Metrics] = None,
        negative_thresholds: typing.Optional[Metrics] = None,
        inplace: bool = True,
        *,
        class_thresholds: typing.Optional[typing.List[Metrics]] = None,
    ) -> Explanation:
        if class_thresholds is not None:
            assert len(class_thresholds) == self.n_classes
            new_lists = [filt(self.class_classifiers[c], class_thresholds[c]) for c in range(self.n_classes)]
        else:
            assert self.n_classes == 2, (
                "Binary filter() form requires n_classes == 2; use class_thresholds= for multi-class"
            )
            new_lists = [
                filt(self.class_classifiers[0], negative_thresholds or Metrics()),
                filt(self.class_classifiers[1], positive_thresholds or Metrics()),
            ]
        return self._modify(new_lists, inplace)

    def rank(
        self,
        positive: typing.Optional[str] = None,
        negative: typing.Optional[str] = None,
        inplace: bool = True,
        *,
        class_rank_by: typing.Optional[typing.List[typing.Optional[str]]] = None,
    ) -> Explanation:
        if class_rank_by is not None:
            assert len(class_rank_by) == self.n_classes
            ranked_by = list(class_rank_by)
        else:
            assert self.n_classes == 2, (
                "Binary rank() form requires n_classes == 2; use class_rank_by= for multi-class"
            )
            ranked_by = [negative, positive]

        new_lists = [
            rank(self.class_classifiers[c], ranked_by[c]) if ranked_by[c] is not None else self.class_classifiers[c]
            for c in range(self.n_classes)
        ]
        result = self._modify(new_lists, inplace)
        for c, rb in enumerate(ranked_by):
            result.class_ranked_by[c] = result.class_ranked_by[c] or rb
        return result

    def trim(
        self,
        positive: typing.Optional[int] = None,
        negative: typing.Optional[int] = None,
        inplace: bool = True,
        *,
        class_top_k: typing.Optional[typing.List[typing.Optional[int]]] = None,
    ) -> Explanation:
        if class_top_k is not None:
            assert len(class_top_k) == self.n_classes
            keep_k = list(class_top_k)
        else:
            assert self.n_classes == 2, (
                "Binary trim() form requires n_classes == 2; use class_top_k= for multi-class"
            )
            keep_k = [negative, positive]

        for c, k in enumerate(keep_k):
            if k is not None:
                assert self.class_ranked_by[c] is not None, (
                    f"The explanation (class {c}) must be ranked before trimming"
                )

        new_lists = [self.class_classifiers[c][: keep_k[c]] for c in range(self.n_classes)]
        return self._modify(new_lists, inplace)

    def keep_top_k(self, top_k: int, inplace: bool = True) -> Explanation:
        for c, ranked_by in enumerate(self.class_ranked_by):
            assert ranked_by is not None, f"The explanation (class {c}) must be ranked before trimming"
        first_ranked_by = self.class_ranked_by[0]
        assert all(rb == first_ranked_by for rb in self.class_ranked_by), (
            "All classes must be ranked by the same metric before trimming"
        )

        cursors = [0] * self.n_classes
        total_available = sum(len(cs) for cs in self.class_classifiers)
        at_most_k = min(top_k, total_available)
        while sum(cursors) < at_most_k:
            best_c, best_score = None, None
            for c in range(self.n_classes):
                if cursors[c] >= len(self.class_classifiers[c]):
                    continue
                score = self.class_classifiers[c][cursors[c]].metrics.score_for_ranking(self.class_ranked_by[c])
                if best_score is None or score > best_score:
                    best_c, best_score = c, score
            if best_c is None:
                break
            cursors[best_c] += 1

        new_lists = [self.class_classifiers[c][: cursors[c]] for c in range(self.n_classes)]
        return self._modify(new_lists, inplace)

    def minimize(self, inplace: bool = True) -> Explanation:
        new_lists = [Classifier.minimize_classifiers(cs) for cs in self.class_classifiers]
        result = self._modify(new_lists, inplace)
        result.class_ranked_by = [None] * self.n_classes
        return result

    def display(self):
        return pandas.DataFrame(
            [
                classifier.to_dict(with_metrics=True)
                for classifiers in self.class_classifiers
                for classifier in classifiers
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

        if self.n_classes == 2:
            positive = pandas.DataFrame(collect_stats(self.class_classifiers[1]).items())
            negative = pandas.DataFrame(collect_stats(self.class_classifiers[0]).items())
            return positive.merge(negative, on=0).rename(columns={0: "feature", "1_x": "positive", "1_y": "negative"})

        per_class = [pandas.DataFrame(collect_stats(cs).items()) for cs in self.class_classifiers]
        result = per_class[0].rename(columns={0: "feature", 1: "class_0"})
        for c in range(1, self.n_classes):
            other = per_class[c].rename(columns={0: "feature", 1: f"class_{c}"})
            result = result.merge(other, on="feature", how="outer")
        return result

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

        per_class_stats = [collect_stats(cs) for cs in self.class_classifiers]
        n_rows, n_cols = graph_layout(len(self.dataset.numeric_columns))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        labels = (
            ["negative", "positive"]
            if self.n_classes == 2
            else [f"class {c}" for c in range(self.n_classes)]
        )
        for ax, feature in zip(axes.flat, self.dataset.numeric_columns):
            arrays = [per_class_stats[c].get(feature, numpy.array([])) for c in range(self.n_classes)]
            non_empty = [a for a in arrays if len(a) > 0]
            if not non_empty:
                continue
            global_min = min(a.min() for a in non_empty)
            global_max = max(a.max() for a in non_empty)
            bins = numpy.linspace(global_min, global_max, 50)
            for c, arr in enumerate(arrays):
                if len(arr) == 0:
                    continue
                ax.hist(arr, bins=bins, alpha=0.5, label=labels[c])
            ax.set_title(feature)
            ax.get_yaxis().set_ticks([])
            ax.legend()
        fig.tight_layout()
        return fig

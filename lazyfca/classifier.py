from __future__ import annotations
import typing
import itertools

import numpy
import numba

from lazyfca.dataset import Sample
from lazyfca.dataset import Dataset


from lazyfca.metrics import Metrics
from lazyfca.metrics import LazyMetrics
from lazyfca.metrics import METADATA


@numba.njit(cache=True)
def cover(
    binary: numpy.ndarray,
    numeric_minimum: numpy.ndarray,
    numeric_maximum: numpy.ndarray,
    subset_binary: numpy.ndarray,
    subset_numeric: numpy.ndarray,
) -> numpy.ndarray:
    result = numpy.empty(subset_binary.shape[0], numba.boolean)
    true_count = 0
    for i in range(subset_binary.shape[0]):
        for j in range(binary.shape[0]):
            if binary[j] and not subset_binary[i, j]:
                result[i] = False
                break
        else:
            for j in range(numeric_minimum.shape[0]):
                value = subset_numeric[i, j]
                if value < numeric_minimum[j] or value > numeric_maximum[j]:
                    result[i] = False
                    break
            else:
                result[i] = True
                true_count += 1

    return result, true_count, subset_binary.shape[0] - true_count


@numba.njit(cache=True)
def is_more_general(
    self_binary: numpy.ndarray,
    self_numeric_minimum: numpy.ndarray,
    self_numeric_maximum: numpy.ndarray,
    other_binary: numpy.ndarray,
    other_numeric_minimum: numpy.ndarray,
    other_numeric_maximum: numpy.ndarray,
) -> bool:
    are_equal = True

    for k in range(len(self_binary)):
        if self_binary[k] and not other_binary[k]:
            return False
        are_equal = are_equal and (self_binary[k] == other_binary[k])
    for k in range(len(self_numeric_minimum)):
        if self_numeric_minimum[k] > other_numeric_minimum[k]:
            return False
        are_equal = are_equal and (self_numeric_minimum[k] == other_numeric_minimum[k])
    for k in range(len(self_numeric_maximum)):
        if self_numeric_maximum[k] < other_numeric_maximum[k]:
            return False
        are_equal = are_equal and (self_numeric_maximum[k] == other_numeric_maximum[k])

    return not are_equal


class Classifier:
    class Type:
        POSITIVE = 1
        NEGATIVE = 2

    @staticmethod
    @numba.njit(cache=True, parallel=True)
    def calculate_classifiers_raw(
        query_binary: numpy.ndarray,
        query_numeric: numpy.ndarray,
        supporters_binary: numpy.ndarray,
        supporters_numeric: numpy.ndarray,
        opposers_binary: numpy.ndarray,
        opposers_numeric: numpy.ndarray,
    ):
        binary = numpy.empty_like(supporters_binary)
        numeric_minimum = numpy.empty_like(supporters_numeric)
        numeric_maximum = numpy.empty_like(supporters_numeric)

        supporters = numpy.empty((supporters_binary.shape[0], supporters_binary.shape[0]), numba.boolean)
        tp = numpy.empty((supporters_binary.shape[0]), numba.int32)
        fp = numpy.empty((supporters_binary.shape[0]), numba.int32)

        opposers = numpy.empty((supporters_binary.shape[0], opposers_binary.shape[0]), numba.boolean)
        tn = numpy.empty((supporters_binary.shape[0]), numba.int32)
        fn = numpy.empty((supporters_binary.shape[0]), numba.int32)

        for i in numba.prange(supporters_binary.shape[0]):
            binary[i] = query_binary & supporters_binary[i]
            numeric_minimum[i] = numpy.minimum(query_numeric, supporters_numeric[i])
            numeric_maximum[i] = numpy.maximum(query_numeric, supporters_numeric[i])
            supporters[i], tp[i], fn[i] = cover(
                binary[i], numeric_minimum[i], numeric_maximum[i], supporters_binary, supporters_numeric
            )
            opposers[i], fp[i], tn[i] = cover(
                binary[i], numeric_minimum[i], numeric_maximum[i], opposers_binary, opposers_numeric
            )
        return binary, numeric_minimum, numeric_maximum, supporters, opposers, tp, fp, tn, fn

    @staticmethod
    def calculate_classifiers(sample: Sample, dataset: Dataset, type: Classifier.Type) -> typing.List[Classifier]:
        match type:
            case Classifier.Type.POSITIVE:
                supporters = dataset.positive
                opposers = dataset.negative
            case Classifier.Type.NEGATIVE:
                supporters = dataset.negative
                opposers = dataset.positive
        raw = Classifier.calculate_classifiers_raw(
            sample.binary, sample.numeric, supporters.binary, supporters.numeric, opposers.binary, opposers.numeric
        )
        return [Classifier(sample, source, dataset, type, *raw) for source, *raw in zip(supporters, *raw)]

    __slots__ = (
        "type",
        "query",
        "source",
        "dataset",
        "supporters",
        "opposers",
        "binary",
        "numeric_minimum",
        "numeric_maximum",
        "supporters_covered",
        "opposers_covered",
        "metrics",
    )

    def __init__(
        self,
        query: Sample,
        source: Sample,
        dataset: Dataset,
        type: Type,
        binary: numpy.ndarray,
        numeric_minimum: numpy.ndarray,
        numeric_maximum: numpy.ndarray,
        supporters: numpy.ndarray,
        opposers: numpy.ndarray,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ):
        self.type = type
        self.query = query
        self.source = source
        self.dataset = dataset
        match type:
            case Classifier.Type.POSITIVE:
                self.supporters = dataset.positive
                self.opposers = dataset.negative
            case Classifier.Type.NEGATIVE:
                self.supporters = dataset.negative
                self.opposers = dataset.positive

        self.binary = binary
        self.numeric_minimum = numeric_minimum
        self.numeric_maximum = numeric_maximum

        self.supporters_covered = supporters
        self.opposers_covered = opposers
        self.metrics = LazyMetrics(self)
        self.metrics.tp = int(tp)
        self.metrics.fp = int(fp)
        self.metrics.tn = int(tn)
        self.metrics.fn = int(fn)

    def get_metrics(self) -> Metrics:
        return self.metrics

    def to_string(self):
        parts = []
        for binary in self.binary:
            parts.append("1" if binary else "0")
        for minimum, maximum in zip(self.numeric_minimum, self.numeric_maximum):
            parts.append(f"[{minimum}, {maximum}]")
        return "; ".join(parts)

    def __repr__(self) -> str:
        lines = [f"Classifier  [{self.type}]"]
        lines.append("=" * 46)
        lines.append(f"  {'Hypothesis':<16}: {self.to_string()}")
        lines.append(f"  {'Supporters':<16}: {len(self.supporters)}")
        lines.append(f"  {'Opposers':<16}: {len(self.opposers)}")

        computed = [
            (m.name, getattr(self.metrics, m.attr), m.is_minimized)
            for m in METADATA
            if getattr(self.metrics, m.attr) is not None
        ]
        if computed:
            lines.append("")
            lines.append("  Computed metrics")
            lines.append("  " + "-" * 42)
            col = max(len(name) for name, _, _ in computed)
            for name, value, minimized in computed:
                fmt = f"{value:.4f}" if isinstance(value, float) else str(value)
                tag = " (↓)" if minimized else ""
                lines.append(f"  {name:<{col}}  {fmt}{tag}")

        lines.append("=" * 46)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self, with_metrics: bool = True) -> dict:
        return {
            "Hypothesis": self.to_string(),
            "Type": "POSITIVE" if self.type == Classifier.Type.POSITIVE else "NEGATIVE",
            "Supporters": len(self.supporters),
            "Opposers": len(self.opposers),
            **(self.metrics.to_dict() if with_metrics else {}),
        }

    def is_more_general_than(self, other: Classifier):
        return is_more_general(
            self.binary,
            self.numeric_minimum,
            self.numeric_maximum,
            other.binary,
            other.numeric_minimum,
            other.numeric_maximum,
        )

    @staticmethod
    @numba.njit(cache=True)
    def minimize_classifiers_raw(
        batch_binary: numpy.ndarray, batch_numeric_minimum: numpy.ndarray, batch_numeric_maximum: numpy.ndarray
    ) -> typing.List[int]:
        # Allocate enough memory for the worst case. Dynamic memory is expensive.
        keep, keep_len = numpy.empty(batch_binary.shape[0], numba.int32), 0
        for i in range(batch_binary.shape[0]):
            found, j = False, 0
            while j < keep_len:
                if not found and is_more_general(
                    batch_binary[keep[j]],
                    batch_numeric_minimum[keep[j]],
                    batch_numeric_maximum[keep[j]],
                    batch_binary[i],
                    batch_numeric_minimum[i],
                    batch_numeric_maximum[i],
                ):
                    # There is already a more general classifier in the collected set
                    found = True
                    break
                if is_more_general(
                    batch_binary[i],
                    batch_numeric_minimum[i],
                    batch_numeric_maximum[i],
                    batch_binary[keep[j]],
                    batch_numeric_minimum[keep[j]],
                    batch_numeric_maximum[keep[j]],
                ):
                    # keep[j] is less general than the current one
                    if found:
                        # current is already in the set
                        # remove keep[j] by moving it to the end
                        keep_len -= 1
                        keep[j] = keep[keep_len]
                        j -= 1
                    else:
                        # current is not in the set, replace keep[j] with current
                        keep[j], found = i, True
                j += 1
            if not found:
                keep[keep_len] = i
                keep_len += 1
        return keep[:keep_len]

    @staticmethod
    def minimize_classifiers(classifiers: typing.List[Classifier]) -> typing.List[Classifier]:
        keep_idx = Classifier.minimize_classifiers_raw(
            numpy.stack([c.binary for c in classifiers]),
            numpy.stack([c.numeric_minimum for c in classifiers]),
            numpy.stack([c.numeric_maximum for c in classifiers]),
        )
        return [classifiers[i] for i in keep_idx]

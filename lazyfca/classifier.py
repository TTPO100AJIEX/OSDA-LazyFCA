from __future__ import annotations

import numpy

from lazyfca.dataset import Sample
from lazyfca.dataset import Subset
from lazyfca.dataset import Dataset


from lazyfca.metrics import Metrics
from lazyfca.metrics import LazyMetrics
from lazyfca.metrics import METADATA


class Hypothesis:
    def __init__(self, lhs: Sample, rhs: Sample):
        self.binary = lhs.binary & rhs.binary
        self._not_binary = ~self.binary
        self.numeric_minimum = numpy.minimum(lhs.numeric, rhs.numeric)
        self.numeric_maximum = numpy.maximum(lhs.numeric, rhs.numeric)

    def covers(self, subset: Subset) -> numpy.ndarray:
        # Returns true/false for every object in the subset
        covers_binary = (subset.binary | self._not_binary).all(axis=1)
        covers_numeric = ((self.numeric_minimum <= subset.numeric) & (subset.numeric <= self.numeric_maximum)).all(
            axis=1
        )
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
        if type == Classifier.Type.POSITIVE:
            self.supporters = dataset.positive
            self.opposers = dataset.negative
        else:
            self.supporters = dataset.negative
            self.opposers = dataset.positive
        self.metrics = LazyMetrics(self)

    def get_metrics(self) -> Metrics:
        return self.metrics

    def __repr__(self) -> str:
        lines = [f"Classifier  [{self.type}]"]
        lines.append("=" * 46)
        lines.append(f"  {'Hypothesis':<16}: {self.hypothesis.to_string()}")
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
            "Hypothesis": self.hypothesis.to_string(),
            "Type": self.type,
            "Supporters": len(self.supporters),
            "Opposers": len(self.opposers),
            **(self.metrics.to_dict() if with_metrics else {}),
        }

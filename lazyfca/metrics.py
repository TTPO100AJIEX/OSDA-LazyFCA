from __future__ import annotations

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from lazyfca.classifier import Classifier

from lazyfca.calculators import contingency_simple
from lazyfca.calculators import contingency_complex
from lazyfca.calculators import matthews_correlation
from lazyfca.calculators import information_gain
from lazyfca.calculators import gini_gain
from lazyfca.calculators import similarity
from lazyfca.calculators import simplicity_prior
from lazyfca.calculators import contingency_expected
from lazyfca.calculators import stability


@dataclasses.dataclass
class Metadata:
    name: str
    attr: str
    lazy_calculator: typing.Callable[[LazyMetrics], None]
    is_minimized: bool = False


METADATA = [
    Metadata(name="True positive", attr="tp", lazy_calculator=contingency_simple),
    Metadata(name="False positive", attr="fp", lazy_calculator=contingency_simple, is_minimized=True),
    Metadata(name="True negative", attr="tn", lazy_calculator=contingency_simple),
    Metadata(name="False negative", attr="fn", lazy_calculator=contingency_simple, is_minimized=True),
    Metadata(name="Supporters covered", attr="supporters_covered", lazy_calculator=contingency_simple),
    Metadata(name="Opposers covered", attr="opposers_covered", lazy_calculator=contingency_simple, is_minimized=True),
    Metadata(name="Supporters to opposers ratio", attr="supporter_opposer_ratio", lazy_calculator=contingency_simple),
    Metadata(name="Support", attr="support", lazy_calculator=contingency_complex),
    Metadata(name="Error rate", attr="error_rate", lazy_calculator=contingency_complex, is_minimized=True),
    Metadata(name="Precision", attr="precision", lazy_calculator=contingency_complex),
    Metadata(name="Lift", attr="lift", lazy_calculator=contingency_complex),
    Metadata(name="WRAcc", attr="wracc", lazy_calculator=contingency_complex),
    Metadata(name="Balanced precision proxy", attr="balanced_precision_proxy", lazy_calculator=contingency_complex),
    Metadata(name="Youden's J", attr="youdens_j", lazy_calculator=contingency_complex),
    Metadata(name="Matthews correlation", attr="matthews_correlation", lazy_calculator=matthews_correlation),
    Metadata(name="Information gain", attr="information_gain", lazy_calculator=information_gain),
    Metadata(name="Gini gain", attr="gini_gain", lazy_calculator=gini_gain),
    Metadata(name="Log odds ratio", attr="log_odds_ratio", lazy_calculator=contingency_complex),
    Metadata(name="Chi squared", attr="chi_squared", lazy_calculator=contingency_expected),
    Metadata(name="G-test", attr="g_test", lazy_calculator=contingency_expected),
    Metadata(name="Interval tightness", attr="interval_tightness", lazy_calculator=similarity),
    Metadata(name="Description volume", attr="description_volume", lazy_calculator=similarity, is_minimized=True),
    Metadata(name="Simplicity prior", attr="simplicity_prior", lazy_calculator=simplicity_prior),
    Metadata(name="Query binary similarity", attr="query_binary_similarity", lazy_calculator=similarity),
    Metadata(name="Query numeric similarity", attr="query_numeric_similarity", lazy_calculator=similarity),
    Metadata(name="Query similarity", attr="query_similarity", lazy_calculator=similarity),
    Metadata(name="Query weighted precision", attr="query_weighted_precision", lazy_calculator=similarity),
    Metadata(name="Query weighted WRAcc", attr="query_weighted_wracc", lazy_calculator=similarity),
    Metadata(name="Stability", attr="stability", lazy_calculator=stability),
    Metadata(name="Robustness", attr="robustness", lazy_calculator=stability),
    Metadata(name="Delta stability", attr="delta_stability", lazy_calculator=stability),
]

MINIMZED_FIELDS = [metadata.attr for metadata in METADATA if metadata.is_minimized]


@dataclasses.dataclass
class Metrics:
    tp: typing.Optional[int] = None
    fp: typing.Optional[int] = None
    tn: typing.Optional[int] = None
    fn: typing.Optional[int] = None
    supporters_covered: typing.Optional[int] = None
    opposers_covered: typing.Optional[int] = None
    supporter_opposer_ratio: typing.Optional[float] = None

    support: typing.Optional[float] = None
    error_rate: typing.Optional[float] = None
    precision: typing.Optional[float] = None
    lift: typing.Optional[float] = None
    wracc: typing.Optional[float] = None
    balanced_precision_proxy: typing.Optional[float] = None
    youdens_j: typing.Optional[float] = None
    matthews_correlation: typing.Optional[float] = None
    information_gain: typing.Optional[float] = None
    gini_gain: typing.Optional[float] = None
    log_odds_ratio: typing.Optional[float] = None
    chi_squared: typing.Optional[float] = None
    g_test: typing.Optional[float] = None
    interval_tightness: typing.Optional[float] = None
    description_volume: typing.Optional[float] = None
    simplicity_prior: typing.Optional[float] = None
    query_binary_similarity: typing.Optional[float] = None
    query_numeric_similarity: typing.Optional[float] = None
    query_similarity: typing.Optional[float] = None
    query_weighted_precision: typing.Optional[float] = None
    query_weighted_wracc: typing.Optional[float] = None
    stability: typing.Optional[float] = None
    robustness: typing.Optional[float] = None
    delta_stability: typing.Optional[float] = None

    def get_metric(self, metric: str):
        if not hasattr(self, metric):
            for metadata in METADATA:
                if metadata.name == metric:
                    metric = metadata.attr
                    break
            else:
                raise NameError(f"Unknown metric: {metric}")
        return getattr(self, metric)

    def to_dict(self):
        return {metadata.name: self.get_metric(metadata.attr) for metadata in METADATA}

    def score_for_ranking(self, field: str) -> float:
        value = self.get_metric(field)
        return -value if field in MINIMZED_FIELDS else value

    @staticmethod
    def from_dict(dictionary: dict) -> Metrics:
        result = Metrics()
        for metadata in METADATA:
            if metadata.name in dictionary:
                setattr(result, metadata.attr, dictionary[metadata.name])
            if metadata.attr in dictionary:
                setattr(result, metadata.attr, dictionary[metadata.attr])
        return result

    def is_better_than(self, other: Metrics) -> bool:
        for metadata in METADATA:
            # If value is not set, consider it as the worst possible value
            other_value = other.get_metric(metadata.attr)
            if other_value is None:
                continue

            self_value = self.get_metric(metadata.attr)
            if self_value is None:
                return False

            if metadata.is_minimized:
                if self_value > other_value:
                    return False
            else:
                if self_value < other_value:
                    return False
        return True


class LazyMetrics(Metrics):
    def __init__(self, classifier: Classifier):
        self.classifier = classifier
        self.metrics = Metrics()

    def get_metric(self, metric: str):
        raw = super().get_metric(metric)
        if raw is not None:
            return raw
        next(item for item in METADATA if item.attr == metric or item.name == metric).lazy_calculator(self)
        return super().get_metric(metric)

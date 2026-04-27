from __future__ import annotations

import typing
import math

if typing.TYPE_CHECKING:
    from lazyfca.metrics import LazyMetrics

import numpy


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator != 0 else default


def _get_basic(metrics: LazyMetrics):
    p, n = len(metrics.classifier.supporters), len(metrics.classifier.opposers)
    tp, fp = metrics.get_metric("tp"), metrics.get_metric("fp")
    fn, tn = metrics.get_metric("fn"), metrics.get_metric("tn")
    return p, n, tp, fp, fn, tn


def _contingency_expected(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float, float]:
    total = tp + fp + fn + tn
    positive_row = tp + fp
    negative_row = fn + tn
    supporter_col = tp + fn
    opposer_col = fp + tn
    return (
        positive_row * supporter_col / total,
        positive_row * opposer_col / total,
        negative_row * supporter_col / total,
        negative_row * opposer_col / total,
    )


def _xlogy(observed: float, expected: float) -> float:
    if observed <= 0 or expected <= 0:
        return 0.0
    return observed * math.log(observed / expected)


def contingency_complex(metrics: LazyMetrics):
    p, n, tp, fp, fn, tn = _get_basic(metrics)
    base_pos_rate = p / (p + n)
    log_tp = math.log1p(tp)
    sqrt_tp = math.sqrt(tp)

    metrics.supporter_opposer_ratio = _safe_div(metrics.tp, metrics.fp, numpy.inf)
    metrics.support = tp / p
    metrics.error_rate = fp / n
    metrics.precision = tp / (tp + fp)
    metrics.precision_log_tp = metrics.precision * log_tp
    metrics.precision_sqrt_tp = metrics.precision * sqrt_tp
    metrics.lift = metrics.precision / base_pos_rate
    metrics.wracc = (tp + fp) / (p + n) * (metrics.precision - base_pos_rate)
    metrics.balanced_precision_proxy = metrics.tp / p - fp / n
    metrics.youdens_j = metrics.tp / (tp + fn) - fp / (fp + tn)
    metrics.log_odds_ratio = (2 * tp + 1) / (2 * fp + 1)
    metrics.log_odds_ratio_log_tp = metrics.log_odds_ratio * log_tp


def matthews_correlation(metrics: LazyMetrics):
    p, n, tp, fp, fn, tn = _get_basic(metrics)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics.matthews_correlation = _safe_div(tp * tn - fp * fn, mcc_denominator)


def information_gain(metrics: LazyMetrics):
    def _entropy(p: int, n: int) -> float:
        total = p + n
        if total == 0 or p == 0 or n == 0:
            return 0.0
        p_ratio = p / total
        n_ratio = n / total
        return -(p_ratio * math.log2(p_ratio) + n_ratio * math.log2(n_ratio))

    p, n, tp, fp, fn, tn = _get_basic(metrics)
    metrics.information_gain = _entropy(p, n) - ((tp + fp) * _entropy(tp, fp) + (fn + tn) * _entropy(fn, tn)) / (p + n)


def gini_gain(metrics: LazyMetrics):
    def _impurity(p: int, n: int) -> float:
        total = p + n
        if total == 0:
            return 0.0
        p_ratio = p / total
        n_ratio = n / total
        return 1.0 - p_ratio**2 - n_ratio**2

    p, n, tp, fp, fn, tn = _get_basic(metrics)
    metrics.gini_gain = _impurity(p, n) - ((tp + fp) * _impurity(tp, fp) + (fn + tn) * _impurity(fn, tn)) / (p + n)


def contingency_expected(metrics: LazyMetrics):
    p, n, tp, fp, fn, tn = _get_basic(metrics)
    expected_tp, expected_fp, expected_fn, expected_tn = _contingency_expected(tp, fp, fn, tn)
    metrics.chi_squared = sum(
        [
            _safe_div((tp - expected_tp) ** 2, expected_tp),
            _safe_div((fp - expected_fp) ** 2, expected_fp),
            _safe_div((fn - expected_fn) ** 2, expected_fn),
            _safe_div((tn - expected_tn) ** 2, expected_tn),
        ]
    )
    metrics.g_test = 2.0 * sum(
        [
            _xlogy(tp, expected_tp),
            _xlogy(fp, expected_fp),
            _xlogy(fn, expected_fn),
            _xlogy(tn, expected_tn),
        ]
    )


def similarity(metrics: LazyMetrics):
    clf = metrics.classifier

    def _query_binary_similarity() -> float:
        query_active = int(clf.query.binary.sum())
        if query_active == 0:
            return 1.0
        matched = int(clf.binary.sum())
        return matched / query_active

    def _interval_tightness() -> tuple[float, float]:
        if len(clf.dataset.numeric_range) == 0:
            return 1.0, 0.0

        widths = clf.numeric_maximum - clf.numeric_minimum
        normalized_widths = numpy.divide(
            widths,
            clf.dataset.numeric_range,
            out=numpy.zeros_like(widths, dtype=numpy.float64),
            where=clf.dataset.numeric_range > 0,
        )
        normalized_widths = numpy.clip(normalized_widths, 0.0, 1.0)
        interval_tightness = 1.0 - float(normalized_widths.mean())
        description_volume = float(numpy.prod(normalized_widths))
        return interval_tightness, description_volume

    query_binary_similarity = _query_binary_similarity()
    interval_tightness, description_volume = _interval_tightness()

    query_numeric_similarity = interval_tightness
    similarity_parts = [query_binary_similarity] if clf.dataset.binary_feature_count > 0 else []
    if clf.dataset.numeric_feature_count > 0:
        similarity_parts.append(query_numeric_similarity)
    query_similarity = float(numpy.mean(similarity_parts)) if similarity_parts else 1.0

    metrics.interval_tightness = interval_tightness
    metrics.description_volume = description_volume
    metrics.query_binary_similarity = query_binary_similarity
    metrics.query_numeric_similarity = query_numeric_similarity
    metrics.query_similarity = query_similarity
    metrics.query_weighted_precision = metrics.get_metric("precision") * query_similarity
    metrics.query_weighted_precision_log_tp = metrics.get_metric("precision_log_tp") * query_similarity
    metrics.query_weighted_precision_sqrt_tp = metrics.get_metric("precision_sqrt_tp") * query_similarity
    metrics.query_weighted_wracc = metrics.get_metric("wracc") * query_similarity


def simplicity_prior(metrics: LazyMetrics):
    clf = metrics.classifier
    interval_tightness = metrics.get_metric("interval_tightness")

    binary_complexity = _safe_div(float(clf.binary.sum()), clf.dataset.binary_feature_count)
    interval_complexity = 1.0 - interval_tightness
    description_complexity = binary_complexity + interval_complexity
    metrics.simplicity_prior = 1.0 / (1.0 + description_complexity)


def stability(metrics: LazyMetrics):
    clf = metrics.classifier

    covered_binary = clf.supporters.binary[clf.supporters_covered]
    covered_numeric = clf.supporters.numeric[clf.supporters_covered]
    witness_sizes = [len(covered_binary)]  # The regenerating subset must be non-empty.

    dropped_binary = clf.query.binary & ~clf.binary
    for index in numpy.flatnonzero(dropped_binary):
        witness_sizes.append(int((~covered_binary[:, index]).sum()))

    for index in range(len(clf.numeric_minimum)):
        min_witnesses = int((covered_numeric[:, index] == clf.numeric_minimum[index]).sum())
        max_witnesses = int((covered_numeric[:, index] == clf.numeric_maximum[index]).sum())
        witness_sizes.append(min_witnesses)
        if clf.numeric_minimum[index] != clf.numeric_maximum[index]:
            witness_sizes.append(max_witnesses)

    witness_sizes = [size for size in witness_sizes if size > 0]
    if not witness_sizes:
        return 0.0, 0.0

    metrics.stability = float(numpy.prod([1.0 - 2.0 ** (-size) for size in witness_sizes]))
    metrics.robustness = metrics.stability
    metrics.delta_stability = float(min(witness_sizes))

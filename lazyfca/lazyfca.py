from __future__ import annotations

import enum
import typing
import dataclasses
import gc

import tqdm
import numpy
import pandas
import joblib

from lazyfca.dataset import Dataset
from lazyfca.explanation import Explanation
from lazyfca.classifier import Classifier
from lazyfca.metrics import Metrics


class LazyFCA:
    """Lazy FCA classifier with native multi-class support.

    The original binary API (``pos_params``, ``neg_params``, ``pos_weight``,
    ``pos_rank_by``, ``neg_rank_by``, ``pos_top_k``, ``neg_top_k``) is fully
    preserved when ``n_classes == 2``. To use more than two classes, set
    ``n_classes`` and pass per-class configuration via ``class_params``,
    ``class_weights``, ``class_rank_by`` and ``class_top_k``.
    """

    Params = Metrics

    class MinimizePolicy(enum.Enum):
        NO_MINIMIZE = 0
        BEFORE_FILTER = 1
        BEFORE_TRIM = 2
        BEFORE_TOPK = 3

    def __init__(
        self,
        pos_params: typing.Optional[Metrics] = None,
        neg_params: typing.Optional[Metrics] = None,
        pos_weight: float = 1.0,
        pos_rank_by: typing.Optional[str] = None,
        neg_rank_by: typing.Optional[str] = None,
        pos_top_k: typing.Optional[int] = None,
        neg_top_k: typing.Optional[int] = None,
        rank_by: typing.Optional[str] = None,
        top_k: typing.Optional[int] = None,
        minimize_policy: MinimizePolicy = MinimizePolicy.NO_MINIMIZE,
        n_classes: int = 2,
        class_params: typing.Optional[typing.List[Metrics]] = None,
        class_weights: typing.Optional[typing.List[float]] = None,
        class_rank_by: typing.Optional[typing.List[typing.Optional[str]]] = None,
        class_top_k: typing.Optional[typing.List[typing.Optional[int]]] = None,
    ):
        assert n_classes >= 2, f"n_classes must be >= 2, got {n_classes}"
        self.n_classes = int(n_classes)
        self.minimize_policy = minimize_policy

        # ---------------- Per-class thresholds (Metrics) ------------------ #
        if class_params is not None:
            assert len(class_params) == self.n_classes, (
                f"class_params has length {len(class_params)}, expected {self.n_classes}"
            )
            self.class_params = list(class_params)
        else:
            assert self.n_classes == 2 or (pos_params is None and neg_params is None), (
                "For n_classes != 2 use class_params instead of pos_params/neg_params"
            )
            self.class_params = [Metrics() for _ in range(self.n_classes)]
            if self.n_classes == 2:
                self.class_params[0] = neg_params if neg_params is not None else Metrics()
                self.class_params[1] = pos_params if pos_params is not None else Metrics()

        # ---------------- Per-class voting weights ------------------------ #
        if class_weights is not None:
            assert len(class_weights) == self.n_classes
            self.class_weights = [float(w) for w in class_weights]
        else:
            self.class_weights = [1.0] * self.n_classes
            if self.n_classes == 2:
                self.class_weights[1] = float(pos_weight)

        # ---------------- Per-class rank_by ------------------------------- #
        if class_rank_by is not None:
            assert len(class_rank_by) == self.n_classes
            self.class_rank_by = list(class_rank_by)
        else:
            assert self.n_classes == 2 or (pos_rank_by is None and neg_rank_by is None), (
                "For n_classes != 2 use class_rank_by instead of pos_rank_by/neg_rank_by"
            )
            self.class_rank_by = [None] * self.n_classes
            if self.n_classes == 2:
                self.class_rank_by[0] = neg_rank_by
                self.class_rank_by[1] = pos_rank_by

        # ---------------- Per-class top_k --------------------------------- #
        if class_top_k is not None:
            assert len(class_top_k) == self.n_classes
            self.class_top_k = list(class_top_k)
        else:
            assert self.n_classes == 2 or (pos_top_k is None and neg_top_k is None), (
                "For n_classes != 2 use class_top_k instead of pos_top_k/neg_top_k"
            )
            self.class_top_k = [None] * self.n_classes
            if self.n_classes == 2:
                self.class_top_k[0] = neg_top_k
                self.class_top_k[1] = pos_top_k

        # ---------------- Cross-class rank/top-k -------------------------- #
        self.rank_by = rank_by
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    # Backwards-compatible binary-only properties                         #
    # ------------------------------------------------------------------ #

    @property
    def pos_params(self) -> Metrics:
        assert self.n_classes == 2
        return self.class_params[1]

    @property
    def neg_params(self) -> Metrics:
        assert self.n_classes == 2
        return self.class_params[0]

    @property
    def pos_weight(self) -> float:
        assert self.n_classes == 2
        return self.class_weights[1]

    @property
    def pos_rank_by(self) -> typing.Optional[str]:
        assert self.n_classes == 2
        return self.class_rank_by[1]

    @property
    def neg_rank_by(self) -> typing.Optional[str]:
        assert self.n_classes == 2
        return self.class_rank_by[0]

    @property
    def pos_top_k(self) -> typing.Optional[int]:
        assert self.n_classes == 2
        return self.class_top_k[1]

    @property
    def neg_top_k(self) -> typing.Optional[int]:
        assert self.n_classes == 2
        return self.class_top_k[0]

    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lines = ["LazyFCA"]
        lines.append("=" * 40)

        fitted = hasattr(self, "dataset")
        lines.append(f"  Status       : {'fitted' if fitted else 'not fitted'}")
        lines.append(f"  Classes      : {self.n_classes}")

        if fitted:
            ds = self.dataset
            sizes = [f"class_{c}={len(ds.subsets[c])}" for c in range(ds.n_classes)]
            total = sum(len(s) for s in ds.subsets)
            lines.append(f"  Dataset      : {total} samples ({', '.join(sizes)})")
            lines.append(f"  Features     : {ds.binary_feature_count} binary, {ds.numeric_feature_count} numeric")

        lines.append("")
        lines.append("  Filtering thresholds")
        lines.append("  " + "-" * 36)

        def fmt_params(label: str, params: Metrics) -> None:
            active = {k: v for k, v in dataclasses.asdict(params).items() if v is not None}
            if active:
                kv = ", ".join(f"{k}={v}" for k, v in active.items())
                lines.append(f"  {label:<14}: {kv}")
            else:
                lines.append(f"  {label:<14}: (none)")

        for c, params in enumerate(self.class_params):
            label = (
                "pos_params"
                if (self.n_classes == 2 and c == 1)
                else "neg_params"
                if (self.n_classes == 2 and c == 0)
                else f"class_{c}_params"
            )
            fmt_params(label, params)

        weight_label = "weights" if self.n_classes != 2 else "pos_weight"
        weight_value = self.class_weights if self.n_classes != 2 else self.class_weights[1]
        lines.append(f"  {weight_label:<14}: {weight_value}")

        lines.append("")
        lines.append("  Ranking & top-k")
        lines.append("  " + "-" * 36)

        def fmt_opt(label: str, value) -> str:
            return f"  {label:<14}: {value if value is not None else '—'}"

        if self.n_classes == 2:
            lines.append(fmt_opt("pos_rank_by", self.class_rank_by[1]))
            lines.append(fmt_opt("neg_rank_by", self.class_rank_by[0]))
            lines.append(fmt_opt("pos_top_k", self.class_top_k[1]))
            lines.append(fmt_opt("neg_top_k", self.class_top_k[0]))
        else:
            lines.append(fmt_opt("class_rank_by", self.class_rank_by))
            lines.append(fmt_opt("class_top_k", self.class_top_k))
        lines.append(fmt_opt("rank_by", self.rank_by))
        lines.append(fmt_opt("top_k", self.top_k))
        lines.append(fmt_opt("minimize_policy", self.minimize_policy))

        lines.append("=" * 40)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    # ------------------------------------------------------------------ #
    # Fit / predict / explain                                             #
    # ------------------------------------------------------------------ #

    def fit(self, X_train: pandas.DataFrame, y_train: pandas.Series):
        self.dataset = Dataset(X_train, y_train, n_classes=self.n_classes)
        # Sync n_classes with whatever the dataset chose (still required to be the same).
        assert self.dataset.n_classes == self.n_classes
        return self

    def classify_explanation(
        self, explanation: Explanation, trust: bool = False, probs: bool = True
    ) -> numpy.ndarray:
        """Return the per-class score vector for ``explanation``.

        When ``probs=False`` the raw count of surviving classifiers per class
        is returned. When ``probs=True`` the counts are weighted by
        :attr:`class_weights` and normalized to sum to 1; if no classifier
        survives for any class, a uniform distribution is returned.

        The output is a one-dimensional array of length ``n_classes`` indexed
        by class label.
        """
        explanation = self._process_explanation(explanation, trust=trust, inplace=False)
        counts = numpy.array(
            [len(explanation.class_classifiers[c]) for c in range(self.n_classes)],
            dtype=numpy.float64,
        )
        if not probs:
            return counts

        weighted = counts * numpy.array(self.class_weights, dtype=numpy.float64)
        total = float(weighted.sum())
        if total == 0:
            return numpy.full(self.n_classes, 1.0 / self.n_classes, dtype=numpy.float64)
        return weighted / total

    def classify_explanations(
        self, explanations: typing.List[Explanation], trust: bool = False, probs: bool = True
    ) -> numpy.ndarray:
        return numpy.array([self.classify_explanation(explanation, trust, probs) for explanation in explanations])

    def classify_sample(self, sample: pandas.Series) -> numpy.ndarray:
        return self.classify_explanation(self.explain_sample(sample), trust=True, probs=True)

    def predict(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> numpy.ndarray:
        return numpy.array(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.classify_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

    def explain_sample(self, sample: pandas.Series) -> Explanation:
        sample = self.dataset.make_sample(sample)
        class_classifiers = [
            Classifier.calculate_classifiers(sample, self.dataset, c) for c in range(self.n_classes)
        ]
        explanation = Explanation(
            self.dataset,
            sample,
            class_classifiers=class_classifiers,
        )
        explanation = self._process_explanation(explanation, trust=False, inplace=True)
        gc.collect()
        return explanation

    def explain(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> typing.List[Explanation]:
        return list(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.explain_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

    def _process_explanation(self, explanation: Explanation, trust: bool, inplace: bool):
        if trust:
            return explanation

        if self.minimize_policy == LazyFCA.MinimizePolicy.BEFORE_FILTER:
            explanation = explanation.minimize(inplace=inplace)
        explanation = explanation.filter(class_thresholds=self.class_params, inplace=inplace)

        # further we can do inplace=True, this is not a bug. If needed, the copy has already been made above
        if self.minimize_policy == LazyFCA.MinimizePolicy.BEFORE_TRIM:
            explanation = explanation.minimize(inplace=True)
        explanation = explanation.rank(class_rank_by=self.class_rank_by, inplace=True)
        explanation = explanation.trim(class_top_k=self.class_top_k, inplace=True)

        if self.minimize_policy == LazyFCA.MinimizePolicy.BEFORE_TOPK:
            explanation = explanation.minimize(inplace=True)
        if self.rank_by is not None:
            explanation = explanation.rank(
                class_rank_by=[self.rank_by] * self.n_classes,
                inplace=True,
            )
        if self.top_k is not None:
            explanation = explanation.keep_top_k(self.top_k, inplace=True)

        return explanation

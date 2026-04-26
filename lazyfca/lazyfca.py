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
    Params = Metrics

    class MinimizePolicy(enum.Enum):
        NO_MINIMIZE = 0
        BEFORE_FILTER = 1
        BEFORE_TRIM = 2
        BEFORE_TOPK = 3

    def __init__(
        self,
        pos_params: LazyFCA.Params = Metrics(),
        neg_params: LazyFCA.Params = Metrics(),
        pos_weight: float = 1.0,
        pos_rank_by: typing.Optional[str] = None,
        neg_rank_by: typing.Optional[str] = None,
        pos_top_k: typing.Optional[int] = None,
        neg_top_k: typing.Optional[int] = None,
        rank_by: typing.Optional[str] = None,
        top_k: typing.Optional[int] = None,
        minimize_policy: MinimizePolicy = MinimizePolicy.NO_MINIMIZE,
    ):
        self.pos_params = pos_params
        self.neg_params = neg_params
        self.pos_weight = pos_weight
        self.pos_rank_by = pos_rank_by
        self.neg_rank_by = neg_rank_by
        self.pos_top_k = pos_top_k
        self.neg_top_k = neg_top_k
        self.rank_by = rank_by
        self.top_k = top_k
        self.minimize_policy = minimize_policy

    def __repr__(self) -> str:
        lines = ["LazyFCA"]
        lines.append("=" * 40)

        fitted = hasattr(self, "dataset")
        lines.append(f"  Status       : {'fitted' if fitted else 'not fitted'}")

        if fitted:
            ds = self.dataset
            total = len(ds.positive) + len(ds.negative)
            lines.append(f"  Dataset      : {total} samples (pos={len(ds.positive)}, neg={len(ds.negative)})")
            lines.append(f"  Features     : {ds.binary_feature_count} binary, {ds.numeric_feature_count} numeric")

        lines.append("")
        lines.append("  Filtering thresholds")
        lines.append("  " + "-" * 36)

        def fmt_params(label: str, params) -> None:
            active = {k: v for k, v in dataclasses.asdict(params).items() if v is not None}
            if active:
                kv = ", ".join(f"{k}={v}" for k, v in active.items())
                lines.append(f"  {label:<14}: {kv}")
            else:
                lines.append(f"  {label:<14}: (none)")

        fmt_params("pos_params", self.pos_params)
        fmt_params("neg_params", self.neg_params)
        lines.append(f"  {'pos_weight':<14}: {self.pos_weight}")

        lines.append("")
        lines.append("  Ranking & top-k")
        lines.append("  " + "-" * 36)

        def fmt_opt(label: str, value) -> str:
            return f"  {label:<14}: {value if value is not None else '—'}"

        lines.append(fmt_opt("pos_rank_by", self.pos_rank_by))
        lines.append(fmt_opt("neg_rank_by", self.neg_rank_by))
        lines.append(fmt_opt("rank_by", self.rank_by))
        lines.append(fmt_opt("pos_top_k", self.pos_top_k))
        lines.append(fmt_opt("neg_top_k", self.neg_top_k))
        lines.append(fmt_opt("top_k", self.top_k))
        lines.append(fmt_opt("minimize_policy", self.minimize_policy))

        lines.append("=" * 40)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def fit(self, X_train: pandas.DataFrame, y_train: pandas.Series):
        self.dataset = Dataset(X_train, y_train)
        return self

    def classify_explanation(
        self, explanation: Explanation, trust: bool = False, probs: bool = True
    ) -> typing.Tuple[float, float]:
        explanation = self._process_explanation(explanation, trust=trust, inplace=False)
        positive, negative = len(explanation.positive_classifiers), len(explanation.negative_classifiers)
        if not probs:
            return (negative, positive)
        positive *= self.pos_weight
        total = negative + positive
        return (0.5, 0.5) if total == 0 else ((negative / total), (positive / total))

    def classify_explanations(
        self, explanations: typing.List[Explanation], trust: bool = False, probs: bool = True
    ) -> numpy.ndarray:
        return numpy.array([self.classify_explanation(explanation, trust, probs) for explanation in explanations])

    def classify_sample(self, sample: pandas.Series) -> typing.Tuple[float, float]:
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
        explanation = Explanation(
            self.dataset,
            sample,
            Classifier.calculate_classifiers(sample, self.dataset, Classifier.Type.POSITIVE),
            Classifier.calculate_classifiers(sample, self.dataset, Classifier.Type.NEGATIVE),
        )
        explanation = self._process_explanation(explanation, trust=False, inplace=True)
        # gc.collect()
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
        explanation = explanation.filter(self.pos_params, self.neg_params, inplace=inplace)

        # further we can do inplace=True, this is not a bug. If needed, the copy has already been made above
        if self.minimize_policy == LazyFCA.MinimizePolicy.BEFORE_TRIM:
            explanation = explanation.minimize(inplace=True)
        explanation = explanation.rank(self.pos_rank_by, self.neg_rank_by, inplace=True)
        explanation = explanation.trim(self.pos_top_k, self.neg_top_k, inplace=True)

        if self.minimize_policy == LazyFCA.MinimizePolicy.BEFORE_TOPK:
            explanation = explanation.minimize(inplace=True)
        if self.rank_by is not None:
            explanation = explanation.rank(self.rank_by, self.rank_by, inplace=True)
        if self.top_k is not None:
            explanation = explanation.keep_top_k(self.top_k, inplace=True)

        return explanation

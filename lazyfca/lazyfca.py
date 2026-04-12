from __future__ import annotations

import typing

import gc
import tqdm
import numpy
import pandas
import joblib

from lazyfca.dataset import Dataset
from lazyfca.dataset import Subset
from lazyfca.explanation import Explanation
from lazyfca.classifier import Classifier
from lazyfca.metrics import Metrics


class LazyFCA:
    Params = Metrics

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

    def _rank(self, classifiers: typing.List[Classifier], rank_by: typing.Optional[str]) -> typing.List[Classifier]:
        return sorted(
            classifiers, key=lambda classifier: classifier.metrics.score_for_ranking(rank_by), reverse=True
        )

    def _rank_and_trim(
        self, classifiers: typing.List[Classifier], rank_by: typing.Optional[str], top_k: typing.Optional[int]
    ) -> typing.List[Classifier]:
        if rank_by is not None:
            classifiers = self._rank(classifiers, rank_by)
        if top_k is not None:
            classifiers = classifiers[:top_k]
        return classifiers

    def _get_top_k(
        self,
        positive_classifiers: typing.List[Classifier],
        negative_classifiers: typing.List[Classifier],
    ) -> typing.Tuple[typing.List[Classifier], typing.List[Classifier]]:
        if self.rank_by is not None:
            positive_classifiers = self._rank(positive_classifiers, self.rank_by)
            negative_classifiers = self._rank(negative_classifiers, self.rank_by)
        if self.top_k is not None:
            top_positive, top_negative = 0, 0
            while top_positive + top_negative < min(self.top_k, len(positive_classifiers) + len(negative_classifiers)):
                next_positive = positive_classifiers[top_positive].metrics.score_for_ranking(self.rank_by)
                next_negative = negative_classifiers[top_negative].metrics.score_for_ranking(self.rank_by)
                if next_positive > next_negative:
                    top_positive += 1
                else:
                    top_negative += 1
            positive_classifiers = positive_classifiers[:top_positive]
            negative_classifiers = negative_classifiers[:top_negative]

        return positive_classifiers, negative_classifiers

    def fit(self, X_train: pandas.DataFrame, y_train: pandas.Series):
        self.dataset = Dataset(X_train, y_train)
        return self

    def classify_explanation(
        self, explanation: Explanation, trust: bool = False, probs: bool = True
    ) -> typing.Tuple[float, float]:
        if trust:
            positive_classifiers = explanation.positive_classifiers
            negative_classifiers = explanation.negative_classifiers
        else:
            positive_classifiers = list(
                filter(
                    lambda classifier: classifier.metrics.is_better_than(self.pos_params),
                    explanation.positive_classifiers,
                )
            )
            negative_classifiers = list(
                filter(
                    lambda classifier: classifier.metrics.is_better_than(self.neg_params),
                    explanation.negative_classifiers,
                )
            )
            positive_classifiers = self._rank_and_trim(positive_classifiers, self.pos_rank_by, self.pos_top_k)
            negative_classifiers = self._rank_and_trim(negative_classifiers, self.neg_rank_by, self.neg_top_k)
            positive_classifiers, negative_classifiers = self._get_top_k(positive_classifiers, negative_classifiers)
        positive, negative = len(positive_classifiers), len(negative_classifiers)
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

        def make_classifiers(
            type: Classifier.Type,
            subset: Subset,
            params: LazyFCA.Params,
            rank_by: typing.Optional[str],
            top_k: typing.Optional[int],
        ):
            classifiers = map(lambda example: Classifier(sample, example, self.dataset, type), subset)
            classifiers = list(filter(lambda classifier: classifier.metrics.is_better_than(params), classifiers))
            return self._rank_and_trim(classifiers, rank_by, top_k)

        positive_classifiers = make_classifiers(
            Classifier.Type.POSITIVE, self.dataset.positive, self.pos_params, self.pos_rank_by, self.pos_top_k
        )
        negative_classifiers = make_classifiers(
            Classifier.Type.NEGATIVE, self.dataset.negative, self.neg_params, self.neg_rank_by, self.neg_top_k
        )
        positive_classifiers, negative_classifiers = self._get_top_k(positive_classifiers, negative_classifiers)
        explanation = Explanation(sample, positive_classifiers, negative_classifiers)
        gc.collect()
        return explanation

    def explain(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> typing.List[Explanation]:
        return list(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.explain_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

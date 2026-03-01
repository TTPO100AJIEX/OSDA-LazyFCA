import typing

import tqdm
import numpy
import pandas
import joblib

from .dataset import Dataset
from .dataset import Subset
from .explanation import Explanation
from .classifier import Classifier


class LazyFCA:
    Params = Classifier.Metrics

    def __init__(self, pos_params: Params = Params(), neg_params: Params = Params(), pos_weight: float = 1.0):
        self.pos_params = pos_params
        self.neg_params = neg_params
        self.pos_weight = pos_weight

    def fit(self, X_train: pandas.DataFrame, y_train: pandas.Series):
        self.dataset = Dataset(X_train, y_train)
        return self

    def classify_explanation(self, explanation: Explanation, trust: bool = True) -> typing.Tuple[float, float]:
        if trust:
            positive = len(explanation.positive_classifiers)
            negative = len(explanation.negative_classifiers)
        else:
            positive = sum([c.get_metrics().is_better_than(self.pos_params) for c in explanation.positive_classifiers])
            negative = sum([c.get_metrics().is_better_than(self.neg_params) for c in explanation.negative_classifiers])
        positive *= self.pos_weight
        total = negative + positive
        return (0.5, 0.5) if total == 0 else ((negative / total), (positive / total))

    def classify_explanations(
        self, explanations: typing.List[Explanation], trust: bool = True
    ) -> numpy.ndarray:
        return numpy.array([
            self.classify_explanation(explanation, trust) for explanation in explanations
        ])

    def classify_sample(self, sample: pandas.Series) -> typing.Tuple[float, float]:
        return self.classify_explanation(self.explain_sample(sample), trust=True)

    def predict(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> numpy.ndarray:
        return numpy.array(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.classify_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

    def explain_sample(self, sample: pandas.Series) -> Explanation:
        sample = self.dataset.make_sample(sample)

        def make_classifiers(type: Classifier.Type, subset: Subset, params: Classifier.Metrics):
            classifiers = map(lambda example: Classifier(sample, example, self.dataset, type), subset)
            return list(filter(lambda classifier: classifier.get_metrics().is_better_than(params), classifiers))

        return Explanation(
            sample,
            make_classifiers(Classifier.Type.POSITIVE, self.dataset.positive, self.pos_params),
            make_classifiers(Classifier.Type.NEGATIVE, self.dataset.negative, self.neg_params),
        )

    def explain(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> typing.List[Explanation]:
        return list(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.explain_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

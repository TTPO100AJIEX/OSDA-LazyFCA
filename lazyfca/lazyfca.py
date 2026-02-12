import tqdm
import numpy
import pandas
import joblib

from .dataset import Dataset
from .classifier import Hypothesis
from .explanation import Explanation
from .classifier import Classifier


class LazyFCA:
    def __init__(
        self,
        min_pos_for_pos_clas: int = 0,
        pos_coef_for_pos_clas: float = 1,
        min_neg_for_neg_clas: int = 0,
        neg_coef_for_neg_clas: float = 1,
        pos_clas_coef: float = 1,
    ):
        self.min_pos_for_pos_clas = min_pos_for_pos_clas
        self.pos_coef_for_pos_clas = pos_coef_for_pos_clas
        self.min_neg_for_neg_clas = min_neg_for_neg_clas
        self.neg_coef_for_neg_clas = neg_coef_for_neg_clas
        self.pos_clas_coef = pos_clas_coef

    def fit(self, X_train: pandas.DataFrame, y_train: pandas.Series):
        self.dataset = Dataset(X_train, y_train)

    def classify_sample(self, sample: pandas.Series):
        sample = self.dataset.make_sample(sample)

        num_positive_classifiers = sum(
            [self._is_positive_classifier(Hypothesis(sample, example)) for example in self.dataset.positive]
        )
        num_negative_classifiers = sum(
            [self._is_negative_classifier(Hypothesis(sample, example)) for example in self.dataset.negative]
        )

        num_positive_classifiers *= self.pos_clas_coef
        total = num_positive_classifiers + num_negative_classifiers
        if total == 0:
            return [0.5, 0.5]
        return [(num_negative_classifiers / total), (num_positive_classifiers / total)]

    def predict(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> numpy.ndarray:
        return numpy.array(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.classify_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

    def explain(self, sample: pandas.Series) -> Explanation:
        sample = self.dataset.make_sample(sample)

        def make_hypothesis(example):
            return Hypothesis(sample, example)

        return Explanation(
            sample,
            [
                Classifier(hypothesis, self.dataset, Classifier.Type.POSITIVE)
                for hypothesis in map(make_hypothesis, self.dataset.positive)
                if self._is_positive_classifier(hypothesis)
            ],
            [
                Classifier(hypothesis, self.dataset, Classifier.Type.NEGATIVE)
                for hypothesis in map(make_hypothesis, self.dataset.negative)
                if self._is_negative_classifier(hypothesis)
            ],
        )

    def _is_positive_classifier(self, hypothesis: Hypothesis):
        num_positive = hypothesis.covers(self.dataset.positive).sum()
        num_negative = hypothesis.covers(self.dataset.negative).sum()
        return self.pos_coef_for_pos_clas * num_positive > num_negative and num_positive >= self.min_pos_for_pos_clas

    def _is_negative_classifier(self, hypothesis: Hypothesis):
        num_positive = hypothesis.covers(self.dataset.positive).sum()
        num_negative = hypothesis.covers(self.dataset.negative).sum()
        return self.neg_coef_for_neg_clas * num_negative > num_positive and num_negative >= self.min_neg_for_neg_clas

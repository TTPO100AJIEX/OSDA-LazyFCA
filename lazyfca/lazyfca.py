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

    def classify_sample(self, sample: pandas.Series):
        explanation = self.explain(sample)

        pos = len(explanation.positive_classifiers) * self.pos_weight
        neg = len(explanation.negative_classifiers)
        total = pos + neg
        return [0.5, 0.5] if total == 0 else [(neg / total), (pos / total)]

    def predict(self, X_test: pandas.DataFrame, n_jobs: int = -1) -> numpy.ndarray:
        return numpy.array(
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.classify_sample)(sample)
                for _, sample in tqdm.tqdm(X_test.iterrows(), total=len(X_test))
            )
        )

    def explain(self, sample: pandas.Series) -> Explanation:
        sample = self.dataset.make_sample(sample)

        def make_classifiers(type: Classifier.Type, subset: Subset, params: Classifier.Metrics):
            classifiers = map(lambda example: Classifier(sample, example, self.dataset, type), subset)
            return list(filter(lambda classifier: classifier.get_metrics().is_better_than(params), classifiers))

        return Explanation(
            sample,
            make_classifiers(Classifier.Type.POSITIVE, self.dataset.positive, self.pos_params),
            make_classifiers(Classifier.Type.NEGATIVE, self.dataset.negative, self.neg_params),
        )

import tqdm
import pandas
import joblib
import numpy
import itertools
import sklearn.compose
import sklearn.preprocessing
import sklearn.model_selection
import matplotlib.pyplot as plt

from lazyfca import LazyFCA

from utils import estimate_quality

data = pandas.read_csv("churn.csv")
data = data.drop(columns = ['customerID'])
data = data[data["TotalCharges"] != ' ']
data["TotalCharges"] = data["TotalCharges"].astype(float)

cols_to_replace = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
data[cols_to_replace] = data[cols_to_replace].replace(['No phone service', 'No internet service'], 'No')

X = data.drop(columns = ["Churn"])
# y = data["Churn"].to_numpy()
y = (data["Churn"] == "Yes").to_numpy()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size = 0.1, stratify = y, random_state = 42
)

numeric = [ "tenure", "MonthlyCharges", "TotalCharges" ]
categorical = list(set(X_train.columns) - set(numeric))
ct = sklearn.compose.ColumnTransformer(
    transformers = [
        ("numeric", 'passthrough', numeric),
        ("categorical", sklearn.preprocessing.OneHotEncoder(dtype = 'bool'), categorical)
    ]
)
X_train = pandas.DataFrame(ct.fit_transform(X_train), columns = ct.get_feature_names_out())
X_test = pandas.DataFrame(ct.transform(X_test), columns = ct.get_feature_names_out())

categorical = [ feature for feature in ct.get_feature_names_out() if feature.startswith("categorical__") ]
X_train[categorical] = X_train[categorical].astype(bool)
X_test[categorical] = X_test[categorical].astype(bool)

y_train = pandas.Series(y_train)
y_test = pandas.Series(y_test)


from lazyfca import LazyFCA

classifier = LazyFCA(
    pos_params=LazyFCA.Params(
        supporters_covered=5,
        supporter_opposer_ratio=1 / 2.75,
    ),
    neg_params=LazyFCA.Params(
        supporters_covered=10,
        supporter_opposer_ratio=4,
    ),
    pos_weight=1.0
)
classifier.fit(X_train, y_train)
all_explanations = classifier.explain(X_test)
scores = pandas.concat(map(lambda explanation: explanation.display(), all_explanations), axis = 0).drop_duplicates()

METRICS = [
    'Supporters', 'Opposers', 'Supporters covered', 'Opposers covered',
    'Support', 'Error rate', 'Precision', 'Lift', 'WRAcc',
    'Balanced precision proxy', 'Youden\'s J', 'Matthews correlation',
    "Information gain", "Gini gain", "Log odds ratio", "Chi squared", "G-test"
]


pos_clas_coef_values = [ 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0 ]

for metric in METRICS:
    positive = scores[scores["Type"] == "POSITIVE"][metric]
    positive = numpy.linspace(positive.min(), positive.max(), 10)
    
    negative = scores[scores["Type"] == "NEGATIVE"][metric]
    negative = numpy.linspace(negative.min(), negative.max(), 10)

    attempts = list(itertools.product(positive, negative, pos_clas_coef_values))

    classif = LazyFCA(
        pos_params=LazyFCA.Params(
            supporters_covered=5,
            supporter_opposer_ratio=1 / 2.75,
        ),
        neg_params=LazyFCA.Params(
            supporters_covered=10,
            supporter_opposer_ratio=4,
        ),
        pos_weight=1.0
    )
    classifier.fit(X_train, y_train)
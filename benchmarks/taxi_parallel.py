import os
import sys
from pathlib import Path

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MPLCONFIGDIR = ROOT / "experiments" / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import tqdm.auto as tqdm

PLOTS_DIR = ROOT / "experiments" / "generated_plots" / "taxi_topk_ranking_raw"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

from lazyfca.lazyfca import LazyFCA
from utils.estimate_quality import estimate_quality


import pandas
import sklearn.compose
import sklearn.model_selection
import sklearn.preprocessing

numeric = [
    'passenger_count',
    'trip_distance',
    'fare_amount',
    'extra',
    'mta_tax',
    'tolls_amount',
    'improvement_surcharge',
    'congestion_surcharge',
    'Airport_fee',
    'cbd_congestion_fee'
]

categorical = [
    'VendorID',
    'RatecodeID',
    'store_and_fwd_flag',
    # 'PULocationID',
    # 'DOLocationID',
    'payment_type',
]

target = 'tip_amount'

data = pandas.read_parquet("../datasets/yellow_tripdata_2026-01.parquet")
data = data[~data['passenger_count'].isna()]
data = pandas.concat([
    data[data['tip_amount'] != 0][:5000],
    data[data['tip_amount'] == 0][:5000]
])

y = (data['tip_amount'] != 0).to_numpy()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    data, y, test_size = 0.1, stratify = y, random_state = 42
)

ct = sklearn.compose.ColumnTransformer(
    transformers = [
        ("numeric", 'passthrough', numeric),
        ("categorical", sklearn.preprocessing.OneHotEncoder(dtype = 'bool', sparse_output = False), categorical)
    ]
)
X_train = pandas.DataFrame(ct.fit_transform(X_train), columns = ct.get_feature_names_out())
X_test = pandas.DataFrame(ct.transform(X_test), columns = ct.get_feature_names_out())
X_train

classifier = LazyFCA(rank_by = 'youdens_j', top_k = 2)
classifier.fit(X_train, y_train)

# Baseline threshold-only performance for reference.
baseline_y_pred = classifier.predict(X_test[:300], n_jobs = -1)
print(estimate_quality(baseline_y_pred, y_test[:300]))
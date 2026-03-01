import numpy

import itertools

supporters_covered = [0, 5, 10, 25]
supporter_opposer_ratio = [ 0.1, 0.33, 0.66, 1.0, 2.5, 4.0, 7.5 ]
support = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4]
error_rate = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.4]
precision = [0.5, 0.625, 0.75, 0.875]
lift = [0.5, 0.75, 1.0, 2.0, 2.5]
wracc = [-0.1, -0.05, -0.015, 0.0, 0.015, 0.05, 0.1]
balanced_precision_proxy = [-0.5, -0.25, -0.05, 0.0, 0.05, 0.25, 0.5]
youdens_j = [-0.5, -0.25, -0.05, 0.0, 0.05, 0.25, 0.5]
matthews_correlation = [-0.5, -0.25, -0.05, 0.0, 0.05, 0.25, 0.5]


values = list(itertools.product(
    supporters_covered,
    supporter_opposer_ratio,
    support,
    error_rate,
    precision,
    lift,
    wracc,
    balanced_precision_proxy,
    youdens_j,
    matthews_correlation,
))

print(len(values))

def process_params(min_pos_for_pos_clas, pos_coef_for_pos_clas, min_neg_for_neg_clas, neg_coef_for_neg_clas, pos_clas_coef):
    y_pred = classifier.predict(X_test)
    estimate_quality(y_pred, y_test)


    def is_positive_classifier(matches):
        num_positive, num_negative = matches
        return pos_coef_for_pos_clas * num_positive > num_negative and num_positive >= min_pos_for_pos_clas

    def is_negative_classifier(matches):
        num_positive, num_negative = matches
        return neg_coef_for_neg_clas * num_negative > num_positive and num_negative >= min_neg_for_neg_clas

    def make_pred(classifiers_lists):
        positive_classifiers_list, negative_classifiers_list = classifiers_lists
        positive_classifiers = sum(map(is_positive_classifier, positive_classifiers_list))
        negative_classifiers = sum(map(is_negative_classifier, negative_classifiers_list))

        positive_classifiers *= pos_clas_coef
        total = negative_classifiers + positive_classifiers
        if total == 0:
            return [0.5, 0.5]
        return [ (negative_classifiers / total), (positive_classifiers / total) ]

    with open('classifier_lists.pkl', 'rb') as f:
        classifier_lists = pickle.load(f)
    y_pred = list(map(make_pred, classifier_lists))
    metrics = estimate_quality(numpy.array(y_pred), y_test)
    return {
        'min_pos_for_pos_clas': min_pos_for_pos_clas,
        'pos_coef_for_pos_clas': pos_coef_for_pos_clas,
        'min_neg_for_neg_clas': min_neg_for_neg_clas,
        'neg_coef_for_neg_clas': neg_coef_for_neg_clas,
        'pos_clas_coef': pos_clas_coef,
        **metrics
    }

df_data = joblib.Parallel(n_jobs = -1)(
    joblib.delayed(process_params)(*params)
    for params in tqdm.tqdm(values)
)
df = pandas.DataFrame(df_data)
df.to_csv("selection_results.csv")
df
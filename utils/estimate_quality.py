import typing

import numpy
import sklearn.metrics
import matplotlib.pyplot as plt


def estimate_quality(
    y_pred_proba: numpy.ndarray,
    y_true: numpy.ndarray,
    ax: typing.Optional[plt.Axes] = None,
    label_names: typing.Optional[typing.List[str]] = None,
    confusion_matrix_include_values: bool = True,
) -> dict:
    if label_names is None:
        label_names = list(range(y_pred_proba.shape[1]))

    y_pred = numpy.argmax(y_pred_proba, axis=1)
    if ax:
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            ax=ax,
            colorbar=False,
            display_labels=label_names,
            include_values=confusion_matrix_include_values,
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    numeric_labels = list(range(y_pred_proba.shape[1]))
    auc_roc_input = y_pred_proba[:, 1]
    multi_class = "raise"
    average = "binary"

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    tn_pos, fp_pos, fn_pos, tp_pos = sklearn.metrics.confusion_matrix(y_true[y_true == 1], y_pred[y_true == 1]).ravel()
    tn_neg, fp_neg, fn_neg, tp_neg = sklearn.metrics.confusion_matrix(y_true[y_true == 0], y_pred[y_true == 0]).ravel()
    P, N = numpy.sum(y_true), numpy.sum(y_true == 0)
    metrics = {
        "total":{
            "Accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
            "Precision": sklearn.metrics.precision_score(y_true, y_pred, average=average, zero_division=0),
            "Recall": sklearn.metrics.recall_score(y_true, y_pred, average=average),
            "AUC-ROC": sklearn.metrics.roc_auc_score(y_true, auc_roc_input, multi_class=multi_class, labels=numeric_labels),
            "F1-score": sklearn.metrics.f1_score(y_true, y_pred, average=average),
            "True Positive": tp,
            "True Negative": tn,
            "False Positive": fp,
            "False Negative": fn,
            "True Negative Rate (Specificity)": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "Negative Predictive Value": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "False Positive Rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "False Discovery Rate": fp / (fp + tp) if (fp + tp) > 0 else 0,
            "Balanced precision proxy": tp / P - fp / N,
            "Youden's J": tp / P - fp / N,
            "Matthews correlation": sklearn.metrics.matthews_corrcoef(y_true, y_pred),
        },
        "positive": {
            "Support": tp_pos / P if P > 0 else 0,
            "Contamination": fp_pos / N if N > 0 else 0,
            "Confidence": tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0,
            "Lift": (tp_pos / (tp_pos + fp_pos)) / (P / (P + N)) if (tp_pos + fp_pos) > 0 else 0,
            "WRAcc": ((tp_pos + fp_pos) / (P + N)) * ((tp_pos / (tp_pos + fp_pos)) - (P / (P + N))) if (tp_pos + fp_pos) > 0 else 0,
            "Balanced precision proxy": tp_pos / P - fp_pos / N,
            "Matthews correlation": (tp_pos * tn_neg - fp_pos * fn_neg) / numpy.sqrt((tp_pos + fn_pos) * (tp_pos + fp_pos) * (tn_neg + fp_neg) * (tn_neg + fn_neg)) if numpy.sqrt((tp_pos + fp_pos) * (tp_pos + fp_pos) * (tn_neg + fp_neg) * (tn_neg + fn_neg)) > 0 else 0,
        },
        "negative": {
            "Support": tn_neg / N if N > 0 else 0,
            "Contamination": fn_neg / P if P > 0 else 0,
            "Confidence": tn_neg / (tn_neg + fn_neg) if (tn_neg + fn_neg) > 0 else 0,
            "Lift": (tn_neg / (tn_neg + fn_neg)) / (N / (P + N)) if (tn_neg + fn_neg) > 0 else 0,
            "WRAcc": ((tn_neg + fn_neg) / (P + N)) * ((tn_neg / (tn_neg + fn_neg)) - (N / (P + N))) if (tn_neg + fn_neg) > 0 else 0,
            "Balanced precision proxy": tn_neg / N - fn_neg / P,
            "Matthews correlation": (tp_neg * tn_neg - fn_neg * fp_neg) / numpy.sqrt((tn_neg + fp_neg) * (tn_neg + fn_neg) * (tp_neg + fn_neg) * (tp_neg + fp_neg)) if numpy.sqrt((tn_neg + fp_neg) * (tn_neg + fn_neg) * (tp_neg + fn_neg) * (tp_neg + fp_neg)) > 0 else 0,
        }
    }
    return metrics

import typing

import numpy
import sklearn.metrics
import matplotlib.pyplot as plt


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0


def _binary_metrics(y_true: numpy.ndarray, y_pred: numpy.ndarray, y_pred_proba: numpy.ndarray) -> dict:
    """Single-vector binary metrics (the original behaviour)."""
    numeric_labels = list(range(y_pred_proba.shape[1]))
    auc_roc_input = y_pred_proba[:, 1]
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=numeric_labels).ravel()
    return {
        "Accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "Precision": sklearn.metrics.precision_score(y_true, y_pred, average="binary", zero_division=0),
        "Recall": sklearn.metrics.recall_score(y_true, y_pred, average="binary", zero_division=0),
        "AUC-ROC": sklearn.metrics.roc_auc_score(y_true, auc_roc_input, labels=numeric_labels),
        "F1-score": sklearn.metrics.f1_score(y_true, y_pred, average="binary", zero_division=0),
        "True Positive": int(tp),
        "True Negative": int(tn),
        "False Positive": int(fp),
        "False Negative": int(fn),
        "True Negative Rate (Specificity)": _safe_div(tn, tn + fp),
        "Negative Predictive Value": _safe_div(tn, tn + fn),
        "False Positive Rate": _safe_div(fp, fp + tn),
        "False Discovery Rate": _safe_div(fp, fp + tp),
    }


def _aggregate_multiclass(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    y_pred_proba: numpy.ndarray,
    average: str,
) -> dict:
    """Single set of metrics for the full dataset, multi-class friendly."""
    numeric_labels = list(range(y_pred_proba.shape[1]))
    metrics = {
        "Accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "Precision": sklearn.metrics.precision_score(
            y_true, y_pred, labels=numeric_labels, average=average, zero_division=0
        ),
        "Recall": sklearn.metrics.recall_score(
            y_true, y_pred, labels=numeric_labels, average=average, zero_division=0
        ),
        "F1-score": sklearn.metrics.f1_score(
            y_true, y_pred, labels=numeric_labels, average=average, zero_division=0
        ),
    }
    try:
        metrics["AUC-ROC"] = sklearn.metrics.roc_auc_score(
            y_true, y_pred_proba, labels=numeric_labels, multi_class="ovr", average=average
        )
    except ValueError:
        metrics["AUC-ROC"] = float("nan")
    return metrics


def _per_class_metrics(
    y_true: numpy.ndarray,
    y_pred: numpy.ndarray,
    y_pred_proba: numpy.ndarray,
    label_names: typing.List,
) -> dict:
    """Per-class metrics (one value of each metric per class)."""
    numeric_labels = list(range(y_pred_proba.shape[1]))
    confusion = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=numeric_labels)

    precision = sklearn.metrics.precision_score(
        y_true, y_pred, labels=numeric_labels, average=None, zero_division=0
    )
    recall = sklearn.metrics.recall_score(
        y_true, y_pred, labels=numeric_labels, average=None, zero_division=0
    )
    f1 = sklearn.metrics.f1_score(y_true, y_pred, labels=numeric_labels, average=None, zero_division=0)

    per_class: dict = {}
    overall_total = confusion.sum()
    for i, c in enumerate(numeric_labels):
        tp = int(confusion[i, i])
        fn = int(confusion[i, :].sum() - tp)
        fp = int(confusion[:, i].sum() - tp)
        tn = int(overall_total - tp - fn - fp)

        try:
            auc = sklearn.metrics.roc_auc_score((y_true == c).astype(int), y_pred_proba[:, i])
        except ValueError:
            auc = float("nan")

        per_class[label_names[i]] = {
            "Precision": float(precision[i]),
            "Recall": float(recall[i]),
            "F1-score": float(f1[i]),
            "AUC-ROC": auc,
            "True Positive": tp,
            "False Positive": fp,
            "True Negative": tn,
            "False Negative": fn,
            "True Negative Rate (Specificity)": _safe_div(tn, tn + fp),
            "Negative Predictive Value": _safe_div(tn, tn + fn),
            "False Positive Rate": _safe_div(fp, fp + tn),
            "False Discovery Rate": _safe_div(fp, fp + tp),
        }

    per_class["Accuracy"] = float(sklearn.metrics.accuracy_score(y_true, y_pred))
    return per_class


def estimate_quality(
    y_pred_proba: numpy.ndarray,
    y_true: numpy.ndarray,
    ax: typing.Optional[plt.Axes] = None,
    label_names: typing.Optional[typing.List[str]] = None,
    confusion_matrix_include_values: bool = True,
    per_class: bool = False,
    average: str = "auto",
) -> dict:
    """Compute classification quality metrics.

    Parameters
    ----------
    y_pred_proba:
        ``(n_samples, n_classes)`` array of predicted class probabilities.
    y_true:
        Ground-truth integer labels in ``[0, n_classes)``.
    ax:
        Optional matplotlib axis on which to draw the confusion matrix.
    label_names:
        Optional human-readable class names. Defaults to ``[0, 1, ..., n_classes-1]``.
    confusion_matrix_include_values:
        Whether to annotate the confusion matrix cells with their numeric values.
    per_class:
        If ``False`` (default) a single dictionary of aggregate metrics for the
        whole dataset is returned (one F1-score, one Precision, ...). If
        ``True``, a per-class breakdown is returned: ``{class_name: {metric: value}}``.
    average:
        Averaging strategy used when ``per_class=False``.
        ``"auto"`` (default) selects ``"binary"`` for two classes and ``"macro"``
        for more than two. May also be set to any value accepted by sklearn
        (``"binary"``, ``"micro"``, ``"macro"``, ``"weighted"``).
    """
    n_classes = int(y_pred_proba.shape[1])
    if label_names is None:
        label_names = list(range(n_classes))

    y_pred = numpy.argmax(y_pred_proba, axis=1)
    if ax is not None:
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            ax=ax,
            colorbar=False,
            display_labels=label_names,
            include_values=confusion_matrix_include_values,
            labels=list(range(n_classes)),
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    if per_class:
        return _per_class_metrics(y_true, y_pred, y_pred_proba, label_names)

    if average == "auto":
        average = "binary" if n_classes == 2 else "macro"

    if average == "binary":
        assert n_classes == 2, f"average='binary' requires exactly 2 classes, got {n_classes}"
        return _binary_metrics(y_true, y_pred, y_pred_proba)

    return _aggregate_multiclass(y_true, y_pred, y_pred_proba, average)

"""Scientific metrics used in the current CarbonDefect experiments."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def metrics_from_predictions(y_true, y_pred, probs=None, num_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)

    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    result = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "error_rate": float(1.0 - accuracy),
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "precision_weighted": float(pw),
        "recall_weighted": float(rw),
        "f1_weighted": float(fw),
        "roc_auc_macro_ovr": np.nan,
        "roc_auc_weighted_ovr": np.nan,
        "pr_auc_macro_ovr": np.nan,
        "pr_auc_weighted_ovr": np.nan,
    }

    if probs is not None:
        probs = np.asarray(probs, dtype=float)
        n = int(num_classes or probs.shape[1])

        try:
            result["roc_auc_macro_ovr"] = float(
                roc_auc_score(
                    y_true, probs, multi_class="ovr",
                    average="macro", labels=list(range(n))
                )
            )
            result["roc_auc_weighted_ovr"] = float(
                roc_auc_score(
                    y_true, probs, multi_class="ovr",
                    average="weighted", labels=list(range(n))
                )
            )
        except Exception:
            pass

        try:
            y_bin = label_binarize(y_true, classes=list(range(n)))
            result["pr_auc_macro_ovr"] = float(
                average_precision_score(y_bin, probs, average="macro")
            )
            result["pr_auc_weighted_ovr"] = float(
                average_precision_score(y_bin, probs, average="weighted")
            )
        except Exception:
            pass

    return result


def choose_defect_threshold(y_true, probs, perfect_idx, target_fpr=0.01):
    """Choose defect threshold with a false-positive-rate ceiling."""
    y_bin = (np.asarray(y_true, dtype=int) != int(perfect_idx)).astype(int)
    score = 1.0 - np.asarray(probs, dtype=float)[:, int(perfect_idx)]

    fpr, tpr, thresholds = roc_curve(y_bin, score)
    valid = np.where(fpr <= target_fpr + 1e-12)[0]

    if len(valid):
        best_tpr = np.nanmax(tpr[valid])
        ties = valid[np.where(np.isclose(tpr[valid], best_tpr))[0]]
        finite = [i for i in ties if np.isfinite(thresholds[i])]
        idx = max(finite, key=lambda i: thresholds[i]) if finite else int(ties[0])
    else:
        idx = int(np.nanargmax(tpr - fpr))

    threshold = float(thresholds[idx]) if np.isfinite(thresholds[idx]) else 1.0
    return threshold, {
        "target_fpr": float(target_fpr),
        "observed_fpr": float(fpr[idx]),
        "defect_recall": float(tpr[idx]),
    }


def binary_defect_metrics(y_true, probs, perfect_idx, threshold):
    """Collapse an 8-class experiment to Perfect vs Any Defect."""
    y_bin = (np.asarray(y_true, dtype=int) != int(perfect_idx)).astype(int)
    score = 1.0 - np.asarray(probs, dtype=float)[:, int(perfect_idx)]
    pred = (score >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_bin, pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "binary_accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "defect_precision": float(precision),
        "defect_recall": float(recall),
        "specificity": float(specificity),
        "defect_f1": float(f1),
        "false_positive_rate": float(1.0 - specificity),
        "false_negative_rate": float(1.0 - recall),
        "roc_auc": float(roc_auc_score(y_bin, score)),
        "pr_auc": float(average_precision_score(y_bin, score)),
    }

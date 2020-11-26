import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix


def auc_score(pos_label):
    def ev(y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred[:, pos_label], pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        return roc_auc
    return ev


def full_classification_report(target_names, labels):
    def ev(y_true, y_pred):
        return classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)
    return ev


def get_confusion_matrix(labels):
    def c(y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=labels)
    return c


def _transform_to_report(prefix, data):
    transformed = {}
    for key, value in data.items():
        if isinstance(value, dict):
            res = _transform_to_report("{}_{}".format(prefix, key), value)
            for k, v in res.items():
                transformed[k] = v
            continue
        transformed["{}_{}".format(prefix, key)] = value
    return transformed


def transform_to_grafana(prefix):
    def doit(metrics):
        return _transform_to_report(prefix, metrics)
    return doit

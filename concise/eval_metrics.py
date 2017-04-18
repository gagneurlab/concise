"""Global evaluation metrics. Simular to metrics.py but not restricted
to keras backend (TensorFlow, Theano) implementation. Hence they allow for
metrics that are not batch-limited as auc, f1 etc...

See also https://github.com/fchollet/keras/issues/5794
"""
import numpy as np

from deepcpg.utils import get_from_module
from deepcpg.evaluation import (cor, kendall, mad, rmse, rrmse,
                                auc, acc, tpr, tnr, mcc, f1, cat_acc,
                                CLA_METRICS, REG_METRICS, CAT_METRICS,
                                evaluate, evaluate_cat, unstack_report
                                )


def mse(y_true, y_pred):
    # equivalent to TF representation
    y_diff = np.where(np.isnan(y_pred) | np.isnan(y_true), 0, y_true - y_pred)
    return ((y_diff) ** 2).mean(axis=None)


# exponentiated root-mean-squared error
def ermse(y_true, y_pred):
    return 10**np.sqrt(mse(y_true, y_pred))


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    return 1 - var_resid / var_y_true


def get(name):
    return get_from_module(name, globals())

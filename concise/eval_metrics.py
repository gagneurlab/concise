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


def mse(x, y):
    # equivalent to TF representation
    y_diff = np.where(np.isnan(y) | np.isnan(x), 0, x - y)
    return ((y_diff) ** 2).mean(axis=None)


# exponentiated root-mean-squared error
def ermse(x, y):
    return 10**np.sqrt(mse(x, y))


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    return 1 - var_resid / var_y_true


def get(name):
    return get_from_module(name, globals())

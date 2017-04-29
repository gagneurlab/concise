"""Global evaluation metrics. Simular to metrics.py but not restricted
to keras backend (TensorFlow, Theano) implementation. Hence they allow for
metrics that are not batch-limited as auc, f1 etc...

See also https://github.com/fchollet/keras/issues/5794
"""
import numpy as np
from concise.utils.helper import get_from_module
import sklearn.metrics as skm
from scipy.stats import kendalltau
# TODO - make them equivalent to metrics
# TODO - test them
# TODO - implement NA values


# Binary classification
# ----

def auc(y, z, round=True):
    if round:
        y = y.round()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return skm.roc_auc_score(y, z)


# def auprc(y, z):
#     return skm.average_precision_score(y, z)


def acc(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.accuracy_score(y, z)


def tpr(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.recall_score(y, z)


def tnr(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    c = skm.confusion_matrix(y, z)
    return c[0, 0] / c[0].sum()


def mcc(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.matthews_corrcoef(y, z)


def f1(y, z, round=True):
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.f1_score(y, z)


# Category classification
# ----

def cat_acc(y, z):
    return np.mean(y.argmax(axis=1) == z.argmax(axis=1))


# Regression
# ----

def cor(y, z):
    """Compute Pearson correlation coefficient."""
    return np.corrcoef(y, z)[0, 1]


def kendall(y, z, nb_sample=100000):
    if len(y) > nb_sample:
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        idx = idx[:nb_sample]
        y = y[idx]
        z = z[idx]
    return kendalltau(y, z)[0]


def mad(y, z):
    return np.mean(np.abs(y - z))


def rmse(y, z):
    return np.sqrt(mse(y, z))


def rrmse(y, z):
    return 1 - rmse(y, z)


def mse(y_true, y_pred):
    # equivalent to TF representation
    y_diff = np.where(np.isnan(y_pred) | np.isnan(y_true), 0, y_true - y_pred)
    return ((y_diff) ** 2).mean(axis=None)


def ermse(y_true, y_pred):
    """Exponentiated root-mean-squared error
    """
    return 10**np.sqrt(mse(y_true, y_pred))


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    return 1 - var_resid / var_y_true


# available eval metrics
# ----


BINARY_CLASS = ["auc", "acc", "tpr", "tnr", "f1", "mcc"]
CATEGORY_CLASS = ["cat_acc"]
REGRESSION = ["mse", "mad", "cor", "ermse", "var_explained"]

AVAILABLE = BINARY_CLASS + CATEGORY_CLASS + REGRESSION


def get(name):
    return get_from_module(name, globals())

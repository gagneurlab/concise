"""Global evaluation metrics. Simular to metrics.py but not restricted
to keras backend (TensorFlow, Theano) implementation. Hence they allow for
metrics that are not batch-limited as auc, f1 etc...

See also https://github.com/fchollet/keras/issues/5794
"""
import numpy as np
from concise.utils.helper import get_from_module
import sklearn.metrics as skm
from scipy.stats import kendalltau
from concise.metrics import MASK_VALUE
# TODO - make them equivalent to metrics
# TODO - test them


# Binary classification

def _mask_nan(y, z):
    mask_array = ~np.isnan(y)
    if np.any(np.isnan(z)):
        print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
              format(np.sum(np.isnan(z)), z.size))
        mask_array = np.logical_and(mask_array, ~np.isnan(z))
    return y[mask_array], z[mask_array]


def _mask_value(y, z, mask=MASK_VALUE):
    mask_array = y != mask
    return y[mask_array], z[mask_array]


def _mask_value_nan(y, z, mask=MASK_VALUE):
    y, z = _mask_nan(y, z)
    return _mask_value(y, z, mask)


def auc(y, z, round=True):
    y, z = _mask_value_nan(y, z)

    if round:
        y = y.round()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return skm.roc_auc_score(y, z)


def auprc(y, z):
    y, z = _mask_value_nan(y, z)
    return skm.average_precision_score(y, z)


def accuracy(y, z, round=True):
    y, z = _mask_value_nan(y, z)
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.accuracy_score(y, z)


def tpr(y, z, round=True):
    y, z = _mask_value_nan(y, z)
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.recall_score(y, z)


def tnr(y, z, round=True):
    y, z = _mask_value_nan(y, z)
    if round:
        y = np.round(y)
        z = np.round(z)
    c = skm.confusion_matrix(y, z)
    return c[0, 0] / c[0].sum()


def mcc(y, z, round=True):
    y, z = _mask_value_nan(y, z)
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.matthews_corrcoef(y, z)


def f1(y, z, round=True):
    y, z = _mask_value_nan(y, z)
    if round:
        y = np.round(y)
        z = np.round(z)
    return skm.f1_score(y, z)


# Category classification

def cat_acc(y, z):
    """Categorical accuracy
    """
    return np.mean(y.argmax(axis=1) == z.argmax(axis=1))


# Regression

def cor(y, z):
    """Compute Pearson correlation coefficient.
    """
    y, z = _mask_nan(y, z)
    return np.corrcoef(y, z)[0, 1]


def kendall(y, z, nb_sample=100000):
    y, z = _mask_nan(y, z)
    if len(y) > nb_sample:
        idx = np.arange(len(y))
        np.random.shuffle(idx)
        idx = idx[:nb_sample]
        y = y[idx]
        z = z[idx]
    return kendalltau(y, z)[0]


def mad(y, z):
    y, z = _mask_nan(y, z)
    return np.mean(np.abs(y - z))


def rmse(y, z):
    return np.sqrt(mse(y, z))


def rrmse(y, z):
    return 1 - rmse(y, z)


def mse(y_true, y_pred):
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return ((y_true - y_pred) ** 2).mean(axis=None)


def ermse(y_true, y_pred):
    """Exponentiated root-mean-squared error
    """
    return 10**np.sqrt(mse(y_true, y_pred))


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    return 1 - var_resid / var_y_true


# available eval metrics --------------------------------------------


BINARY_CLASS = ["auc", "auprc", "accuracy", "tpr", "tnr", "f1", "mcc"]
CATEGORY_CLASS = ["cat_acc"]
REGRESSION = ["mse", "mad", "cor", "ermse", "var_explained"]

AVAILABLE = BINARY_CLASS + CATEGORY_CLASS + REGRESSION


def get(name):
    return get_from_module(name, globals())

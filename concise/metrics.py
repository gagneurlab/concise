"""
Loss-metrics
"""
import keras.backend as K
from deepcpg.metrics import (cat_acc, mse, mae, CPG_NAN,
                             _sample_weights, _cat_sample_weights)
from concise.utils.helper import get_from_module
from concise.losses import MASK_VALUE
# TODO - specify NAN - now implicitly taken from deepcpg
# TODO - subset vector with respect to the mask in K
#        - write a wrapper function like in concise.losses.MaskLoss


# TODO - fix this function
def contingency_table(y, z, mask=MASK_VALUE):
    y = K.round(y)
    z = K.round(z)

    weights = _sample_weights(y, mask=mask)

    def count_matches(a, b):
        import pdb
        pdb.set_trace()

        tmp = K.concatenate([a, b])
        return K.sum(K.cast(K.all(tmp, -1), K.floatx()) * weights) / K.sum(weights)

    ones = K.ones_like(y)
    zeros = K.zeros_like(y)
    y_ones = K.equal(y, ones)
    y_zeros = K.equal(y, zeros)
    z_ones = K.equal(z, ones)
    z_zeros = K.equal(z, zeros)

    tp = count_matches(y_ones, z_ones)
    tn = count_matches(y_zeros, z_zeros)
    fp = count_matches(y_zeros, z_ones)
    fn = count_matches(y_ones, z_zeros)

    return (tp, tn, fp, fn)


def prec(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fp)


def tpr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)


def tnr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp)


def fpr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (fp + tn)


def fnr(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return fn / (fn + tp)


def f1(y, z):
    _tpr = tpr(y, z)
    _prec = prec(y, z)
    return 2 * (_prec * _tpr) / (_prec + _tpr)


def mcc(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp * tn - fp * fn) /\
        K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def acc(y, z):
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp + tn) / (tp + tn + fp + fn)


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    var_resid = K.var(y_true - y_pred)
    var_y_true = K.var(y_true)
    return 1 - var_resid / var_y_true


# TODO - fix these metrics by masking them?
AVAILABLE = ["var_explained", "prec", "tpr", "tnr", "fpr",
             "fnr", "f1", "mcc", "acc", "cat_acc", "mse", "mae"]

# TODO - add their masked equivalent
# - subset y_pred, y_true by their masks and pass them to your metrics
# K.gather(reference, indices)?


def get(name):
    return get_from_module(name, globals())

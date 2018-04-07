"""
Loss-metrics
"""
import keras.backend as K
from concise.utils.helper import get_from_module
from concise.losses import MASK_VALUE

# binary classification
# -----

# adopted from:
# https://github.com/cangermueller/deepcpg/blob/master/deepcpg/metrics.py
#
# TODO - add Christophs license here


def contingency_table(y, z):
    """Note:  if y and z are not rounded to 0 or 1, they are ignored
    """
    y = K.cast(K.round(y), K.floatx())
    z = K.cast(K.round(z), K.floatx())

    def count_matches(y, z):
        return K.sum(K.cast(y, K.floatx()) * K.cast(z, K.floatx()))

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


def tpr(y, z):
    """True positive rate `tp / (tp + fn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fn)


def tnr(y, z):
    """True negative rate `tn / (tn + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tn / (tn + fp)


def fpr(y, z):
    """False positive rate `fp / (fp + tn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (fp + tn)


def fnr(y, z):
    """False negative rate `fn / (fn + tp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fn / (fn + tp)


def precision(y, z):
    """Precision `tp / (tp + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return tp / (tp + fp)


def fdr(y, z):
    """False discovery rate `fp / (tp + fp)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return fp / (tp + fp)


def accuracy(y, z):
    """Classification accuracy `(tp + tn) / (tp + tn + fp + fn)`
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp + tn) / (tp + tn + fp + fn)


sensitivity = recall = tpr
specificity = tnr


def f1(y, z):
    """F1 score: `2 * (p * r) / (p + r)`, where p=precision and r=recall.
    """
    _recall = recall(y, z)
    _prec = precision(y, z)
    return 2 * (_prec * _recall) / (_prec + _recall)


def mcc(y, z):
    """Matthews correlation coefficient
    """
    tp, tn, fp, fn = contingency_table(y, z)
    return (tp * tn - fp * fn) / K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


# weights helper

def _sample_weights(y, mask=None):
    if mask is None:
        weights = K.ones_like(y)
    else:
        weights = 1 - K.cast(K.equal(y, mask), K.floatx())
    return weights


def _cat_sample_weights(y, mask=None):
    return 1 - K.cast(K.equal(K.sum(y, axis=-1), 0), K.floatx())

# multi-class classification
# -----


def cat_acc(y, z):
    """Classification accuracy for multi-categorical case
    """
    weights = _cat_sample_weights(y)
    _acc = K.cast(K.equal(K.argmax(y, axis=-1),
                          K.argmax(z, axis=-1)),
                  K.floatx())
    _acc = K.sum(_acc * weights) / K.sum(weights)
    return _acc

# regression
# -----


# def mse(y, z, mask=MASK_VALUE):
#     weights = _sample_weights(y, mask)
#     _mse = K.sum(K.square(y - z) * weights) / K.sum(weights)
#     return _mse


# def mae(y, z, mask=MASK_VALUE):
#     weights = _sample_weights(y, mask)
#     _mae = K.sum(K.abs(y - z) * weights) / K.sum(weights)
#     return _mae


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    var_resid = K.var(y_true - y_pred)
    var_y_true = K.var(y_true)
    return 1 - var_resid / var_y_true

# available metrics
# ----


BINARY_CLASS = ["tpr", "tnr", "fpr", "fnr",
                "precision", "fdr", "recall", "sensitivity", "specificity",
                "f1", "mcc", "accuracy"]
CATEGORY_CLASS = ["cat_acc"]
REGRESSION = ["var_explained"]  # , "mse", "mae"]
AVAILABLE = BINARY_CLASS + CATEGORY_CLASS + REGRESSION

# make sure all metrics have the same __name__ (targeting aliases)
for v in AVAILABLE:
    globals()[v].__name__ = v


def get(name):
    return get_from_module(name, globals())

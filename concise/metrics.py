"""
Loss-metrics
"""
import keras.backend as K
from deepcpg.metrics import (contingency_table, prec,
                             tpr, tnr, fpr, fnr,
                             f1, mcc, acc, cat_acc,
                             mse, mae, CPG_NAN
                             )
from concise.utils.helper import get_from_module

# TODO - save those to the model?

# TODO - specify NAN

# TODO - subset vector with respect to the mask in K


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    var_resid = K.var(y_true - y_pred)
    var_y_true = K.var(y_true)
    return 1 - var_resid / var_y_true


AVAILABLE = ["var_explained", "prec", "tpr", "tnr", "fpr",
             "fnr", "f1", "mcc", "acc", "cat_acc", "mse", "mae"]


def get(name):
    return get_from_module(name, globals())

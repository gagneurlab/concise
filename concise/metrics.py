"""
Loss-metrics
"""
import keras.backend as K


def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    var_resid = K.var(y_true - y_pred)
    var_y_true = K.var(y_true)
    return 1 - var_resid / var_y_true

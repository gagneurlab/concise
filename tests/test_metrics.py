import numpy as np
import concise.metrics as cm
import concise.eval_metrics as cem
from concise.losses import MASK_VALUE
import keras.backend as K
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object, get_custom_objects


def test_metrics():
    expect = [0.0, 2.0, 1.0, 2.0]
    y_pred = np.array([0, 0.2, 0.6, 0.4, 1, 0])
    y_true = np.array([1, 0, -1, 1, 0, 0])
    y_true_mask = y_true[y_true != MASK_VALUE]
    y_pred_mask = y_pred[y_true != MASK_VALUE]

    y_true_r = y_true.reshape((-1, 2))
    y_pred_r = y_pred.reshape((-1, 2))
    y_true_mask_r = y_true_mask.reshape((-1, 1))
    y_pred_mask_r = y_pred_mask.reshape((-1, 1))
    res1 = [K.eval(x) for x in cm.contingency_table(y_true, y_pred)]
    res2 = [K.eval(x) for x in cm.contingency_table(y_true_mask, y_pred_mask)]
    res3 = [K.eval(x) for x in cm.contingency_table(y_true_r, y_pred_r)]
    res4 = [K.eval(x) for x in cm.contingency_table(y_true_mask_r, y_pred_mask_r)]

    assert sum(res1) == 5
    assert sum(res2) == 5
    assert sum(res3) == 5
    assert sum(res4) == 5
    assert res1 == expect
    assert res2 == expect
    assert res3 == expect
    assert res4 == expect

    assert np.allclose(K.eval(cm.tpr(y_true, y_pred)), cem.tpr(y_true, y_pred))
    assert np.allclose(K.eval(cm.accuracy(y_true, y_pred)), cem.accuracy(y_true, y_pred))

    # other metrics
    assert cem.auprc(y_true, y_pred) > 0
    assert cem.recall_at_precision(y_true, y_pred, .5) > 0

    # test serialization
    s = serialize_keras_object(cm.accuracy)
    a = deserialize_keras_object(s)
    assert a == cm.accuracy


def test_classification_metrics():
    y_pred = np.array([0, 0.2, 0.6, 0.4, 1, 0])
    y_true = np.array([1, 0, -1, 1, 0, 0])


def test_regression_metrics():
    y_true = np.arange(10, dtype=float)
    y_pred = np.arange(10, dtype=float)
    y_true[:2] = np.nan

    assert cem.var_explained(y_true, y_pred) == 1
    # can't handle NA values for now
    #
    # assert np.allclose(K.eval(cm.var_explained(K.cast(y_true, K.floatx()),
    #                                            K.cast(y_pred, K.floatx()))),
    #                    cem.var_explained(y_true, y_pred))
    # assert np.allclose(K.eval(cm.mse(K.cast(y_true, K.floatx()),
    #                                  K.cast(y_pred, K.floatx()))),
    #                    cem.mse(y_true, y_pred))

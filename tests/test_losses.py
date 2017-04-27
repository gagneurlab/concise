import keras.losses as kloss
from concise.losses import MaskLoss, MASK_VALUE
import numpy as np
import keras.backend as K
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object, get_custom_objects

def test_MaskLoss():
    l = MaskLoss("binary_crossentropy")
    y_pred = np.array([0, 0.2, 0.6, 0.4, 1])
    y_true = np.array([1, 0, -1, 1, 0.0])

    y_true_mask = K.cast(y_true[y_true != MASK_VALUE], K.floatx())
    y_pred_mask = K.cast(y_pred[y_true != MASK_VALUE], K.floatx())
    y_true_cast = K.cast(y_true, K.floatx())
    y_pred_cast = K.cast(y_pred, K.floatx())

    res = K.eval(l(y_true, y_pred))

    res_mask = K.eval(kloss.binary_crossentropy(y_true_mask, y_pred_mask))

    assert np.allclose(res, res_mask)

    # test serialization
    s = serialize_keras_object(l)
    a = deserialize_keras_object(s, custom_objects={"MaskLoss": MaskLoss})
    assert a.loss == l.loss
    assert a.mask_value == l.mask_value
    res2 = K.eval(a(y_true, y_pred))
    assert np.allclose(res, res2)

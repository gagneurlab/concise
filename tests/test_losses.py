import keras.losses as kloss
import concise.losses as closs
from concise.losses import MASK_VALUE
import numpy as np
import keras.backend as K
import keras.layers as kl
from keras.models import Model, load_model
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object, get_custom_objects

def test_MaskLoss():
    l = closs.binary_crossentropy_masked
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
    a = deserialize_keras_object(s)
    # assert a.loss == l.loss
    # assert a.mask_value == l.mask_value
    res2 = K.eval(a(y_true, y_pred))
    assert np.allclose(res, res2)


def test_ConvDNAQuantitySplines(tmpdir):

    x = np.vstack([np.arange(15), np.arange(15)])
    y = np.arange(2)

    inl = kl.Input((15,))

    o = kl.Dense(1)(inl)
    model = Model(inl, o)
    model.compile("Adam", loss=closs.binary_crossentropy_masked)
    model.fit(x, y)

    filepath = str(tmpdir.mkdir('data').join('test_keras.h5'))

    # load and save the model
    model.save(filepath)
    m = load_model(filepath)
    assert isinstance(m, Model)

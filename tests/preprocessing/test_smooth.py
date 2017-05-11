import pytest
import numpy as np
import keras.backend as K
from keras import layers as kl
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object
from keras.models import Sequential, model_from_json
import concise.layers as cl
import concise.regularizers as cr
from concise.preprocessing.smooth import encodeSplines, _trunc

def test_trunc():
    x = np.arange(10)

    assert np.allclose(_trunc(x, minval=2),
                       np.array([2, 2, 2, 3, 4, 5, 6, 7, 8, 9])
                       )
    assert np.allclose(_trunc(x, maxval=6),
                       np.array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])
                       )
    assert np.allclose(_trunc(x, minval=3, maxval=6),
                       np.array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
                       )

def test_encodeSplines():
    x = np.arange(100)
    x_trunc = np.copy(x)

    assert np.allclose(encodeSplines(x, start=-1, end=50).sum(2), 1)
    assert np.allclose(encodeSplines(x, start=-1, end=120).sum(2), 1)
    assert np.allclose(encodeSplines(x, start=10, end=120).sum(2), 1)
    assert np.allclose(encodeSplines(x).sum(2), 1)


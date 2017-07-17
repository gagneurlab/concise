"""
test layer saving and loading
"""
import pytest
import numpy as np
from concise.preprocessing import EncodeSplines
import keras.layers as kl
import concise.regularizers as cr
import concise.layers as cl
from keras.models import Model, load_model


@pytest.mark.parametrize("shared_weights", (True, False))
@pytest.mark.parametrize("init", ("zeros", "ones"))
@pytest.mark.parametrize("shape", [
    ((100, 3)),
    ((4,)),
    ((100, 101, 102))
])
def test_one(shape, init, shared_weights):
    # shape = (100, 3)
    batch = 11
    n_bases = 10
    x = np.random.uniform(size=(batch, ) + shape)
    x_spl = EncodeSplines(n_bases=10).fit_transform(x)
    inp = kl.Input(shape + (n_bases,))
    out = cl.SplineT(shared_weights=shared_weights, kernel_initializer=init)(inp)
    m = Model(inp, out)
    y = m.predict(x_spl)
    assert y.shape == x.shape
    if init == "zeros":
        assert np.allclose(y, 0)
    elif init == "ones":
        assert np.allclose(y, 1)
    else:
        raise ValueError


def test_serialization(tmpdir):
    shape = (100, 3)
    n_bases = 10
    batch = 32
    inp = kl.Input(shape + (n_bases,))
    x = np.random.uniform(size=(batch, ) + shape)
    x_spl = EncodeSplines(n_bases=10).fit_transform(x)
    out = cl.SplineT(kernel_regularizer=cr.SplineSmoother(l2_smooth=1),
                     kernel_initializer="glorot_uniform")(inp)
    m = Model(inp, out)
    m.compile("Adam", "mse")
    m.fit(x_spl, x)
    assert isinstance(m.get_config(), dict)
    filepath = str(tmpdir.mkdir('data').join('test_keras.h5'))

    # load and save the model
    m.save(filepath)
    m = load_model(filepath)
    assert isinstance(m, Model)

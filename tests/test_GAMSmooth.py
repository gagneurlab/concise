import pytest
import numpy as np
import keras.backend as K
from keras import layers as kl
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object
from keras.models import Sequential, model_from_json
import concise.layers as cl
import concise.regularizers as cr
from concise.preprocessing import encodeSplines


# TODO - compare GAMSmooth with Concise on simulated data
# TODO - compare GAMSmooth with encodeSplines + ConvDNAQuantitySplines:
#          smooth_input = cl.InputDNAQuantitySplines(seq_length, n_bases)
#          x = cl.ConvDNAQuantitySplines(filters=3)(smooth_input)
#          x = kl.Lambda(lambda x: x + 1)(x)


def test_serialization():

    seq_length = 100
    input_shape = (None, seq_length, 4)  # (batch_size, steps, input_dim)
    # input_shape = (seq_length, 4)  # (batch_size, steps, input_dim)

    # output_shape = (None, steps, filters)

    conv_l = kl.Conv1D(filters=15, kernel_size=11,
                       padding="valid",
                       activation="relu",
                       batch_input_shape=input_shape,
                       )

    # output_shape: (batch_size, new_steps, filters)
    # (new_)steps = length along the sequence, might changed due to padding
    model = Sequential()
    model.add(conv_l)
    model.add(cl.GAMSmooth())
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    js = model.to_json()
    js
    # a = model_from_json(js, custom_objects={"Conv1D": kl.Conv1D})
    a = model_from_json(js)
    assert np.all(a.layers[1].get_weights()[0] == 0)

    # check just layer serialization:
    conv_l.build(input_shape)
    s = serialize_keras_object(cl.GAMSmooth())

    a = deserialize_keras_object(s, custom_objects={"Conv1D": kl.Conv1D})
    a.get_config()

    # serialization was successfull
    assert isinstance(a.get_config(), dict)


def test_encodeSplines():

    # check that encodeSplines + conv1d do the same thing as GAMSmooth layer
    start = 0
    end = 100
    seq_length = end
    n_bases = 10
    pos = np.arange(start, end)
    n_features = 1

    posx = np.broadcast_to(pos, (n_features, end - start))
    posx = pos.reshape((1, -1))

    x_spline = encodeSplines(posx, n_bases=n_bases)

    assert x_spline.shape == (n_features, end, n_bases)
    # display the blocks
    x_spline[0, :40]
    x_spline[0, -40:]
    # edges are 0
    assert x_spline[0, 0, -1] == 0.0
    assert x_spline[0, -1, 0] == 0.0

    sm_l = cl.GAMSmooth(n_bases=n_bases)
    sm_l.build(input_shape=(None, end, n_features))

    X_spline = sm_l.X_spline

    assert np.allclose(x_spline[0].sum(axis=1), 1)
    assert np.allclose(X_spline.sum(axis=1), 1)
    assert np.allclose(X_spline, x_spline[0])
    # X_splines are the same

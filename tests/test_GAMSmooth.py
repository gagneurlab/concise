import pytest
from keras import layers as kl
from concise.layers import GAMSmooth
import numpy as np
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object
from keras.models import Sequential, model_from_json

# TODO - compare with Concise


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
    model.add(GAMSmooth())
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    js = model.to_json()
    js
    # a = model_from_json(js, custom_objects={"Conv1D": kl.Conv1D})
    a = model_from_json(js)
    assert np.all(a.layers[1].get_weights()[0] == 1)

    # check just layer serialization:
    conv_l.build(input_shape)
    s = serialize_keras_object(GAMSmooth())

    a = deserialize_keras_object(s, custom_objects={"Conv1D": kl.Conv1D})
    a.get_config()

    # serialization was successfull
    assert isinstance(a.get_config(), dict)



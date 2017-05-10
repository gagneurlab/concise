import pytest
from keras.regularizers import Regularizer, L1L2
from keras import layers as kl
import concise.initializers as ci
from concise.utils import PWM
import numpy as np
from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object, get_custom_objects
from keras.models import Sequential, model_from_json


@pytest.mark.parametrize("kernel_initializer,bias_initializer", [
    (ci.PWMKernelInitializer, ci.PWMBiasInitializer),
    (ci.PSSMKernelInitializer, ci.PSSMBiasInitializer)
])
def test_correct_initialization(kernel_initializer, bias_initializer):
    pwm_list = [PWM(np.array([[1, 2, 3, 4], [2, 4, 4, 5]])),
                PWM(np.array([[1, 2, 1, 4], [2, 10, 4, 5]]))]

    # add no noise
    conv_l = kl.Conv1D(filters=128, kernel_size=11,
                       kernel_regularizer=L1L2(l1=1, l2=1),  # Regularization
                       padding="valid",
                       activation="relu",
                       kernel_initializer=kernel_initializer(pwm_list, stddev=0.1),
                       bias_initializer=bias_initializer(pwm_list, kernel_size=11),
                       )

    seq_length = 100
    input_shape = (None, seq_length, 4)  # (batch_size, steps, input_dim)
    # output_shape: (batch_size, new_steps, filters)
    # (new_)steps = length along the sequence, might changed due to padding

    conv_l.build(input_shape)

    # weights
    w = conv_l.get_weights()[0]
    w
    assert w.shape == (11, 4, 128)  # (kernel_size, 4, filters)

    # bias
    b = conv_l.get_weights()[1]
    b
    assert b.shape == (128,)  # (filters,)

    mean_larger_zero = np.mean(np.sum(np.mean(w, axis=1), axis=0) + b > 0)
    if kernel_initializer == ci.PWMKernelInitializer:
        assert mean_larger_zero > .3
        assert mean_larger_zero < .7
    else:
        assert mean_larger_zero < 0.1


@pytest.mark.parametrize("kernel_initializer,bias_initializer", [
    (ci.PWMKernelInitializer, ci.PWMBiasInitializer),
    (ci.PSSMKernelInitializer, ci.PSSMBiasInitializer)
])
def test_init_serialization(kernel_initializer, bias_initializer):
    pwm_list = [PWM([[1, 2, 3, 4],
                     [2, 4, 4, 5]]),
                PWM([[1, 2, 1, 4],
                     [2, 10, 4, 5]])]

    # should work out of the box
    # get_custom_objects()['PWMKernelInitializer'] = PWMKernelInitializer
    # get_custom_objects()['PWMBiasInitializer'] = PWMBiasInitializer

    seq_length = 100
    input_shape = (None, seq_length, 4)  # (batch_size, steps, input_dim)
    # input_shape = (seq_length, 4)  # (batch_size, steps, input_dim)

    # output_shape = (None, steps, filters)

    conv_l = kl.Conv1D(filters=15, kernel_size=11,
                       kernel_regularizer=L1L2(l1=1, l2=1),  # Regularization
                       padding="valid",
                       activation="relu",
                       kernel_initializer=kernel_initializer(pwm_list, stddev=0.1),
                       bias_initializer=bias_initializer(pwm_list, kernel_size=11),
                       batch_input_shape=input_shape,
                       )

    # output_shape: (batch_size, new_steps, filters)
    # (new_)steps = length along the sequence, might changed due to padding
    model = Sequential()
    model.add(conv_l)
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    js = model.to_json()
    js
    # a = model_from_json(js, custom_objects={"Conv1D": kl.Conv1D})
    a = model_from_json(js)
    assert np.all(a.layers[0].kernel_initializer.pwm_list[0].pwm == pwm_list[0].pwm)

    # check just layer serialization:
    conv_l.build(input_shape)
    s = serialize_keras_object(conv_l)

    a = deserialize_keras_object(s, custom_objects={"Conv1D": kl.Conv1D})

    conv_l.get_config()

    # serialization was successfull
    assert np.all(a.kernel_initializer.pwm_list[0].pwm == pwm_list[0].pwm)


@pytest.mark.parametrize("kernel_initializer,bias_initializer", [
    (ci.PWMKernelInitializer, ci.PWMBiasInitializer),
    (ci.PSSMKernelInitializer, ci.PSSMBiasInitializer)
])
def test_empty_pwm_list(kernel_initializer, bias_initializer):
    pwm_list = []
    pass
    seq_length = 100
    input_shape = (None, seq_length, 4)  # (batch_size, steps, input_dim)
    # input_shape = (seq_length, 4)  # (batch_size, steps, input_dim)

    # output_shape = (None, steps, filters)

    conv_l = kl.Conv1D(filters=15, kernel_size=11,
                       kernel_regularizer=L1L2(l1=1, l2=1),  # Regularization
                       padding="valid",
                       activation="relu",
                       kernel_initializer=kernel_initializer(pwm_list, stddev=0.1),
                       bias_initializer=bias_initializer(pwm_list, kernel_size=11),
                       batch_input_shape=input_shape,
                       )


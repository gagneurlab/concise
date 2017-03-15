import pytest
from keras.regularizers import Regularizer, L1L2
from keras import layers as kl
from concise.initializers import PWMBiasInitializer, PWMKernelInitializer
from concise.utils import PWM
import numpy as np


from keras.utils.generic_utils import deserialize_keras_object, serialize_keras_object

from keras.layers import deserialize as deserialize_layer

def test_correct_initialization():
    pwm_list = [PWM(np.array([[1, 2, 3, 4], [2, 4, 4, 5]])),
                PWM(np.array([[1, 2, 1, 4], [2, 10, 4, 5]]))]

    conv_l = kl.Conv1D(filters=128, kernel_size=11,
                       kernel_regularizer=L1L2(l1=1, l2=1),  # Regularization
                       padding="valid",
                       activation="relu",
                       kernel_initializer=PWMKernelInitializer(pwm_list, stddev=0.1),  # TODO
                       bias_initializer=PWMBiasInitializer(pwm_list)
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
    assert mean_larger_zero > .3
    assert mean_larger_zero < .7


def test_init_serialization(self):
    pwm_list = [PWM([[1, 2, 3, 4],
                     [2, 4, 4, 5]]),
                PWM([[1, 2, 1, 4],
                     [2, 10, 4, 5]])]

    conv_l = kl.Conv1D(filters=4, kernel_size=11,
                       kernel_regularizer=L1L2(l1=1, l2=1),  # Regularization
                       padding="valid",
                       activation="relu",
                       kernel_initializer=PWMKernelInitializer(pwm_list, stddev=0.1),  # TODO
                       bias_initializer=PWMBiasInitializer(pwm_list)
                       )

    seq_length = 100
    input_shape = (None, seq_length, 4)  # (batch_size, steps, input_dim)
    # output_shape: (batch_size, new_steps, filters)
    # (new_)steps = length along the sequence, might changed due to padding

    conv_l.build(input_shape)

    # TODO - serialize the layer
    # TODO - de-serialize the layer

    c = conv_l.get_config()
    # TODO - add other things to
    s = serialize_keras_object(conv_l)

    # TODO - why is this not working
    deserialize_keras_object(s, custom_objects={"PWMKernelInitializer": PWMKernelInitializer,
                                                "PWMBiasInitializer": PWMBiasInitializer,
                                                "Conv1D": kl.Conv1D
                                                })

    # Improper config?

    # TODO - why are there so many pwm_list elements?
    pass


# TODO - save the layer
# TODO - load the layer

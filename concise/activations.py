import keras.backend as K
from concise.utils.helper import get_from_module


def exponential(x):
    return K.exp(x)


AVAILABLE = ["exponential"]


def get(name):
    return get_from_module(name, globals())

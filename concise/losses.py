import numpy as np
import keras.backend as K
import keras.layers as kl
import keras.losses as kloss
from concise.utils.helper import get_from_module
MASK_VALUE = -1


class MaskLoss:

    def __init__(self, loss, mask_value=MASK_VALUE):
        """
        Compile masked loss function
        This function ignores values where y_true == mask_value.

        Arguments:
            loss = loss function from keras.losses
            mask_value = numeric value to be masked away (np.nan not supported for now)

        Inspired by: https://github.com/fchollet/keras/issues/3893
        """
        self.loss = kloss.deserialize(loss)  # TODO - add the ability to create your own loss functions
        self.mask_value = mask_value

    def __call__(self, y_true, y_pred):
        # currently not suppoerd with NA's:
        #  - there is no K.is_nan impolementation in keras.backend
        #  - https://github.com/fchollet/keras/issues/1628
        mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())

        # we divide by the mean to correct for the number of done loss evaluations
        return self.loss(y_true * mask, y_pred * mask) / K.mean(mask)

    def get_config(self):
        return {"loss": kloss.serialize(self.loss),
                "mask_value": self.mask_value
                }


AVAILABLE = ["MaskLoss"]


def get(name):
    return get_from_module(name, globals())

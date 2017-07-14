import numpy as np
import keras.backend as K
import keras.layers as kl
import keras.losses as kloss
from concise.utils.helper import get_from_module
MASK_VALUE = -1


def mask_loss(loss, mask_value=MASK_VALUE):
    """Generates a new loss function that ignores values where `y_true == mask_value`.

    # Arguments
        loss: str; name of the keras loss function from `keras.losses`
        mask_value: int; which values should be masked

    # Returns
        function; Masked version of the `loss`

    # Example
        ```python
                categorical_crossentropy_masked = mask_loss("categorical_crossentropy")
        ```
    """
    loss_fn = kloss.deserialize(loss)

    def masked_loss_fn(y_true, y_pred):
        # currently not suppoerd with NA's:
        #  - there is no K.is_nan impolementation in keras.backend
        #  - https://github.com/fchollet/keras/issues/1628
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())

        # we divide by the mean to correct for the number of done loss evaluations
        return loss_fn(y_true * mask, y_pred * mask) / K.mean(mask)

    masked_loss_fn.__name__ = loss + "_masked"
    return masked_loss_fn


# Bellow would be the most general case and wouldn't reqire hard-coding the values
# However, it doesn't work with the classes as the current serialization is with .__name__
# and not with serialize_keras_object
# https://github.com/fchollet/keras/blob/master/keras/metrics.py#L48
#
# class MaskLoss:

#     __name__ = "MaskLoss"

#     def __init__(self, loss, mask_value=MASK_VALUE):
#         """
#         Compile masked loss function
#         This function ignores values where y_true == mask_value.

#         Arguments:
#             loss = loss function from keras.losses
#             mask_value = numeric value to be masked away (np.nan not supported for now)

#         Inspired by: https://github.com/fchollet/keras/issues/3893
#         """
#         self.loss = kloss.deserialize(loss)  # TODO - add the ability to create your own loss functions
#         self.mask_value = mask_value

#     def __call__(self, y_true, y_pred):
#         # currently not suppoerd with NA's:
#         #  - there is no K.is_nan impolementation in keras.backend
#         #  - https://github.com/fchollet/keras/issues/1628
#         mask = K.cast(K.not_equal(y_true, self.mask_value), K.floatx())

#         # we divide by the mean to correct for the number of done loss evaluations
#         return self.loss(y_true * mask, y_pred * mask) / K.mean(mask)

#     def get_config(self):
#         return {"loss": kloss.serialize(self.loss),
#                 "mask_value": self.mask_value
#                 }


# masked loss functions
AVAILABLE = [  # "mean_squared_error_masked",
    # "mean_absolute_error_masked",
    # "mean_absolute_percentage_error_masked",
    # "mean_squared_logarithmic_error_masked",
    # "squared_hinge_masked",
    # "hinge_masked",
    "categorical_crossentropy_masked",
    "sparse_categorical_crossentropy_masked",
    "binary_crossentropy_masked",
    "kullback_leibler_divergence_masked"]

# NOTE - name has to be <loss>_mask
# TODO - take care of which masking value you are using
#         - use nan for numeric values
# mean_squared_error_masked = mask_loss("mean_squared_error")
# mean_absolute_error_masked = mask_loss("mean_absolute_error")
# mean_absolute_percentage_error_masked = mask_loss("mean_absolute_percentage_error")
# mean_squared_logarithmic_error_masked = mask_loss("mean_squared_logarithmic_error")
# squared_hinge_masked = mask_loss("squared_hinge")
# hinge_masked = mask_loss("hinge")
categorical_crossentropy_masked = mask_loss("categorical_crossentropy")
sparse_categorical_crossentropy_masked = mask_loss("sparse_categorical_crossentropy")
binary_crossentropy_masked = mask_loss("binary_crossentropy")
kullback_leibler_divergence_masked = mask_loss("kullback_leibler_divergence")


def get(name):
    try:
        return kloss.get(name)
    except ValueError:
        return get_from_module(name, globals())
